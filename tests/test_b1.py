"""Tests for Battery 1 — business task execution."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from llmtest.batteries.b1_business import B1Business
from llmtest.batteries import WorkItem
from llmtest.batteries.b1_fixtures import load_unit_tasks
from llmtest import schema

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    """Fake store for plan() tests."""
    def iter_rows(self):
        return []


def _cybersecurity_task():
    """Load the real exemplar task via the loader (source of truth for fixture_sha)."""
    tasks = load_unit_tasks(ROOT, "cybersecurity")
    return next(t for t in tasks if t.id == "cybersecurity-01")


def _make_exec_item(cfg, task, run_n=1):
    """Build a WorkItem matching what plan() now produces (per-task fixture_sha,
    prompt/signals/cls riding in payload so execute() doesn't re-load YAML)."""
    condition = "runtime=fork;spec=ngram32;kv=q8;ctx=16k;cond=B1"
    suite_version = "suite-v2.0.0-shakedown"
    row_id = schema.compute_row_id(
        suite_version=suite_version,
        model_id="gpt-oss-20b",
        quant_sha256=cfg.registry["models"]["gpt-oss-20b"]["provenance"]["sha256"],
        battery=1,
        task_id="b1.cybersecurity-01",
        fixture_sha=task.fixture_sha,
        condition=condition,
        run_n=run_n
    )
    return WorkItem(
        row_id=row_id,
        model_id="gpt-oss-20b",
        battery=1,
        task_id="b1.cybersecurity-01",
        condition=condition,
        run_n=run_n,
        payload={"model": cfg.registry["models"]["gpt-oss-20b"],
                 "task_id": task.id,
                 "fixture_sha": task.fixture_sha,
                 "suite_version": suite_version,
                 "prompt": task.prompt,
                 "signals": task.signals,
                 "cls": task.cls}
    )


def test_plan_covers_11_models_excluding_quant_arm(tmp_path):
    """plan() excludes the ONE model with role=quant-arm, covers 11 models × 1 exemplar task × 3 runs."""
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    # Verify registry has 12 models
    models = cfg.registry["models"]
    assert len(models) == 12

    # Verify exactly ONE has role=quant-arm
    quant_arm_models = [mid for mid, m in models.items() if m.get("role") == "quant-arm"]
    assert len(quant_arm_models) == 1
    assert quant_arm_models[0] == "gemma-4-26b-a4b-mxfp4"

    # plan() should exclude quant-arm models
    store = FakeStore()
    b1 = B1Business()
    items = b1.plan(cfg, store)

    # 11 models × 120 tasks (cybersecurity-01..08, it_infra-01..08, helpdesk-01..08,
    # knowledge_mgmt-01..08, coding-01..08, finance-01..08, operations-01..08,
    # data_analytics-01..08, project_mgmt-01..08, marketing-01..08, seo-01..08,
    # sales-01..08, outreach-01..08, legal_compliance-01..08, hr_people_ops-01..08) × 3 runs
    # = 3960 items
    assert len(items) == 3960

    # Verify quant-arm model is excluded
    model_ids = {item.model_id for item in items}
    assert "gemma-4-26b-a4b-mxfp4" not in model_ids
    assert len(model_ids) == 11

    # Verify all items have correct battery, task_id prefix
    for item in items:
        assert item.battery == 1
        assert item.task_id.startswith("b1.")

    # Verify canonical condition string is canonical
    order = cfg.suite["condition_order"]
    for item in items:
        # Should parse and round-trip correctly
        parts = dict(p.split("=") for p in item.condition.split(";"))
        canonical = schema.canonical_condition(parts, order)
        assert item.condition == canonical


def test_execute_produces_judging_row_with_artifact(tmp_path, monkeypatch):
    """execute() produces a row with needs_judging=True, det_checks pass, artifact saved."""
    from llmtest.registry import load_config
    from llmtest.batteries.b1_business import B1Business

    cfg = load_config(ROOT)

    # Create stub endpoint handle
    class StubHandle:
        session_id = "s-stub"
        normalized_config = {}
        def chat(self, messages, **kwargs):
            return {
                "choices": [{"message": {"content": "Enable MFA. CVE-2024-3400 patched. $4,200."}}],
                "timings": {"predicted_n": 50, "predicted_per_second": 100.0}
            }

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    # Mock artifacts root
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()

    ctx = SimpleNamespace(
        cfg=cfg,
        server_manager=lambda: StubMgr(),
        root=tmp_path
    )

    # Create a WorkItem for the exemplar task, using the loader's real per-task
    # fixture_sha (content hash of the task YAML bytes) — NOT a hand-built
    # unit-name hash. prompt/signals/cls ride in payload like plan() now does.
    task = _cybersecurity_task()
    item = _make_exec_item(cfg, task, run_n=1)

    b1 = B1Business()
    rows = b1.execute(item, ctx)

    assert len(rows) == 1
    row = rows[0]

    # Verify row_id and fixture_sha are exactly what plan() computed/carried —
    # execute() must not recompute fixture_sha from the unit name.
    assert row["row_id"] == item.row_id
    assert row["fixture_sha"] == task.fixture_sha

    # Verify row structure
    assert row["needs_judging"] is True
    assert row["status"] == "ok"
    assert row["battery"] == 1
    assert row["task_id"] == "b1.cybersecurity-01"

    # Verify det_checks passed for all signals
    assert "contains-0" in row["det_checks"]
    assert "regex-1" in row["det_checks"]
    assert "numeric-2" in row["det_checks"]
    assert all(row["det_checks"][k]["pass"] for k in row["det_checks"])

    # Verify metrics recorded
    assert "chars" in row["metrics"]
    assert row["metrics"]["chars"] == len("Enable MFA. CVE-2024-3400 patched. $4,200.")

    # Verify sampling records max_tokens (temperature omitted from request)
    assert row["sampling"].get("max_tokens") == 900  # short class max_tokens

    # Verify artifact was written under the canonical "response" key
    # (TESTPLAN shape; Finding 2 of the P3 Task 5 review).
    assert "response" in row["artifacts"]
    artifact_info = row["artifacts"]["response"]
    assert "sha256" in artifact_info
    assert "relpath" in artifact_info


def test_preflight_missing_unit_returns_error_row(tmp_path):
    """preflight() with a missing unit dir returns an error selftest row for that unit."""
    from llmtest.registry import load_config
    from llmtest.batteries.b1_business import B1Business

    cfg = load_config(ROOT)

    # Build a minimal context with a tree missing most unit dirs
    missing_unit_tree = tmp_path / "missing_units"
    missing_unit_tree.mkdir()

    # Copy only the cybersecurity unit
    src_unit = ROOT / "suite" / "b1_business" / "cybersecurity"
    dst_unit = missing_unit_tree / "suite" / "b1_business" / "cybersecurity"
    dst_unit.mkdir(parents=True)
    for f in src_unit.glob("*.yaml"):
        (dst_unit / f.name).write_bytes(f.read_bytes())

    ctx = SimpleNamespace(cfg=cfg, root=missing_unit_tree)

    b1 = B1Business()
    rows = b1.preflight(ctx)

    # Should have 15 selftest rows (one per unit in b1.units_tier1)
    assert len(rows) == 15

    # cybersecurity should have status="ok"
    cybersecurity_rows = [r for r in rows if r["task_id"] == "b1.selftest.cybersecurity"]
    assert len(cybersecurity_rows) == 1
    assert cybersecurity_rows[0]["status"] == "ok"

    # All other units should have status="error"
    error_rows = [r for r in rows if r["status"] == "error"]
    assert len(error_rows) == 14

    for row in rows:
        assert row["tags"] == ["selftest"]
        assert row["condition"] == "cond=SELFTEST"
        assert row["run_n"] == 1
        assert row["battery"] == 1


def test_plan_force_bumps_run_n_condition_scoped(tmp_path):
    """--force plans exactly ONE new item per (model, task), at
    run_n = max(existing run_n for that (model_id, task_id, condition)) + 1.
    A same-model/same-task row under a DIFFERENT condition must NOT influence
    the bump (condition-scoped, not just model+task scoped)."""
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    model_id = "gpt-oss-20b"
    task_id = "b1.cybersecurity-01"
    target_condition = "runtime=fork;spec=ngram32;kv=q8;ctx=16k;cond=B1"
    other_condition = "runtime=fork;spec=ngram32;kv=q8;ctx=16k;cond=OTHER"

    seeded_rows = [
        {"model_id": model_id, "task_id": task_id, "condition": target_condition,
         "run_n": 1, "row_id": "seed-run1"},
        {"model_id": model_id, "task_id": task_id, "condition": target_condition,
         "run_n": 2, "row_id": "seed-run2"},
        # Same model+task, DIFFERENT condition, much higher run_n — must be ignored.
        {"model_id": model_id, "task_id": task_id, "condition": other_condition,
         "run_n": 7, "row_id": "seed-run7"},
    ]

    class SeededStore:
        def iter_rows(self):
            return seeded_rows

    b1 = B1Business()
    items = b1.plan(cfg, SeededStore(), model_filter=model_id, force=True)

    matching = [it for it in items if it.model_id == model_id and it.task_id == task_id]
    assert len(matching) == 1  # exactly ONE new item per (model, task) when forced

    item = matching[0]
    assert item.run_n == 3  # max(1, 2) + 1 for the matching condition — NOT 8
    assert item.row_id not in {"seed-run1", "seed-run2", "seed-run7"}


def test_sampling_records_runtime_default_temp(tmp_path):
    """sampling must record that temperature was omitted (runtime default), not silently drop it."""
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    class StubHandle:
        session_id = "s-stub"
        normalized_config = {}
        def chat(self, messages, **kwargs):
            return {
                "choices": [{"message": {"content": "Enable MFA. CVE-2024-3400 patched. $4,200."}}],
                "timings": {}
            }

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: StubMgr(), root=tmp_path)

    task = _cybersecurity_task()
    item = _make_exec_item(cfg, task, run_n=1)

    rows = B1Business().execute(item, ctx)
    row = rows[0]

    assert row["sampling"] == {"temp": "runtime-default", "max_tokens": 900}
