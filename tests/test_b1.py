"""Tests for Battery 1 — business task execution."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from llmtest.batteries.b1_business import B1Business
from llmtest.batteries import WorkItem
from llmtest import schema

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    """Fake store for plan() tests."""
    def iter_rows(self):
        return []


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

    # 11 models × 1 task (cybersecurity-01) × 3 runs = 33 items
    assert len(items) == 33

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
                "choices": [{"message": {"content": "Enable MFA. CVE-2026-1234. $4,200."}}],
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

    # Create a WorkItem for the exemplar task
    task_sha = hashlib.sha256(b"cybersecurity").hexdigest()
    row_id = schema.compute_row_id(
        suite_version="suite-v2.0.0-shakedown",
        model_id="gpt-oss-20b",
        quant_sha256="4e4f9cd88d6456e4f389e7262eca4a8d565211e2b22ece9ca7a8556168ff3c66",
        battery=1,
        task_id="b1.cybersecurity-01",
        fixture_sha=task_sha,
        condition="runtime=fork;spec=ngram32;kv=q8;ctx=16k;cond=B1",
        run_n=1
    )

    item = WorkItem(
        row_id=row_id,
        model_id="gpt-oss-20b",
        battery=1,
        task_id="b1.cybersecurity-01",
        condition="runtime=fork;spec=ngram32;kv=q8;ctx=16k;cond=B1",
        run_n=1,
        payload={"model": cfg.registry["models"]["gpt-oss-20b"],
                 "task_id": "cybersecurity-01",
                 "fixture_sha": task_sha,
                 "suite_version": "suite-v2.0.0-shakedown"}
    )

    b1 = B1Business()
    rows = b1.execute(item, ctx)

    assert len(rows) == 1
    row = rows[0]

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
    assert row["metrics"]["chars"] == len("Enable MFA. CVE-2026-1234. $4,200.")

    # Verify sampling records max_tokens (temperature omitted from request)
    assert row["sampling"].get("max_tokens") == 900  # short class max_tokens

    # Verify artifact was written
    assert "b1" in row["artifacts"]
    artifact_info = row["artifacts"]["b1"]
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
