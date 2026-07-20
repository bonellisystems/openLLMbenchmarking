"""Tests for Battery 3 — hallucination curve."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtest.batteries.b3_hallucination import B3Hallucination
from llmtest.batteries import WorkItem
from llmtest.batteries.b3_fixtures import load_tasks, score_hallucination
from llmtest import schema

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    """Fake store for plan() tests."""
    def iter_rows(self):
        return []


def _hedge_task():
    """Load a real 'hedge' exemplar task via the loader (source of truth for fixture_sha)."""
    tasks = load_tasks(ROOT)
    return next(t for t in tasks if t.id == "hallucination-03")


def _answer_task():
    """Load a real 'answer' exemplar task via the loader."""
    tasks = load_tasks(ROOT)
    return next(t for t in tasks if t.id == "hallucination-01")


def _multi_turn_task():
    tasks = load_tasks(ROOT)
    return next(t for t in tasks if t.id == "hallucination-13")


def _make_exec_item(cfg, task, run_n=1, model_id="gpt-oss-20b"):
    """Build a WorkItem matching what plan() produces for a given task."""
    condition = "runtime=fork;spec=ngram32;kv=q8;ctx=32k;cond=B3"
    suite_version = "suite-v2.0.0-shakedown"
    row_id = schema.compute_row_id(
        suite_version=suite_version,
        model_id=model_id,
        quant_sha256=cfg.registry["models"][model_id]["provenance"]["sha256"],
        battery=3,
        task_id=f"b3.{task.id}",
        fixture_sha=task.fixture_sha,
        condition=condition,
        run_n=run_n
    )
    return WorkItem(
        row_id=row_id,
        model_id=model_id,
        battery=3,
        task_id=f"b3.{task.id}",
        condition=condition,
        run_n=run_n,
        payload={"model": cfg.registry["models"][model_id],
                 "task_id": task.id,
                 "fixture_sha": task.fixture_sha,
                 "suite_version": suite_version,
                 "turns": task.turns,
                 "expect": task.expect,
                 "cls": task.cls,
                 "category": task.category,
                 "difficulty": task.difficulty,
                 "hedge_signals": task.hedge_signals,
                 "trap_signals": task.trap_signals,
                 "answer_signals": task.answer_signals}
    )


# --- fixture loader tests -----------------------------------------------

def test_load_tasks_returns_13_tasks_sorted():
    tasks = load_tasks(ROOT)
    assert len(tasks) == 13
    ids = [t.id for t in tasks]
    assert ids == sorted(ids)
    assert ids[0] == "hallucination-01"
    assert ids[-1] == "hallucination-13"


def test_load_tasks_categories_cover_configured_vocab():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    tasks = load_tasks(ROOT)
    seen = {t.category for t in tasks}
    assert seen == set(cfg.suite["b3"]["categories"])


def test_load_tasks_difficulty_tiers_present():
    tasks = load_tasks(ROOT)
    diffs = {t.difficulty for t in tasks}
    assert diffs == {"easy", "medium", "hard"}


def test_load_tasks_malformed_fixture_raises_loud(tmp_path):
    """A malformed fixture (missing required key) must raise ValueError, never
    be silently skipped."""
    bad_dir = tmp_path / "suite" / "b3_hallucination"
    bad_dir.mkdir(parents=True)
    (bad_dir / "task-01.yaml").write_text(
        "id: hallucination-01\ncategory: abstention_unanswerable\n"
        "difficulty: easy\nclass: short\n"
        # industry and expect deliberately omitted
        "prompt: test\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_tasks(tmp_path)


def test_load_tasks_requires_prompt_xor_turns(tmp_path):
    bad_dir = tmp_path / "suite" / "b3_hallucination"
    bad_dir.mkdir(parents=True)
    (bad_dir / "task-01.yaml").write_text(
        "id: hallucination-01\ncategory: abstention_unanswerable\n"
        "difficulty: easy\nclass: short\nindustry: generic_smb\nexpect: hedge\n"
        "prompt: test\nturns: [a, b]\n"
        "trap_signals: [{type: contains, value: x}]\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one of"):
        load_tasks(tmp_path)


def test_load_tasks_hedge_expect_requires_trap_signals(tmp_path):
    bad_dir = tmp_path / "suite" / "b3_hallucination"
    bad_dir.mkdir(parents=True)
    (bad_dir / "task-01.yaml").write_text(
        "id: hallucination-01\ncategory: abstention_unanswerable\n"
        "difficulty: easy\nclass: short\nindustry: generic_smb\nexpect: hedge\n"
        "prompt: test\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="trap_signals"):
        load_tasks(tmp_path)


def test_load_tasks_missing_dir_returns_empty(tmp_path):
    assert load_tasks(tmp_path) == []


# --- validate lint tests --------------------------------------------------

def test_validate_clean_on_real_repo():
    """The 13 authored fixtures must pass the same lint CI runs."""
    from llmtest.validate_cmd import run_validate
    assert run_validate(ROOT) == 0


def test_fixture_lint_catches_bad_category(tmp_path, capsys, monkeypatch):
    import shutil
    import yaml as _yaml
    from llmtest.validate_cmd import run_validate

    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "TESTPLAN.md").write_text("# Test Plan\n", encoding="utf-8")
    b3_dir = tmp_path / "suite" / "b3_hallucination"
    b3_dir.mkdir(parents=True)
    (b3_dir / "task-01.yaml").write_text(
        "id: hallucination-01\ncategory: not_a_real_category\n"
        "difficulty: easy\nclass: short\nindustry: generic_smb\nexpect: hedge\n"
        "prompt: test\ntrap_signals: [{type: contains, value: x}]\n",
        encoding="utf-8")

    exit_code = run_validate(tmp_path)
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "task-01.yaml" in out and "category" in out


# --- scoring tests ---------------------------------------------------------

def test_score_hallucination_hedge_correct_vs_fabricated():
    task = _hedge_task()
    correct_text = "I don't have access to the ticketing system right now, so I can't give you an exact count."
    fabricated_text = "Sure! Your helpdesk closed 47 tickets last Tuesday."

    good = score_hallucination(correct_text, task)
    bad = score_hallucination(fabricated_text, task)

    assert good["hedged"]["pass"] is True
    assert good["fabricated"]["pass"] is False
    assert good["correct"]["pass"] is True

    assert bad["hedged"]["pass"] is False
    assert bad["fabricated"]["pass"] is True
    assert bad["correct"]["pass"] is False


def test_score_hallucination_answer_correct_vs_wrong():
    task = _answer_task()
    correct_text = "The standard HTTPS port is 443."
    wrong_confident_text = "The standard HTTPS port is 8080."

    good = score_hallucination(correct_text, task)
    bad = score_hallucination(wrong_confident_text, task)

    assert good["correct"]["pass"] is True
    assert good["fabricated"]["pass"] is False

    assert bad["correct"]["pass"] is False
    # Confidently wrong on a KNOWABLE fact, no hedge -> counted as fabricated.
    assert bad["fabricated"]["pass"] is True


def test_score_hallucination_answer_hedge_is_not_fabrication():
    """Over-caution on a knowable fact is scored incorrect, but NOT double
    counted as fabrication (that would conflate two different failure modes)."""
    task = _answer_task()
    hedged_text = "I'm not sure, sorry."
    result = score_hallucination(hedged_text, task)
    assert result["correct"]["pass"] is False
    assert result["fabricated"]["pass"] is False
    assert result["hedged"]["pass"] is True


# --- plan() tests -----------------------------------------------------------

def test_plan_covers_11_models_excluding_quant_arm():
    """plan() excludes every model with role=quant-arm, covers the rest of the
    roster x 13 tasks x 3 runs. Roster size is read from the registry itself
    (not hardcoded) so this survives roster growth."""
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    non_quant_arm_models = [mid for mid, m in cfg.registry["models"].items()
                             if m.get("role") != "quant-arm"]

    store = FakeStore()
    b3 = B3Hallucination()
    items = b3.plan(cfg, store)

    assert len(items) == len(non_quant_arm_models) * 13 * 3

    model_ids = {item.model_id for item in items}
    assert "gemma-4-26b-a4b-mxfp4" not in model_ids
    assert len(model_ids) == len(non_quant_arm_models)

    for item in items:
        assert item.battery == 3
        assert item.task_id.startswith("b3.")

    order = cfg.suite["condition_order"]
    for item in items:
        parts = dict(p.split("=") for p in item.condition.split(";"))
        canonical = schema.canonical_condition(parts, order)
        assert item.condition == canonical
        assert parts["cond"] == "B3"


def test_plan_force_bumps_run_n_condition_scoped():
    """--force plans exactly ONE new item per (model, task), at
    run_n = max(existing run_n for that (model_id, task_id, condition)) + 1."""
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    model_id = "gpt-oss-20b"
    task_id = "b3.hallucination-01"
    target_condition = "runtime=fork;spec=ngram32;kv=q8;ctx=32k;cond=B3"
    other_condition = "runtime=fork;spec=ngram32;kv=q8;ctx=32k;cond=OTHER"

    seeded_rows = [
        {"model_id": model_id, "task_id": task_id, "condition": target_condition,
         "run_n": 1, "row_id": "seed-run1"},
        {"model_id": model_id, "task_id": task_id, "condition": target_condition,
         "run_n": 2, "row_id": "seed-run2"},
        {"model_id": model_id, "task_id": task_id, "condition": other_condition,
         "run_n": 9, "row_id": "seed-run9"},
    ]

    class SeededStore:
        def iter_rows(self):
            return seeded_rows

    b3 = B3Hallucination()
    items = b3.plan(cfg, SeededStore(), model_filter=model_id, force=True)

    matching = [it for it in items if it.model_id == model_id and it.task_id == task_id]
    assert len(matching) == 1

    item = matching[0]
    assert item.run_n == 3
    assert item.row_id not in {"seed-run1", "seed-run2", "seed-run9"}


# --- preflight() tests -------------------------------------------------------

def test_preflight_all_categories_ok_on_real_repo():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    ctx = SimpleNamespace(cfg=cfg, root=ROOT)

    b3 = B3Hallucination()
    rows = b3.preflight(ctx)

    assert len(rows) == len(cfg.suite["b3"]["categories"])
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["tags"] == ["selftest"] for r in rows)
    assert all(r["condition"] == "cond=SELFTEST" for r in rows)
    assert all(r["battery"] == 3 for r in rows)


def test_preflight_missing_category_returns_error_row(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    # Tree with only ONE b3 fixture (category closed_domain_control) --
    # every other configured category should error.
    dst = tmp_path / "suite" / "b3_hallucination"
    dst.mkdir(parents=True)
    src_task = ROOT / "suite" / "b3_hallucination" / "task-01.yaml"
    (dst / "task-01.yaml").write_bytes(src_task.read_bytes())

    ctx = SimpleNamespace(cfg=cfg, root=tmp_path)
    b3 = B3Hallucination()
    rows = b3.preflight(ctx)

    assert len(rows) == len(cfg.suite["b3"]["categories"])
    ok_rows = [r for r in rows if r["status"] == "ok"]
    error_rows = [r for r in rows if r["status"] == "error"]
    assert len(ok_rows) == 1
    assert ok_rows[0]["task_id"] == "b3.selftest.closed_domain_control"
    assert len(error_rows) == len(cfg.suite["b3"]["categories"]) - 1


# --- execute() tests ----------------------------------------------------------

class _StubHandle:
    def __init__(self, text, timings=None):
        self.session_id = "s-stub"
        self.normalized_config = {}
        self._text = text
        self._timings = timings or {"predicted_n": 20, "predicted_per_second": 90.0}

    def chat(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self._text}}],
                "timings": self._timings}


class _StubMgr:
    def __init__(self, handle):
        self._handle = handle

    def request_endpoint(self, *a, **k):
        return self._handle


def test_execute_hedge_task_correct_response_scores_correct(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    handle = _StubHandle("I don't have access to the ticketing system right now, "
                          "so I can't give you an exact count.")
    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: _StubMgr(handle), root=tmp_path)

    task = _hedge_task()
    item = _make_exec_item(cfg, task)

    rows = B3Hallucination().execute(item, ctx)
    assert len(rows) == 1
    row = rows[0]

    assert row["row_id"] == item.row_id
    assert row["fixture_sha"] == task.fixture_sha
    assert row["needs_judging"] is False
    assert row["status"] == "ok"
    assert row["battery"] == 3
    assert row["det_checks"]["correct"]["pass"] is True
    assert row["det_checks"]["fabricated"]["pass"] is False
    assert row["metrics"]["correct"] is True
    assert row["metrics"]["fabricated"] is False
    assert row["metrics"]["category"] == "abstention_unanswerable"
    assert row["metrics"]["difficulty"] == "easy"
    assert "response" in row["artifacts"]


def test_execute_hedge_task_fabricating_response_scores_fabricated(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    handle = _StubHandle("Sure! Your helpdesk closed 47 tickets last Tuesday.")
    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: _StubMgr(handle), root=tmp_path)

    task = _hedge_task()
    item = _make_exec_item(cfg, task)

    rows = B3Hallucination().execute(item, ctx)
    row = rows[0]

    assert row["det_checks"]["fabricated"]["pass"] is True
    assert row["det_checks"]["correct"]["pass"] is False
    assert row["metrics"]["fabricated"] is True

    # Artifact transcript actually saved to disk under artifacts/b3/.
    artifact_relpath = row["artifacts"]["response"]["relpath"]
    assert artifact_relpath == f"b3/{row['row_id']}.txt"
    assert (tmp_path / "artifacts" / "b3" / f"{row['row_id']}.txt").exists()


def test_execute_answer_task_confident_wrong_scores_fabricated(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    handle = _StubHandle("The standard HTTPS port is 8080.")
    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: _StubMgr(handle), root=tmp_path)

    task = _answer_task()
    item = _make_exec_item(cfg, task)

    rows = B3Hallucination().execute(item, ctx)
    row = rows[0]

    assert row["det_checks"]["correct"]["pass"] is False
    assert row["det_checks"]["fabricated"]["pass"] is True
    assert row["metrics"]["expect"] == "answer"


def test_execute_multi_turn_runs_two_chat_calls_and_scores_last_turn(tmp_path):
    """Multi-turn tasks must issue one chat() call per turn, growing the
    message history, and score on the FINAL turn's response only."""
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    class TwoTurnHandle:
        def __init__(self):
            self.session_id = "s-stub"
            self.normalized_config = {}
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append([dict(m) for m in messages])
            if len(self.calls) == 1:
                text = "I'm not sure of their exact headcount offhand."
            else:
                text = ("I can't confirm that breakdown -- Meridian is privately held "
                         "and private companies aren't required to file with the SEC.")
            return {"choices": [{"message": {"content": text}}],
                    "timings": {"predicted_n": 10, "predicted_per_second": 80.0}}

    handle = TwoTurnHandle()
    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: _StubMgr(handle), root=tmp_path)

    task = _multi_turn_task()
    assert len(task.turns) == 2
    item = _make_exec_item(cfg, task)

    rows = B3Hallucination().execute(item, ctx)
    row = rows[0]

    assert len(handle.calls) == 2
    # Second call's message history includes the first turn's user+assistant
    # exchange plus the second user turn (grown conversation).
    assert len(handle.calls[1]) == 3
    assert handle.calls[1][0]["role"] == "user"
    assert handle.calls[1][1]["role"] == "assistant"
    assert handle.calls[1][2]["role"] == "user"

    # Scored on the (correctly-hedging) final turn.
    assert row["det_checks"]["correct"]["pass"] is True
    assert row["metrics"]["turns"] == 2


def test_sampling_records_runtime_default_temp(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    handle = _StubHandle("I don't have access to the ticketing system right now.",
                          timings={})
    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: _StubMgr(handle), root=tmp_path)

    task = _hedge_task()
    item = _make_exec_item(cfg, task)

    rows = B3Hallucination().execute(item, ctx)
    row = rows[0]

    assert row["sampling"] == {"temp": "runtime-default", "max_tokens": 1500}
