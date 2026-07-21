"""Tests for scripts/classify_b8_local.py (task-b8classify) -- the pass
that actually RUNS `classify_first_failure` over real failed B8 rows and
records the verdict in the sibling `results/b8_classifications-<suite>.jsonl`
store, which `scripts/p8_report.py::build_b8_section` then reads.

`--fake` mode only here -- NO live classifier CLI anywhere in this file
(mirrors tests/test_failure_class.py's discipline). Imports both scripts by
path (scripts/ isn't a package), same trick tests/test_report_b8.py already
uses for scripts/p8_report.py.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


classify_b8_local = _load_script_module("classify_b8_local", "scripts/classify_b8_local.py")
p8_report = _load_script_module("p8_report", "scripts/p8_report.py")

from llmtest import schema                            # noqa: E402
from llmtest.harness.trace import Trace, TraceEvent   # noqa: E402
from llmtest.store import Store                        # noqa: E402

SUITE = "suite-v2.1.0"
TASK_SUFFIX = "edit-01"  # suite/b8_harness/task-01.yaml's real id


def _seed_repo(tmp_path: Path) -> None:
    """Enough of a repo for load_config() + load_b8_tasks() to work: the
    real config/ (all of it -- load_config reads every config/*.yaml
    unconditionally) and the real suite/b8_harness/ task manifests."""
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    shutil.copytree(REPO_ROOT / "suite" / "b8_harness", tmp_path / "suite" / "b8_harness")
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)


def _completed_but_wrong_logic_trace() -> Trace:
    """Harness completed cleanly (parsed tool call, tool ran, terminal
    'completed') -- only the oracle's completed=False (passed separately)
    marks this a FAILURE, so it routes to the panel (the (c) shape)."""
    events = [
        TraceEvent(kind="turn", payload={}),
        TraceEvent(kind="tool_call", payload={
            "tool": "write_file", "callID": "1",
            "input": {"path": "greet.sh", "content": "#!/bin/bash\necho hi\n"},
            "parsed": True,
        }),
        TraceEvent(kind="tool_result", payload={"status": "completed", "output": "ok"}),
        TraceEvent(kind="terminal", payload={"finish": "stop"}),
    ]
    return Trace.from_events(events, terminal_status="completed",
                              tokens_prompt=40, tokens_completion=20, subagent_spawned="no")


def _unparsed_tool_call_trace() -> Trace:
    """A deterministic (a) failure -- the panel must NEVER be consulted for
    this one, proven via --fake mode still producing 'a'/'deterministic'
    even though FakeClassifier would answer 'c' if it were ever asked."""
    events = [
        TraceEvent(kind="turn", payload={}),
        TraceEvent(kind="tool_call", payload={
            "tool": "write_file", "callID": "1", "input": "not-valid-json{{{",
            "parsed": False,
        }),
        TraceEvent(kind="tool_result", payload={"status": "error", "output": None}),
        TraceEvent(kind="terminal", payload={}),
    ]
    return Trace.from_events(events, terminal_status="failed-task",
                              tokens_prompt=10, tokens_completion=0, subagent_spawned="no")


def _seed_failed_row(tmp_path: Path, *, trace: Trace, task_suffix: str = TASK_SUFFIX,
                      model_id: str = "model-a", run_n: int = 1,
                      with_trace_ref: bool = True) -> dict:
    """Writes one schema-valid, completion=False B8 row via the real Store
    (so append()'s own schema validation -- including the row_id/hash
    invariant -- is exercised too, not bypassed), plus -- unless
    `with_trace_ref` is False -- persists `trace` to
    artifacts/b8_traces/<row_id>.json exactly the way
    llmtest.batteries.b8_harness.execute() does, and stamps
    response_meta.trace_ref to match. row_id is computed via
    schema.compute_row_id (the SAME preimage b8_harness.py's execute()
    builds), not a hand-picked string -- Store.append() rejects a row_id
    that doesn't hash-match its own fields."""
    condition = (f"cond=B8;harness=opencode;task={task_suffix};"
                 f"attempt_id=att-{run_n};execution_provenance_sha=" + "0" * 64)
    row_id = schema.compute_row_id(
        suite_version=SUITE, model_id=model_id, quant_sha256="qsha", battery=8,
        task_id=f"b8.{task_suffix}", fixture_sha="sha", condition=condition, run_n=run_n)

    response_meta = {}
    if with_trace_ref:
        trace_relpath = f"b8_traces/{row_id}.json"
        trace_path = tmp_path / "artifacts" / trace_relpath
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace.to_dict()), encoding="utf-8")
        response_meta = {"trace_ref": trace_relpath}

    row = {
        "schema_version": 1, "row_id": row_id, "parent_id": None,
        "suite_version": SUITE, "fixture_sha": "sha", "code_sha": "unknown",
        "battery": 8, "task_id": f"b8.{task_suffix}", "condition": condition,
        "run_n": run_n, "model_id": model_id, "hf_repo": "o/r", "quant_file": "q.gguf",
        "quant_sha256": "qsha", "tier": "T1", "session_id": "s",
        "sampling": {"harness": "opencode"}, "ts": "2026-07-20T00:00:00+00:00",
        "request": {}, "response_meta": response_meta,
        "det_checks": {"oracle": {"pass": False, "detail": "solution logic wrong"}},
        "needs_judging": False,
        "metrics": {
            "completion": False, "steps": trace.steps,
            "tokens_prompt": trace.tokens_prompt, "tokens_completion": trace.tokens_completion,
            "terminal_status": trace.terminal_status, "subagent_spawned": trace.subagent_spawned,
        },
        "timing_authoritative": False, "artifacts": {}, "status": "ok",
        "error_detail": None, "tags": [],
    }
    store = Store(tmp_path / "results")
    assert store.append(row), "seeded row failed Store.append() (schema-invalid seed data)"
    return row


def _cfg(root: Path):
    from llmtest.registry import load_config
    return load_config(root)


# ---------------------------------------------------------------------------
# 1. Loader helpers (task_suffix / load_task_map / load_trace_for_row)
# ---------------------------------------------------------------------------


def test_task_suffix_strips_b8_prefix():
    assert classify_b8_local.task_suffix("b8.py-bugfix-01") == "py-bugfix-01"
    assert classify_b8_local.task_suffix("no-prefix") == "no-prefix"


def test_load_task_map_finds_real_fixture_by_id(tmp_path):
    _seed_repo(tmp_path)
    tasks_by_id = classify_b8_local.load_task_map(tmp_path)
    assert TASK_SUFFIX in tasks_by_id
    assert tasks_by_id[TASK_SUFFIX].id == TASK_SUFFIX


def test_load_trace_for_row_round_trips_persisted_trace(tmp_path):
    _seed_repo(tmp_path)
    trace = _completed_but_wrong_logic_trace()
    row = _seed_failed_row(tmp_path, trace=trace)

    restored = classify_b8_local.load_trace_for_row(tmp_path, row)
    assert restored == trace


def test_load_trace_for_row_missing_trace_ref_returns_none(tmp_path):
    _seed_repo(tmp_path)
    row = _seed_failed_row(tmp_path, trace=_completed_but_wrong_logic_trace(), with_trace_ref=False)
    assert classify_b8_local.load_trace_for_row(tmp_path, row) is None


# ---------------------------------------------------------------------------
# 2. classify_pending_rows -- --fake mode end to end
# ---------------------------------------------------------------------------


def test_fake_mode_classifies_a_failed_row_and_writes_sibling_store(tmp_path):
    _seed_repo(tmp_path)
    trace = _completed_but_wrong_logic_trace()
    row = _seed_failed_row(tmp_path, trace=trace)
    cfg = _cfg(tmp_path)

    summary = classify_b8_local.classify_pending_rows(
        tmp_path, cfg, SUITE, fake=True, log=lambda *a, **k: None)

    assert summary == {"failed_total": 1, "pending": 1, "classified": 1,
                        "skipped_no_trace": 0, "skipped_no_task": 0}

    out_path = classify_b8_local.classifications_path(tmp_path, SUITE)
    assert out_path.exists()
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    rec = records[0]
    assert rec["row_id"] == row["row_id"]
    assert rec["label"] == "c"          # FakeClassifier("c"), routed to the panel
    assert rec["source"] == "panel"
    assert rec["suite_version"] == SUITE


def test_fake_mode_deterministic_failure_never_reaches_fake_panel(tmp_path):
    """A (a) schema-never-parsed trace is deterministic -- classify_pending_rows
    must record 'a'/'deterministic' even in --fake mode, proving the fake
    panel (which always answers 'c') was never actually consulted."""
    _seed_repo(tmp_path)
    row = _seed_failed_row(tmp_path, trace=_unparsed_tool_call_trace())
    cfg = _cfg(tmp_path)

    classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                            log=lambda *a, **k: None)

    out_path = classify_b8_local.classifications_path(tmp_path, SUITE)
    rec = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert (rec["label"], rec["source"]) == ("a", "deterministic")


def test_second_run_skips_already_classified_rows_unless_redo(tmp_path):
    _seed_repo(tmp_path)
    row = _seed_failed_row(tmp_path, trace=_completed_but_wrong_logic_trace())
    cfg = _cfg(tmp_path)

    s1 = classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                                 log=lambda *a, **k: None)
    assert s1["classified"] == 1

    s2 = classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                                 log=lambda *a, **k: None)
    assert s2["pending"] == 0
    assert s2["classified"] == 0

    out_path = classify_b8_local.classifications_path(tmp_path, SUITE)
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 1  # no duplicate record

    s3 = classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                                 redo=True, log=lambda *a, **k: None)
    assert s3["classified"] == 1
    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 2  # --redo appends again


def test_only_completion_false_rows_are_ever_considered(tmp_path):
    _seed_repo(tmp_path)
    # a PASSING row -- must never be picked up (Store schema requires a
    # consistent oracle/completion pairing, but the classify pass filters
    # on completion regardless of det_checks content).
    passing_trace = _completed_but_wrong_logic_trace()
    row = _seed_failed_row(tmp_path, trace=passing_trace, run_n=1)
    # flip it to a passing row after the fact (still schema-valid: Store
    # doesn't cross-validate det_checks against metrics)
    store = Store(tmp_path / "results")
    rows_path = tmp_path / "results" / f"rows-{SUITE}.jsonl"
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    recs = [json.loads(ln) for ln in lines]
    recs[0]["metrics"]["completion"] = True
    rows_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    summary = classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                                       log=lambda *a, **k: None)
    assert summary["failed_total"] == 0
    assert summary["pending"] == 0


def test_row_with_missing_trace_is_skipped_not_crashed(tmp_path):
    _seed_repo(tmp_path)
    _seed_failed_row(tmp_path, trace=_completed_but_wrong_logic_trace(), with_trace_ref=False)
    cfg = _cfg(tmp_path)

    summary = classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                                       log=lambda *a, **k: None)
    assert summary["skipped_no_trace"] == 1
    assert summary["classified"] == 0
    out_path = classify_b8_local.classifications_path(tmp_path, SUITE)
    assert not out_path.exists()


def test_row_with_unknown_task_id_is_skipped_not_crashed(tmp_path):
    _seed_repo(tmp_path)
    _seed_failed_row(tmp_path, task_suffix="no-such-task-id",
                      trace=_completed_but_wrong_logic_trace())
    cfg = _cfg(tmp_path)

    summary = classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                                       log=lambda *a, **k: None)
    assert summary["skipped_no_task"] == 1
    assert summary["classified"] == 0


# ---------------------------------------------------------------------------
# 3. End-to-end: classify pass output feeds build_b8_section's real distribution
# ---------------------------------------------------------------------------


def test_classify_pass_output_shows_up_in_build_b8_section(tmp_path):
    _seed_repo(tmp_path)
    row = _seed_failed_row(tmp_path, trace=_completed_but_wrong_logic_trace())
    cfg = _cfg(tmp_path)

    classify_b8_local.classify_pending_rows(tmp_path, cfg, SUITE, fake=True,
                                            log=lambda *a, **k: None)

    caveats: list = []
    rows = p8_report.load_rows(tmp_path, SUITE, caveats)
    # the row itself carries NO metrics.first_failure_class -- only the
    # sibling classification store does.
    assert "first_failure_class" not in rows[0]["metrics"]

    section = p8_report.build_b8_section(tmp_path, cfg, rows)

    fc_block = section.split("**First-failure-class distribution", 1)[1]
    fc_row = next(ln for ln in fc_block.splitlines() if "model-a" in ln and "opencode" in ln)
    cells = [c.strip() for c in fc_row.strip().strip("|").split("|")]
    assert cells[4] == "1"     # class c
    assert cells[-1] == "0"    # (unclassified) -- the classify pass covered it


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
