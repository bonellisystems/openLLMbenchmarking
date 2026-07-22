"""Scorer-conformance suite (Wave 3b, B8 validity program): "add scripted
pass/fail/timeout/tamper agents as harness conformance tests" -- scripted
"agent behaviors" (NO LLM anywhere in this file: the post-run workspace
and/or a synthetic `Trace` are constructed directly) that exercise the FULL
completion scorer and assert each is scored correctly.

Two distinct scorer entry points are exercised, matching which one each
scenario is actually about:

- PASS / FAIL agents call `llmtest.harness.tasks.run_oracle` DIRECTLY, with
  `oracle_image="python:3.11-slim"` (Docker-gated) -- NOT through
  `B8Harness.execute()`. `_resolve_run_oracle`'s injected-callable seam
  (`llmtest/batteries/b8_harness.py`) drops `oracle_image` for an injected
  `ctx.b8_run_oracle` (every existing `ctx.b8_run_oracle` test double
  predates the oracle_image param and has the old 3-arg signature), so
  routing a real Python-task oracle run through `execute()` with an
  injected oracle callable would silently run the oracle under the
  Python-less pinned CUDA image instead of `python:3.11-slim` -- calling
  `run_oracle` directly is both simpler and avoids that trap.
- TIMEOUT/budget is the one scenario the brief explicitly routes through
  `B8Harness.execute()` (real battery-level plan()/execute(), with an
  injected mock adapter + an injected `run_oracle` stub) -- what's under
  test here is execute()'s own budget-crediting/terminal_status logic, not
  the oracle itself, so the injected-oracle seam is exactly the right tool
  and its oracle_image-dropping is inert (terminal_status="killed"/
  "budget-exceeded" forces `completed_final=False` regardless of what the
  injected oracle says). Hermetic -- no Docker.
- TAMPER exercises `run_oracle`'s HARD-CAP steps ((a) protected-file
  hash check, (b) tamper detection) directly -- these never construct a
  `Sandbox` at all (see `llmtest.harness.tasks`'s module docstring), so
  they're hermetic like TIMEOUT, and the whole point under test is that
  they fire BEFORE the behavioral oracle ever would.

Uses `py-bugfix-01`'s own `check_fixtures` (Wave 3b, see
`suite/b8_harness/_schema.md`) as the PASS/FAIL agents' solutions, rather
than hand-duplicating them here.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtest.batteries.b8_harness import B8Harness
from llmtest.harness import tasks as t
from llmtest.harness.base import MockHarnessAdapter
from llmtest.harness.failure_class import classify_first_failure
from llmtest.harness.trace import Trace, TraceEvent
from llmtest.registry import load_config

ROOT = Path(__file__).resolve().parents[1]


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


requires_docker = pytest.mark.skipif(not _docker_available(), reason="Docker not reachable")


def _load(task_id: str) -> t.B8Task:
    return next(x for x in t.load_b8_tasks(ROOT) if x.id == task_id)


def _apply(ws: Path, solution: dict) -> None:
    for rel, content in solution.items():
        (ws / rel).write_bytes(content.encode("utf-8"))


# ============================================================================
# 1. PASS agent -- writes a correct solution into the workspace -> run_oracle
#    -> completion True, oracle pass=True, structured stage/reason present.
# ============================================================================


@requires_docker
def test_pass_agent_correct_solution_is_credited_complete(tmp_path):
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply(ws, task.check_fixtures["reference"])

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    completed, detail = result  # legacy 2-tuple view still honored
    assert completed is True
    assert result.pass_ is True
    assert detail == "PASS"
    # Structured stage/reason present (Wave 3a machine-readable convention)
    # -- proves this reached the real behavioral oracle, not a hard cap.
    assert result.stage == "behavior"
    assert result.reason_code is None


# ============================================================================
# 2. FAIL agent -- writes a plausible-but-wrong solution -> completion
#    False, reason_code set (e.g. wrong_output), actual bounded.
# ============================================================================


@requires_docker
def test_fail_agent_plausible_wrong_solution_is_rejected_with_reason_code(tmp_path):
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    # check_fixtures.wrong[1]: syntactically valid, RUNS fine, wrong answer
    # (an off-by-one denominator) -- the canonical "ran but produced the
    # wrong output" FAIL agent, distinct from wrong[0] (a compile-stage
    # SyntaxError, covered by test_run_oracle_crashed_solution_yields_
    # compile_stage in tests/test_harness_tasks.py already).
    _apply(ws, task.check_fixtures["wrong"][1])

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    completed, detail = result
    assert completed is False
    assert result.pass_ is False
    assert result.stage == "behavior"
    assert result.reason_code == "wrong_output"
    assert result.case
    assert result.expected
    # `actual` is candidate-controlled output -- bounded per Wave 3a's
    # _ACTUAL_TRUNCATE_CHARS contract (this wave's brief: "actual bounded"),
    # not merely usually-short.
    assert result.actual is not None
    assert len(result.actual) <= t._ACTUAL_TRUNCATE_CHARS + len("...<TRUNCATED 9999 more chars>")
    assert "FAIL:" in detail


# ============================================================================
# 3. TIMEOUT/budget agent -- a Trace with terminal_status="killed" (or
#    over-budget steps) -> B8Harness.execute() (injected mock adapter +
#    injected run_oracle) yields completion=False, terminal_status
#    reflecting budget/killed, and classify_first_failure -> deterministic
#    "e" (panel NOT consulted). Hermetic -- no Docker, no live harness.
# ============================================================================


class _RaisingClassifier:
    """Mirrors tests/test_failure_class.py's RaisingClassifier -- fails the
    test if the panel is EVER consulted for what must be a deterministic
    verdict."""

    def classify(self, blinded_text: str) -> str:
        raise AssertionError("panel must not be consulted for a deterministic (e) verdict")


class _FakeStore:
    """No seeded rows -- plan() emits WorkItems for every configured
    replicate, mirrors tests/test_b8.py's FakeStore."""

    def iter_rows(self):
        return []


class _StubHandle:
    session_id = "s-stub"
    base_url = "http://127.0.0.1:9/"
    normalized_config = {"ctx": 32768, "kv_dtype": "q8_0"}


class _StubMgr:
    def request_endpoint(self, *a, **k):
        return _StubHandle()


def _versioned_adapter(scripted_events, *, terminal_status,
                       tokens_prompt=120, tokens_completion=80):
    class _Adapter(MockHarnessAdapter):
        def version(self) -> str:
            return "mock-conformance-1.0"

    return _Adapter(scripted_events=scripted_events, terminal_status=terminal_status,
                    tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
                    subagent_spawned="no")


def _execute_scripted_run(tmp_path, *, task_id, adapter,
                          oracle_result=(True, "stub-would-have-passed")):
    """Drive real B8Harness.plan() -> execute() end to end with EVERY seam
    injected (adapter, oracle, endpoint) -- no live harness process, no live
    endpoint, no Docker. Returns (row_dict, task, persisted_trace) --
    `persisted_trace` is reloaded from the artifacts/b8_traces/<row_id>.json
    execute() itself writes, exactly the round-trip
    scripts/classify_b8_local.py performs for a real run."""
    cfg = load_config(ROOT)
    cfg.suite["b8"]["tasks"] = [task_id]
    model_id = cfg.suite["b8"]["models"][0]

    items = B8Harness().plan(cfg, _FakeStore(), model_filter=model_id)
    assert items, "plan() produced no WorkItems -- fixtures/config missing?"
    item = items[0]
    task = item.payload["task"]

    ctx = SimpleNamespace(
        cfg=cfg, root=tmp_path, server_manager=lambda: _StubMgr(),
        b8_adapters={item.payload["harness"]: adapter},
        b8_run_oracle=lambda task_, workspace, root=".": oracle_result,
        b8_attempt_id="conformance-fixed-attempt")

    rows = B8Harness().execute(item, ctx)
    assert len(rows) == 1
    row = rows[0]
    trace_path = tmp_path / "artifacts" / row["response_meta"]["trace_ref"]
    persisted_trace = Trace.from_dict(json.loads(trace_path.read_text(encoding="utf-8")))
    return row, task, persisted_trace


def test_timeout_agent_killed_trace_is_not_credited_and_classifies_deterministic_e(tmp_path):
    """The harness's own wall-clock kill (terminal_status="killed") --
    completion must be False EVEN THOUGH the injected oracle says
    "would have passed" (proving terminal_status, not the oracle verdict
    alone, gates credit), and classify_first_failure must resolve to "e"
    deterministically without ever invoking the panel."""
    events = [TraceEvent(kind="turn"), TraceEvent(kind="tool_call", payload={"tool": "bash_exec"})]
    adapter = _versioned_adapter(events, terminal_status="killed")

    row, task, trace = _execute_scripted_run(
        tmp_path, task_id="py-bugfix-01", adapter=adapter,
        oracle_result=(True, "stub-would-have-passed"))

    assert row["metrics"]["completion"] is False
    assert row["metrics"]["terminal_status"] == "killed"
    assert trace.terminal_status == "killed"

    label, source = classify_first_failure(
        trace, task, completed=row["metrics"]["completion"],
        classifiers=[_RaisingClassifier()])
    assert (label, source) == ("e", "deterministic")


def test_budget_exceeded_agent_over_step_budget_is_not_credited_and_classifies_deterministic_e(tmp_path):
    """The OTHER source of a TIMEOUT/budget agent (brief: "or over-budget
    steps") -- the harness itself reports a clean "completed" finish, but
    B8Harness.execute()'s own post-hoc step-budget check overrides
    terminal_status to "budget-exceeded" because the scripted trace burned
    more turn-events (steps) than cfg.suite['b8']['budgets']['steps']
    allows -- the exact "stumbled for N steps and got lucky" case Wave 1a's
    budget enforcement targets (mirrors tests/test_b8.py's
    test_execute_steps_over_budget_forces_completion_false, re-verified
    here through classify_first_failure too)."""
    cfg = load_config(ROOT)
    budget_steps = cfg.suite["b8"]["budgets"]["steps"]
    over_steps = budget_steps + 3
    events = [TraceEvent(kind="turn") for _ in range(over_steps)]
    adapter = _versioned_adapter(events, terminal_status="completed")

    row, task, trace = _execute_scripted_run(
        tmp_path, task_id="py-bugfix-01", adapter=adapter,
        oracle_result=(True, "stub-would-have-passed"))

    assert row["metrics"]["completion"] is False
    assert row["metrics"]["budget_exceeded"] is True
    assert row["metrics"]["terminal_status"] == "budget-exceeded"
    assert trace.terminal_status == "budget-exceeded"
    # The oracle's own (stubbed) verdict is still recorded honestly --
    # det_checks is the RAW oracle result; only the row's scoring
    # `completion` is forced False by the budget check.
    assert row["det_checks"]["oracle"]["pass"] is True

    label, source = classify_first_failure(
        trace, task, completed=row["metrics"]["completion"],
        classifiers=[_RaisingClassifier()])
    assert (label, source) == ("e", "deterministic")


# ============================================================================
# 4. TAMPER agent -- deletes a setup_repo file / plants a disallowed
#    symlink / edits a protected_paths file -> run_oracle -> completion
#    False with the SPECIFIC tamper reason, BEFORE the behavioral oracle
#    runs (hard-cap precedence). Hermetic -- these steps never construct a
#    Sandbox at all.
# ============================================================================
# Every scenario below ALSO applies the genuinely-correct fix
# (check_fixtures.reference) on top of the tamper -- "behavior would
# otherwise pass" -- so a completion=False verdict here can only be
# explained by the hard cap firing FIRST, never by the fix itself being
# wrong (mirrors task-3-brief.md's original Step-1 scenario).


def test_tamper_agent_edits_protected_file_is_rejected_before_oracle_runs(tmp_path):
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply(ws, task.check_fixtures["reference"])  # genuinely correct fix
    (ws / "NOTES.md").write_bytes(b"tampered by the scripted agent\n")

    result = t.run_oracle(task, ws)  # no oracle_image -- never reached
    assert result.pass_ is False
    assert "protected file tampered" in result.detail.lower()
    assert "NOTES.md" in result.detail
    # Hard-cap path: never populates the behavioral oracle's structured
    # fields -- proof this never reached step (c) at all.
    assert result.stage is None
    assert result.reason_code is None


def test_tamper_agent_deletes_setup_repo_file_is_rejected_before_oracle_runs(tmp_path):
    """"Deletes a setup_repo file" -- NOTES.md is a real setup_repo entry
    (agent-visible, protected). Its deletion is caught by run_oracle step
    (a)'s protected-hash check ("... (missing)"), the same hard cap that
    covers an edited-but-still-present protected file above -- a deleted
    protected file is simply the more extreme case of "does not match its
    recorded hash.\""""
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply(ws, task.check_fixtures["reference"])  # genuinely correct fix
    (ws / "NOTES.md").unlink()

    result = t.run_oracle(task, ws)
    assert result.pass_ is False
    assert "protected file tampered" in result.detail.lower()
    assert "NOTES.md" in result.detail
    assert "missing" in result.detail.lower()
    assert result.stage is None


def test_tamper_agent_plants_disallowed_symlink_is_rejected_before_oracle_runs(tmp_path):
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply(ws, task.check_fixtures["reference"])  # genuinely correct fix
    try:
        os.symlink(str(ws / "stats.py"), ws / "sneaky_link.py")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"os.symlink not permitted on this host: {e!r}")

    result = t.run_oracle(task, ws)
    assert result.pass_ is False
    assert "disallowed symlink" in result.detail.lower()
    assert "sneaky_link.py" in result.detail
    assert result.stage is None


def test_tamper_agent_hard_cap_paths_never_touch_docker(tmp_path, monkeypatch):
    """All three TAMPER scenarios above run without Docker -- assert this
    directly (mirrors test_harness_tasks.py's
    test_run_oracle_no_docker_needed_for_hard_cap_paths) by making any
    subprocess.run call fail the test."""
    import llmtest.harness.sandbox as sandbox_mod

    def _boom(*a, **k):
        raise AssertionError("TAMPER hard-cap path must not shell out to Docker")

    monkeypatch.setattr(sandbox_mod.subprocess, "run", _boom)

    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply(ws, task.check_fixtures["reference"])
    (ws / "NOTES.md").write_bytes(b"tampered\n")

    result = t.run_oracle(task, ws)
    assert result.pass_ is False
