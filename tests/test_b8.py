"""Tests for Battery 8 -- real agent-harness execution + its additive row
identity (execution_provenance_sha / attempt_id / replicate_n).

Mirrors test_b6.py's / test_b7.py's `FakeStore`/`_stub_ctx`-injected-ctx
pattern -- no live harness process, no live endpoint, no Docker anywhere in
this file: `execute()` is exercised entirely through the `ctx.b8_adapters` /
`ctx.b8_run_oracle` / `ctx.b8_attempt_id` seams `llmtest/batteries/
b8_harness.py` defines for exactly this purpose.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtest import batteries, schema
from llmtest.batteries import b8_fixtures as b8f
from llmtest.batteries.b8_harness import (
    B8Harness, _base_condition, _execution_provenance_sha, _full_condition,
)
from llmtest.harness.base import MockHarnessAdapter
from llmtest.harness.trace import Trace, TraceEvent
from llmtest.registry import load_config

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    """Fake store for plan() tests -- no seeded rows."""
    def iter_rows(self):
        return []


def _tasks_for_plan(cfg):
    """The exact set of B8Task objects `B8Harness.plan()` will cross for
    `cfg` -- `b8f.load_tasks(ROOT)` narrowed by the optional `b8.tasks`
    allowlist (task-b8expand), mirroring `plan()`'s own filter. Tests that
    pick "the first task" or count expected WorkItems must go through this
    (not a bare `b8f.load_tasks(ROOT)[0]`) now that suite.yaml's real
    `b8.tasks` allowlist excludes the 5 bash placeholder manifests from
    what plan() actually produces."""
    tasks = b8f.load_tasks(ROOT)
    allowlist = cfg.suite["b8"].get("tasks")
    if allowlist:
        allowed = set(allowlist)
        tasks = [t for t in tasks if t.id in allowed]
    return tasks


def _stub_ctx(tmp_path, cfg, *, adapter=None, run_oracle_result=(True, "stub-pass"),
             attempt_id=None):
    """ctx with every B8 execute() seam wired to a controllable stand-in:
    a stub ServerManager/EndpointHandle (never makes a real HTTP call --
    the injected adapter never touches it either), an injected
    MockHarnessAdapter (no real harness subprocess), a controllable
    run_oracle result (no Docker), and an optional fixed attempt_id."""
    class StubHandle:
        session_id = "s-stub"
        base_url = "http://127.0.0.1:9/"
        normalized_config = {"ctx": 32768, "kv_dtype": "q8_0"}

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    kwargs = dict(cfg=cfg, root=tmp_path, server_manager=lambda: StubMgr(),
                 b8_run_oracle=lambda task, workspace, root=".": run_oracle_result)
    if adapter is not None:
        kwargs["b8_adapters"] = {"opencode": adapter}
    if attempt_id is not None:
        kwargs["b8_attempt_id"] = attempt_id
    return SimpleNamespace(**kwargs)


def _mock_adapter(*, version_str="mock-1.0", terminal_status="completed",
                  tokens_prompt=120, tokens_completion=80, subagent_spawned="no"):
    events = [TraceEvent(kind="turn"),
             TraceEvent(kind="tool_call", payload={"tool": "bash_exec"}),
             TraceEvent(kind="tool_result", payload={"status": "completed"}),
             TraceEvent(kind="turn")]

    class _Versioned(MockHarnessAdapter):
        def version(self) -> str:
            return version_str

    return _Versioned(scripted_events=events, terminal_status=terminal_status,
                      tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
                      subagent_spawned=subagent_spawned)


def _first_item(cfg, model_id=None, store=None):
    model_id = model_id or cfg.suite["b8"]["models"][0]
    items = B8Harness().plan(cfg, store or FakeStore(), model_filter=model_id)
    assert items, "plan() produced no WorkItems -- fixtures/config missing?"
    return items[0]


# --- Battery registry -------------------------------------------------------


def test_battery_8_registers():
    b = batteries.get(8)
    assert isinstance(b, B8Harness)
    assert b.id == 8


# --- suite.yaml wiring -------------------------------------------------------


def test_suite_yaml_has_b8_block_and_condition_additions():
    cfg = load_config(ROOT)
    assert "b8" in cfg.suite
    # Target is >=5 (task-3-brief.md / spec ss2.8, real Wilson intervals)
    # -- was TEMPORARILY lowered to 3 for the first real local validation
    # run (task-b8local), raised back to >=5 (task-b8expand) now that the
    # OpenCode/gpt-oss-20b pipeline is confirmed stable.
    assert cfg.suite["b8"]["replicates"] >= 5
    assert cfg.suite["b8"]["models"]
    assert cfg.suite["b8"]["harnesses"]
    for k in ("harness", "task", "attempt_id", "execution_provenance_sha"):
        assert k in cfg.suite["condition_order"]
    assert "B8" in cfg.suite["condition_vocab"]["cond"]


def test_suite_yaml_b8_tasks_allowlist_is_the_five_hard_python_task_ids():
    """The real suite.yaml (task-b8hard) restricts a live run to exactly
    the 5 HARDER Python task ids -- the original 6 real Python tasks
    (task-06..11.yaml) and the 5 bash placeholder manifests
    (task-01..05.yaml) are all loaded (load_b8_tasks still returns them)
    but never crossed by plan(). The original 6 are all solved 30/30 by
    gpt-oss-20b and don't discriminate; the 5 harder manifests
    (task-12..16.yaml) are calibrated so a capable 20B model is genuinely
    expected to fail some fraction of the time."""
    cfg = load_config(ROOT)
    allowlist = cfg.suite["b8"].get("tasks")
    assert allowlist, "suite.yaml b8.tasks allowlist must be set (task-b8hard)"
    assert set(allowlist) == {
        "py-hard-bugfix-01", "py-hard-algo-01", "py-hard-edge-01",
        "py-hard-multifile-01", "py-hard-toolheavy-01",
    }
    all_task_ids = {task.id for task in b8f.load_tasks(ROOT)}
    for task_id in allowlist:
        assert task_id in all_task_ids, f"{task_id} not found by load_tasks"


# --- Battery.plan() -----------------------------------------------------------


def test_plan_item_count_matches_models_x_harnesses_x_tasks_x_replicates():
    """Step-1 assertion #1: plan() item count == len(models) x len(harnesses)
    x len(tasks) x replicates -- `tasks` here is `_tasks_for_plan(cfg)`
    (the b8.tasks-allowlist-narrowed set), not every loaded manifest,
    since suite.yaml's real b8.tasks allowlist (task-b8expand) excludes
    the 5 bash placeholders from what plan() actually crosses."""
    cfg = load_config(ROOT)
    b8cfg = cfg.suite["b8"]
    tasks = _tasks_for_plan(cfg)
    assert tasks, "no suite/b8_harness/task-*.yaml fixtures found (post-allowlist)"

    items = B8Harness().plan(cfg, FakeStore())
    expected = len(b8cfg["models"]) * len(b8cfg["harnesses"]) * len(tasks) * b8cfg["replicates"]
    assert expected > 0
    assert len(items) == expected

    for item in items:
        assert item.battery == 8
        assert item.task_id.startswith("b8.")
        assert 1 <= item.run_n <= b8cfg["replicates"]


def test_plan_model_filter():
    cfg = load_config(ROOT)
    model_id = cfg.suite["b8"]["models"][0]
    items = B8Harness().plan(cfg, FakeStore(), model_filter=model_id)
    assert items
    assert {i.model_id for i in items} == {model_id}


# --- Battery.plan(): b8.tasks allowlist (task-b8expand) -----------------------


def test_plan_tasks_allowlist_restricts_to_listed_ids():
    """A non-empty b8.tasks allowlist restricts plan() to exactly those
    task ids -- proven against a SUBSET distinct from the real suite.yaml
    default, so this doesn't just coincidentally pass because of what
    suite.yaml already has configured."""
    cfg = load_config(ROOT)
    b8cfg = cfg.suite["b8"]
    model_id = b8cfg["models"][0]
    harness_name = b8cfg["harnesses"][0]
    subset = ["py-bugfix-01", "py-edit-01"]
    cfg.suite["b8"]["tasks"] = subset

    items = B8Harness().plan(cfg, FakeStore(), model_filter=model_id)
    assert items
    seen_task_ids = {i.task_id for i in items}
    assert seen_task_ids == {f"b8.{tid}" for tid in subset}
    assert len(items) == len(subset) * b8cfg["replicates"]
    for item in items:
        parts = dict(p.split("=", 1) for p in item.condition.split(";"))
        assert parts["harness"] == harness_name
        assert parts["task"] in subset


def test_plan_tasks_allowlist_single_id_excludes_everything_else():
    cfg = load_config(ROOT)
    b8cfg = cfg.suite["b8"]
    model_id = b8cfg["models"][0]
    cfg.suite["b8"]["tasks"] = ["py-toolheavy-01"]

    items = B8Harness().plan(cfg, FakeStore(), model_filter=model_id)
    assert items
    assert {i.task_id for i in items} == {"b8.py-toolheavy-01"}
    assert len(items) == b8cfg["replicates"]


@pytest.mark.parametrize("allowlist_value", [[], None])
def test_plan_tasks_allowlist_empty_or_absent_falls_back_to_all_tasks(allowlist_value):
    """Additivity: an empty list (or the key removed entirely) must
    reproduce byte-for-byte the pre-task-b8expand behavior -- every
    loaded manifest crossed, unchanged."""
    cfg = load_config(ROOT)
    b8cfg = cfg.suite["b8"]
    model_id = b8cfg["models"][0]
    if allowlist_value is None:
        cfg.suite["b8"].pop("tasks", None)
    else:
        cfg.suite["b8"]["tasks"] = allowlist_value

    items = B8Harness().plan(cfg, FakeStore(), model_filter=model_id)
    all_tasks = b8f.load_tasks(ROOT)
    expected_task_ids = {f"b8.{task.id}" for task in all_tasks}
    assert {i.task_id for i in items} == expected_task_ids
    assert len(items) == len(all_tasks) * b8cfg["replicates"]


def _seed_full_condition(order, harness_name, task_id):
    return schema.canonical_condition(
        {"cond": "B8", "harness": harness_name, "task": task_id,
         "attempt_id": "seed-attempt", "execution_provenance_sha": "0" * 64}, order)


def test_plan_resume_skips_already_recorded_replicates():
    """Resume/dedup lives in plan() for B8 (see b8_harness.py module
    docstring's RESUME note) -- unlike B1/B6/B7, a B8 WorkItem's row_id
    never matches its own eventual row's row_id (attempt_id/exec_sha are
    stamped only at execute() time), so run_cmd.py's row_id-membership
    resume check is a structural no-op for this battery; plan() itself
    must be the thing that skips already-recorded replicate_ns."""
    cfg = load_config(ROOT)
    b8cfg = cfg.suite["b8"]
    order = cfg.suite["condition_order"]
    model_id = b8cfg["models"][0]
    harness_name = b8cfg["harnesses"][0]
    task = _tasks_for_plan(cfg)[0]
    task_id = f"b8.{task.id}"

    full_condition = _seed_full_condition(order, harness_name, task.id)
    seeded_rows = [
        {"model_id": model_id, "task_id": task_id, "condition": full_condition,
         "run_n": n, "row_id": f"seed-{n}"}
        for n in (1, 2, 3)
    ]

    class SeededStore:
        def iter_rows(self):
            return seeded_rows

    items = B8Harness().plan(cfg, SeededStore(), model_filter=model_id)
    matching = [i for i in items if i.task_id == task_id]
    assert sorted(i.run_n for i in matching) == list(range(4, b8cfg["replicates"] + 1))


def test_plan_force_bumps_one_run_n_beyond_existing():
    cfg = load_config(ROOT)
    b8cfg = cfg.suite["b8"]
    order = cfg.suite["condition_order"]
    model_id = b8cfg["models"][0]
    harness_name = b8cfg["harnesses"][0]
    task = _tasks_for_plan(cfg)[0]
    task_id = f"b8.{task.id}"

    full_condition = _seed_full_condition(order, harness_name, task.id)
    seeded_rows = [
        {"model_id": model_id, "task_id": task_id, "condition": full_condition,
         "run_n": n, "row_id": f"seed-{n}"}
        for n in (1, 2, 3)
    ]

    class SeededStore:
        def iter_rows(self):
            return seeded_rows

    items = B8Harness().plan(cfg, SeededStore(), model_filter=model_id, force=True)
    matching = [i for i in items if i.task_id == task_id]
    assert len(matching) == 1
    assert matching[0].run_n == 4


# --- Battery.execute() ---------------------------------------------------------


def test_execute_produces_schema_valid_row(tmp_path):
    """Step-1 assertion #2: execute() with an injected MockHarnessAdapter (+
    a controllable run_oracle completion) produces a schema-valid row,
    battery==8, needs_judging==False, metrics populated from the Trace."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)
    adapter = _mock_adapter(terminal_status="completed", tokens_prompt=120,
                            tokens_completion=80, subagent_spawned="no")
    ctx = _stub_ctx(tmp_path, cfg, adapter=adapter, run_oracle_result=(True, "stub-pass"),
                    attempt_id="attempt-fixed")

    rows = B8Harness().execute(item, ctx)
    assert len(rows) == 1
    row = rows[0]

    errs = schema.validate_row(row)
    assert errs == [], errs
    assert row["battery"] == 8
    assert row["needs_judging"] is False
    assert row["status"] == "ok"
    assert row["run_n"] == item.run_n

    assert row["metrics"]["completion"] is True
    assert row["metrics"]["steps"] == 2               # 2 "turn" events scripted
    assert row["metrics"]["tokens_prompt"] == 120
    assert row["metrics"]["tokens_completion"] == 80
    assert row["metrics"]["terminal_status"] == "completed"
    assert row["metrics"]["subagent_spawned"] == "no"

    assert adapter.calls == ["setup", "run", "teardown"]

    parts = dict(p.split("=", 1) for p in row["condition"].split(";"))
    assert parts["attempt_id"] == "attempt-fixed"
    assert len(parts["execution_provenance_sha"]) == 64
    assert parts["harness"] == "opencode"


def test_execute_persists_trace_and_stamps_trace_ref_on_the_row(tmp_path):
    """task-b8classify: execute() must persist the FULL Trace (not just the
    summary counts already in metrics) to artifacts/b8_traces/<row_id>.json
    and reference it additively via response_meta.trace_ref -- row_id and
    schema-validity must be unaffected."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)
    adapter = _mock_adapter(terminal_status="completed", tokens_prompt=120,
                            tokens_completion=80, subagent_spawned="no")
    ctx = _stub_ctx(tmp_path, cfg, adapter=adapter, run_oracle_result=(True, "stub-pass"),
                    attempt_id="attempt-fixed")

    row = B8Harness().execute(item, ctx)[0]

    assert schema.validate_row(row) == []

    trace_ref = row["response_meta"].get("trace_ref")
    assert trace_ref, "execute() must stamp response_meta.trace_ref"
    assert row["row_id"] in trace_ref

    trace_path = tmp_path / "artifacts" / trace_ref
    assert trace_path.exists()

    restored = Trace.from_dict(json.loads(trace_path.read_text(encoding="utf-8")))
    assert restored.terminal_status == "completed"
    assert restored.tokens_prompt == 120
    assert restored.tokens_completion == 80
    assert restored.subagent_spawned == "no"
    assert restored.steps == row["metrics"]["steps"]


def test_execute_persisted_trace_reflects_failing_run(tmp_path):
    """Same trace-persistence path, but for a FAILED (oracle-rejected) run
    -- the exact case scripts/classify_b8_local.py will actually read."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)
    adapter = _mock_adapter(terminal_status="killed", tokens_prompt=30,
                            tokens_completion=5, subagent_spawned="no")
    ctx = _stub_ctx(tmp_path, cfg, adapter=adapter,
                    run_oracle_result=(False, "timed out"), attempt_id="attempt-fixed")

    row = B8Harness().execute(item, ctx)[0]

    trace_path = tmp_path / "artifacts" / row["response_meta"]["trace_ref"]
    restored = Trace.from_dict(json.loads(trace_path.read_text(encoding="utf-8")))
    assert restored.terminal_status == "killed"
    assert row["metrics"]["completion"] is False


def test_execute_different_rows_get_distinct_trace_files(tmp_path):
    """Two different physical attempts (different attempt_id -> different
    row_id) must never collide on the same trace file."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)

    ctx_a = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(), attempt_id="attempt-A")
    ctx_b = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(), attempt_id="attempt-B")

    row_a = B8Harness().execute(item, ctx_a)[0]
    row_b = B8Harness().execute(item, ctx_b)[0]

    ref_a = row_a["response_meta"]["trace_ref"]
    ref_b = row_b["response_meta"]["trace_ref"]
    assert ref_a != ref_b
    assert (tmp_path / "artifacts" / ref_a).exists()
    assert (tmp_path / "artifacts" / ref_b).exists()


def test_execute_completion_false_when_oracle_fails(tmp_path):
    """Also covers the coordinator's Minor #1: the oracle's failure detail
    (WHY run_oracle rejected the run) must not be discarded -- Task 8
    (first-failure classification) reads it from det_checks.oracle.detail,
    not just the bare completion bool in metrics."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)
    adapter = _mock_adapter(terminal_status="completed")
    ctx = _stub_ctx(tmp_path, cfg, adapter=adapter,
                    run_oracle_result=(False, "out-of-bounds edit: sneaky.sh"),
                    attempt_id="attempt-fixed")

    row = B8Harness().execute(item, ctx)[0]
    assert row["metrics"]["completion"] is False
    assert schema.validate_row(row) == []
    assert row["det_checks"]["oracle"]["pass"] is False
    assert row["det_checks"]["oracle"]["detail"] == "out-of-bounds edit: sneaky.sh"


def test_execute_completion_true_still_carries_oracle_detail(tmp_path):
    """Flip side: a passing oracle's detail (e.g. a stub-pass reason) is
    threaded through too, not just the failure case."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)
    adapter = _mock_adapter(terminal_status="completed")
    ctx = _stub_ctx(tmp_path, cfg, adapter=adapter,
                    run_oracle_result=(True, "PASS"), attempt_id="attempt-fixed")

    row = B8Harness().execute(item, ctx)[0]
    assert row["det_checks"]["oracle"]["pass"] is True
    assert row["det_checks"]["oracle"]["detail"] == "PASS"


@pytest.mark.parametrize("terminal_status", ["killed", "infra-error", "budget-exceeded"])
def test_execute_handles_non_completed_terminal_status(tmp_path, terminal_status):
    """Coordinator's Minor #2: killed/infra-error/budget-exceeded are legal
    Trace.terminal_status values (llmtest.harness.trace.
    VALID_TERMINAL_STATUSES) and execute() passes trace.terminal_status
    straight through with no branching -- but nothing pinned that a
    non-"completed" status still produces a schema-valid row instead of
    crashing. A real budget-exceeding/killed harness run is exactly the
    case B8 most needs to measure (reasoning models regularly run long),
    so this must not raise."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)
    adapter = _mock_adapter(terminal_status=terminal_status, tokens_prompt=50,
                            tokens_completion=0, subagent_spawned="no")
    ctx = _stub_ctx(tmp_path, cfg, adapter=adapter,
                    run_oracle_result=(False, f"terminal_status={terminal_status}"),
                    attempt_id="attempt-fixed")

    row = B8Harness().execute(item, ctx)[0]
    assert schema.validate_row(row) == []
    assert row["battery"] == 8
    assert row["metrics"]["terminal_status"] == terminal_status
    assert row["metrics"]["completion"] is False
    assert row["det_checks"]["oracle"]["detail"] == f"terminal_status={terminal_status}"


def test_execute_different_attempt_ids_never_collide(tmp_path):
    """Step-1 assertion #3 (append-only invariant): same cell, two different
    attempt_ids -> different row_id."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)

    ctx_a = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(), attempt_id="attempt-A")
    ctx_b = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(), attempt_id="attempt-B")

    row_a = B8Harness().execute(item, ctx_a)[0]
    row_b = B8Harness().execute(item, ctx_b)[0]

    assert row_a["row_id"] != row_b["row_id"]
    assert row_a["condition"] != row_b["condition"]
    # everything else about the cell is identical -- only attempt_id differs
    assert row_a["task_id"] == row_b["task_id"]
    assert row_a["model_id"] == row_b["model_id"]
    assert row_a["run_n"] == row_b["run_n"]


def test_execute_same_attempt_id_reproduces_same_row_id(tmp_path):
    """The flip side of the append-only invariant: replaying the SAME
    attempt_id under the same harness/profile/prompt must reproduce the
    SAME row_id -- exec_sha has no timestamp/randomness in its inputs."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)

    ctx1 = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(), attempt_id="attempt-same")
    ctx2 = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(), attempt_id="attempt-same")

    row1 = B8Harness().execute(item, ctx1)[0]
    row2 = B8Harness().execute(item, ctx2)[0]

    assert row1["row_id"] == row2["row_id"]
    assert row1["condition"] == row2["condition"]


def test_execute_execution_provenance_sha_changes_when_harness_version_changes(tmp_path):
    """Step-1 assertion #4 (integration level): execution_provenance_sha
    changes when the (mock) harness version() changes, with attempt_id held
    fixed so the effect is isolated to the version change alone."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)

    ctx1 = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(version_str="opencode-v1"),
                     attempt_id="attempt-fixed")
    ctx2 = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(version_str="opencode-v2"),
                     attempt_id="attempt-fixed")

    row1 = B8Harness().execute(item, ctx1)[0]
    row2 = B8Harness().execute(item, ctx2)[0]

    parts1 = dict(p.split("=", 1) for p in row1["condition"].split(";"))
    parts2 = dict(p.split("=", 1) for p in row2["condition"].split(";"))
    assert parts1["execution_provenance_sha"] != parts2["execution_provenance_sha"]
    assert row1["row_id"] != row2["row_id"]


def test_execution_provenance_sha_changes_when_harness_version_changes_unit():
    """Same assertion as above, at the unit level -- calls
    _execution_provenance_sha directly rather than round-tripping through
    execute()."""
    common = dict(litellm_version="", server_profile={"flags": {}, "template_sha": "x" * 64},
                 rendered_prompt="do the thing")
    sha_a = _execution_provenance_sha(harness_version="opencode-0.1.0", **common)
    sha_b = _execution_provenance_sha(harness_version="opencode-0.2.0", **common)
    assert sha_a != sha_b
    assert len(sha_a) == len(sha_b) == 64


def test_execution_provenance_sha_deterministic_for_same_inputs():
    kwargs = dict(harness_version="opencode-0.1.0", litellm_version="",
                 server_profile={"flags": {"ctx": 32768}, "template_sha": "y" * 64},
                 rendered_prompt="do the thing")
    assert _execution_provenance_sha(**kwargs) == _execution_provenance_sha(**kwargs)


def test_execute_raises_notimplementederror_when_sandbox_enabled(tmp_path):
    """Sandbox seam: flipping suite.yaml b8.sandbox.enabled to true before a
    Node-capable Sandbox image exists must fail loudly, not silently keep
    running on the host."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)
    cfg.suite["b8"]["sandbox"]["enabled"] = True
    ctx = _stub_ctx(tmp_path, cfg, adapter=_mock_adapter(), attempt_id="attempt-fixed")

    with pytest.raises(NotImplementedError):
        B8Harness().execute(item, ctx)


# --- condition helpers --------------------------------------------------------


def test_base_condition_omits_attempt_and_exec_sha():
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    condition = _base_condition(order, "opencode", "edit-01")
    parts = dict(p.split("=", 1) for p in condition.split(";"))
    assert parts == {"cond": "B8", "harness": "opencode", "task": "edit-01"}


def test_full_condition_includes_attempt_and_exec_sha():
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    condition = _full_condition(order, "opencode", "edit-01", "att-1", "f" * 64)
    parts = dict(p.split("=", 1) for p in condition.split(";"))
    assert parts["attempt_id"] == "att-1"
    assert parts["execution_provenance_sha"] == "f" * 64


# --- Regression: compute_row_id additivity (Step-1 assertion #5) -------------


def test_regression_known_b1_b6_row_ids_unchanged():
    """A B8 touch to config/suite.yaml's condition_order/condition_vocab
    must NEVER change what compute_row_id returns for B1/B6 (or any other
    pre-existing battery) -- canonical_condition only emits keys present in
    the caller's own `parts` dict, so appending harness/task/attempt_id/
    execution_provenance_sha to condition_order is inert for every battery
    that never puts those keys in its parts. These two digests are pinned
    literals (fully self-contained inputs, not loaded from live
    config/fixtures) computed against llmtest.schema.compute_row_id BEFORE
    this task's changes landed -- if this test ever fails, compute_row_id's
    hashing algorithm itself changed, which must never happen silently."""
    b6_row_id = schema.compute_row_id(
        suite_version="suite-v2.0.0", model_id="gpt-oss-20b",
        quant_sha256="4e4f9cd88d6456e4f389e7262eca4a8d565211e2b22ece9ca7a8556168ff3c66",
        battery=6, task_id="b6.scratch-01",
        fixture_sha="a" * 64,
        condition="runtime=fork;spec=ngram32;kv=q8;ctx=32k;cond=B6", run_n=1)
    assert b6_row_id == "fd856a152b2863ce0e93984b07f4a25e2b18d50feb794506bf1f6191cd0f9432"

    b1_row_id = schema.compute_row_id(
        suite_version="suite-v2.0.0", model_id="gpt-oss-20b",
        quant_sha256="4e4f9cd88d6456e4f389e7262eca4a8d565211e2b22ece9ca7a8556168ff3c66",
        battery=1, task_id="b1.coding.short.01",
        fixture_sha="b" * 64,
        condition="runtime=fork;spec=ngram32;kv=q8;ctx=32k;cond=B1", run_n=1)
    assert b1_row_id == "d73d39811df299d7510538fe75ef9e994f86609e0023c7225768026f4975b66d"


def test_regression_real_b6_fixture_row_id_unchanged_by_condition_order_growth():
    """Same idea, but against the REAL, currently-loaded suite.yaml/registry
    -- proves growing condition_order for B8 didn't perturb an actual B6
    plan()-computed row_id (B6's condition parts never include the new
    keys, so canonical_condition's output -- and therefore compute_row_id's
    -- is unaffected)."""
    from llmtest.batteries.b6_agenticcoding import B6AgenticCoding

    cfg = load_config(ROOT)
    items = B6AgenticCoding().plan(cfg, FakeStore(), model_filter="gpt-oss-20b")
    assert items
    item = items[0]
    assert "harness=" not in item.condition
    assert "attempt_id=" not in item.condition
    assert "execution_provenance_sha=" not in item.condition
