"""Tests for task-b8local -- real local B8 run enablement:

  1. The endpoint-injection seam (`ctx.b8_endpoint`) added to
     `llmtest/batteries/b8_harness.py::execute()`.
  2. The `b8.sandbox.oracle_image` config -> execute() -> the REAL
     `llmtest.harness.tasks.run_oracle` threading proof (a lower-level,
     docker-argv-capturing proof of the same threading also lives in
     tests/test_harness_tasks.py, mirroring test_harness_sandbox.py's
     container-hardening test pattern).
  3. The 3 new real Python task manifests (task-06..08.yaml, task-b8local)
     PLUS the 3 more added by task-b8expand (task-09..11.yaml) load via
     `load_b8_tasks` with every required key, and their hidden oracles are
     never written into the agent-visible workspace. (The task-b8expand
     Python-oracle subprocess-isolation hardening itself is exercised in
     tests/test_harness_tasks.py, both hermetically and -- since Docker is
     reachable here -- against the real python:3.11-slim container.)
  4. A dry-run-safe smoke test of `scripts/run_b8_local.py` -- importing it
     and parsing `--help` (or a --task filter that matches nothing) must
     never launch a subprocess, hit the network, or touch Docker.

Mirrors tests/test_b8.py's FakeStore/SimpleNamespace-ctx-injection pattern
throughout -- no live harness process, no live endpoint, no Docker anywhere
in this file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import llmtest.batteries.b8_harness as b8h
from llmtest import schema
from llmtest.batteries import b8_fixtures as b8f
from llmtest.batteries.b8_harness import B8Harness
from llmtest.harness.base import MockHarnessAdapter
from llmtest.harness.tasks import materialize_repo
from llmtest.harness.trace import TraceEvent
from llmtest.registry import load_config

ROOT = Path(__file__).resolve().parents[1]

# The first 3 real Python task manifests (task-b8local, task-06..08.yaml).
_NEW_PY_TASK_IDS = ("py-bugfix-01", "py-fromscratch-01", "py-edit-01")
# The 3 more real Python task manifests added by task-b8expand
# (task-09..11.yaml) -- multi-file/tool-heavy in Python for the first
# time, plus a harder from-scratch task.
_B8EXPAND_NEW_PY_TASK_IDS = ("py-multifile-01", "py-toolheavy-01", "py-fromscratch-02")
# All 6 original real Python task manifests. NOTE (task-b8hard): these are
# no longer suite.yaml's b8.tasks allowlist -- that now targets 5 HARDER
# manifests instead (task-12..16.yaml, ids `py-hard-*`; see
# test_suite_yaml_b8_tasks_allowlist_is_the_five_hard_python_task_ids in
# test_b8.py) because these original 6 are all solved 30/30 by gpt-oss-20b
# and don't discriminate. This tuple is still exactly right for THIS
# file's own purpose: exercising the task-b8local/task-b8expand real-run
# enablement plumbing against manifests known to load/materialize/execute
# correctly, independent of whatever suite.yaml's allowlist currently
# targets.
_ALL_PY_TASK_IDS = _NEW_PY_TASK_IDS + _B8EXPAND_NEW_PY_TASK_IDS


class FakeStore:
    """No seeded rows -- mirrors test_b8.py's FakeStore, for plan()."""
    def iter_rows(self):
        return []


def _mock_adapter(*, terminal_status="completed", tokens_prompt=10,
                  tokens_completion=5, subagent_spawned="no"):
    events = [TraceEvent(kind="turn"),
             TraceEvent(kind="tool_call", payload={"tool": "bash_exec"}),
             TraceEvent(kind="tool_result", payload={"status": "completed"})]
    return MockHarnessAdapter(scripted_events=events, terminal_status=terminal_status,
                              tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
                              subagent_spawned=subagent_spawned)


def _first_item(cfg, model_id=None):
    model_id = model_id or cfg.suite["b8"]["models"][0]
    items = B8Harness().plan(cfg, FakeStore(), model_filter=model_id)
    assert items, "plan() produced no WorkItems -- fixtures/config missing?"
    return items[0]


def _explode(*_a, **_k):
    raise AssertionError("server_manager() must not be called when ctx.b8_endpoint is set")


# ---------------------------------------------------------------------------
# 1. Endpoint-injection seam
# ---------------------------------------------------------------------------


def test_execute_uses_injected_endpoint_and_never_calls_server_manager(tmp_path):
    """`ctx.b8_endpoint`, when present, is used AS-IS; `ctx.server_manager()`
    is never invoked at all -- proven by making it raise if it were."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)

    endpoint = SimpleNamespace(base_url="http://127.0.0.1:9999", session_id="s-manual",
                               normalized_config={"runtime": "manual", "ctx": 40960})
    ctx = SimpleNamespace(
        cfg=cfg, root=tmp_path, server_manager=_explode,
        b8_endpoint=endpoint, b8_adapters={"opencode": _mock_adapter()},
        b8_run_oracle=lambda task, workspace, root=".": (True, "stub-pass"),
        b8_attempt_id="attempt-fixed")

    row = B8Harness().execute(item, ctx)[0]

    assert schema.validate_row(row) == []
    assert row["session_id"] == "s-manual"
    assert row["sampling"]["harness"] == "opencode"
    assert row["metrics"]["completion"] is True


def test_execute_falls_back_to_server_manager_when_no_endpoint_injected(tmp_path):
    """Flip side (regression against every OTHER B8 test / the real
    run_cmd.py path): a ctx with NO b8_endpoint attribute at all must still
    reach ctx.server_manager().request_endpoint(...), unchanged."""
    cfg = load_config(ROOT)
    item = _first_item(cfg)

    calls = []

    class StubHandle:
        session_id = "s-stub"
        base_url = "http://127.0.0.1:9/"
        normalized_config = {"ctx": 32768, "kv_dtype": "q8_0"}

    class StubMgr:
        def request_endpoint(self, *a, **k):
            calls.append((a, k))
            return StubHandle()

    ctx = SimpleNamespace(
        cfg=cfg, root=tmp_path, server_manager=lambda: StubMgr(),
        b8_adapters={"opencode": _mock_adapter()},
        b8_run_oracle=lambda task, workspace, root=".": (True, "stub-pass"),
        b8_attempt_id="attempt-fixed")
    # deliberately no ctx.b8_endpoint

    row = B8Harness().execute(item, ctx)[0]
    assert calls, "server_manager().request_endpoint() was never called"
    assert row["session_id"] == "s-stub"


# ---------------------------------------------------------------------------
# 2. oracle_image threading (execute() -> the REAL run_oracle)
# ---------------------------------------------------------------------------


def test_execute_threads_suite_yaml_oracle_image_to_real_run_oracle(tmp_path, monkeypatch):
    """Proves suite.yaml's b8.sandbox.oracle_image reaches the REAL
    llmtest.harness.tasks.run_oracle (not an injected ctx.b8_run_oracle,
    which _resolve_run_oracle deliberately wraps to swallow the kwarg --
    every existing test's injected callable keeps its original,
    oracle_image-agnostic signature). Monkeypatches b8_harness's own
    module-level `run_oracle` name (bound at import time via `from
    llmtest.harness.tasks import ... run_oracle`) with a spy, so this needs
    no Docker."""
    cfg = load_config(ROOT)
    cfg.suite["b8"]["sandbox"]["oracle_image"] = "python:3.11-slim"
    item = _first_item(cfg)

    captured = {}

    def fake_run_oracle(task, workspace, *, root=".", oracle_image=None):
        captured["oracle_image"] = oracle_image
        return True, "stub-pass"

    monkeypatch.setattr(b8h, "run_oracle", fake_run_oracle)

    endpoint = SimpleNamespace(base_url="http://127.0.0.1:9999", session_id="s-manual",
                               normalized_config={})
    ctx = SimpleNamespace(cfg=cfg, root=tmp_path, b8_endpoint=endpoint,
                          b8_adapters={"opencode": _mock_adapter()},
                          b8_attempt_id="attempt-fixed")
    # deliberately NOT setting ctx.b8_run_oracle -- exercises the real
    # resolution path (_resolve_run_oracle falls through to the
    # module-level `run_oracle` name, monkeypatched above).

    row = B8Harness().execute(item, ctx)[0]
    assert captured.get("oracle_image") == "python:3.11-slim"
    assert row["metrics"]["completion"] is True


def test_execute_passes_none_oracle_image_when_suite_yaml_key_absent(tmp_path, monkeypatch):
    """Additivity: if b8.sandbox.oracle_image is ever removed from
    suite.yaml, execute() must still call the real run_oracle cleanly with
    oracle_image=None (the pin fallback), not KeyError."""
    cfg = load_config(ROOT)
    cfg.suite["b8"]["sandbox"].pop("oracle_image", None)
    item = _first_item(cfg)

    captured = {}

    def fake_run_oracle(task, workspace, *, root=".", oracle_image=None):
        captured["oracle_image"] = oracle_image
        return True, "stub-pass"

    monkeypatch.setattr(b8h, "run_oracle", fake_run_oracle)

    endpoint = SimpleNamespace(base_url="http://127.0.0.1:9999", session_id="s-manual",
                               normalized_config={})
    ctx = SimpleNamespace(cfg=cfg, root=tmp_path, b8_endpoint=endpoint,
                          b8_adapters={"opencode": _mock_adapter()},
                          b8_attempt_id="attempt-fixed")

    row = B8Harness().execute(item, ctx)[0]
    assert captured.get("oracle_image") is None
    assert row["metrics"]["completion"] is True


# ---------------------------------------------------------------------------
# 3. The Python task manifests (task-b8local's 3 + task-b8expand's 3 more)
# ---------------------------------------------------------------------------


def test_all_three_new_python_task_ids_are_discovered():
    all_ids = {task.id for task in b8f.load_tasks(ROOT)}
    for task_id in _NEW_PY_TASK_IDS:
        assert task_id in all_ids, (
            f"{task_id} not discovered by load_b8_tasks -- filename glob "
            f"mismatch? (loader matches 'task-*.yaml', not 'pytask-*.yaml')")


def test_all_three_b8expand_new_python_task_ids_are_discovered():
    all_ids = {task.id for task in b8f.load_tasks(ROOT)}
    for task_id in _B8EXPAND_NEW_PY_TASK_IDS:
        assert task_id in all_ids, (
            f"{task_id} not discovered by load_b8_tasks -- filename glob "
            f"mismatch? (loader matches 'task-*.yaml', not 'pytask-*.yaml')")


@pytest.mark.parametrize("task_id", _ALL_PY_TASK_IDS)
def test_new_python_manifest_has_required_fields(task_id):
    all_tasks = {task.id: task for task in b8f.load_tasks(ROOT)}
    task = all_tasks[task_id]

    assert task.shape in {"bugfix", "from-scratch", "edit", "multi-file", "tool-heavy"}
    assert task.setup_repo_sha and len(task.setup_repo_sha) == 64
    assert isinstance(task.allowed_tools, list) and task.allowed_tools
    for key in ("wall_clock_s", "tokens", "steps"):
        assert key in task.budgets
    assert task.protected_shas, "protected_paths must be non-empty"
    assert task.oracle and task.oracle[0] == "bash"
    # oracle argv mirrors the bash manifests' exact convention, swapping
    # "bash oracle_test.sh" for "python3 oracle_test.py"
    oracle_cmd = " ".join(task.oracle)
    assert "cp -r /oracle /tmp/work" in oracle_cmd
    assert "python3 oracle_test.py" in oracle_cmd
    assert "oracle_test.py" in task.oracle_files
    assert "oracle_test.sh" not in task.oracle_files


@pytest.mark.parametrize("task_id", _ALL_PY_TASK_IDS)
def test_new_python_manifest_oracle_withheld_from_agent_workspace(tmp_path, task_id):
    """Mirrors test_harness_tasks.py's
    test_materialize_repo_never_writes_oracle_files, scoped to just the 6
    real Python manifests: materialize_repo must never write any
    oracle_files path into the agent-visible workspace."""
    all_tasks = {task.id: task for task in b8f.load_tasks(ROOT)}
    task = all_tasks[task_id]

    ws = tmp_path / task_id
    materialize_repo(task, ws)

    for oracle_path in task.oracle_files:
        assert not (ws / oracle_path).exists(), (
            f"{task_id}: oracle file {oracle_path!r} leaked into the "
            f"agent-visible workspace")


def test_new_python_manifests_use_stdlib_only_no_pytest():
    """The oracle scripts must run under bare `python:3.11-slim` (no
    pytest installed) -- guards against a future edit accidentally adding
    a pytest dependency to one of these oracle scripts."""
    all_tasks = {task.id: task for task in b8f.load_tasks(ROOT)}
    for task_id in _ALL_PY_TASK_IDS:
        oracle_src = all_tasks[task_id].oracle_files["oracle_test.py"]
        assert "pytest" not in oracle_src
        assert "import sys" in oracle_src


def test_new_python_manifests_use_subprocess_isolation_not_bare_import():
    """task-b8expand hardening (codex review Important #1's Python
    analog): every Python oracle_test.py must run the candidate solution
    in a SUBPROCESS (`subprocess.run([sys.executable, ...])`), never a
    bare top-level `import <solution module>` straight into the
    checker's own process -- the latter lets a module-level
    `sys.exit(0)` in the solution kill the whole checker with exit code
    0 before any check runs (`SystemExit` is not caught by `except
    Exception`), which `Sandbox.hidden_validate` (exit-code-only) would
    wrongly register as a pass."""
    all_tasks = {task.id: task for task in b8f.load_tasks(ROOT)}
    for task_id in _ALL_PY_TASK_IDS:
        oracle_src = all_tasks[task_id].oracle_files["oracle_test.py"]
        assert "import subprocess" in oracle_src, (
            f"{task_id}: oracle_test.py does not import subprocess -- "
            f"not hardened against in-process import short-circuiting")
        assert "subprocess.run" in oracle_src


# ---------------------------------------------------------------------------
# 4. scripts/run_b8_local.py -- dry-run-safe smoke
# ---------------------------------------------------------------------------


def _load_runner_module():
    """Import scripts/run_b8_local.py by path (scripts/ isn't a package) --
    mirrors tests/test_report_b8.py's identical trick for scripts/p8_report.py.
    Module-level code in run_b8_local.py is limited to imports/defs (no
    top-level side effects), so this exec is itself dry-run-safe."""
    spec = importlib.util.spec_from_file_location(
        "run_b8_local", ROOT / "scripts" / "run_b8_local.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_b8_local", mod)
    spec.loader.exec_module(mod)
    return mod


def test_runner_imports_cleanly_with_no_side_effects():
    mod = _load_runner_module()
    assert hasattr(mod, "main")
    assert hasattr(mod, "build_parser")
    assert hasattr(mod, "ManualEndpoint")


def test_runner_help_exits_zero_without_launching_anything():
    mod = _load_runner_module()
    with pytest.raises(SystemExit) as exc_info:
        mod.build_parser().parse_args(["--help"])
    assert exc_info.value.code == 0


def test_runner_normalize_base_url_strips_trailing_v1():
    """The double-/v1 risk: OpenCodeAdapter._write_opencode_config appends
    '/v1' itself, so an --endpoint-url already ending in /v1 must be
    stripped down to the bare origin before it becomes ctx.b8_endpoint.
    base_url, or every real OpenCode call breaks silently."""
    mod = _load_runner_module()
    assert mod._normalize_base_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert mod._normalize_base_url("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    assert mod._normalize_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert mod._normalize_base_url("http://127.0.0.1:8080/") == "http://127.0.0.1:8080"


def test_runner_manual_endpoint_has_required_contract_fields():
    """The injected-endpoint contract OpenCodeAdapter.setup() and
    b8_harness.execute() actually read: base_url, session_id,
    normalized_config."""
    mod = _load_runner_module()
    ep = mod.ManualEndpoint(base_url="http://127.0.0.1:8080", session_id="s1",
                            normalized_config={"ctx": 40960})
    assert ep.base_url == "http://127.0.0.1:8080"
    assert ep.session_id == "s1"
    assert ep.normalized_config == {"ctx": 40960}


def test_runner_main_returns_early_when_task_filter_matches_nothing(capsys):
    """Exercises the REAL main() end-to-end (real load_config/plan(), no
    mocking) with a --task filter that matches zero WorkItems -- proves the
    empty-filter path returns before ever reaching battery.execute(), so
    this stays dry-run-safe (no network, no Docker, no opencode subprocess)
    despite calling the real entry point."""
    mod = _load_runner_module()
    rc = mod.main(["--endpoint-url", "http://127.0.0.1:1",
                   "--task", "definitely-not-a-real-b8-task-id"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0 planned" in out


# ---------------------------------------------------------------------------
# 4b. main() loop -- infra-error retry (Codex review): an infra-error is a
# transient HARNESS/serving-layer failure, so the SAME item is re-executed
# up to 3 attempts total and only the LAST attempt's rows are appended. All
# fakes (no live endpoint / Docker / opencode) via module-level monkeypatch,
# same discipline as the rest of this file.
# ---------------------------------------------------------------------------


class _FakeItem:
    model_id = "model-a"
    task_id = "b8.py-bugfix-01"
    run_n = 1


def _fake_metrics_row(terminal_status, completion):
    infra = terminal_status == "infra-error"
    return {"row_id": f"rid-{terminal_status}-{completion}",
            "metrics": {"completion": completion, "terminal_status": terminal_status,
                        "steps": 1 if infra else 4,
                        "tokens_prompt": 0 if infra else 100,
                        "tokens_completion": 0 if infra else 50}}


class _FakeBattery:
    """plan() yields one WorkItem; execute() returns whatever the scripted
    `terminal_statuses` sequence says (one row per call), recording how many
    times it was called."""
    def __init__(self, terminal_statuses):
        self._scripted = list(terminal_statuses)
        self.calls = 0

    def plan(self, cfg, store, model_filter=None, force=False):
        return [_FakeItem()]

    def execute(self, item, ctx):
        ts = self._scripted[self.calls]
        self.calls += 1
        return [_fake_metrics_row(ts, completion=(ts == "completed"))]


class _FakeCfg:
    suite = {"b8": {"ctx": 40960, "kv": "q8_0"}}


def _install_runner_fakes(mod, monkeypatch, terminal_statuses):
    """Swap the runner module's B8Harness/Store/RunContext/load_config for
    fakes, returning (fake_battery, appended_rows). Nothing live is touched:
    battery.execute() is scripted, Store.append() just records."""
    appended: list = []

    class _FakeStore:
        def __init__(self, *a, **k):
            pass

        def append(self, row):
            appended.append(row)
            return True

    class _FakeCtx:
        def __init__(self, **k):
            pass

    battery = _FakeBattery(terminal_statuses)
    monkeypatch.setattr(mod, "load_config", lambda root: _FakeCfg())
    monkeypatch.setattr(mod, "Store", _FakeStore)
    monkeypatch.setattr(mod, "RunContext", _FakeCtx)
    monkeypatch.setattr(mod, "B8Harness", lambda: battery)
    return battery, appended


def test_runner_retries_infra_error_then_appends_only_the_completed_row(monkeypatch, capsys):
    """Two infra-errors then a `completed` row: execute() is called 3 times
    and EXACTLY ONE row -- the final `completed` one -- is appended (the two
    discarded infra-error attempts are never stored)."""
    mod = _load_runner_module()
    battery, appended = _install_runner_fakes(
        mod, monkeypatch, ["infra-error", "infra-error", "completed"])

    rc = mod.main(["--endpoint-url", "http://127.0.0.1:8080"])

    assert battery.calls == 3                       # 1 initial + 2 retries
    assert len(appended) == 1                        # only the LAST attempt
    assert appended[0]["metrics"]["terminal_status"] == "completed"
    out = capsys.readouterr().out
    assert "RETRY infra-error attempt 1/3" in out
    assert "RETRY infra-error attempt 2/3" in out
    assert rc == 0


def test_runner_recovers_on_first_retry_without_exhausting_attempts(monkeypatch, capsys):
    """One infra-error then a `completed` row: the retry recovers on
    attempt 2, so execute() is called only twice (no needless 3rd attempt)
    and just the completed row is appended."""
    mod = _load_runner_module()
    battery, appended = _install_runner_fakes(
        mod, monkeypatch, ["infra-error", "completed"])

    rc = mod.main(["--endpoint-url", "http://127.0.0.1:8080"])

    assert battery.calls == 2
    assert len(appended) == 1
    assert appended[0]["metrics"]["terminal_status"] == "completed"
    out = capsys.readouterr().out
    assert "RETRY infra-error attempt 1/3" in out
    assert "RETRY infra-error attempt 2/3" not in out
    assert rc == 0


def test_runner_appends_last_infra_error_when_all_attempts_exhausted(monkeypatch, capsys):
    """All 3 attempts infra-error: execute() is called 3 times and the LAST
    infra-error row IS appended (kept as excluded provenance -- p8_report's
    eligibility rule drops it from the k/N denominator, not the run loop)."""
    mod = _load_runner_module()
    battery, appended = _install_runner_fakes(
        mod, monkeypatch, ["infra-error", "infra-error", "infra-error"])

    rc = mod.main(["--endpoint-url", "http://127.0.0.1:8080"])

    assert battery.calls == 3
    assert len(appended) == 1
    assert appended[0]["metrics"]["terminal_status"] == "infra-error"
    out = capsys.readouterr().out
    assert "RETRY infra-error attempt 1/3" in out
    assert "RETRY infra-error attempt 2/3" in out
    assert rc == 0                                    # infra-error is not an EXEC-ERROR


def test_runner_does_not_retry_a_completed_row(monkeypatch, capsys):
    """A first-attempt `completed` row is appended immediately with no
    retry (guards against retrying non-infra terminal statuses)."""
    mod = _load_runner_module()
    battery, appended = _install_runner_fakes(mod, monkeypatch, ["completed"])

    rc = mod.main(["--endpoint-url", "http://127.0.0.1:8080"])

    assert battery.calls == 1
    assert len(appended) == 1
    out = capsys.readouterr().out
    assert "RETRY infra-error" not in out
    assert rc == 0
