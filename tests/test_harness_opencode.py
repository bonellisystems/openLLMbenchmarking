"""Fault-injection contract test for the OpenCode harness adapter (Task 4,
Part 2 Phase 2) -- the first LIVE `HarnessAdapter`. No real OpenCode
process, no real endpoint, no Docker: `_launch` (the subprocess seam) is
monkeypatched or fed a `FakeProcess`, and `_read_trace` (the sqlite seam) is
exercised against REAL, minimal on-disk sqlite fixture databases built by
`_make_db` below -- mirroring the actual `session`/`message`/`part` shape
confirmed against a live `~/.local/share/opencode/opencode.db` (see
`docs/superpowers/notes/b8-spike-serverprofile.md`), not a guessed one.
Exercising the real sqlite/JSON parsing against a fixture db (rather than
mocking `_read_trace` itself away) is deliberate: that mapping is the core
deliverable of this task, so the test suite should actually drive it.

The live smoke (real OpenCode, real endpoint) is a separate, manual Step 4
-- not part of this file.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from llmtest.harness import opencode as oc
from llmtest.harness.tasks import B8Task
from llmtest.harness.trace import Trace


# -- fixtures ----------------------------------------------------------------


def _make_task(prompt: str = "Create hello.txt containing HELLO",
                wall_clock_s: int = 120) -> B8Task:
    """A minimal real B8Task -- not a dict -- since `materialize_repo`
    (called from `OpenCodeAdapter.setup`) needs real attribute access
    (`task.setup_repo`), matching `llmtest.harness.tasks`'s actual
    dataclass contract rather than the loose dict `MockHarnessAdapter`'s
    own contract test uses."""
    return B8Task(
        id="edit-01", shape="edit", setup_repo_sha="sha-x",
        allowed_tools=["read_file", "write_file"],
        budgets={"wall_clock_s": wall_clock_s, "tokens": 2000, "steps": 6},
        oracle=["bash", "-c", "true"], protected_shas={},
        task_version="1.0.0", fixture_sha="sha-x",
        setup_repo={"greet.sh": "#!/bin/bash\necho hi\n"},
        oracle_files={"oracle_test.sh": "#!/bin/bash\necho PASS\n"},
        protected_paths=[], allowed_diff_paths=["greet.sh"],
        prompt=prompt, path=None,
    )


def _make_db(path: Path, *, session_id: str, directory: str, session_time: int,
             messages: list[tuple[str, int, dict]],
             parts: list[tuple[str, str, int, dict]]) -> None:
    """Build a minimal real sqlite db with exactly the columns
    `OpenCodeAdapter._read_trace` actually selects -- confirmed against a
    live opencode.db (session.directory/time_created as real columns;
    message.data/part.data as JSON-blob TEXT columns), not the full
    production schema (irrelevant extra columns deliberately omitted so
    this fixture can't accidentally depend on them)."""
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE session (id TEXT, directory TEXT, time_created INTEGER)")
        cur.execute("CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
        cur.execute("CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
        cur.execute("INSERT INTO session VALUES (?, ?, ?)", (session_id, directory, session_time))
        for i, (mid, tc, data) in enumerate(messages):
            cur.execute("INSERT INTO message VALUES (?, ?, ?, ?)",
                        (mid, session_id, tc, json.dumps(data)))
        for i, (pid, mid, tc, data) in enumerate(parts):
            cur.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                        (pid, mid, session_id, tc, json.dumps(data)))
        con.commit()
    finally:
        con.close()


def _make_empty_db(path: Path) -> None:
    _make_db(path, session_id="unused", directory="Z:\\nowhere", session_time=0,
             messages=[], parts=[])
    # Wipe the placeholder session row too -- an "empty" db means NO session
    # matches, not one with a directory that happens never to be queried for.
    con = sqlite3.connect(str(path))
    con.execute("DELETE FROM session")
    con.commit()
    con.close()


class FakeProcess:
    """Stand-in for `subprocess.Popen`'s return value. `.wait()` raises
    `TimeoutExpired` (like a real hung process) when constructed with
    `hang=True`; otherwise returns immediately with `returncode`."""

    def __init__(self, pid: int = 4242, hang: bool = False, returncode: int = 0):
        self.pid = pid
        self.returncode = returncode
        self._hang = hang

    def wait(self, timeout=None):
        if self._hang:
            raise subprocess.TimeoutExpired(cmd="opencode", timeout=timeout)
        return self.returncode


def _new_adapter(tmp_path: Path, db_path: Path, **kwargs) -> oc.OpenCodeAdapter:
    adapter = oc.OpenCodeAdapter(model="gpt-oss-20b", db_path=db_path, **kwargs)
    ws = tmp_path / "ws"
    adapter.setup(_make_task(), endpoint="http://127.0.0.1:8080", workspace=ws)
    return adapter


# -- 1. hang -> killed, and teardown leaves no lingering child --------------


def test_hang_yields_killed_status_and_launch_kills_the_process(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path, wall_clock_s=1)

    fake_proc = FakeProcess(pid=13131, hang=True)
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **k: fake_proc)
    killed = []
    monkeypatch.setattr(oc.subprocess, "run",
                         lambda argv, **k: killed.append(argv) or subprocess.CompletedProcess(argv, 0))

    trace = adapter.run()

    assert isinstance(trace, Trace)
    assert trace.terminal_status in ("killed", "budget-exceeded")
    # _launch must have killed the hung process tree itself (not left it for
    # some later, unspecified cleanup) -- assert via the mocked process
    # handle: taskkill was invoked against exactly this pid.
    assert any(str(fake_proc.pid) in str(argv) for argv in killed)

    # teardown() afterward must not blow up and must leave no dangling
    # process reference on the adapter.
    adapter.teardown()
    assert adapter.process is None


# -- 2. malformed tool call -> recorded, no crash ---------------------------


def test_malformed_tool_call_is_recorded_not_crashed(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    ws = tmp_path / "ws"
    directory = str(ws.resolve())
    session_id = "ses_test1"
    messages = [
        ("m1", 100, {"role": "user"}),
        ("m2", 200, {"role": "assistant", "finish": "stop",
                     "tokens": {"input": 10, "output": 5, "reasoning": 0}}),
    ]
    parts = [
        ("p1", "m2", 150, {"type": "step-start"}),
        # state.status == "error" -- the documented malformed/failed case.
        ("p2", "m2", 160, {"type": "tool", "callID": "c1", "tool": "bash",
                            "state": {"status": "error", "input": {"command": "false"},
                                      "output": "exit code 1"}}),
        # missing/invalid state entirely -- must ALSO not crash.
        ("p3", "m2", 170, {"type": "tool", "callID": "c2", "tool": "write"}),
    ]
    _make_db(db_path, session_id=session_id, directory=directory, session_time=50,
              messages=messages, parts=parts)

    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))
    adapter._since_ts = 0  # fixture predates "now" -- accept any time_created

    trace = adapter.run()  # must not raise

    assert trace.terminal_status == "completed"
    tool_calls = [e for e in trace.events if e.kind == "tool_call"]
    tool_results = [e for e in trace.events if e.kind == "tool_result"]
    assert len(tool_calls) == 2
    assert len(tool_results) == 2
    statuses = {e.payload.get("status") for e in tool_results}
    assert "error" in statuses
    # the missing-state part must be reported as a failure too, not silently
    # dropped or misreported as success.
    assert all(s != "completed" for s in statuses)


# -- 3. endpoint disconnect -> infra-error ----------------------------------


def test_nonzero_launch_exit_yields_infra_error(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (1, False, None))

    trace = adapter.run()

    assert trace.terminal_status == "infra-error"


def test_launch_raising_oserror_yields_infra_error(monkeypatch, tmp_path):
    # e.g. opencode binary missing, or the launch itself throws before a
    # process ever exists -- a distinct failure shape from a clean nonzero exit.
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch",
                         lambda argv, cwd, timeout: (None, False, "OSError: not found"))

    trace = adapter.run()

    assert trace.terminal_status == "infra-error"


def test_provider_error_message_with_no_finish_yields_infra_error(monkeypatch, tmp_path):
    # Confirmed live shape (oc_db_bak fixture): a provider-side failure
    # (e.g. context overflow) surfaces as an assistant message with an
    # "error" field and finish left unset -- opencode itself still exits
    # 0. Must NOT be misclassified as "completed".
    db_path = tmp_path / "opencode.db"
    ws = tmp_path / "ws"
    directory = str(ws.resolve())
    session_id = "ses_test_err"
    messages = [
        ("m1", 100, {"role": "user"}),
        ("m2", 200, {"role": "assistant",
                     "error": {"name": "ContextOverflowError", "data": {"message": "boom"}}}),
    ]
    _make_db(db_path, session_id=session_id, directory=directory, session_time=50,
              messages=messages, parts=[])

    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))
    adapter._since_ts = 0

    trace = adapter.run()

    assert trace.terminal_status == "infra-error"


def test_error_on_earlier_message_does_not_leak_into_completed_status(monkeypatch, tmp_path):
    # A session where an earlier assistant message errored but a LATER one
    # (still in the same session) finished cleanly must be judged by the
    # LAST assistant message only, per the brief's "the last assistant
    # message has a finish" -- not by "an error was seen anywhere".
    db_path = tmp_path / "opencode.db"
    ws = tmp_path / "ws"
    directory = str(ws.resolve())
    session_id = "ses_recover"
    messages = [
        ("m1", 100, {"role": "user"}),
        ("m2", 200, {"role": "assistant",
                     "error": {"name": "ContextOverflowError", "data": {}}}),
        ("m3", 300, {"role": "user"}),
        ("m4", 400, {"role": "assistant", "finish": "stop",
                     "tokens": {"input": 5, "output": 5, "reasoning": 0}}),
    ]
    _make_db(db_path, session_id=session_id, directory=directory, session_time=50,
              messages=messages, parts=[])

    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))
    adapter._since_ts = 0

    trace = adapter.run()

    assert trace.terminal_status == "completed"


# -- 4. missing usage -> tokens fall back, no crash -------------------------


def test_message_missing_tokens_falls_back_to_zero(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    ws = tmp_path / "ws"
    directory = str(ws.resolve())
    session_id = "ses_test2"
    messages = [
        ("m1", 100, {"role": "user"}),
        # no "tokens" key at all -- the documented missing-usage case.
        ("m2", 200, {"role": "assistant", "finish": "stop"}),
    ]
    _make_db(db_path, session_id=session_id, directory=directory, session_time=50,
              messages=messages, parts=[("p1", "m2", 150, {"type": "step-start"})])

    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))
    adapter._since_ts = 0

    trace = adapter.run()  # must not raise (e.g. no KeyError on tokens["input"])

    assert trace.terminal_status == "completed"
    assert trace.tokens_prompt == 0
    assert trace.tokens_completion == 0
    # recorded, not silently swallowed -- lives in the terminal event payload
    # since Trace's own schema (Task 1) is frozen and has no such field.
    terminal_events = [e for e in trace.events if e.kind == "terminal"]
    assert terminal_events and terminal_events[0].payload.get("missing_usage") is True


# -- 5. teardown always leaves no lingering child ---------------------------


def test_teardown_kills_lingering_process_and_clears_handle(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)

    fake_proc = FakeProcess(pid=9999, hang=True)
    adapter.process = fake_proc  # simulate a still-live process left over from a prior run
    killed = []
    monkeypatch.setattr(oc.subprocess, "run",
                         lambda argv, **k: killed.append(argv) or subprocess.CompletedProcess(argv, 0))

    adapter.teardown()

    assert any(str(fake_proc.pid) in str(argv) for argv in killed)
    assert adapter.process is None

    # idempotent -- a second teardown() with nothing left to kill must not
    # blow up and must not re-invoke taskkill needlessly.
    adapter.teardown()
    assert len(killed) == 1


def test_teardown_with_no_process_is_a_safe_no_op(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)
    calls = []
    monkeypatch.setattr(oc.subprocess, "run", lambda *a, **k: calls.append(a) or None)

    adapter.teardown()  # never launched anything -- must not raise

    assert calls == []


# -- supporting: happy-path sqlite mapping (turns/tool events/subagent) -----


def test_happy_path_maps_steps_tool_events_and_subagent_spawn(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    ws = tmp_path / "ws"
    directory = str(ws.resolve())
    session_id = "ses_happy"
    messages = [
        ("m1", 100, {"role": "user"}),
        ("m2", 200, {"role": "assistant", "finish": "stop",
                     "tokens": {"input": 40, "output": 12, "reasoning": 3}}),
    ]
    parts = [
        ("p1", "m2", 110, {"type": "step-start"}),
        ("p2", "m2", 120, {"type": "tool", "callID": "c1", "tool": "write",
                            "state": {"status": "completed",
                                      "input": {"filePath": "hello.txt", "content": "HELLO"},
                                      "output": "Wrote file successfully."}}),
        ("p3", "m2", 130, {"type": "tool", "callID": "c2", "tool": "task",
                            "state": {"status": "completed", "input": {}, "output": "done"}}),
        ("p4", "m2", 140, {"type": "step-start"}),
    ]
    _make_db(db_path, session_id=session_id, directory=directory, session_time=50,
              messages=messages, parts=parts)

    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))
    adapter._since_ts = 0

    trace = adapter.run()

    assert trace.terminal_status == "completed"
    assert trace.steps == 2  # two step-start parts
    assert trace.tokens_prompt == 40
    assert trace.tokens_completion == 12
    assert trace.subagent_spawned == "yes"  # tool == "task" was invoked
    subagent_events = [e for e in trace.events if e.kind == "subagent_spawn"]
    assert len(subagent_events) == 1
    tool_calls = [e for e in trace.events if e.kind == "tool_call"]
    assert {e.payload["tool"] for e in tool_calls} == {"write", "task"}


def test_no_matching_session_yields_empty_trace_not_crash(monkeypatch, tmp_path):
    # opencode wrote nothing correlating to this workspace at all (e.g. it
    # crashed before ever creating a session) -- must not raise.
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))

    trace = adapter.run()

    assert trace.steps == 0
    assert trace.tokens_prompt == 0
    assert trace.tokens_completion == 0
    assert trace.subagent_spawned == "no"


def test_version_reports_opencode_cli_version(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)

    def fake_run(argv, **kwargs):
        assert argv[-1] == "--version"
        return subprocess.CompletedProcess(argv, 0, stdout="1.2.15\n", stderr="")

    monkeypatch.setattr(oc.subprocess, "run", fake_run)
    assert adapter.version() == "1.2.15"


def test_version_falls_back_when_opencode_binary_missing(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)

    def fake_run(argv, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(oc.subprocess, "run", fake_run)
    assert adapter.version() == "unknown"


# -- npm .cmd shim unwrap (the live-smoke-surfaced Windows bug) -------------


def test_unwrap_npm_cmd_shim_finds_node_exe_and_entry(tmp_path):
    # Mirrors the real shim content read live from
    # C:\Users\...\AppData\Roaming\npm\opencode.cmd (the bug this fixes).
    shim = tmp_path / "opencode.cmd"
    shim.write_text(
        '@ECHO off\r\n'
        'GOTO start\r\n'
        ':find_dp0\r\n'
        'SET dp0=%~dp0\r\n'
        'EXIT /b\r\n'
        ':start\r\n'
        'SETLOCAL\r\n'
        'CALL :find_dp0\r\n\r\n'
        'IF EXIST "%dp0%\\node.exe" (\r\n'
        '  SET "_prog=%dp0%\\node.exe"\r\n'
        ') ELSE (\r\n'
        '  SET "_prog=node"\r\n'
        ')\r\n\r\n'
        'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  '
        '"%dp0%\\node_modules\\opencode-ai\\bin\\opencode" %*\r\n',
        encoding="utf-8")
    (tmp_path / "node.exe").write_bytes(b"")  # just needs to exist
    entry_dir = tmp_path / "node_modules" / "opencode-ai" / "bin"
    entry_dir.mkdir(parents=True)
    (entry_dir / "opencode").write_bytes(b"")

    result = oc._unwrap_npm_cmd_shim(str(shim))

    assert result == [str(tmp_path / "node.exe"), str(entry_dir / "opencode")]


def test_unwrap_npm_cmd_shim_returns_none_for_non_shim_file(tmp_path):
    plain = tmp_path / "something.cmd"
    plain.write_text("@ECHO off\r\necho hello\r\n", encoding="utf-8")
    assert oc._unwrap_npm_cmd_shim(str(plain)) is None


def test_unwrap_npm_cmd_shim_returns_none_for_non_cmd_extension(tmp_path):
    exe = tmp_path / "opencode.exe"
    exe.write_bytes(b"")
    assert oc._unwrap_npm_cmd_shim(str(exe)) is None


def test_launch_uses_unwrapped_node_argv_not_the_cmd_shim(monkeypatch, tmp_path):
    # End-to-end through _launch (not just the parser): a multi-line
    # prompt must reach Popen as ONE argv element targeting node.exe
    # directly, never routed through the .cmd shim (which would silently
    # truncate it at the first newline -- the actual live bug).
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path, wall_clock_s=5)

    shim_dir = tmp_path / "npmbin"
    shim_dir.mkdir()
    shim = shim_dir / "opencode.cmd"
    shim.write_text('"%dp0%\\node_modules\\opencode-ai\\bin\\opencode" %*\r\n', encoding="utf-8")
    (shim_dir / "node.exe").write_bytes(b"")
    entry_dir = shim_dir / "node_modules" / "opencode-ai" / "bin"
    entry_dir.mkdir(parents=True)
    (entry_dir / "opencode").write_bytes(b"")

    monkeypatch.setattr(oc.shutil, "which", lambda cmd: str(shim) if cmd == "opencode" else None)
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        return FakeProcess(pid=1, hang=False, returncode=0)

    monkeypatch.setattr(oc.subprocess, "Popen", fake_popen)

    multiline_prompt = "line one\nline two\nline three"
    adapter._launch(["opencode", "run", multiline_prompt, "-m", "local/gpt-oss-20b"],
                     cwd=adapter.workspace, timeout=5)

    assert captured["argv"][0] == str(shim_dir / "node.exe")
    assert captured["argv"][1] == str(entry_dir / "opencode")
    # the full multi-line prompt survives intact as ONE argv element, and
    # the trailing -m flag is still present -- neither casualty from the
    # cmd-shim bug this test guards against.
    assert multiline_prompt in captured["argv"]
    assert captured["argv"][-2:] == ["-m", "local/gpt-oss-20b"]


# -- #1 (Important, post-review): opencode.json must NOT land in the -------
# -- agent-visible workspace -- delivered via OPENCODE_CONFIG env var only -


def test_opencode_config_not_written_into_agent_workspace(tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)

    # run_oracle's diff constraint (tasks.py) flags any new file not in
    # allowed_diff_paths as an out-of-bounds edit -- a leftover
    # opencode.json in the agent-visible workspace would silently FAIL an
    # otherwise-correct run. Only the task's own materialized files may be
    # present after setup().
    workspace_files = {p.name for p in adapter.workspace.iterdir()}
    assert "opencode.json" not in workspace_files
    assert workspace_files == {"greet.sh"}


def test_opencode_config_written_as_sibling_and_passed_via_env(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)

    # config exists on disk, but OUTSIDE the workspace (a sibling, mirroring
    # _log_path), never under /workspace.
    assert adapter._config_path is not None
    assert adapter._config_path.exists()
    assert adapter._config_path.parent == adapter.workspace.parent
    cfg = json.loads(adapter._config_path.read_text(encoding="utf-8"))
    assert cfg["provider"]["local"]["options"]["baseURL"] == "http://127.0.0.1:8080/v1"

    captured = {}

    def fake_popen(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return FakeProcess(pid=1, hang=False, returncode=0)

    monkeypatch.setattr(oc.subprocess, "Popen", fake_popen)
    adapter._launch(["opencode", "run", "hi", "-m", "local/gpt-oss-20b"],
                     cwd=adapter.workspace, timeout=5)

    assert captured["env"] is not None
    assert captured["env"].get("OPENCODE_CONFIG") == str(adapter._config_path)


# -- Wave 1a: native OpenCode step-limit config (agent.build.steps) --------


def test_write_opencode_config_sets_agent_build_steps_when_max_steps_given(tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path, max_steps=20)

    cfg = json.loads(adapter._config_path.read_text(encoding="utf-8"))
    # "build" -- the built-in agent `opencode run` uses when no --agent
    # flag is passed (confirmed against the real anomalyco/opencode
    # source); OpenCode's config layer overrides a built-in agent's fields
    # by reusing its name as the config key.
    assert cfg["agent"]["build"]["steps"] == 20


def test_write_opencode_config_omits_agent_key_when_max_steps_not_given(tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)  # no max_steps kwarg -> None

    cfg = json.loads(adapter._config_path.read_text(encoding="utf-8"))
    assert "agent" not in cfg


# -- #2 (Important, post-review): session correlation must not misclassify -
# -- a genuinely completed run as infra-error -------------------------------


def test_session_correlated_via_git_root_ancestor_not_misclassified(monkeypatch, tmp_path):
    # workspace nested inside some ancestor dir OpenCode recorded as
    # `session.directory` instead of the literal workspace cwd (the
    # hypothesized git-worktree-root shape) -- must still correlate to a
    # completed run, not silently fall to infra-error.
    db_path = tmp_path / "opencode.db"
    ws = tmp_path / "repo" / "ws"
    ws.mkdir(parents=True)
    ancestor_dir = str((tmp_path / "repo").resolve())
    session_id = "ses_repo"
    messages = [
        ("m1", 100, {"role": "user"}),
        ("m2", 200, {"role": "assistant", "finish": "stop",
                     "tokens": {"input": 10, "output": 5, "reasoning": 0}}),
    ]
    _make_db(db_path, session_id=session_id, directory=ancestor_dir, session_time=50,
              messages=messages, parts=[("p1", "m2", 150, {"type": "step-start"})])

    adapter = oc.OpenCodeAdapter(model="gpt-oss-20b", db_path=db_path)
    adapter.setup(_make_task(), endpoint="http://127.0.0.1:8080", workspace=ws)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))
    adapter._since_ts = 0

    trace = adapter.run()

    assert trace.terminal_status == "completed"
    assert trace.steps == 1


def test_session_time_based_fallback_when_directory_correlation_totally_fails(monkeypatch, tmp_path):
    # directory matches NOTHING -- not exact, not case-insensitive, not an
    # ancestor (e.g. a future OpenCode version records `directory` in some
    # other shape entirely). As long as exactly one session was created in
    # the post-launch time window (the adapter launches and awaits exactly
    # one opencode subprocess per run), it must still be used -- never
    # losing a genuinely-completed run to a directory-correlation miss.
    db_path = tmp_path / "opencode.db"
    ws = tmp_path / "ws"
    session_id = "ses_unrelated_dir"
    messages = [
        ("m1", 100, {"role": "user"}),
        ("m2", 200, {"role": "assistant", "finish": "stop",
                     "tokens": {"input": 3, "output": 2, "reasoning": 0}}),
    ]
    _make_db(db_path, session_id=session_id, directory="Z:\\totally\\unrelated\\path",
              session_time=50, messages=messages, parts=[])

    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(adapter, "_launch", lambda argv, cwd, timeout: (0, False, None))
    adapter._since_ts = 0

    trace = adapter.run()

    assert trace.terminal_status == "completed"


# -- #5 (trivial, post-review): _launch resets self.process once handled --


def test_launch_resets_process_handle_after_clean_exit(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(oc.subprocess, "Popen",
                         lambda *a, **k: FakeProcess(pid=1, hang=False, returncode=0))

    adapter._launch(["opencode", "run", "hi", "-m", "local/gpt-oss-20b"],
                     cwd=adapter.workspace, timeout=5)

    assert adapter.process is None


def test_launch_resets_process_handle_after_kill_so_teardown_is_a_true_noop(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    _make_empty_db(db_path)
    adapter = _new_adapter(tmp_path, db_path)
    monkeypatch.setattr(oc.subprocess, "Popen",
                         lambda *a, **k: FakeProcess(pid=1, hang=True))
    killed = []
    monkeypatch.setattr(oc.subprocess, "run",
                         lambda argv, **k: killed.append(argv) or subprocess.CompletedProcess(argv, 0))

    adapter._launch(["opencode", "run", "hi", "-m", "local/gpt-oss-20b"],
                     cwd=adapter.workspace, timeout=1)
    assert adapter.process is None
    assert len(killed) == 1

    adapter.teardown()  # must be a genuine no-op now -- no second taskkill
    assert len(killed) == 1
