"""Contract test for the container sandbox (Task 2, Part 2 Phase 1 B8): the
security boundary a real agent-harness run will (Phase 2, deferred) execute
inside. Docker Desktop/WSL2 required for the container-behavior tests --
`pytest.mark.skipif` guards those so CI without Docker skips cleanly. The
pin-loading test needs no Docker and always runs.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from llmtest.harness.sandbox import Sandbox


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


requires_docker = pytest.mark.skipif(not _docker_available(), reason="Docker not reachable")


def test_default_pin_loaded_from_runtime_pins_yaml():
    """No Docker needed -- just config wiring. Sandbox() with no explicit
    image/digest reads config/runtime_pins.yaml's `sandbox:` block."""
    sbx = Sandbox(workspace=Path("does-not-need-to-exist-for-this-check"))
    assert sbx.image == "nvidia/cuda:12.6.2-base-ubuntu24.04"
    assert sbx.digest == (
        "sha256:631ec7090c36ab846cf021073ff4a64fb9cffa90b4f9f0083799288c607073ce"
    )


def test_hidden_validate_command_oracle_container_is_hardened(tmp_path, monkeypatch):
    """Task 3 hardening (forward-noted in the Task 2 report): the throwaway
    oracle container in hidden_validate's command-oracle path must carry
    the same hardening flags as the main `__enter__` container, since it
    now runs agent-produced code (Task 3's oracles compile/run the post-run
    workspace). Monkeypatches subprocess.run to capture the `docker run`
    argv without needing a real container -- runs everywhere, no Docker
    required, matching the brief's "test the hard-cap paths without
    Docker" intent applied to this hardening specifically."""
    ws = tmp_path / "ws-hardening"
    ws.mkdir()
    (ws / "f.txt").write_text("x")

    captured = {}
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        if argv[:2] == ["docker", "run"]:
            captured["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    sbx = Sandbox(workspace=ws, cpus=1.5, mem_limit="1g")
    ok, detail = sbx.hidden_validate(["true"], ws)
    assert ok is True, detail

    argv = captured.get("argv")
    assert argv is not None, "docker run was never invoked"
    assert "--cap-drop" in argv and argv[argv.index("--cap-drop") + 1] == "ALL"
    assert ("--security-opt" in argv
            and argv[argv.index("--security-opt") + 1] == "no-new-privileges")
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "1.5"
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == "1g"


@requires_docker
def test_workspace_write_persists_and_outside_write_fails(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    with Sandbox(workspace=ws) as sbx:
        code, out, err = sbx.run_in(
            ["bash", "-c", "echo hello > /workspace/inside.txt && cat /workspace/inside.txt"])
        assert code == 0
        assert "hello" in out
        # persisted to the host dir (the bind mount, not a container-only write)
        assert (ws / "inside.txt").read_text().strip() == "hello"

        code2, out2, err2 = sbx.run_in(["bash", "-c", "echo bad > /outside.txt"])
        assert code2 != 0
        assert "read-only" in (out2 + err2).lower()


@requires_docker
def test_spawned_process_killed_on_container_teardown(tmp_path):
    ws = tmp_path / "ws2"
    ws.mkdir()
    sbx = Sandbox(workspace=ws)
    sbx.__enter__()
    cid = sbx.container_id
    try:
        assert cid
        code, out, _ = sbx.run_in(["bash", "-c", "sleep 300 & disown; echo spawned"])
        assert code == 0 and "spawned" in out
        # confirm the background process is actually alive before teardown
        code2, out2, _ = sbx.run_in(
            ["bash", "-c", "pgrep -f 'sleep 300' >/dev/null && echo ALIVE"])
        assert code2 == 0 and "ALIVE" in out2
    finally:
        sbx.__exit__(None, None, None)

    # container force-removed => its whole process tree (incl. the
    # backgrounded sleep, reparented but still in the same cgroup) is gone
    ps = subprocess.run(["docker", "ps", "-a", "-q", "--filter", f"id={cid}"],
                        capture_output=True, text=True)
    assert ps.stdout.strip() == ""


@requires_docker
def test_network_none_blocks_non_endpoint_egress(tmp_path):
    ws = tmp_path / "ws3"
    ws.mkdir()
    with Sandbox(workspace=ws) as sbx:
        code, out, err = sbx.run_in(
            ["bash", "-c", "exec 3<>/dev/tcp/1.1.1.1/80"], timeout=5)
        assert code != 0
        assert "unreachable" in (out + err).lower() or "network" in (out + err).lower()


@requires_docker
def test_run_in_timeout_kills_the_exec(tmp_path):
    ws = tmp_path / "ws4"
    ws.mkdir()
    with Sandbox(workspace=ws) as sbx:
        code, _, _ = sbx.run_in(["sleep", "10"], timeout=2)
        assert code != 0  # killed by wall-clock timeout, not a clean 0 exit


@requires_docker
def test_snapshot_workspace_reflects_current_files(tmp_path):
    ws = tmp_path / "ws5"
    ws.mkdir()
    with Sandbox(workspace=ws) as sbx:
        sbx.run_in(["bash", "-c",
                    "echo one > /workspace/a.txt && mkdir -p /workspace/sub "
                    "&& echo two > /workspace/sub/b.txt"])
        snap = sbx.snapshot_workspace()
    assert snap["a.txt"] == b"one\n"
    assert snap["sub/b.txt"] == b"two\n"


@requires_docker
def test_hidden_validate_callable_oracle_sees_post_run_state(tmp_path):
    ws = tmp_path / "ws6"
    ws.mkdir()
    with Sandbox(workspace=ws) as sbx:
        sbx.run_in(["bash", "-c", "echo final-answer > /workspace/answer.txt"])

    def oracle(copy_path: Path):
        content = (copy_path / "answer.txt").read_text().strip()
        return content == "final-answer", f"content={content!r}"

    ok, detail = Sandbox(workspace=ws).hidden_validate(oracle, ws)
    assert ok is True, detail


@requires_docker
def test_hidden_validate_command_oracle_mount_isolated_and_read_only(tmp_path):
    ws = tmp_path / "ws7"
    ws.mkdir()
    (ws / "out.txt").write_text("expected-value")
    sbx = Sandbox(workspace=ws)

    # oracle references /oracle -- a path distinct from /workspace, proving
    # it runs against an isolated mount, not the agent's writable one
    ok, detail = sbx.hidden_validate(["bash", "-c", "grep -q expected-value /oracle/out.txt"], ws)
    assert ok is True, detail

    # the oracle mount is read-only: an attempted write fails
    ok2, detail2 = sbx.hidden_validate(["bash", "-c", "echo y > /oracle/out.txt"], ws)
    assert ok2 is False, detail2

    # tampering with the real workspace after the fact is caught on re-validate
    (ws / "out.txt").write_text("tampered")
    ok3, detail3 = sbx.hidden_validate(["bash", "-c", "grep -q expected-value /oracle/out.txt"], ws)
    assert ok3 is False, detail3


@requires_docker
def test_hidden_validate_command_oracle_timeout_returns_false_and_leaves_no_container(tmp_path):
    """I-1 fix (whole-branch review): agent-produced code runs inside the
    oracle container -- a busy/infinite loop there previously hung `docker
    run` forever (--cpus only throttles; --memory never trips on a
    non-allocating loop). `timeout=` now wraps the oracle command with the
    in-container `timeout -s KILL <n>` coreutil, matching `run_in`'s
    existing pattern. Also proves no container leak: `docker run --rm`
    only auto-removes a container that exits on its own -- here the
    in-container kill IS what makes that happen (SIGKILL propagates as the
    container's own exit code, --rm reaps it), and hidden_validate's
    finally-block cleanup covers the case where it doesn't."""
    ws = tmp_path / "ws-timeout"
    ws.mkdir()
    sbx = Sandbox(workspace=ws)

    ok, detail = sbx.hidden_validate(["sleep", "30"], ws, timeout=2)
    assert ok is False
    assert "timeout" in detail.lower()

    # no leaked container: filter by the deterministic name PREFIX
    # hidden_validate uses (the suffix is a random uuid per call, so this
    # matches regardless of which exact call produced it)
    check = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "name=llmtest-b8-oracle-"],
        capture_output=True, text=True)
    assert check.stdout.strip() == "", f"leaked oracle container(s): {check.stdout!r}"


# -- symlink security (post-review fix: hidden_validate/snapshot_workspace
# must not resolve a workspace-planted symlink on the HOST) -----------------


@requires_docker
def test_hidden_validate_symlink_to_host_secret_not_leaked(tmp_path):
    """An agent-planted symlink pointing at a real host file (absolute host
    path, e.g. what `ln -s /etc/passwd leak` or a smuggled host path would
    produce) must never let the oracle observe the host file's content. The
    HOST harness process must copy the symlink AS a symlink (never follow
    it during the host-side copy); if it resolves at all, it can only
    resolve INSIDE the isolated oracle container's own filesystem."""
    secret_dir = tmp_path / "hostsecret"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("SECRET-HOST-CONTENT")

    ws = tmp_path / "ws8"
    ws.mkdir()
    os.symlink(str(secret_file), ws / "leak_abs_host.txt")
    os.symlink("/etc/passwd", ws / "leak_etc_passwd.txt")   # only meaningful inside a Linux container

    sbx = Sandbox(workspace=ws)
    # passes (exit 0) only if the host secret text is NOT visible inside the container
    ok, detail = sbx.hidden_validate(
        ["bash", "-c",
         "! grep -qr SECRET-HOST-CONTENT /oracle/ 2>/dev/null"], ws)
    assert ok is True, detail

    # /etc/passwd, if it resolves at all, resolves to the CONTAINER's own
    # /etc/passwd (never the host's -- Windows has no such path) -- sanity
    # check it looks like a real (container-local) passwd file, not empty/host data
    ok2, detail2 = sbx.hidden_validate(
        ["bash", "-c", "grep -q '^root:' /oracle/leak_etc_passwd.txt"], ws)
    assert ok2 is True, detail2


@requires_docker
def test_hidden_validate_dangling_symlink_does_not_raise(tmp_path):
    """A dangling symlink in the workspace must not crash `hidden_validate`
    (previously: default `shutil.copytree` follows symlinks and raises on a
    dangling target -- a crash-DoS an agent could trigger deliberately)."""
    ws = tmp_path / "ws9"
    ws.mkdir()
    os.symlink(str(tmp_path / "does-not-exist.txt"), ws / "dangling.txt")
    (ws / "real.txt").write_text("present")

    sbx = Sandbox(workspace=ws)
    # must return cleanly, not raise; the dangling target is unreachable
    # inside the container, so the oracle command itself fails cleanly
    ok, detail = sbx.hidden_validate(["bash", "-c", "cat /oracle/dangling.txt"], ws)
    assert ok is False, detail

    # the rest of the workspace still copies and validates fine
    def oracle(copy_path: Path):
        return (copy_path / "real.txt").read_text() == "present", "ok"

    ok2, detail2 = sbx.hidden_validate(oracle, ws)
    assert ok2 is True, detail2


@requires_docker
def test_snapshot_workspace_skips_symlinks_no_leak_no_traversal(tmp_path):
    """`snapshot_workspace` must not follow a symlinked file (would read
    host content) or descend into a symlinked directory (would traverse
    outside the workspace) -- both must be silently skipped, not just
    'not returned after being read'."""
    secret_dir = tmp_path / "hostsecret2"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("SECRET-HOST-CONTENT")

    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "outside_file.txt").write_text("SECRET-HOST-CONTENT")

    ws = tmp_path / "ws10"
    ws.mkdir()
    os.symlink(str(secret_file), ws / "leak.txt")
    os.symlink(str(outside_dir), ws / "linked_dir", target_is_directory=True)

    with Sandbox(workspace=ws) as sbx:
        sbx.run_in(["bash", "-c", "echo real > /workspace/real.txt"])
        snap = sbx.snapshot_workspace()

    assert snap.get("real.txt") == b"real\n"
    assert "leak.txt" not in snap
    assert not any(k.startswith("linked_dir/") for k in snap)
    assert all(v != b"SECRET-HOST-CONTENT" for v in snap.values())
