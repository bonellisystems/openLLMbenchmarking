"""Tests for B8 task manifests + the anti-gaming completion oracle (Task 3,
Part 2 Phase 1). Mirrors test_b6.py's loader-test shape (real fixtures via
ROOT + malformed-fixture tests via tmp_path) and test_harness_sandbox.py's
`requires_docker` skip pattern for the container-backed oracle-pass path.

Design invariant under test (the brief's Step-1 scenario): protected-file
tamper is a HARD CAP checked BEFORE the behavioral oracle ever runs, so it
must return False even when the tampered oracle itself would trivially
"pass" -- and that check must not require Docker at all (it never
constructs a Sandbox).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from llmtest.harness import tasks as t

ROOT = Path(__file__).resolve().parents[1]


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


requires_docker = pytest.mark.skipif(not _docker_available(), reason="Docker not reachable")


def _load(task_id: str) -> t.B8Task:
    all_tasks = t.load_b8_tasks(ROOT)
    return next(x for x in all_tasks if x.id == task_id)


# -- manifest loading --------------------------------------------------------


def test_load_b8_tasks_returns_all_manifests_with_required_fields():
    """8 manifests total (task-b8local, was 5): the original 5 bash
    placeholders (task-01..05.yaml) plus 3 real Python task manifests
    (task-06..08.yaml) added for the real local-run enablement pass."""
    all_tasks = t.load_b8_tasks(ROOT)
    assert len(all_tasks) == 8
    assert [x.id for x in all_tasks] == sorted(x.id for x in all_tasks)
    for task in all_tasks:
        assert task.id
        assert task.shape in t._VALID_SHAPES
        assert task.setup_repo_sha and len(task.setup_repo_sha) == 64
        assert isinstance(task.allowed_tools, list) and task.allowed_tools
        assert isinstance(task.budgets, dict)
        for key in ("wall_clock_s", "tokens", "steps"):
            assert key in task.budgets
        assert isinstance(task.oracle, list) and task.oracle
        assert isinstance(task.protected_shas, dict) and task.protected_shas
        assert task.task_version
        assert task.fixture_sha and len(task.fixture_sha) == 64


def test_load_b8_tasks_covers_all_shapes():
    """Every valid shape appears at least once -- no longer EXACTLY once
    (task-b8local): the 3 new Python manifests reuse the bugfix/from-scratch/
    edit shapes (their bash counterparts already cover multi-file/tool-heavy),
    so those three shapes now appear twice. Coverage, not cardinality."""
    all_tasks = t.load_b8_tasks(ROOT)
    shapes = {x.shape for x in all_tasks}
    assert shapes == t._VALID_SHAPES


def test_fixture_sha_is_sha256_of_manifest_bytes():
    all_tasks = t.load_b8_tasks(ROOT)
    for task in all_tasks:
        expected = hashlib.sha256(task.path.read_bytes()).hexdigest()
        assert task.fixture_sha == expected


def test_setup_repo_sha_changes_when_repo_content_changes(tmp_path):
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    _write_minimal_manifest(task_dir / "task-01.yaml", extra_file_content="one")
    a = t.load_b8_tasks(tmp_path)[0].setup_repo_sha

    task_dir2 = tmp_path / "b" / "suite" / "b8_harness"
    task_dir2.mkdir(parents=True)
    _write_minimal_manifest(task_dir2 / "task-01.yaml", extra_file_content="two")
    b = t.load_b8_tasks(tmp_path / "b")[0].setup_repo_sha

    assert a != b


def _write_minimal_manifest(path: Path, extra_file_content: str) -> None:
    path.write_text(f"""\
id: edit-99
shape: edit
task_version: "1.0.0"
prompt: "do a thing"
allowed_tools: [read_file, write_file]
budgets: {{wall_clock_s: 60, tokens: 500, steps: 4}}
setup_repo:
  main.sh: |
    echo {extra_file_content}
  notes.txt: |
    protected, agent-visible
oracle_files:
  oracle_test.sh: |
    echo PASS
protected_paths: [notes.txt]
allowed_diff_paths: [main.sh]
oracle:
  type: command
  argv: ["bash", "-c", "cp -r /oracle /tmp/work && cd /tmp/work && bash oracle_test.sh"]
""", encoding="utf-8")


# -- malformed manifests (mirrors test_b6.py's fail-loud pattern) -----------


def test_loader_raises_on_missing_required_key(tmp_path):
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    (task_dir / "task-01.yaml").write_text(
        "id: edit-99\nshape: edit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed manifest"):
        t.load_b8_tasks(tmp_path)


def test_loader_raises_on_unknown_shape(tmp_path):
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    _write_minimal_manifest(task_dir / "task-01.yaml", extra_file_content="x")
    text = (task_dir / "task-01.yaml").read_text(encoding="utf-8")
    (task_dir / "task-01.yaml").write_text(
        text.replace("shape: edit", "shape: not_a_real_shape"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid shape"):
        t.load_b8_tasks(tmp_path)


def test_loader_raises_when_protected_path_not_in_setup_repo(tmp_path):
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    _write_minimal_manifest(task_dir / "task-01.yaml", extra_file_content="x")
    text = (task_dir / "task-01.yaml").read_text(encoding="utf-8")
    text = text.replace("protected_paths: [notes.txt]",
                         "protected_paths: [does_not_exist.sh]")
    (task_dir / "task-01.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="not found in setup_repo"):
        t.load_b8_tasks(tmp_path)


def test_loader_raises_when_path_in_both_setup_repo_and_oracle_files(tmp_path):
    """The hidden-oracle architecture (post-review hardening) is only
    meaningful if a manifest author can't accidentally leave the oracle
    agent-visible by also listing its path in setup_repo -- the loader must
    reject that outright rather than let one silently shadow the other."""
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    _write_minimal_manifest(task_dir / "task-01.yaml", extra_file_content="x")
    text = (task_dir / "task-01.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "setup_repo:\n  main.sh:",
        "setup_repo:\n  oracle_test.sh: |\n    leaked\n  main.sh:",
    )
    (task_dir / "task-01.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="both setup_repo .* and oracle_files"):
        t.load_b8_tasks(tmp_path)


def test_loader_missing_dir_returns_empty():
    assert t.load_b8_tasks(Path("/definitely/not/a/real/path")) == []


# -- oracle withholding (the review's Important finding) --------------------
# "hidden validators live OUTSIDE the writable workspace" is a DISTINCT
# constraint from "protected files hash-checked" -- these tests prove the
# former is now structurally true, not just the latter.


def test_materialize_repo_never_writes_oracle_files():
    """The materialized agent workspace must not contain any oracle_files
    path, for every one of the 5 real manifests -- an agent given this
    workspace has literally nothing to read that reveals what the hidden
    oracle checks."""
    all_tasks = t.load_b8_tasks(ROOT)
    for task in all_tasks:
        assert task.oracle_files, f"{task.id}: oracle_files must be non-empty"
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / task.id
            t.materialize_repo(task, ws)
            for oracle_path in task.oracle_files:
                assert not (ws / oracle_path).exists(), (
                    f"{task.id}: oracle file {oracle_path!r} leaked into "
                    f"the agent-visible workspace")


def test_toolheavy_overfit_attack_has_no_oracle_file_to_read(tmp_path):
    """Concrete disproof from the review: with the OLD layout
    (oracle_test.sh inside setup_repo), an agent could READ the hidden
    test, see it writes data/f_extra.txt and expects total 200, and write
    sum_all.sh as `if [ -f data/f_extra.txt ]; then echo 200; else echo
    150; fi` -- passing both checks with zero real summing logic. Prove
    this is now structurally impossible: the materialized agent workspace
    has no oracle_test.sh (or any oracle_files path) anywhere to read, and
    the specific probe values the attack needs (the extra filename, the
    post-injection total) appear nowhere in it."""
    task = _load("toolheavy-01")
    assert "oracle_test.sh" in task.oracle_files  # sanity: still exists, just hidden
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    assert not (ws / "oracle_test.sh").exists()

    all_text = "\n".join(p.read_text(encoding="utf-8") for p in ws.rglob("*") if p.is_file())
    assert "f_extra" not in all_text
    assert "200" not in all_text


# -- run_oracle: precedence / anti-gaming (the brief's Step-1 scenario) -----
# These run WITHOUT Docker: the protected-hash check and the diff-constraint
# check both short-circuit before any Sandbox is constructed.


def test_run_oracle_false_on_protected_tamper_even_though_behavior_would_pass(tmp_path):
    """The brief's Step-1 test, adapted to the post-review architecture: the
    hidden oracle (oracle_test.sh) is no longer even IN the agent workspace
    (see the "oracle withholding" tests below) -- so the hash-cap's genuine
    target is now the agent-VISIBLE protected fixture (NOTES.md). A gamer
    correctly fixes the bug (behavior would genuinely pass) but also tampers
    with NOTES.md; the protected-hash check must still catch that and
    return False -- BEFORE the behavioral oracle ever runs."""
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)

    # apply the correct fix to stats.sh (behavior would genuinely pass)
    fixed = task.setup_repo["stats.sh"].replace(
        "    echo $((sum / $#))\n\nsummarize",
        "    echo $((sum / $#))\n}\n\nsummarize",
    )
    assert fixed != task.setup_repo["stats.sh"]
    (ws / "stats.sh").write_bytes(fixed.encode("utf-8"))

    # tamper with the protected, agent-visible fixture file
    (ws / "NOTES.md").write_bytes(b"tampered\n")

    completed, detail = t.run_oracle(task, ws)
    assert completed is False
    assert "protected" in detail.lower()
    assert "NOTES.md" in detail


def test_run_oracle_false_on_out_of_bounds_edit(tmp_path):
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    # bugfix-01's only allowed_diff_path is stats.sh; writing a new,
    # unrelated file must be rejected by the diff constraint.
    (ws / "sneaky.sh").write_bytes(b"echo hi\n")

    completed, detail = t.run_oracle(task, ws)
    assert completed is False
    assert "out-of-bounds" in detail.lower()
    assert "sneaky.sh" in detail


def test_run_oracle_diff_constraint_does_not_traverse_symlinked_dir(tmp_path):
    """The diff-constraint walk reads the agent-controlled workspace on the
    HOST, so it must not descend into a symlinked directory -- mirrors the
    exact traversal Sandbox.snapshot_workspace already guards against
    (Task 2 precedent). Paired with a real, unrelated out-of-bounds file
    (sneaky.sh) so this stays deterministic and Docker-free: under the
    (fixed) symlink-safe walk, `linked_dir` is pruned before descent and
    never appears in the failure reason at all; under the old
    `Path.rglob`-based walk it would have been traversed and reported
    first (sorts before "sneaky.sh"), leaking the host directory's
    existence/content into `detail`."""
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "outside_file.txt").write_text("SECRET-HOST-CONTENT")

    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    os.symlink(str(outside_dir), ws / "linked_dir", target_is_directory=True)
    (ws / "sneaky.sh").write_bytes(b"echo hi\n")

    completed, detail = t.run_oracle(task, ws)
    assert completed is False
    assert "sneaky.sh" in detail
    assert "SECRET-HOST-CONTENT" not in detail
    assert "linked_dir" not in detail
    assert "outside_file" not in detail


def test_run_oracle_no_docker_needed_for_hard_cap_paths(tmp_path, monkeypatch):
    """Both hard-cap checks above must never touch Sandbox/docker -- assert
    this directly by making any subprocess.run call fail the test."""
    import llmtest.harness.sandbox as sandbox_mod

    def _boom(*a, **k):
        raise AssertionError("run_oracle must not shell out for a hard-cap failure")

    monkeypatch.setattr(sandbox_mod.subprocess, "run", _boom)

    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "NOTES.md").write_bytes(b"tampered\n")
    completed, _ = t.run_oracle(task, ws)
    assert completed is False


# -- run_oracle: oracle_image threading (task-b8local) -----------------------
# No Docker needed for either test below -- subprocess.run is fully
# monkeypatched (every "docker ..." call, not just "docker run") so these
# stay hermetic regardless of whether a Docker daemon is reachable.


def _fake_docker_run(captured: dict, real_run):
    def fake_run(argv, **kwargs):
        if argv and argv[0] == "docker":
            if argv[:2] == ["docker", "run"]:
                captured["argv"] = argv
                return subprocess.CompletedProcess(argv, 0, stdout="PASS", stderr="")
            # docker rm -f / docker ps -a -q --filter ... (cleanup +
            # cleanup-verification) -- succeed with empty output so
            # hidden_validate's own post-cleanup verification doesn't
            # think a container leaked.
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, **kwargs)
    return fake_run


def _correctly_fixed_bugfix01_workspace(ws: Path, task: "t.B8Task") -> None:
    t.materialize_repo(task, ws)
    fixed = task.setup_repo["stats.sh"].replace(
        "    echo $((sum / $#))\n\nsummarize",
        "    echo $((sum / $#))\n}\n\nsummarize",
    )
    (ws / "stats.sh").write_bytes(fixed.encode("utf-8"))


def test_run_oracle_threads_configured_oracle_image_into_sandbox(tmp_path, monkeypatch):
    """task-b8local: b8_harness.py's execute() threads suite.yaml's
    b8.sandbox.oracle_image through run_oracle -> Sandbox, so a
    Python-shaped B8 manifest's oracle (needs python3) doesn't run inside
    the pinned, Python-less nvidia/cuda image. Monkeypatches subprocess.run
    (mirrors test_harness_sandbox.py's container-hardening test) to capture
    the `docker run` argv without needing a real container."""
    import llmtest.harness.sandbox as sandbox_mod

    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    _correctly_fixed_bugfix01_workspace(ws, task)

    captured: dict = {}
    monkeypatch.setattr(sandbox_mod.subprocess, "run",
                        _fake_docker_run(captured, subprocess.run))

    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is True, detail

    argv = captured.get("argv")
    assert argv is not None, "docker run was never invoked"
    # Bare tag, no "@sha256:..." digest -- run_oracle's oracle_image
    # docstring: digest="" avoids pairing the override image with the CUDA
    # pin's unrelated digest.
    assert "python:3.11-slim" in argv
    assert not any(a.startswith("python:3.11-slim@") for a in argv)
    assert not any("nvidia/cuda" in a for a in argv)


def test_run_oracle_default_oracle_image_still_uses_pinned_cuda_image(tmp_path, monkeypatch):
    """Additivity regression: oracle_image=None (the default -- every
    caller that predates task-b8local, including every OTHER test in this
    file) must reach Sandbox with no image/digest override at all, so it
    still resolves to the pinned CUDA image+digest, unchanged."""
    import llmtest.harness.sandbox as sandbox_mod

    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    _correctly_fixed_bugfix01_workspace(ws, task)

    captured: dict = {}
    monkeypatch.setattr(sandbox_mod.subprocess, "run",
                        _fake_docker_run(captured, subprocess.run))

    completed, detail = t.run_oracle(task, ws)   # oracle_image not passed
    assert completed is True, detail

    argv = captured.get("argv")
    assert argv is not None, "docker run was never invoked"
    assert any(a.startswith("nvidia/cuda:12.6.2-base-ubuntu24.04@sha256:") for a in argv)


# -- run_oracle: the real behavioral pass path (needs Docker) ---------------


@requires_docker
def test_run_oracle_reinjects_oracle_files_before_validating(tmp_path):
    """Direct proof of re-injection (review item (b)): oracle_test.sh is
    never in the agent workspace (materialize_repo doesn't write it), yet a
    correct fix still gets validated successfully -- which is only possible
    if run_oracle put a fresh copy of it back before calling
    Sandbox.hidden_validate. Also locks in that `ws` itself (the agent's
    real, real workspace) is never mutated by that re-injection -- it
    happens on a private copy."""
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    assert not (ws / "oracle_test.sh").exists()

    fixed = task.setup_repo["stats.sh"].replace(
        "    echo $((sum / $#))\n\nsummarize",
        "    echo $((sum / $#))\n}\n\nsummarize",
    )
    (ws / "stats.sh").write_bytes(fixed.encode("utf-8"))

    completed, detail = t.run_oracle(task, ws)
    assert completed is True, detail
    assert not (ws / "oracle_test.sh").exists()


@requires_docker
def test_run_oracle_respects_wall_clock_budget_and_leaves_no_container(tmp_path):
    """I-1 fix (whole-branch review): run_oracle must plumb the manifest's
    budgets.wall_clock_s through to Sandbox.hidden_validate's `timeout` --
    without it, agent-produced code in the oracle (every real manifest
    executes some) has no wall-clock bound at all. Uses a synthetic
    manifest with a small wall_clock_s (2s) and an oracle command that
    sleeps well past it (30s) so this stays fast, and confirms no oracle
    container leaks."""
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    (task_dir / "task-01.yaml").write_text("""\
id: edit-99
shape: edit
task_version: "1.0.0"
prompt: "do a thing"
allowed_tools: [read_file, write_file]
budgets: {wall_clock_s: 2, tokens: 500, steps: 4}
setup_repo:
  main.sh: |
    echo hi
  notes.txt: |
    protected, agent-visible
oracle_files:
  oracle_test.sh: |
    ignored -- the oracle argv below never references this file
protected_paths: [notes.txt]
allowed_diff_paths: [main.sh]
oracle:
  type: command
  argv: ["sleep", "30"]
""", encoding="utf-8")

    task = t.load_b8_tasks(tmp_path)[0]
    assert task.budgets["wall_clock_s"] == 2

    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)

    completed, detail = t.run_oracle(task, ws)
    assert completed is False
    assert "timeout" in detail.lower()

    check = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "name=llmtest-b8-oracle-"],
        capture_output=True, text=True)
    assert check.stdout.strip() == "", f"leaked oracle container(s): {check.stdout!r}"


@requires_docker
def test_run_oracle_reinjection_does_not_write_through_symlink(tmp_path):
    """I-2 fix (whole-branch review): `shutil.copytree(..., symlinks=True)`
    used to preserve an agent-planted symlink in the re-injection copy;
    the subsequent per-oracle-file `write_bytes()` would then follow it,
    writing oracle content to an arbitrary HOST path outside the temp dir
    -- if the agent guesses an oracle_files path (e.g. "oracle_test.sh",
    which every one of these 5 manifests uses) and symlinks it to a
    writable location. `copy_real_files` never preserves the symlink at
    all, so there is nothing left to write through.

    Needs Docker: the planted symlink is silently skipped (not flagged) by
    the diff-constraint (documented limitation, step (b)), so this only
    reaches the write-through-risk code in step (c)."""
    outside_target = tmp_path / "outside_writable.txt"
    outside_target.write_text("SENTINEL-BEFORE")

    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)

    try:
        os.symlink(str(outside_target), ws / "oracle_test.sh")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"os.symlink not permitted on this host: {e!r}")

    # The bug is left unfixed on purpose -- this test is about the
    # write-through, not the behavioral outcome; run_oracle must simply
    # return cleanly (a plain (bool, str), no raise) either way, and must
    # never touch the outside file regardless of that outcome.
    completed, detail = t.run_oracle(task, ws)
    assert isinstance(completed, bool), detail

    assert outside_target.read_text() == "SENTINEL-BEFORE"


@requires_docker
def test_run_oracle_true_for_correct_bugfix_workspace(tmp_path):
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    fixed = task.setup_repo["stats.sh"].replace(
        "    echo $((sum / $#))\n\nsummarize",
        "    echo $((sum / $#))\n}\n\nsummarize",
    )
    (ws / "stats.sh").write_bytes(fixed.encode("utf-8"))

    completed, detail = t.run_oracle(task, ws)
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_when_bug_still_present(tmp_path):
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    # no edit at all -- bug still present
    completed, detail = t.run_oracle(task, ws)
    assert completed is False, detail


@requires_docker
def test_run_oracle_true_for_correct_edit_workspace(tmp_path):
    task = _load("edit-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "greet.sh").write_bytes(b"""\
#!/bin/bash
name="$1"
echo "Hello, ${name}!"
""")
    completed, detail = t.run_oracle(task, ws)
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_still_hardcoded_edit_workspace(tmp_path):
    task = _load("edit-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)  # no edit -- still hardcoded to "World"
    completed, detail = t.run_oracle(task, ws)
    assert completed is False, detail


@requires_docker
def test_run_oracle_true_for_correct_multifile_workspace(tmp_path):
    task = _load("multifile-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "lib_math.sh").write_bytes(b"""\
#!/bin/bash
add() {
    echo $(( $1 + $2 ))
}

multiply() {
    echo $(( $1 * $2 ))
}
""")
    (ws / "lib_format.sh").write_bytes(b"""\
#!/bin/bash
fmt_result() {
    echo "Result: $1"
}
""")
    completed, detail = t.run_oracle(task, ws)
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_multifile_workspace_missing_one_fix(tmp_path):
    """Only lib_math.sh fixed -- lib_format.sh still has the wrong prefix.
    Proves the oracle genuinely requires BOTH edits, not just one."""
    task = _load("multifile-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "lib_math.sh").write_bytes(b"""\
#!/bin/bash
add() {
    echo $(( $1 + $2 ))
}

multiply() {
    echo $(( $1 * $2 ))
}
""")
    completed, detail = t.run_oracle(task, ws)
    assert completed is False, detail


@requires_docker
def test_run_oracle_true_for_correct_scratch_workspace(tmp_path):
    task = _load("scratch-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "is_prime.sh").write_bytes(b"""\
#!/bin/bash
is_prime() {
    local n="$1"
    if [ "$n" -lt 2 ]; then echo false; return; fi
    local i=2
    while [ $((i * i)) -le "$n" ]; do
        if [ $((n % i)) -eq 0 ]; then echo false; return; fi
        i=$((i + 1))
    done
    echo true
}
""")
    completed, detail = t.run_oracle(task, ws)
    assert completed is True, detail


@requires_docker
def test_run_oracle_true_for_correct_toolheavy_workspace_generalizes(tmp_path):
    """Anti-gaming check for the tool-heavy shape: a hardcoded-total solution
    must fail once the oracle adds an extra data file, proving the oracle
    genuinely exercises dynamic discovery, not a fixed-file no-op."""
    task = _load("toolheavy-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "sum_all.sh").write_bytes(b"""\
#!/bin/bash
total=0
for f in data/*.txt; do
    for n in $(cat "$f"); do
        total=$((total + n))
    done
done
echo "$total"
""")
    completed, detail = t.run_oracle(task, ws)
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_hardcoded_toolheavy_solution(tmp_path):
    task = _load("toolheavy-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "sum_all.sh").write_bytes(b'#!/bin/bash\necho 150\n')
    completed, detail = t.run_oracle(task, ws)
    assert completed is False, detail
