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


def test_load_b8_tasks_returns_five_tasks_with_required_fields():
    all_tasks = t.load_b8_tasks(ROOT)
    assert len(all_tasks) == 5
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


def test_load_b8_tasks_covers_all_five_shapes_exactly_once():
    all_tasks = t.load_b8_tasks(ROOT)
    shapes = sorted(x.shape for x in all_tasks)
    assert shapes == sorted(t._VALID_SHAPES)


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
