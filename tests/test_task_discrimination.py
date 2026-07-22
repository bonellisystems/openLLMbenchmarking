"""Per-task discrimination proof (Wave 3b, B8 validity program): "give
every task a reference solution, at least one alternate valid solution, and
several plausible incorrect or shortcut patches" -- then prove, against the
REAL `run_oracle` (real Docker container, real `python:3.11-slim` oracle
image, real hard-cap precedence), that every task's oracle actually
DISCRIMINATES: a genuinely correct solution passes, and every
plausible-but-flawed one fails.

Fixtures live in each manifest's own `check_fixtures` block (see
`suite/b8_harness/_schema.md`'s "check_fixtures" section) -- a test-only,
additive manifest key `load_b8_tasks` loads into `B8Task.check_fixtures`
but that `materialize_repo`/`run_oracle`/`B8Harness` never read. This test
is parametrized over every one of the 11 real Python task manifests
(task-06..16.yaml) that carries one.

This is the mechanism the Wave-3b brief calls for: "If any task's oracle
FAILS to discriminate ... that's a real oracle bug -- FIX the oracle or the
fixture and note it." Every fixture below was hermetically verified (real
oracle_test.py, host python, no Docker -- the fast iteration loop) before
being committed to its manifest; this file re-proves the same property
through the REAL run_oracle path (real Docker container, real hard-cap
precedence ahead of the behavioral oracle), which is the strongest form of
the proof and the one that would catch a Docker/Sandbox-layer regression
the hermetic check can't see.
"""
from __future__ import annotations

import subprocess

import pytest

from llmtest.harness import tasks as t

# Tamper/hard-cap phrasing (run_oracle steps (a)/(b)/(c) -- see
# llmtest/harness/tasks.py's module docstring) that must NEVER appear in a
# `wrong` fixture's rejection detail -- if it does, the fixture failed for
# the WRONG reason (a hard-cap short-circuit before the behavioral oracle
# even ran), which proves nothing about whether the oracle itself
# discriminates. Every check_fixtures solution is designed to touch only
# `allowed_diff_paths` files, so none of these should ever fire; asserted
# directly here as a fixture-authoring regression guard.
_HARD_CAP_PHRASES = ("protected file tampered", "protected/setup file deleted",
                    "replaced with a directory", "disallowed symlink",
                    "out-of-bounds edit")


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


requires_docker = pytest.mark.skipif(not _docker_available(), reason="Docker not reachable")

ROOT = None  # set below, mirrors test_harness_tasks.py's module-level ROOT


def _root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def _tasks_with_check_fixtures() -> list[t.B8Task]:
    return [task for task in t.load_b8_tasks(_root()) if task.check_fixtures]


_TASKS = _tasks_with_check_fixtures()
_TASK_IDS = [task.id for task in _TASKS]


def test_at_least_the_11_real_python_tasks_carry_check_fixtures():
    """Sanity check on the parametrization source itself -- if this is
    empty or short, every test below silently no-ops (pytest parametrize
    over an empty list just collects zero tests), which would look like
    "everything passed" while actually testing nothing."""
    assert len(_TASK_IDS) == 11, _TASK_IDS
    assert set(_TASK_IDS) == {
        "py-bugfix-01", "py-fromscratch-01", "py-edit-01", "py-multifile-01",
        "py-toolheavy-01", "py-fromscratch-02", "py-hard-bugfix-01",
        "py-hard-algo-01", "py-hard-edge-01", "py-hard-multifile-01",
        "py-hard-toolheavy-01",
    }


def _apply_overlay(task: t.B8Task, ws, solution: dict) -> None:
    """Materialize is assumed already done. Overlay `solution` (a
    check_fixtures reference/alternate/wrong entry) on top -- writes only
    the files the solution specifies, leaving every other setup_repo file
    at its original materialized content (the deliberate "left one file
    unfixed" shape some `wrong` fixtures use)."""
    for rel, content in solution.items():
        (ws / rel).write_bytes(content.encode("utf-8"))


@requires_docker
@pytest.mark.parametrize("task_id", _TASK_IDS)
def test_reference_solution_passes(task_id, tmp_path):
    task = next(x for x in _TASKS if x.id == task_id)
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply_overlay(task, ws, task.check_fixtures["reference"])

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert result.pass_ is True, (
        f"{task_id}: reference solution FAILED the oracle -- {result.detail}")


@requires_docker
@pytest.mark.parametrize("task_id", _TASK_IDS)
def test_alternate_solution_passes_when_present(task_id, tmp_path):
    task = next(x for x in _TASKS if x.id == task_id)
    alternate = task.check_fixtures.get("alternate")
    if alternate is None:
        pytest.skip(f"{task_id}: no alternate fixture (optional per check_fixtures schema)")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply_overlay(task, ws, alternate)

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert result.pass_ is True, (
        f"{task_id}: alternate solution FAILED the oracle -- {result.detail}")


def _wrong_params():
    """(task_id, wrong_index) for every (task, wrong-fixture) pair --
    flattened so a failure names exactly which wrong entry broke, not just
    which task."""
    out = []
    for task in _TASKS:
        for i in range(len(task.check_fixtures["wrong"])):
            out.append((task.id, i))
    return out


@requires_docker
@pytest.mark.parametrize("task_id,wrong_index", _wrong_params())
def test_wrong_solution_fails_behaviorally(task_id, wrong_index, tmp_path):
    """The load-bearing discrimination proof: a plausible-but-flawed
    solution must be REJECTED, and rejected for the right reason -- the
    BEHAVIORAL oracle (stage/reason_code present, no hard-cap tamper
    phrasing in `detail`), never a tamper/out-of-bounds short-circuit. Every
    check_fixtures `wrong` entry only ever writes `allowed_diff_paths`
    files, so a hard-cap hit here would itself be a fixture-authoring bug
    (asserted directly, not just implied by the pass_/detail check)."""
    task = next(x for x in _TASKS if x.id == task_id)
    wrong = task.check_fixtures["wrong"][wrong_index]
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    _apply_overlay(task, ws, wrong)

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert result.pass_ is False, (
        f"{task_id} wrong[{wrong_index}]: a plausible-but-WRONG solution was "
        f"WRONGLY accepted -- the oracle does not discriminate here. detail={result.detail!r}")
    lowered = result.detail.lower()
    for phrase in _HARD_CAP_PHRASES:
        assert phrase not in lowered, (
            f"{task_id} wrong[{wrong_index}]: rejected via a HARD-CAP tamper check "
            f"({phrase!r} in detail), not the behavioral oracle -- fixture touches a "
            f"path outside allowed_diff_paths. detail={result.detail!r}")
    # A genuine behavioral-oracle rejection carries a stage/reason_code
    # (Wave 3a's machine-readable convention -- every one of these 11
    # oracles emits it) -- confirms this reached step (c), not a step
    # (a)/(b) hard cap that never populates these fields.
    assert result.stage in ("compile", "import", "behavior"), (
        f"{task_id} wrong[{wrong_index}]: expected a behavioral-oracle stage, "
        f"got {result.stage!r} -- detail={result.detail!r}")
    assert result.reason_code, (
        f"{task_id} wrong[{wrong_index}]: expected a reason_code, got None -- "
        f"detail={result.detail!r}")
