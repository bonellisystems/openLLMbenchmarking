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
import json
import os
import subprocess
import sys
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
    """16 manifests total (task-b8hard, was 11): the original 5 bash
    placeholders (task-01..05.yaml), the first 3 real Python task
    manifests (task-06..08.yaml, task-b8local), 3 more real Python task
    manifests (task-09..11.yaml, task-b8expand) covering the multi-file/
    tool-heavy shapes in Python for the first time and a harder
    from-scratch task, plus 5 HARDER Python manifests (task-12..16.yaml,
    task-b8hard, ids `py-hard-*`) calibrated so a capable 20B model is
    genuinely expected to fail some fraction of the time -- the original 6
    real Python tasks are all solved 30/30 by gpt-oss-20b and don't
    discriminate."""
    all_tasks = t.load_b8_tasks(ROOT)
    # >= 16: Wave-4 breadth batches add manifests over time; this test is an
    # all-manifests-load sanity check, not an exact-count pin (per-task
    # discrimination is covered by tests/test_task_discrimination.py).
    assert len(all_tasks) >= 16
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
    (task-b8local/task-b8expand): with the task-b8expand additions, all 5
    shapes now have BOTH a bash and a Python manifest (from-scratch has
    two Python manifests too, py-fromscratch-01 and -02). Coverage, not
    cardinality."""
    all_tasks = t.load_b8_tasks(ROOT)
    shapes = {x.shape for x in all_tasks}
    assert shapes == t._VALID_SHAPES


def test_fixture_sha_is_sha256_of_manifest_bytes():
    """Wave 3b: fixture_sha is sha256 of the manifest bytes with any
    check_fixtures block excluded (t._strip_check_fixtures_for_hash) --
    a no-op for every one of the 5 bash placeholders (task-01..05.yaml,
    no check_fixtures key at all), so this is still, in effect, "sha256 of
    the raw file" for them; the 11 real Python manifests now have a
    check_fixtures block, so their fixture_sha is the sha256 of the file
    WITHOUT that block (see test_check_fixtures_does_not_change_fixture_sha
    below for the direct before/after proof that adding it left fixture_sha
    unchanged)."""
    all_tasks = t.load_b8_tasks(ROOT)
    for task in all_tasks:
        raw = task.path.read_bytes()
        expected = hashlib.sha256(t._strip_check_fixtures_for_hash(raw)).hexdigest()
        assert task.fixture_sha == expected


# -- Wave 3b: check_fixtures -- test-only, additive, excluded from row -----
# identity (fixture_sha/setup_repo_sha) -----------------------------------


def test_load_b8_tasks_loads_check_fixtures_for_the_11_real_python_tasks():
    """Every one of the 11 real Python manifests (task-06..16.yaml) now
    carries a check_fixtures block -- reference (+ optionally alternate)
    solution(s), plus 1-3 plausible-but-wrong ones -- and load_b8_tasks
    surfaces it as B8Task.check_fixtures, unvalidated against
    setup_repo/oracle content (a pure test aid, per its own docstring)."""
    all_tasks = t.load_b8_tasks(ROOT)
    python_task_ids = {"py-bugfix-01", "py-fromscratch-01", "py-edit-01",
                       "py-multifile-01", "py-toolheavy-01", "py-fromscratch-02",
                       "py-hard-bugfix-01", "py-hard-algo-01", "py-hard-edge-01",
                       "py-hard-multifile-01", "py-hard-toolheavy-01"}
    seen = set()
    for task in all_tasks:
        if task.id not in python_task_ids:
            continue
        seen.add(task.id)
        assert "reference" in task.check_fixtures, task.id
        assert isinstance(task.check_fixtures["reference"], dict) and task.check_fixtures["reference"]
        assert "wrong" in task.check_fixtures, task.id
        wrong = task.check_fixtures["wrong"]
        assert isinstance(wrong, list) and 1 <= len(wrong) <= 3, task.id
        for solution in wrong:
            assert isinstance(solution, dict) and solution, task.id
    assert seen == python_task_ids, f"missing check_fixtures for: {python_task_ids - seen}"


def test_load_b8_tasks_check_fixtures_defaults_to_empty_dict_when_absent():
    """The 5 bash placeholder manifests (task-01..05.yaml) predate this
    convention and never get a check_fixtures block -- additive, so they
    must load exactly as before with an empty dict, not a missing
    attribute/KeyError."""
    all_tasks = t.load_b8_tasks(ROOT)
    bash_ids = {"edit-01", "multifile-01", "bugfix-01", "scratch-01", "toolheavy-01"}
    for task in all_tasks:
        if task.id in bash_ids:
            assert task.check_fixtures == {}


def test_check_fixtures_does_not_change_fixture_sha(tmp_path):
    """The direct, load-bearing row-identity proof (Wave 3b brief:
    "Keep check_fixtures out of fixture_sha/setup_repo_sha/row identity"):
    a manifest's fixture_sha, computed BEFORE any check_fixtures block
    exists, is byte-for-byte the SAME after check_fixtures is added (or
    later edited) -- adding/fixing test-only fixtures must never look like
    a fixture/oracle content change to row identity."""
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    manifest_path = task_dir / "task-01.yaml"
    _write_minimal_manifest(manifest_path, extra_file_content="one")

    sha_before = t.load_b8_tasks(tmp_path)[0].fixture_sha
    setup_repo_sha_before = t.load_b8_tasks(tmp_path)[0].setup_repo_sha

    text = manifest_path.read_text(encoding="utf-8")
    text += (
        "check_fixtures:\n"
        "  reference:\n"
        "    main.sh: |\n"
        "      echo one\n"
        "  wrong:\n"
        "    - main.sh: |\n"
        "        echo nope\n"
    )
    manifest_path.write_text(text, encoding="utf-8")

    tasks_after = t.load_b8_tasks(tmp_path)
    assert len(tasks_after) == 1
    task_after = tasks_after[0]
    assert task_after.check_fixtures  # actually loaded, not silently dropped
    assert task_after.fixture_sha == sha_before
    assert task_after.setup_repo_sha == setup_repo_sha_before

    # And editing an EXISTING check_fixtures block again -- not just adding
    # one for the first time -- must also leave fixture_sha unchanged.
    text2 = text.replace("echo nope", "echo still nope, edited")
    manifest_path.write_text(text2, encoding="utf-8")
    task_after_edit = t.load_b8_tasks(tmp_path)[0]
    assert task_after_edit.fixture_sha == sha_before


def test_check_fixtures_block_stripped_regardless_of_position(tmp_path):
    """_strip_check_fixtures_for_hash must find and remove the
    check_fixtures block wherever it appears in the file, not merely when
    it's the LAST top-level key -- Wave 4 authors are not required to
    place it at the end."""
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    manifest_path = task_dir / "task-01.yaml"
    _write_minimal_manifest(manifest_path, extra_file_content="one")
    sha_before = t.load_b8_tasks(tmp_path)[0].fixture_sha

    text = manifest_path.read_text(encoding="utf-8")
    # Insert check_fixtures in the MIDDLE of the file, followed by more
    # real content (protected_paths/allowed_diff_paths/oracle), not at EOF.
    text = text.replace(
        "protected_paths: [notes.txt]",
        "check_fixtures:\n"
        "  reference:\n"
        "    main.sh: |\n"
        "      echo one\n"
        "protected_paths: [notes.txt]",
    )
    manifest_path.write_text(text, encoding="utf-8")

    task_after = t.load_b8_tasks(tmp_path)[0]
    assert task_after.check_fixtures
    assert task_after.fixture_sha == sha_before


def test_loader_raises_on_check_fixtures_not_a_mapping(tmp_path):
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    manifest_path = task_dir / "task-01.yaml"
    _write_minimal_manifest(manifest_path, extra_file_content="one")
    text = manifest_path.read_text(encoding="utf-8")
    text += "check_fixtures: not_a_mapping\n"
    manifest_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="check_fixtures must be a mapping"):
        t.load_b8_tasks(tmp_path)


def test_loader_raises_on_check_fixtures_wrong_wrong_count(tmp_path):
    task_dir = tmp_path / "suite" / "b8_harness"
    task_dir.mkdir(parents=True)
    manifest_path = task_dir / "task-01.yaml"
    _write_minimal_manifest(manifest_path, extra_file_content="one")
    text = manifest_path.read_text(encoding="utf-8")
    text += "check_fixtures:\n  wrong: []\n"
    manifest_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="check_fixtures.wrong must be a list of 1-3"):
        t.load_b8_tasks(tmp_path)


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


# -- Python oracle subprocess isolation (task-b8expand hardening) ----------
# Hermetic, NO Docker: each Python oracle_test.py is run directly against
# the HOST python (mirrors the `python3 oracle_test.py` step the bash
# oracle wrapper runs inside the python:3.11-slim sandbox container, minus
# the container -- subprocess isolation is a property of the oracle_test.py
# SOURCE itself, testable directly). Proves the hardening (_schema.md's
# task-b8expand update; codex review Important #1's Python analog) holds
# for all 6 Python task manifests: a solution with a module-level
# `sys.exit(0)` must NOT be marked complete, and a genuinely correct
# solution must still pass.

_ALL_PY_TASK_IDS = ("py-bugfix-01", "py-fromscratch-01", "py-edit-01",
                    "py-multifile-01", "py-toolheavy-01", "py-fromscratch-02")

# The 5 HARDER Python manifests (task-b8hard, task-12..16.yaml) -- kept as
# a separate tuple (rather than folded into _ALL_PY_TASK_IDS) so the
# generic "correct solution passes" / "sys.exit(0) gaming rejected" tests
# below can be parametrized over ALL 11 real Python tasks at once
# (_ALL_PY_TASK_IDS + _HARD_PY_TASK_IDS) while task-specific "known-flawed
# solution fails" tests (further below, one per hard task -- the flawed
# solutions differ in shape per task, so a single generic parametrization
# doesn't fit) can target just these 5.
_HARD_PY_TASK_IDS = ("py-hard-bugfix-01", "py-hard-algo-01", "py-hard-edge-01",
                     "py-hard-multifile-01", "py-hard-toolheavy-01")

# task id -> {relative path: correct file content}. A REFERENCE solution
# for each task, mirroring its own `notes:`/prompt contract -- used only by
# this test file, never shipped to an agent.
_CORRECT_SOLUTIONS = {
    "py-bugfix-01": {
        "stats.py": (
            "def average(nums):\n"
            "    total = sum(nums)\n"
            "    return total / len(nums)\n\n"
            "def summarize(nums):\n"
            "    avg = average(nums)\n"
            "    return f\"summary: avg={avg:.2f}\"\n"
        ),
    },
    "py-fromscratch-01": {
        "textutils.py": (
            "def is_palindrome(s):\n"
            "    cleaned = [c.lower() for c in s if c.isalnum()]\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
    },
    "py-edit-01": {
        "greet.py": "def greet(name):\n    return f\"Hello, {name}!\"\n",
    },
    "py-multifile-01": {
        "calc.py": (
            "def add(a, b):\n    return a + b\n\n"
            "def multiply(a, b):\n    return a * b\n"
        ),
        "formatter.py": "def fmt_result(x):\n    return f\"Result: {x}\"\n",
    },
    "py-toolheavy-01": {
        "handler_charlie.py": "def transform(x):\n    return x * 2 + 1\n",
    },
    "py-fromscratch-02": {
        "exprcalc.py": (
            "def evaluate_expression(expr):\n"
            "    tokens = _tokenize(expr)\n"
            "    pos = [0]\n\n"
            "    def parse_expr():\n"
            "        value = parse_term()\n"
            "        while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):\n"
            "            op = tokens[pos[0]]\n"
            "            pos[0] += 1\n"
            "            rhs = parse_term()\n"
            "            value = value + rhs if op == '+' else value - rhs\n"
            "        return value\n\n"
            "    def parse_term():\n"
            "        value = parse_factor()\n"
            "        while pos[0] < len(tokens) and tokens[pos[0]] == '*':\n"
            "            pos[0] += 1\n"
            "            rhs = parse_factor()\n"
            "            value = value * rhs\n"
            "        return value\n\n"
            "    def parse_factor():\n"
            "        tok = tokens[pos[0]]\n"
            "        if tok == '(':\n"
            "            pos[0] += 1\n"
            "            value = parse_expr()\n"
            "            pos[0] += 1\n"
            "            return value\n"
            "        pos[0] += 1\n"
            "        return int(tok)\n\n"
            "    return parse_expr()\n\n\n"
            "def _tokenize(expr):\n"
            "    tokens = []\n"
            "    i = 0\n"
            "    while i < len(expr):\n"
            "        c = expr[i]\n"
            "        if c.isspace():\n"
            "            i += 1\n"
            "            continue\n"
            "        if c in '+-*()':\n"
            "            tokens.append(c)\n"
            "            i += 1\n"
            "            continue\n"
            "        if c.isdigit():\n"
            "            j = i\n"
            "            while j < len(expr) and expr[j].isdigit():\n"
            "                j += 1\n"
            "            tokens.append(expr[i:j])\n"
            "            i = j\n"
            "            continue\n"
            "        raise ValueError(f'unexpected character: {c!r}')\n"
            "    return tokens\n"
        ),
    },
    "py-hard-bugfix-01": {
        "search_utils.py": (
            "def first_occurrence(nums, target):\n"
            "    lo, hi = 0, len(nums) - 1\n"
            "    result = -1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if nums[mid] == target:\n"
            "            result = mid\n"
            "            hi = mid - 1\n"
            "        elif nums[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return result\n"
        ),
    },
    "py-hard-algo-01": {
        "lru_cache.py": (
            "from collections import OrderedDict\n\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.capacity = capacity\n"
            "        self._data = OrderedDict()\n\n"
            "    def get(self, key):\n"
            "        if key not in self._data:\n"
            "            return -1\n"
            "        self._data.move_to_end(key)\n"
            "        return self._data[key]\n\n"
            "    def put(self, key, value):\n"
            "        if key in self._data:\n"
            "            self._data.move_to_end(key)\n"
            "        self._data[key] = value\n"
            "        if len(self._data) > self.capacity:\n"
            "            self._data.popitem(last=False)\n"
        ),
    },
    "py-hard-edge-01": {
        "rankfind.py": (
            "def second_largest_unique(nums):\n"
            "    largest = second = None\n"
            "    for x in nums:\n"
            "        if largest is None or x > largest:\n"
            "            if largest is not None and largest != x:\n"
            "                second = largest\n"
            "            largest = x\n"
            "        elif x != largest and (second is None or x > second):\n"
            "            second = x\n"
            "    return second\n"
        ),
    },
    "py-hard-multifile-01": {
        "checkout.py": (
            "from pricing import apply_discount, apply_tax\n\n\n"
            "def checkout_total(cart, flat_discount, tax_rate):\n"
            "    subtotal = cart.subtotal()\n"
            "    discounted = apply_discount(subtotal, flat_discount)\n"
            "    total = apply_tax(discounted, tax_rate)\n"
            "    return round(total, 2)\n"
        ),
    },
    "py-hard-toolheavy-01": {
        "letter_grade.py": (
            "def letter_grade(score):\n"
            "    if score >= 90:\n"
            "        return \"A\"\n"
            "    elif score >= 80:\n"
            "        return \"B\"\n"
            "    elif score >= 70:\n"
            "        return \"C\"\n"
            "    elif score >= 60:\n"
            "        return \"D\"\n"
            "    else:\n"
            "        return \"F\"\n"
        ),
    },
}

# Known-FLAWED reference solutions for the 5 hard tasks -- each is a
# plausible-but-wrong solution missing exactly the edge the task's hidden
# oracle traps (see each manifest's own `notes:` for the full rationale).
# For py-hard-bugfix-01/py-hard-multifile-01/py-hard-toolheavy-01 this
# happens to be identical to the manifest's own shipped (unfixed)
# setup_repo content -- the simplest, most direct proof that "doing
# nothing" is correctly rejected. py-hard-algo-01/py-hard-edge-01 are
# from-scratch tasks (no setup_repo file to leave unfixed), so their
# flawed variants are constructed explicitly below.
_HARD_FLAWED_SOLUTIONS = {
    "py-hard-bugfix-01": {
        "search_utils.py": (
            "def first_occurrence(nums, target):\n"
            "    lo, hi = 0, len(nums) - 1\n"
            "    result = -1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if nums[mid] == target:\n"
            "            result = mid\n"
            "            lo = mid + 1\n"  # BUG: should be `hi = mid - 1`
            "        elif nums[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return result\n"
        ),
    },
    "py-hard-algo-01": {
        "lru_cache.py": (
            "from collections import OrderedDict\n\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.capacity = capacity\n"
            "        self._data = OrderedDict()\n\n"
            "    def get(self, key):\n"
            "        if key not in self._data:\n"
            "            return -1\n"
            "        self._data.move_to_end(key)\n"
            "        return self._data[key]\n\n"
            "    def put(self, key, value):\n"
            # BUG: never move_to_end() an EXISTING key -- put() doesn't
            # refresh recency, only get() does.
            "        self._data[key] = value\n"
            "        if len(self._data) > self.capacity:\n"
            "            self._data.popitem(last=False)\n"
        ),
    },
    "py-hard-edge-01": {
        "rankfind.py": (
            "def second_largest_unique(nums):\n"
            # BUG: sentinel 0 instead of None -- breaks on empty/single/
            # all-duplicate input (returns 0, not None) and on all-negative
            # input (no value ever exceeds the 0 sentinel).
            "    largest = second = 0\n"
            "    for x in nums:\n"
            "        if x > largest:\n"
            "            second = largest\n"
            "            largest = x\n"
            "        elif x > second and x != largest:\n"
            "            second = x\n"
            "    return second\n"
        ),
    },
    "py-hard-multifile-01": {
        "checkout.py": (
            "from pricing import apply_discount, apply_tax\n\n\n"
            "def checkout_total(cart, flat_discount, tax_rate):\n"
            "    subtotal = cart.subtotal()\n"
            # BUG: tax applied to the full subtotal BEFORE the discount is
            # subtracted -- wrong order per the stated business rule.
            "    taxed = apply_tax(subtotal, tax_rate)\n"
            "    total = apply_discount(taxed, flat_discount)\n"
            "    return round(total, 2)\n"
        ),
    },
    "py-hard-toolheavy-01": {
        "letter_grade.py": (
            "def letter_grade(score):\n"
            # BUG: `>` instead of `>=` on every boundary -- a curved score
            # of exactly 90/80/70/60 gets the WRONG (one-lower) grade.
            "    if score > 90:\n"
            "        return \"A\"\n"
            "    elif score > 80:\n"
            "        return \"B\"\n"
            "    elif score > 70:\n"
            "        return \"C\"\n"
            "    elif score > 60:\n"
            "        return \"D\"\n"
            "    else:\n"
            "        return \"F\"\n"
        ),
    },
}


def _run_oracle_locally(task_id: str, extra_files: dict, tmp_path: Path):
    """Materialize `task_id`'s setup_repo into a fresh workspace, overlay
    `extra_files` (the candidate solution) on top, write the task's own
    (normally-hidden) oracle_test.py into that SAME workspace, and run it
    with the HOST python -- exactly the `python3 oracle_test.py` step the
    bash oracle wrapper runs inside the sandbox container, minus the
    container."""
    task = _load(task_id)
    ws = tmp_path / task_id
    t.materialize_repo(task, ws)
    for rel, content in extra_files.items():
        (ws / rel).write_text(content, encoding="utf-8")
    (ws / "oracle_test.py").write_text(task.oracle_files["oracle_test.py"], encoding="utf-8")
    return subprocess.run([sys.executable, "oracle_test.py"], cwd=ws,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("task_id", _ALL_PY_TASK_IDS + _HARD_PY_TASK_IDS)
def test_python_oracle_passes_for_a_genuinely_correct_solution(task_id, tmp_path):
    """Regression: the subprocess-isolation hardening must not break a
    real pass -- a genuinely correct solution still gets PASS/exit 0."""
    r = _run_oracle_locally(task_id, _CORRECT_SOLUTIONS[task_id], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


@pytest.mark.parametrize("task_id", _ALL_PY_TASK_IDS + _HARD_PY_TASK_IDS)
def test_python_oracle_rejects_sys_exit_zero_at_import_solution(task_id, tmp_path):
    """The gaming vector task-b8expand closes (codex review Important #1's
    Python analog of the bash `source is_prime.sh; exit 0` finding): a
    solution whose module-level code calls `sys.exit(0)` before its real
    logic must NOT be marked complete. Built by prepending `import sys;
    sys.exit(0)` to each of the task's correct solution file(s) -- the
    real, working logic is still there AFTER the exit call, so this
    isolates the effect of the early exit itself, not a missing
    implementation. Under the pre-hardening (import-into-self) oracle
    design this would have raised SystemExit(0) -- uncaught by `except
    Exception` -- and exited the whole checker with code 0 before any
    check ran, which `Sandbox.hidden_validate` (exit-code-only) would have
    wrongly registered as a pass."""
    malicious = {rel: "import sys\nsys.exit(0)\n\n" + content
                for rel, content in _CORRECT_SOLUTIONS[task_id].items()}
    r = _run_oracle_locally(task_id, malicious, tmp_path)
    assert r.returncode != 0, (
        f"{task_id}: a sys.exit(0)-at-import solution was WRONGLY marked "
        f"complete -- stdout={r.stdout!r}")
    assert "PASS" not in r.stdout


# -- hard tasks (task-b8hard): known-flawed solutions must FAIL ------------
# The load-bearing proof that these 5 manifests actually DISCRIMINATE (the
# whole point of authoring them): a genuinely correct solution passes
# (covered generically above, `_ALL_PY_TASK_IDS + _HARD_PY_TASK_IDS`
# includes all 5), and a PLAUSIBLE-BUT-WRONG solution -- each missing
# exactly the edge its manifest's `notes:` documents -- fails. Hermetic
# (no Docker): runs the oracle's own subprocess-isolated logic directly
# against the host python, same as `_run_oracle_locally`'s other callers.


@pytest.mark.parametrize("task_id", _HARD_PY_TASK_IDS)
def test_hard_task_known_flawed_solution_is_rejected(task_id, tmp_path):
    r = _run_oracle_locally(task_id, _HARD_FLAWED_SOLUTIONS[task_id], tmp_path)
    assert r.returncode != 0, (
        f"{task_id}: known-flawed solution was WRONGLY marked complete -- "
        f"stdout={r.stdout!r}")
    assert "PASS" not in r.stdout
    # The failure detail must actually name a discriminating case (not just
    # "something failed") -- this is the text that becomes
    # det_checks.oracle.detail for the first-failure classifier.
    assert "FAIL:" in r.stdout


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
    HOST, so it must not DESCEND into a symlinked directory -- mirrors the
    exact traversal Sandbox.snapshot_workspace already guards against
    (Task 2 precedent): `Path.rglob` has no way in Python 3.10 to stop
    descending into a symlinked directory, which is exactly why this walk
    doesn't use it either. Paired with a real, unrelated out-of-bounds file
    (sneaky.sh) so this stays deterministic and Docker-free.

    Wave 3a fix: a disallowed symlinked directory is no longer silently
    pruned-and-ignored (the pre-Wave-3a "LIMITATION (documented, not
    fixed)" this test originally proved) -- it is now REPORTED, by name,
    as a `"disallowed symlink: linked_dir"` tamper failure, found and
    returned BEFORE the walk ever reaches `sneaky.sh` (dirs are checked
    before files at each os.walk level). The security property under test
    is unchanged and, if anything, strengthened: the symlink is still
    never TRAVERSED (no descent into it, ever, on the host), so neither
    the outside directory's existence details nor its content can leak
    into `detail` -- only the symlink's OWN path within the workspace
    (`linked_dir`) is named."""
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
    assert "disallowed symlink" in detail
    assert "linked_dir" in detail
    assert "SECRET-HOST-CONTENT" not in detail
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


# -- run_oracle: task-b8expand Python subprocess-isolation, REAL container -
# The strongest possible proof of the hardening: the FULL run_oracle path
# (real Docker container via oracle_image="python:3.11-slim", real
# Sandbox.hidden_validate) -- not just the local-host-subprocess execution
# the tests above this section use -- against a candidate solution that
# tries to short-circuit the checker with a module-level `sys.exit(0)`.


@requires_docker
def test_run_oracle_true_for_correct_py_multifile_workspace(tmp_path):
    """Real, end-to-end proof (via the actual python:3.11-slim container)
    that a genuinely correct Python multi-file solution passes through the
    REAL run_oracle path -- mirrors the bash multifile-01 pair above."""
    task = _load("py-multifile-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "calc.py").write_bytes(b"""\
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b
""")
    (ws / "formatter.py").write_bytes(b"""\
def fmt_result(x):
    return f"Result: {x}"
""")
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_py_multifile_sys_exit_zero_gaming_attempt(tmp_path):
    """The task-3-brief.md-style Step-1 scenario, for the task-b8expand
    Python-oracle-hardening finding: run the REAL run_oracle (real Docker
    container, real Sandbox.hidden_validate) against a `calc.py` that
    tries to short-circuit the checker with a module-level `sys.exit(0)`
    placed BEFORE its otherwise-correct `multiply()` -- must NOT be marked
    complete. Under the pre-hardening (import-into-self) oracle design,
    this exact `sys.exit(0)` would have killed the whole oracle_test.py
    process with exit code 0 before any check ran, and `hidden_validate`
    (which only inspects the process exit code) would have wrongly
    reported a pass."""
    task = _load("py-multifile-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "calc.py").write_bytes(b"""\
import sys
sys.exit(0)


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b
""")
    (ws / "formatter.py").write_bytes(b"""\
def fmt_result(x):
    return f"Result: {x}"
""")
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False, detail


# -- run_oracle: task-b8hard, the 5 harder Python manifests, REAL container -
# Full end-to-end proof (real Docker container, real Sandbox.hidden_validate,
# real protected-hash + diff-constraint hard caps) for every one of the 5
# harder manifests: a genuinely correct fix passes, and the manifest's own
# documented flawed/unfixed variant fails -- the strongest form of the
# "oracle discriminates" proof the task-b8hard build calls for, on top of
# the hermetic (no-Docker) versions above.


@requires_docker
def test_run_oracle_true_for_correct_hard_bugfix_workspace(tmp_path):
    task = _load("py-hard-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "search_utils.py").write_bytes(
        _CORRECT_SOLUTIONS["py-hard-bugfix-01"]["search_utils.py"].encode("utf-8"))
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_hard_bugfix_unfixed_workspace(tmp_path):
    """No edit at all -- the last-occurrence bug (duplicates) is still
    present."""
    task = _load("py-hard-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False, detail


@requires_docker
def test_run_oracle_true_for_correct_hard_algo_workspace(tmp_path):
    task = _load("py-hard-algo-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "lru_cache.py").write_bytes(
        _CORRECT_SOLUTIONS["py-hard-algo-01"]["lru_cache.py"].encode("utf-8"))
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_hard_algo_workspace_missing_put_mru_update(tmp_path):
    """The `put()`-on-an-existing-key-must-refresh-recency edge, via the
    REAL container path -- the flaw the hermetic parametrized test above
    also covers, re-verified end to end."""
    task = _load("py-hard-algo-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "lru_cache.py").write_bytes(
        _HARD_FLAWED_SOLUTIONS["py-hard-algo-01"]["lru_cache.py"].encode("utf-8"))
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False, detail


@requires_docker
def test_run_oracle_true_for_correct_hard_edge_workspace(tmp_path):
    task = _load("py-hard-edge-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "rankfind.py").write_bytes(
        _CORRECT_SOLUTIONS["py-hard-edge-01"]["rankfind.py"].encode("utf-8"))
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_hard_edge_sentinel_zero_workspace(tmp_path):
    task = _load("py-hard-edge-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "rankfind.py").write_bytes(
        _HARD_FLAWED_SOLUTIONS["py-hard-edge-01"]["rankfind.py"].encode("utf-8"))
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False, detail


@requires_docker
def test_run_oracle_true_for_correct_hard_multifile_workspace(tmp_path):
    task = _load("py-hard-multifile-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "checkout.py").write_bytes(
        _CORRECT_SOLUTIONS["py-hard-multifile-01"]["checkout.py"].encode("utf-8"))
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_hard_multifile_unfixed_workspace(tmp_path):
    """No edit at all -- checkout.py still applies tax before the
    discount."""
    task = _load("py-hard-multifile-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False, detail


@requires_docker
def test_run_oracle_false_for_hard_multifile_protected_tamper_even_with_correct_fix(tmp_path):
    """The "real protected_paths fixture" requirement: cart.py/pricing.py
    are agent-visible but hash-checked. A gamer correctly fixes checkout.py
    (behavior would genuinely pass) but also tampers with the protected
    pricing.py -- the hard cap must catch that BEFORE the behavioral
    oracle ever runs, mirroring the brief's Step-1 scenario."""
    task = _load("py-hard-multifile-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "checkout.py").write_bytes(
        _CORRECT_SOLUTIONS["py-hard-multifile-01"]["checkout.py"].encode("utf-8"))
    (ws / "pricing.py").write_bytes(b"tampered\n")
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False
    assert "protected" in detail.lower()
    assert "pricing.py" in detail


@requires_docker
def test_run_oracle_true_for_correct_hard_toolheavy_workspace(tmp_path):
    task = _load("py-hard-toolheavy-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "letter_grade.py").write_bytes(
        _CORRECT_SOLUTIONS["py-hard-toolheavy-01"]["letter_grade.py"].encode("utf-8"))
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is True, detail


@requires_docker
def test_run_oracle_false_for_hard_toolheavy_unfixed_workspace(tmp_path):
    """No edit at all -- the `>`-vs-`>=` boundary bug is still present."""
    task = _load("py-hard-toolheavy-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False, detail


@requires_docker
def test_run_oracle_false_for_hard_toolheavy_decoy_tamper_even_with_correct_fix(tmp_path):
    """The decoy-protection requirement this manifest is specifically
    designed to exercise: `curve.py` is an unusual-looking but CORRECT
    file (the deliberate red herring) and is agent-visible/protected. A
    model that correctly fixes `letter_grade.py` (behavior would
    genuinely pass) but ALSO "fixes" the decoy `curve.py` must still be
    rejected by the hard cap, before the behavioral oracle ever runs --
    exactly the failure mode a model that doesn't trust the instructions
    ("only modify letter_grade.py") would hit."""
    task = _load("py-hard-toolheavy-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "letter_grade.py").write_bytes(
        _CORRECT_SOLUTIONS["py-hard-toolheavy-01"]["letter_grade.py"].encode("utf-8"))
    (ws / "curve.py").write_bytes(b"tampered\n")
    completed, detail = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert completed is False
    assert "protected" in detail.lower()
    assert "curve.py" in detail


# -- Wave 3a: machine-readable oracle results + tamper detection ------------
# "Make the completion scorer DEFENSIBLE": (1) the oracle's own JSON result
# line parses into det_checks.oracle's structured fields (stage/reason_code/
# case/expected/actual), with `actual` bounded and a clean `detail` derived
# from the structured fields rather than raw stdout/stderr; (2) the
# diff-constraint now also catches deletions, disallowed symlinks, and
# setup-file type changes -- gaps the pre-Wave-3a walk (which only ever
# inspected files that still EXIST, unchanged) could not see at all.


def test_oracle_result_unpacks_like_the_legacy_two_tuple():
    """`OracleResult.__iter__` is the whole backward-compat mechanism this
    wave relies on -- pin it directly, not just via every call site that
    happens to exercise it."""
    result = t.OracleResult(pass_=False, detail="out-of-bounds edit: x",
                            stage="behavior", reason_code="wrong_output",
                            case="f(1)", expected="2", actual="3")
    completed, detail = result
    assert completed is False
    assert detail == "out-of-bounds edit: x"
    assert result.stage == "behavior"
    assert result.reason_code == "wrong_output"


def test_parse_oracle_json_result_finds_last_json_line():
    stdout = "some noise\nFAIL: f(1) -> 3 (want 2)\n" + \
             '{"pass": false, "stage": "behavior", "reason_code": "wrong_output", ' \
             '"case": "f(1)", "expected": "2", "actual": "3", "exit_status": 1}\n'
    parsed = t._parse_oracle_json_result(stdout)
    assert parsed == {"pass": False, "stage": "behavior", "reason_code": "wrong_output",
                      "case": "f(1)", "expected": "2", "actual": "3", "exit_status": 1}


def test_parse_oracle_json_result_tolerates_trailing_blank_line():
    stdout = '{"pass": true}\n\n'
    assert t._parse_oracle_json_result(stdout) == {"pass": True}


def test_parse_oracle_json_result_returns_none_for_no_json():
    assert t._parse_oracle_json_result("PASS\n") is None
    assert t._parse_oracle_json_result("") is None
    assert t._parse_oracle_json_result(None) is None


def test_parse_oracle_json_result_ignores_json_without_bool_pass_key():
    """A coincidentally JSON-shaped debug line with no boolean `pass` key
    is not mistaken for the real result -- requiring the `pass` key (and
    that it's specifically a bool, not e.g. the string "true") is the
    guard against that."""
    assert t._parse_oracle_json_result('{"note": "just some debug json"}\n') is None
    assert t._parse_oracle_json_result('{"pass": "true"}\n') is None


def test_oracle_result_from_validation_malformed_json_falls_back_gracefully(monkeypatch):
    """The 5 bash placeholder manifests (task-01..05.yaml) predate the
    Wave 3a JSON convention and never will emit it -- `run_oracle` must not
    crash on their stdout, and must fall back to the EXACT pre-Wave-3a
    `ValidateResult.detail` shape (the free-form `exit=... stdout=...
    stderr=...` dump), with every structured field left `None`."""
    from llmtest.harness.sandbox import ValidateResult
    vr = ValidateResult(True, "exit=0 stdout='PASS\\n' stderr=''", stdout="PASS\n")
    result = t._oracle_result_from_validation(vr)
    assert result.pass_ is True
    assert result.detail == vr.detail
    assert result.stage is None
    assert result.reason_code is None
    assert result.case is None
    assert result.expected is None
    assert result.actual is None


def test_oracle_result_from_validation_no_stdout_falls_back_gracefully():
    """`ValidateResult.stdout=None` (callable oracle, timeout, or a
    pre-oracle setup failure -- see that dataclass's docstring) must not
    crash the JSON scan either."""
    from llmtest.harness.sandbox import ValidateResult
    vr = ValidateResult(False, "oracle timeout: exceeded wall-clock budget (...)", stdout=None)
    result = t._oracle_result_from_validation(vr)
    assert result.pass_ is False
    assert result.detail == vr.detail
    assert result.stage is None


def test_oracle_result_from_validation_parses_pass_and_derives_clean_detail():
    from llmtest.harness.sandbox import ValidateResult
    stdout = ('PASS\n'
              '{"pass": true, "stage": "behavior", "reason_code": null, '
              '"case": null, "expected": null, "actual": null, "exit_status": 0}\n')
    vr = ValidateResult(True, f"exit=0 stdout={stdout!r} stderr=''", stdout=stdout)
    result = t._oracle_result_from_validation(vr)
    assert result.pass_ is True
    assert result.detail == "PASS"
    assert result.stage == "behavior"
    assert result.reason_code is None


def test_oracle_result_from_validation_parses_fail_and_bounds_actual():
    """`actual` (a candidate's own printed output, per the schema) is
    truncated to `_ACTUAL_TRUNCATE_CHARS`, and the derived `detail` is
    built from the ALREADY-bounded value -- never the raw, unbounded one."""
    from llmtest.harness.sandbox import ValidateResult
    long_actual = "Z" * 500
    stdout = ('FAIL: f(1) -> ... (want 2)\n' +
             json.dumps({"pass": False, "stage": "behavior", "reason_code": "wrong_output",
                        "case": "f(1)", "expected": "2", "actual": long_actual,
                        "exit_status": 1}) + "\n")
    vr = ValidateResult(False, f"exit=1 stdout={stdout!r} stderr=''", stdout=stdout)
    result = t._oracle_result_from_validation(vr)
    assert result.pass_ is False
    assert result.stage == "behavior"
    assert result.reason_code == "wrong_output"
    assert len(result.actual) <= len("Z" * 200) + len("...<TRUNCATED 300 more chars>")
    assert result.actual.endswith("<TRUNCATED 300 more chars>")
    # detail is derived FROM the bounded actual, not the raw 500-char one.
    assert long_actual not in result.detail
    assert "TRUNCATED" in result.detail


def test_truncate_leaves_short_text_unchanged():
    assert t._truncate("short") == "short"


# -- run_oracle: new tamper checks (deletion / disallowed symlink / type --
# change), no Docker needed for any of these -- they're all hard-cap
# checks step (c) never reaches.


def _write_third_category_manifest(task_dir: Path) -> None:
    """A manifest with a setup_repo path (`readonly_helper.sh`) that is
    NEITHER `protected_paths` NOR `allowed_diff_paths` -- every one of the
    16 real manifests happens to fully partition setup_repo between those
    two categories, so this THIRD category (exercised by the new Wave 3a
    deletion/type-change checks below) needs a synthetic manifest to prove
    it's covered at all."""
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task-01.yaml").write_text("""\
id: edit-99
shape: edit
task_version: "1.0.0"
prompt: "do a thing"
allowed_tools: [read_file, write_file]
budgets: {wall_clock_s: 60, tokens: 500, steps: 4}
setup_repo:
  main.sh: |
    echo hi
  notes.txt: |
    protected, agent-visible
  readonly_helper.sh: |
    echo helper
oracle_files:
  oracle_test.sh: |
    echo PASS
protected_paths: [notes.txt]
allowed_diff_paths: [main.sh]
oracle:
  type: command
  argv: ["bash", "-c", "cp -r /oracle /tmp/work && cd /tmp/work && bash oracle_test.sh"]
""", encoding="utf-8")


def test_run_oracle_false_on_setup_file_deleted(tmp_path):
    """Deletion detection: a setup_repo path that is neither protected nor
    diff-allowed, simply removed from the post-run workspace, must be
    caught -- pre-Wave-3a, the os.walk-based diff-constraint only ever
    inspected files that still EXIST, so an outright deletion was
    invisible to it."""
    _write_third_category_manifest(tmp_path / "suite" / "b8_harness")
    task = t.load_b8_tasks(tmp_path)[0]
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "readonly_helper.sh").unlink()

    completed, detail = t.run_oracle(task, ws)
    assert completed is False
    assert "deleted" in detail.lower()
    assert "readonly_helper.sh" in detail


def test_run_oracle_deletion_check_exempts_allowed_diff_paths(tmp_path):
    """The flip side: deleting an `allowed_diff_paths` member is NOT a
    deletion-tamper failure by itself (a from-scratch deliverable may
    legitimately never get written) -- it falls through to whatever the
    rest of the pipeline says (here, no other files changed, so it's still
    schema-clean up to the behavioral oracle, which this test doesn't
    reach)."""
    _write_third_category_manifest(tmp_path / "suite" / "b8_harness")
    task = t.load_b8_tasks(tmp_path)[0]
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "main.sh").unlink()   # main.sh IS in allowed_diff_paths

    completed, detail = t.run_oracle(task, ws)
    # Must NOT be rejected for "deleted" -- whatever run_oracle ultimately
    # decides (it will go on to try the behavioral oracle, which may fail
    # for its own reasons), the deletion-tamper message must never appear.
    assert "deleted" not in detail.lower()


def test_run_oracle_false_on_setup_file_replaced_with_directory(tmp_path):
    """Type/mode change: an EMPTY directory silently replacing a
    setup_repo file produces neither a content diff (nothing under it to
    walk) nor a deletion-check hit (the path still exists) -- it needs its
    own explicit check."""
    _write_third_category_manifest(tmp_path / "suite" / "b8_harness")
    task = t.load_b8_tasks(tmp_path)[0]
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "readonly_helper.sh").unlink()
    (ws / "readonly_helper.sh").mkdir()

    completed, detail = t.run_oracle(task, ws)
    assert completed is False
    assert "directory" in detail.lower()
    assert "readonly_helper.sh" in detail


def test_run_oracle_false_on_disallowed_symlink_at_file_path(tmp_path):
    """Disallowed-symlink detection (file path, not a directory -- the
    directory case is covered by
    test_run_oracle_diff_constraint_does_not_traverse_symlinked_dir
    above): a symlink planted at a path that is neither protected nor
    diff-allowed must be flagged BY NAME, not silently skipped the way the
    pre-Wave-3a walk treated every symlinked file."""
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    try:
        os.symlink(str(ws / "stats.sh"), ws / "sneaky_link.sh")
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"os.symlink not permitted on this host: {e!r}")

    completed, detail = t.run_oracle(task, ws)
    assert completed is False
    assert "disallowed symlink" in detail.lower()
    assert "sneaky_link.sh" in detail


def test_run_oracle_symlink_at_allowed_diff_path_not_flagged_as_disallowed(tmp_path):
    """The flip side: a symlink planted AT an `allowed_diff_paths` path is
    not itself flagged as a disallowed-symlink tamper (no manifest today
    opts a path INTO symlink-permitted status explicitly, so this is
    simply "not on the disallow list") -- whatever happens next (the
    behavioral oracle almost certainly still fails on it) is a separate
    concern from tamper detection."""
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    other = ws / "NOTES.md"
    try:
        os.remove(ws / "stats.sh")
        os.symlink(str(other), ws / "stats.sh")   # stats.sh IS allowed_diff
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"os.symlink not permitted on this host: {e!r}")

    completed, detail = t.run_oracle(task, ws)
    assert "disallowed symlink" not in detail.lower()


# -- Wave 3a: real python:3.11-slim round-trip (needs Docker) ---------------


@requires_docker
def test_run_oracle_correct_solution_yields_structured_pass(tmp_path):
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    (ws / "stats.py").write_bytes(
        _CORRECT_SOLUTIONS["py-bugfix-01"]["stats.py"].encode("utf-8"))

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert result.pass_ is True
    assert result.detail == "PASS"
    assert result.stage == "behavior"
    assert result.reason_code is None


@requires_docker
def test_run_oracle_wrong_solution_yields_structured_fail_with_bounded_actual(tmp_path):
    """The end-to-end proof (real container, real oracle_test.py, real
    run_oracle) that a wrong solution's det_checks.oracle carries
    structured, USEFUL fields -- not just a bare `False` -- and that a
    candidate solution printing well over `_ACTUAL_TRUNCATE_CHARS` worth
    of output has `actual` genuinely bounded, not merely usually-short."""
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    long_wrong = (
        "def average(nums):\n"
        "    total = sum(nums)\n"
        "    return total / len(nums)\n\n"
        "def summarize(nums):\n"
        f"    return {'Z' * 500!r}\n"
    )
    (ws / "stats.py").write_bytes(long_wrong.encode("utf-8"))

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert result.pass_ is False
    assert result.stage == "behavior"
    assert result.reason_code == "wrong_output"
    assert result.case == "summarize([2, 4, 6])"
    assert result.expected == "summary: avg=4.00"
    assert len(result.actual) <= t._ACTUAL_TRUNCATE_CHARS + len("...<TRUNCATED 300 more chars>")
    assert "TRUNCATED" in result.actual
    assert "FAIL:" in result.detail
    assert "TRUNCATED" in result.detail


@requires_docker
def test_run_oracle_crashed_solution_yields_compile_stage(tmp_path):
    """The unfixed py-bugfix-01 workspace (the planted SyntaxError, never
    touched) -- the canonical `compile`-stage/`syntax_error` case."""
    task = _load("py-bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)   # bug left in place -- stats.py has a SyntaxError

    result = t.run_oracle(task, ws, oracle_image="python:3.11-slim")
    assert result.pass_ is False
    assert result.stage == "compile"
    assert result.reason_code == "syntax_error"


@requires_docker
def test_run_oracle_bash_placeholder_manifest_falls_back_to_free_form_detail(tmp_path):
    """The 5 bash placeholder manifests (task-01..05.yaml) predate the
    Wave 3a JSON convention and never will emit the JSON result line --
    real, end-to-end proof (not just the hermetic
    test_oracle_result_from_validation_malformed_json_falls_back_gracefully
    above) that `run_oracle` still scores them correctly, with every
    structured field left `None`."""
    task = _load("bugfix-01")
    ws = tmp_path / "ws"
    t.materialize_repo(task, ws)
    fixed = task.setup_repo["stats.sh"].replace(
        "    echo $((sum / $#))\n\nsummarize",
        "    echo $((sum / $#))\n}\n\nsummarize",
    )
    (ws / "stats.sh").write_bytes(fixed.encode("utf-8"))

    result = t.run_oracle(task, ws)
    assert result.pass_ is True
    assert result.stage is None
    assert result.reason_code is None
    assert result.case is None
