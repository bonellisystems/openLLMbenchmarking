"""Versioned B8 task manifests + the deterministic, anti-gaming completion
oracle (Task 3, Part 2 Phase 1) -- what decides, after a real agent-harness
run (Phase 2, deferred) exits, whether the task actually got done.

Mirrors `llmtest.batteries.b6_fixtures`'s loader shape (a `@dataclass` task
+ `load_*` that fails loud on a malformed fixture, one exception wrapping
per file) but the manifest itself is richer: each `suite/b8_harness/
task-*.yaml` embeds a small INITIAL REPO (`setup_repo`: relative path ->
full file content, inline block scalars -- same idea as B6's `buggy_code`,
just multi-file) plus a hidden behavioral ORACLE and the anti-gaming
metadata needed to run it.

LANGUAGE NOTE -- why these manifests are bash, not the Python B6 uses:
the pinned sandbox image (`nvidia/cuda:12.6.2-base-ubuntu24.04`, Task 2)
has NO Python, gcc, or node -- only bash/sh/perl/coreutils (confirmed by
running the image directly; `apt-get install python3` also fails, offline
+ `--network none`). Design Decisions #2/#3 (task-3-brief.md) require the
behavioral-oracle-pass path to run INSIDE that container, since it executes
agent-produced code -- so a host-side Python `compile()`/`exec()` callable
is not an option (it wouldn't need Docker, defeating the sandbox boundary).
Bash is in `b6_fixtures._VALID_LANGUAGES` too. The brief explicitly licenses
this: "the SHAPE ... is what matters, not literally snake" / "planted-bug
-style" -- so B6's shapes (missing-terminator bugfix, from-scratch codegen)
are reused faithfully, ported to a language the pinned image can actually
execute. See the report for the full rationale.

Anti-gaming precedence (`run_oracle`, task-3-brief.md Design Decision #1) --
HARD CAPS run first, before the behavioral oracle, and without Docker:
  (a) protected-file hash check: any `protected_shas` mismatch (or missing
      file) -> immediate `(False, ...)`, even if the behavioral task would
      otherwise pass. This is what stops a gamer from editing the hidden
      oracle script itself to force a pass.
  (b) diff constraint: any file that differs from `setup_repo`'s initial
      content (or is new) and is not in `allowed_diff_paths` (nor
      `protected_shas`, already covered by (a)) -> `(False, ...)`.
  (c) only once (a) and (b) both pass: the behavioral oracle runs, via
      `Sandbox.hidden_validate(task.oracle, workspace)` -- a FRESH,
      read-only copy in a throwaway container outside the agent's reach.
Steps (a)/(b) never construct a `Sandbox`, so they need no Docker at all;
only (c) does (guarded by `@requires_docker` in the test).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llmtest.harness.sandbox import Sandbox

_TASKS_DIR = "b8_harness"

_VALID_SHAPES = {"edit", "multi-file", "bugfix", "tool-heavy", "from-scratch"}

_REQUIRED_KEYS = (
    "id", "shape", "task_version", "prompt", "allowed_tools", "budgets",
    "setup_repo", "protected_paths", "allowed_diff_paths", "oracle",
)
_REQUIRED_BUDGET_KEYS = ("wall_clock_s", "tokens", "steps")


@dataclass
class B8Task:
    """One B8 task manifest -- an immutable, versioned (initial repo,
    hidden oracle) pair. Field order leads with the brief's exact interface
    list (`id, shape, setup_repo_sha, allowed_tools, budgets, oracle,
    protected_shas, task_version, fixture_sha`); the remaining fields are
    additive and required to actually MATERIALIZE a workspace and enforce
    the diff constraint -- `run_oracle`'s Design Decision #1b cannot be
    implemented without the initial repo content to diff against."""

    id: str
    shape: str
    setup_repo_sha: str
    allowed_tools: list[str]
    budgets: dict
    oracle: list[str]
    protected_shas: dict[str, str]
    task_version: str
    fixture_sha: str
    setup_repo: dict[str, str] = field(default_factory=dict)
    protected_paths: list[str] = field(default_factory=list)
    allowed_diff_paths: list[str] = field(default_factory=list)
    prompt: str = ""
    path: Path | None = None


def _hash_setup_repo(setup_repo: dict[str, str]) -> str:
    """Deterministic hash over the initial repo tree: sorted (path,
    content) pairs, NUL-separated so no path/content concatenation can
    collide across different splits."""
    h = hashlib.sha256()
    for rel_path in sorted(setup_repo):
        h.update(rel_path.encode("utf-8"))
        h.update(b"\x00")
        h.update(setup_repo[rel_path].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_b8_tasks(root: Path) -> list[B8Task]:
    """Load all B8 task manifests from suite/b8_harness/task-*.yaml.

    Fail-loud (mirrors b6_fixtures.load_tasks): a missing required key, an
    unknown shape, or a protected path absent from setup_repo raises
    ValueError rather than silently skipping the manifest -- a
    silently-dropped anti-gaming check is worse than no check at all.
    """
    tasks_dir = Path(root) / "suite" / _TASKS_DIR
    if not tasks_dir.exists():
        return []

    out: list[B8Task] = []
    for task_file in sorted(tasks_dir.glob("task-*.yaml")):
        try:
            raw = task_file.read_bytes()
            fixture_sha = hashlib.sha256(raw).hexdigest()
            data = yaml.safe_load(raw.decode("utf-8"))

            for key in _REQUIRED_KEYS:
                if key not in data:
                    raise ValueError(f"missing required key: {key}")

            shape = data["shape"]
            if shape not in _VALID_SHAPES:
                raise ValueError(f"invalid shape: {shape!r} (must be one of {sorted(_VALID_SHAPES)})")

            setup_repo = data["setup_repo"]
            if not isinstance(setup_repo, dict) or not setup_repo:
                raise ValueError("setup_repo must be a non-empty mapping of path -> content")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in setup_repo.items()):
                raise ValueError("setup_repo keys/values must all be strings")

            protected_paths = data["protected_paths"]
            if not isinstance(protected_paths, list) or not protected_paths:
                raise ValueError("protected_paths must be a non-empty list")
            for p in protected_paths:
                if p not in setup_repo:
                    raise ValueError(f"protected path not found in setup_repo: {p}")

            allowed_diff_paths = data["allowed_diff_paths"]
            if not isinstance(allowed_diff_paths, list) or not allowed_diff_paths:
                raise ValueError("allowed_diff_paths must be a non-empty list")
            overlap = set(allowed_diff_paths) & set(protected_paths)
            if overlap:
                raise ValueError(f"paths cannot be both protected and diff-allowed: {sorted(overlap)}")

            allowed_tools = data["allowed_tools"]
            if not isinstance(allowed_tools, list) or not allowed_tools:
                raise ValueError("allowed_tools must be a non-empty list")

            budgets = data["budgets"]
            if not isinstance(budgets, dict):
                raise ValueError("budgets must be a mapping")
            for key in _REQUIRED_BUDGET_KEYS:
                if key not in budgets:
                    raise ValueError(f"budgets missing required key: {key}")

            oracle_spec = data["oracle"]
            if not isinstance(oracle_spec, dict) or "type" not in oracle_spec:
                raise ValueError("oracle must be a mapping with a 'type' key")
            if oracle_spec["type"] != "command":
                raise ValueError(f"unsupported oracle type: {oracle_spec['type']!r} (only 'command' is supported)")
            argv = oracle_spec.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
                raise ValueError("oracle.argv must be a non-empty list of strings")

            protected_shas = {p: hashlib.sha256(setup_repo[p].encode("utf-8")).hexdigest()
                               for p in protected_paths}

            task = B8Task(
                id=data["id"], shape=shape,
                setup_repo_sha=_hash_setup_repo(setup_repo),
                allowed_tools=list(allowed_tools), budgets=dict(budgets),
                oracle=list(argv), protected_shas=protected_shas,
                task_version=str(data["task_version"]), fixture_sha=fixture_sha,
                setup_repo=dict(setup_repo), protected_paths=list(protected_paths),
                allowed_diff_paths=list(allowed_diff_paths),
                prompt=data["prompt"], path=task_file)
            out.append(task)
        except Exception as e:  # noqa: BLE001 - wrap with file context, mirrors b6_fixtures
            raise ValueError(f"malformed manifest {task_file}: {e}") from e

    return sorted(out, key=lambda x: x.id)


def materialize_repo(task: B8Task, dest: str | Path) -> Path:
    """Write `task.setup_repo` out to `dest` as real files -- the initial
    workspace a harness adapter would hand an agent (Phase 2) or a test
    builds by hand. Always `write_bytes` (never `write_text`): on Windows,
    text-mode writes translate `\\n` -> `\\r\\n`, which would silently
    corrupt the on-disk bytes relative to what `protected_shas` /
    `setup_repo` hash, and would break bash scripts executed inside the
    (Linux) sandbox container.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel_path, content in task.setup_repo.items():
        p = dest / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))
    return dest


def run_oracle(task: B8Task, workspace: str | Path, *, root: str | Path = ".") -> tuple[bool, str]:
    """Decide whether `task` was actually completed in the post-run
    `workspace`. See the module docstring for the full precedence
    rationale; in short: protected-file tamper and out-of-bounds edits are
    HARD CAPS checked first (no Docker needed), and only then does the
    hidden behavioral oracle run, in a fresh container-isolated copy.
    """
    ws = Path(workspace)

    # (a) protected-file hash check -- HARD CAP, before anything else.
    for rel_path in sorted(task.protected_shas):
        f = ws / rel_path
        if not f.is_file():
            return False, f"protected file tampered: {rel_path} (missing)"
        actual = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual != task.protected_shas[rel_path]:
            return False, f"protected file tampered: {rel_path}"

    # (b) diff constraint -- only allowed_diff_paths (or protected, already
    # verified unchanged above) may differ from the initial repo. Walked the
    # same way Sandbox.snapshot_workspace walks a workspace (Task 2
    # precedent): os.walk(followlinks=False), symlinked subdirectories
    # pruned BEFORE os.walk descends into them, symlinked files skipped
    # directly. This reads the agent-controlled workspace on the HOST, so a
    # planted symlink (a file pointing at an arbitrary host path, or a
    # directory pointing outside the workspace) must never be followed or
    # traversed here -- Path.rglob has no way in Python 3.10 to stop
    # descending into a symlinked directory, which is exactly why
    # snapshot_workspace does not use it either.
    allowed = set(task.allowed_diff_paths) | set(task.protected_shas)
    for dirpath, dirnames, filenames in os.walk(ws, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_symlink():
                continue
            rel = full.relative_to(ws).as_posix()
            if rel in task.protected_shas:
                continue
            initial = task.setup_repo.get(rel)
            current = full.read_bytes()
            changed = initial is None or current != initial.encode("utf-8")
            if changed and rel not in allowed:
                return False, f"out-of-bounds edit: {rel}"

    # (c) behavioral oracle -- fresh, read-only, isolated copy.
    sbx = Sandbox(workspace=ws, root=root)
    return sbx.hidden_validate(task.oracle, ws)
