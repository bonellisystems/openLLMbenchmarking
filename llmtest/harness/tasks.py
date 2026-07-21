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

THREE file categories, not two (post-review hardening -- see task-3-report.md
"Fix note" for the finding this closes): a manifest's files split into
  - `setup_repo` -- the agent-VISIBLE, agent-modifiable initial workspace.
    `materialize_repo` writes ONLY this to disk; this is everything a real
    harness adapter (Phase 2) would hand an agent.
  - `oracle_files` -- the hidden behavioral-oracle script(s). NEVER written
    by `materialize_repo`, NEVER present in the agent's workspace at any
    point. `run_oracle` re-injects them into a private copy immediately
    before running the oracle (step (c) below) -- the agent workspace it
    was handed, and the post-run workspace it hands back, never contained
    them, so there is nothing for the agent to read and overfit against.
  - `protected_paths` (drawn from `setup_repo`, never from `oracle_files`)
    -- agent-VISIBLE files that must nonetheless remain byte-identical
    after the run (e.g. a fixture file the agent isn't supposed to touch).
    The loader enforces `protected_paths ⊆ setup_repo.keys()`, which
    structurally guarantees a manifest author cannot accidentally "protect"
    a hidden oracle file instead of withholding it.
This closes a real gap the first pass of this module had: putting
`oracle_test.sh` inside `setup_repo` (hash-protected, but still
agent-readable) satisfied only ONE of the two DISTINCT global constraints
("protected files hash-checked") and not the other ("hidden validators
live OUTSIDE the writable workspace") -- an agent could read the hidden
test and hand-craft a workspace that satisfies it (e.g. checking for a
specific planted filename) without doing the real task. Withholding
`oracle_files` from the agent workspace entirely, and re-injecting only at
validation time from a copy the agent never sees, makes "hidden" structurally
true rather than a naming convention.

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
      otherwise pass. This is what stops a gamer from editing a protected,
      agent-visible fixture file to force a pass.
  (b) diff constraint: any file that differs from `setup_repo`'s initial
      content (or is new) and is not in `allowed_diff_paths` (nor
      `protected_shas`, already covered by (a)) -> `(False, ...)`.
  (c) only once (a) and (b) both pass: `oracle_files` are re-injected into
      a private copy of the (already-validated) workspace, and the
      behavioral oracle runs against THAT copy via
      `Sandbox.hidden_validate(task.oracle, ...)` -- a FRESH, read-only
      copy in a throwaway container outside the agent's reach, that now
      also contains the oracle the agent never got to see.
Steps (a)/(b) never construct a `Sandbox`, so they need no Docker at all;
only (c) does (guarded by `@requires_docker` in the test).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llmtest.harness.sandbox import Sandbox, copy_real_files

_TASKS_DIR = "b8_harness"

_VALID_SHAPES = {"edit", "multi-file", "bugfix", "tool-heavy", "from-scratch"}

_REQUIRED_KEYS = (
    "id", "shape", "task_version", "prompt", "allowed_tools", "budgets",
    "setup_repo", "oracle_files", "protected_paths", "allowed_diff_paths",
    "oracle",
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
    implemented without the initial repo content to diff against.

    `oracle_files` is additive for the same reason and is the field that
    makes withholding possible: `materialize_repo` never touches it, only
    `run_oracle`'s step (c) does (re-injecting into a copy the agent never
    sees). `setup_repo_sha` is computed over `setup_repo` ONLY -- it never
    includes `oracle_files`, so it reflects exactly what the agent could
    ever have observed."""

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
    oracle_files: dict[str, str] = field(default_factory=dict)
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
    unknown shape, a protected path absent from setup_repo, or a path
    claimed by both setup_repo and oracle_files raises ValueError rather
    than silently skipping the manifest -- a silently-dropped anti-gaming
    check (or a hidden-oracle file that accidentally leaked into the
    agent-visible tree) is worse than no check at all.
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

            oracle_files = data["oracle_files"]
            if not isinstance(oracle_files, dict) or not oracle_files:
                raise ValueError("oracle_files must be a non-empty mapping of path -> content")
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in oracle_files.items()):
                raise ValueError("oracle_files keys/values must all be strings")
            file_overlap = set(setup_repo) & set(oracle_files)
            if file_overlap:
                raise ValueError(
                    f"paths cannot be in both setup_repo (agent-visible) and "
                    f"oracle_files (hidden): {sorted(file_overlap)}")

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
                setup_repo=dict(setup_repo), oracle_files=dict(oracle_files),
                protected_paths=list(protected_paths),
                allowed_diff_paths=list(allowed_diff_paths),
                prompt=data["prompt"], path=task_file)
            out.append(task)
        except Exception as e:  # noqa: BLE001 - wrap with file context, mirrors b6_fixtures
            raise ValueError(f"malformed manifest {task_file}: {e}") from e

    return sorted(out, key=lambda x: x.id)


def materialize_repo(task: B8Task, dest: str | Path) -> Path:
    """Write `task.setup_repo` out to `dest` as real files -- the initial
    workspace a harness adapter would hand an agent (Phase 2) or a test
    builds by hand. Writes ONLY `setup_repo` -- `task.oracle_files` is
    NEVER written here, by design: the agent workspace must never contain
    the hidden oracle. (`run_oracle`'s step (c) re-injects `oracle_files`
    into a private, agent-never-saw-it copy, immediately before running the
    behavioral oracle.) Always `write_bytes` (never `write_text`): on
    Windows, text-mode writes translate `\\n` -> `\\r\\n`, which would
    silently corrupt the on-disk bytes relative to what `protected_shas` /
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


def run_oracle(task: B8Task, workspace: str | Path, *, root: str | Path = ".",
                oracle_image: str | None = None) -> tuple[bool, str]:
    """Decide whether `task` was actually completed in the post-run
    `workspace`. See the module docstring for the full precedence
    rationale; in short: protected-file tamper and out-of-bounds edits are
    HARD CAPS checked first (no Docker needed), and only then does the
    hidden behavioral oracle run -- against a copy that has `task.
    oracle_files` re-injected into it, since `workspace` itself (what the
    agent actually had) never contained them.

    The manifest's `budgets["wall_clock_s"]` is passed straight through as
    `Sandbox.hidden_validate`'s `timeout` -- agent-produced code executes
    inside the oracle container (Task 3's oracles compile/run the post-run
    workspace), so an unbounded busy/infinite loop there would otherwise
    hang indefinitely. Falls back to a sane 60s if `wall_clock_s` is ever
    absent (the loader requires it, so this only matters for a `B8Task`
    built outside `load_b8_tasks`, e.g. in a test).

    `oracle_image` (additive, default `None`): overrides the pinned
    `config/runtime_pins.yaml` sandbox image (`nvidia/cuda:...-base`, which
    has no Python -- see the module docstring's LANGUAGE NOTE) for the
    oracle container ONLY -- e.g. `python:3.11-slim` for a Python-shaped B8
    task's oracle, which needs `python3` to import/execute the agent's
    solution. `None` (the default) preserves the exact prior behavior: no
    `image`/`digest` kwargs reach `Sandbox`, which falls back to the pin as
    it always has. When set, `digest` is passed as `""` (falsy), NOT left
    as `None` -- `Sandbox.__init__` reads the pin file whenever `image is
    None or digest is None` and falls back to `pin.get("digest")` for any
    falsy digest; a bare `image=oracle_image` with no digest override would
    therefore still resolve `self.digest` to the CUDA pin's OWN digest,
    pairing an unrelated image with the wrong digest into an unpullable
    `image@digest` ref (`Sandbox._image_ref`). `digest=""` short-circuits
    both: `_image_ref` returns the bare tag (no `@digest` suffix at all),
    and the pin file is never even read for this call.
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
    #
    # LIMITATION (documented, not fixed -- matches the Task 2 precedent
    # this mirrors): a symlinked entry at a disallowed path is skipped
    # entirely, not reported as an out-of-bounds edit. This is a coverage
    # gap in the diff-constraint's REPORTING only, not a content leak: step
    # (c) below copies the workspace into the re-injection copy via
    # `copy_real_files` (real files only, symlinks never included at all),
    # and `hidden_validate`'s OWN internal copy (`copytree(...,
    # symlinks=True)`) only ever gets READ afterward (via a `:ro` container
    # mount or a read-only callable) -- nothing downstream ever writes
    # through a preserved symlink, so a symlink skipped here cannot smuggle
    # host content into, or a host write out of, either copy.
    allowed = set(task.allowed_diff_paths) | set(task.protected_shas)
    for dirpath, dirnames, filenames in os.walk(ws, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if not os.path.islink(os.path.join(dirpath, d))
                       and d not in ("__pycache__", ".pytest_cache", ".mypy_cache")]
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_symlink():
                continue
            # Transient tooling artifacts an agent's OWN test run auto-creates
            # (Python bytecode is the common one -- `python solution.py` writes
            # __pycache__/*.pyc into the workspace). These are never a meaningful
            # edit; without this exclusion the diff-constraint false-flags a
            # CORRECT solution as an "out-of-bounds edit" and reports a false
            # task failure (B8 local-run finding, 2026-07-21: it dominated
            # gpt-oss-20b's apparent failures, all "out-of-bounds edit:
            # __pycache__/*.pyc"). Note: .pyc are excluded from the OUT-OF-BOUNDS
            # walk only; protected_paths hash-checks (step a) still cover any
            # protected source file, so this doesn't widen the gaming surface.
            if name.endswith((".pyc", ".pyo")):
                continue
            rel = full.relative_to(ws).as_posix()
            if rel in task.protected_shas:
                continue
            initial = task.setup_repo.get(rel)
            current = full.read_bytes()
            changed = initial is None or current != initial.encode("utf-8")
            if changed and rel not in allowed:
                return False, f"out-of-bounds edit: {rel}"

    # (c) behavioral oracle -- re-inject the pristine, never-agent-visible
    # `oracle_files` into a private copy of the (already hard-cap-verified)
    # workspace, THEN hand that copy to Sandbox.hidden_validate, which
    # copies it again into a fresh, read-only, isolated container. The
    # agent's real workspace (`ws`) is never touched or read again after
    # this point.
    #
    # Uses `copy_real_files` (real files only -- see its docstring), NOT
    # `shutil.copytree(..., symlinks=True)`: this step subsequently does
    # `(inject_root / rel_path).write_bytes(...)` for every oracle_files
    # path, and copytree would preserve an agent-planted symlink AS a
    # symlink -- if an agent guesses an oracle_files path (e.g.
    # "oracle_test.sh", which every one of these 5 manifests uses) and
    # plants a symlink there pointing at an arbitrary writable host path,
    # that write_bytes call would follow it and write oracle content to
    # THAT host path instead of into the copy. copy_real_files never
    # produces a symlink in inject_root, so there is nothing for the
    # write to collide with.
    try:
        with tempfile.TemporaryDirectory(prefix="llmtest-b8-oracle-inject-") as tmp:
            inject_root = Path(tmp) / "ws"
            copy_real_files(ws, inject_root)
            for rel_path, content in task.oracle_files.items():
                p = inject_root / rel_path
                if p.is_symlink():
                    # Should be unreachable -- copy_real_files never creates
                    # a symlink anywhere under inject_root. Refuse loudly
                    # rather than silently write through one, in case that
                    # invariant is ever broken by a future change.
                    return False, f"oracle re-injection blocked: {rel_path} is unexpectedly a symlink"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content.encode("utf-8"))
            wall_clock_s = task.budgets.get("wall_clock_s", 60)
            sbx_kwargs = {"workspace": inject_root, "root": root}
            if oracle_image is not None:
                sbx_kwargs["image"] = oracle_image
                sbx_kwargs["digest"] = ""   # see docstring: avoids pairing
                                            # oracle_image with the CUDA pin's digest
            sbx = Sandbox(**sbx_kwargs)
            return sbx.hidden_validate(task.oracle, inject_root, timeout=wall_clock_s)
    except Exception as e:  # noqa: BLE001 - a setup failure before the oracle
        # even runs is a validation result, not a crash -- mirrors
        # hidden_validate's own copy-step error handling.
        return False, f"oracle re-injection setup failed: {e!r}"
