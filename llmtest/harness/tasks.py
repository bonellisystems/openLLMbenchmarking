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
  (b) tamper detection (Wave 3a, B8 measurement-validity -- codex review:
      "detect deletions, symlinks, special files, mode changes, and all
      out-of-scope modifications"): a `setup_repo` path DELETED (missing
      from the workspace) or replaced with a DIRECTORY -> `(False, ...)`;
      a symlink anywhere in the workspace at a disallowed path -> `(False,
      "disallowed symlink: ...")`, REPORTED rather than silently pruned
      (the diff constraint below only ever walked files that still
      EXIST -- none of these three were visible to it at all pre-Wave-3a).
  (c) diff constraint: any remaining file that differs from `setup_repo`'s
      initial content (or is new) and is not in `allowed_diff_paths` (nor
      `protected_shas`, already covered by (a)) -> `(False, ...)`.
  (d) only once (a)-(c) all pass: `oracle_files` are re-injected into
      a private copy of the (already-validated) workspace, and the
      behavioral oracle runs against THAT copy via
      `Sandbox.hidden_validate(task.oracle, ...)` -- a FRESH, read-only
      copy in a throwaway container outside the agent's reach, that now
      also contains the oracle the agent never got to see. `run_oracle`
      then parses the oracle's own machine-readable JSON result line (see
      `OracleResult`/`_parse_oracle_json_result`/`suite/b8_harness/
      _schema.md`'s Wave 3a section) into structured `stage`/`reason_code`/
      `case`/`expected`/`actual` fields, falling back gracefully (no
      crash) to the original free-form detail when no such line parses
      (every one of the 5 bash placeholder manifests, task-01..05.yaml,
      included -- they predate this convention).
Steps (a)-(c) never construct a `Sandbox`, so they need no Docker at all;
only (d) does (guarded by `@requires_docker` in the test). `run_oracle`
returns an `OracleResult` (Wave 3a), an iterable-dataclass that unpacks
exactly like the legacy `(completed: bool, detail: str)` 2-tuple shown
throughout this docstring -- see that dataclass's own docstring.
"""
from __future__ import annotations

import hashlib
import json
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


@dataclass
class OracleResult:
    """`run_oracle`'s return value (Wave 3a, B8 measurement-validity:
    "make the completion scorer DEFENSIBLE"). Iterable-dataclass, not a
    plain tuple: `__iter__` yields exactly `(pass_, detail)`, so every
    existing `completed, detail = run_oracle(...)` call site (this
    module's docstrings, `suite/b8_harness/_schema.md`, every test in
    tests/test_harness_tasks.py, `llmtest.batteries.b8_harness.execute()`,
    and every injected `ctx.b8_run_oracle` test double that returns a bare
    `(bool, str)` 2-tuple instead of this type) keeps working completely
    unchanged -- tuple-unpacking (`a, b = x`) works against ANY 2-item
    iterable, not just an actual `tuple`.

    `stage`/`reason_code`/`case`/`expected`/`actual` (additive, all
    default `None`) are populated ONLY when step (c), the behavioral
    oracle, both ran AND printed the B8 oracle's machine-readable JSON
    result line (see `_parse_oracle_json_result` and `_schema.md`'s Wave
    3a convention). The hard-cap failures (protected-file tamper,
    deletion/type-change/disallowed-symlink tamper, out-of-bounds edit --
    steps (a)/(b)) never populate them; their `detail` string alone
    remains the full explanation for those paths, byte-for-byte the same
    shape as pre-Wave-3a. `llmtest.batteries.b8_harness.execute()` reads
    these via `getattr(result, "stage", None)` etc, not direct attribute
    access, so an injected 2-tuple test double -- which has none of these
    attributes -- degrades to "no structured fields" rather than raising.
    `actual` is bounded to `_ACTUAL_TRUNCATE_CHARS` (it may echo a
    candidate solution's own printed output) so it can never become an
    oversized/injectable blob in the Wave-1b classifier's TRUSTED
    evidence section (`render_blinded_trace`'s "Oracle rejection detail")."""
    pass_: bool
    detail: str
    stage: str | None = None
    reason_code: str | None = None
    case: str | None = None
    expected: str | None = None
    actual: str | None = None

    def __iter__(self):
        return iter((self.pass_, self.detail))


# Hard cap on OracleResult.actual (Wave 3a) -- see OracleResult's own
# docstring. Applied here, in run_oracle (the TRUSTED layer that builds
# det_checks.oracle), rather than trusted to every individual oracle_test.py
# authored across the suite (11 today, ~40-50 more incoming per Wave 4) --
# one choke point that can never be forgotten at the source.
_ACTUAL_TRUNCATE_CHARS = 200


def _truncate(text: str, limit: int = _ACTUAL_TRUNCATE_CHARS) -> str:
    """Hard-truncate `text` to `limit` characters with a trailing
    `...<TRUNCATED n more chars>` marker when it doesn't fit -- mirrors
    `llmtest.harness.failure_class._truncate_repr`'s shape (same bounding
    idea, applied to a plain string here rather than a `repr()`)."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<TRUNCATED {len(text) - limit} more chars>"


def _parse_oracle_json_result(stdout: str | None) -> dict | None:
    """Scan `stdout` for the B8 oracle's machine-readable JSON result line
    (Wave 3a convention -- see `suite/b8_harness/_schema.md`), from the
    LAST line backward, and return the first line that parses as a JSON
    object carrying a boolean `pass` key. Scanning backward (rather than
    assuming the literal last line) tolerates a trailing blank line or any
    stray output emitted after the result line; requiring a boolean `pass`
    key (not just "any JSON object") avoids mistaking a coincidental
    JSON-shaped line elsewhere in an oracle's own debug output for the
    real result.

    Returns `None` -- never raises -- for `stdout=None` (no process stdout
    captured at all, e.g. a callable-oracle or a timed-out run) or when no
    line satisfies this: exactly the "older/broken oracle" case
    `_oracle_result_from_validation` falls back on gracefully rather than
    crash (every one of the 5 bash placeholder manifests, task-01..
    05.yaml, included -- they predate this convention and are left as-is)."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
            return obj
    return None


def _oracle_result_from_validation(vr) -> OracleResult:
    """Build `run_oracle` step (c)'s final `OracleResult` from `Sandbox.
    hidden_validate`'s `ValidateResult` (`vr`). Parses `vr.stdout` for the
    B8 oracle's JSON result line (`_parse_oracle_json_result`) and, when
    found, derives a clean, BOUNDED `detail` string from the STRUCTURED
    fields themselves -- never from raw stdout/stderr (the "avoid
    free-form stderr as the authoritative explanation" fix this wave
    targets) -- rather than `vr.detail`'s much noisier `exit=... stdout=...
    stderr=...` dump. Falls back to `vr.detail` UNCHANGED (pre-Wave-3a
    behavior, still fully backward compatible with the Wave-1b classifier,
    which only ever reads `det_checks.oracle.detail`) whenever no
    parseable JSON line is present -- an older/broken oracle or a genuine
    parse failure; this is the "don't crash on malformed/absent JSON"
    guarantee."""
    parsed = _parse_oracle_json_result(vr.stdout)
    if parsed is None:
        return OracleResult(pass_=vr.ok, detail=vr.detail)

    passed = bool(parsed["pass"])
    stage = parsed.get("stage")
    stage = stage if isinstance(stage, str) else None
    reason_code = parsed.get("reason_code")
    reason_code = reason_code if isinstance(reason_code, str) else None
    case = parsed.get("case")
    case = str(case) if case is not None else None
    expected = parsed.get("expected")
    expected = str(expected) if expected is not None else None
    actual = parsed.get("actual")
    actual = _truncate(str(actual)) if actual is not None else None

    if passed:
        detail = "PASS"
    elif case is not None or expected is not None or actual is not None:
        detail = f"FAIL: {case} -> {actual!r} (want {expected!r})"
    else:
        # A FAIL result with no case/expected/actual at all (e.g. a
        # compile/import-stage oracle that only ever reports
        # stage/reason_code) -- a reason-code-based summary beats a blank
        # "FAIL:  -> None (want None)".
        detail = f"FAIL: {reason_code or 'oracle reported failure'}"

    return OracleResult(pass_=passed, detail=detail, stage=stage,
                        reason_code=reason_code, case=case, expected=expected,
                        actual=actual)


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
                oracle_image: str | None = None) -> OracleResult:
    """Decide whether `task` was actually completed in the post-run
    `workspace`. See the module docstring for the full precedence
    rationale; in short: protected-file tamper and out-of-bounds edits are
    HARD CAPS checked first (no Docker needed), and only then does the
    hidden behavioral oracle run -- against a copy that has `task.
    oracle_files` re-injected into it, since `workspace` itself (what the
    agent actually had) never contained them.

    Returns an `OracleResult` (Wave 3a), not a plain tuple -- it unpacks
    exactly like the legacy `(completed, detail)` 2-tuple via `__iter__`
    (see that dataclass's own docstring), so this is a behavior-preserving
    change for every existing caller.

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
            return OracleResult(False, f"protected file tampered: {rel_path} (missing)")
        actual_sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual_sha != task.protected_shas[rel_path]:
            return OracleResult(False, f"protected file tampered: {rel_path}")

    # (b-i) Deletion detection (Wave 3a, codex review: "detect deletions...
    # and all out-of-scope modifications" -- an agent must not delete a
    # fixture to dodge a check). The os.walk-based diff-constraint below
    # only ever walks files that still EXIST in `ws`; it has no way to
    # notice one that's simply gone. `allowed_diff_paths` members are
    # exempt (an agent may legitimately leave one absent, e.g. never
    # having written a from-scratch deliverable at all -- that is scored
    # as a behavioral failure downstream, not a tamper attempt).
    # `protected_paths` deletions are already caught above in step (a)
    # ("... (missing)") and can never actually reach here for a protected
    # path, but this check is written generally (any non-allowed-diff
    # setup_repo path, not "unprotected only") so it also covers a THIRD
    # category a manifest is free to define: an agent-visible file that is
    # neither hash-protected nor diff-allowed, which must simply remain
    # present.
    allowed_diff_set = set(task.allowed_diff_paths)
    for rel_path in sorted(task.setup_repo):
        if rel_path in allowed_diff_set:
            continue
        if not (ws / rel_path).exists():
            return OracleResult(False, f"protected/setup file deleted: {rel_path}")

    # (b-ii) Type/mode change (Wave 3a, optional-but-cheap per the codex
    # review): a setup_repo path silently replaced by a DIRECTORY. An
    # EMPTY replacement directory in particular produces neither a
    # file-content diff (the walk below finds no files under it to flag)
    # nor a deletion-check hit above (the path still `.exists()`), so it
    # would otherwise slip through both checks undetected. Checked for
    # every setup_repo path (not just non-allowed-diff ones): replacing an
    # editable file with a directory is never a legitimate "diff" of that
    # file's content, regardless of which category the path belongs to.
    # A path replaced by a SYMLINK is instead caught by (b-iii) below (the
    # os.walk symlink check), which fires first for any symlinked entry.
    for rel_path in sorted(task.setup_repo):
        candidate = ws / rel_path
        if candidate.is_symlink():
            continue
        if candidate.is_dir():
            return OracleResult(False, f"setup file replaced with a directory: {rel_path}")

    # (b-iii) diff constraint + disallowed-symlink detection -- only
    # allowed_diff_paths (or protected, already verified unchanged above)
    # may differ from the initial repo. Walked the same way Sandbox.
    # snapshot_workspace walks a workspace (Task 2 precedent):
    # os.walk(followlinks=False) -- this reads the agent-controlled
    # workspace on the HOST, so a planted symlink (a file pointing at an
    # arbitrary host path, or a directory pointing outside the workspace)
    # must never be FOLLOWED or TRAVERSED here -- Path.rglob has no way in
    # Python 3.10 to stop descending into a symlinked directory, which is
    # exactly why snapshot_workspace does not use it either. A symlinked
    # subdirectory is still pruned BEFORE os.walk descends into it (never
    # traversed on the host, unchanged from before), but now REPORTED
    # (Wave 3a fix, codex review: "detect ... symlinks ... and all
    # out-of-scope modifications") rather than silently vanishing --
    # pre-Wave-3a this was a documented, not-fixed coverage gap in the
    # diff-constraint's REPORTING (never a content-leak risk: neither this
    # walk nor step (c)'s re-injection copy, via `copy_real_files`, ever
    # follows or preserves a symlink). A symlink at a path already in
    # `allowed` (an `allowed_diff_paths`/protected path) is left alone --
    # no manifest today opts a path INTO symlink-permitted status
    # explicitly, so this is the same "allowed path, no further diff-
    # shape restriction" treatment every other allowed-diff edit already
    # gets.
    allowed = set(task.allowed_diff_paths) | set(task.protected_shas)
    for dirpath, dirnames, filenames in os.walk(ws, followlinks=False):
        real_dirnames = []
        for d in dirnames:
            dpath = Path(dirpath) / d
            if os.path.islink(dpath):
                if d not in ("__pycache__", ".pytest_cache", ".mypy_cache"):
                    rel = dpath.relative_to(ws).as_posix()
                    if rel not in allowed:
                        return OracleResult(False, f"disallowed symlink: {rel}")
                continue  # pruned either way -- never descended into
            if d not in ("__pycache__", ".pytest_cache", ".mypy_cache"):
                real_dirnames.append(d)
        dirnames[:] = real_dirnames

        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(ws).as_posix()
            if full.is_symlink():
                # Transient bytecode can theoretically be a symlink on an
                # exotic setup; excluded the same way a REAL .pyc is below.
                if name.endswith((".pyc", ".pyo")):
                    continue
                if rel not in allowed:
                    return OracleResult(False, f"disallowed symlink: {rel}")
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
            if rel in task.protected_shas:
                continue
            initial = task.setup_repo.get(rel)
            current = full.read_bytes()
            changed = initial is None or current != initial.encode("utf-8")
            if changed and rel not in allowed:
                return OracleResult(False, f"out-of-bounds edit: {rel}")

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
                    return OracleResult(
                        False, f"oracle re-injection blocked: {rel_path} is unexpectedly a symlink")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content.encode("utf-8"))
            wall_clock_s = task.budgets.get("wall_clock_s", 60)
            sbx_kwargs = {"workspace": inject_root, "root": root}
            if oracle_image is not None:
                sbx_kwargs["image"] = oracle_image
                sbx_kwargs["digest"] = ""   # see docstring: avoids pairing
                                            # oracle_image with the CUDA pin's digest
            sbx = Sandbox(**sbx_kwargs)
            vr = sbx.hidden_validate(task.oracle, inject_root, timeout=wall_clock_s)
            # Wave 3a: parse the oracle's own machine-readable JSON result
            # line (if any) out of `vr.stdout` into structured det_checks
            # fields, deriving a clean `detail` from THEM rather than
            # `vr`'s raw `exit=... stdout=... stderr=...` dump; falls back
            # to `vr` unchanged (pre-Wave-3a shape) when no JSON line
            # parses (e.g. the 5 bash placeholder manifests, which predate
            # this convention).
            return _oracle_result_from_validation(vr)
    except Exception as e:  # noqa: BLE001 - a setup failure before the oracle
        # even runs is a validation result, not a crash -- mirrors
        # hidden_validate's own copy-step error handling.
        return OracleResult(False, f"oracle re-injection setup failed: {e!r}")
