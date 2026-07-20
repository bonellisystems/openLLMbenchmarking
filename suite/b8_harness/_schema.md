# B8 Task Manifest Schema

## Overview

B8 manifests are YAML task definitions for the agentic-harness battery — a
versioned, immutable `(initial repo, hidden behavioral oracle)` pair a real
agent harness (Phase 2, deferred) runs an agent against inside the
disposable Docker sandbox (`llmtest/harness/sandbox.py`, Task 2). Each
manifest lives at `suite/b8_harness/task-<NN>.yaml`, one of five task
*shapes*: `edit`, `multi-file`, `bugfix`, `tool-heavy`, `from-scratch`.

Unlike B1/B6 fixtures (graded by static text signals on a model's raw
completion), B8 manifests carry their own **initial repo** inline and are
graded by an **anti-gaming completion oracle** (`llmtest.harness.tasks.
run_oracle`) that actually diffs and executes the post-run workspace —
see "Anti-gaming semantics" below.

**Language note:** manifest content is bash, not Python. The pinned sandbox
image (`nvidia/cuda:12.6.2-base-ubuntu24.04`, Task 2) has no Python/gcc/
node interpreter — only bash/sh/perl/coreutils — and the behavioral oracle
must execute inside that same container (it runs agent-produced code, per
Design Decision #3). See `llmtest/harness/tasks.py`'s module docstring.

## Two distinct anti-gaming mechanisms, both required

The global constraints list TWO SEPARATE properties a B8 task must have,
and a manifest must satisfy both:

1. **"Hidden validators live OUTSIDE the writable workspace"** — the agent
   must never be able to READ the oracle. Satisfied by `oracle_files`
   (below): those paths are never written into the agent's workspace by
   `materialize_repo`, at any point. There is nothing to read and overfit
   against (e.g. hand-crafting output that satisfies a memorized check
   without doing the real task).
2. **"Protected files hash-checked"** — a DIFFERENT property: some
   agent-VISIBLE files must not change. Satisfied by `protected_paths` /
   `protected_shas` (below): the agent CAN read these, but any edit is
   caught as a hard-cap failure before the behavioral oracle even runs.

An earlier version of this schema put the hidden test script inside
`setup_repo` (agent-visible) and only protected it via hash-check —
satisfying mechanism (2) but not (1). That is insufficient on its own: an
agent could still read the test, see exactly what it probes for, and
construct a workspace that satisfies the literal check without doing the
work (concrete example: `toolheavy-01`'s old oracle wrote a specific extra
data file and expected a specific new total; a model that read the test
could special-case exactly that file/total). The schema below is the
corrected, three-category version that closes this.

## Required Keys

- **id** (string): Unique task identifier, `<shape-word>-<NN>` (e.g.
  `edit-01`, `bugfix-01`).
- **shape** (string): One of `edit`, `multi-file`, `bugfix`, `tool-heavy`,
  `from-scratch`.
- **task_version** (string): Manifest version (e.g. `"1.1.0"`). Bump this
  when the initial repo or oracle content changes.
- **prompt** (string): The full, self-contained task prompt handed to the
  agent.
- **allowed_tools** (list of strings): Which tool categories the harness
  should expose for this task (e.g. `read_file`, `write_file`,
  `list_files`, `bash_exec`, `subagent_spawn`). Documentation for the
  Phase-2 adapter wiring; not enforced by the loader beyond "non-empty
  list."
- **budgets** (mapping): Per-run limits — `wall_clock_s`, `tokens`, `steps`
  (all required keys; Phase-2/Task-7 concern to actually enforce these).
- **setup_repo** (mapping): `relative/path -> full file content`, the
  AGENT-VISIBLE, agent-modifiable INITIAL workspace. Inline block scalars,
  same idea as `b6_fixtures`'s `buggy_code:`, just multi-file. For a
  `from-scratch` task, the deliverable file is intentionally absent here
  (it does not exist until the agent creates it). `materialize_repo` writes
  ONLY this mapping to disk — this is everything a real harness adapter
  (Phase 2) would ever hand an agent.
- **oracle_files** (mapping): `relative/path -> full file content`, the
  HIDDEN behavioral-oracle script(s) (mechanism 1 above). Same shape as
  `setup_repo`, but `materialize_repo` never touches it. `run_oracle`
  re-injects these paths into a private copy of the workspace immediately
  before running the behavioral oracle (step 3 below) — the agent's real
  workspace, before and after the run, never contains them. A manifest
  cannot list the same path in both `setup_repo` and `oracle_files` (loader
  raises `ValueError`).
- **protected_paths** (list of strings): Paths that must already exist in
  `setup_repo` (never `oracle_files` — the loader enforces
  `protected_paths ⊆ setup_repo.keys()`, which structurally prevents
  "protecting" a file instead of withholding it) whose content must remain
  BYTE-IDENTICAL after the run (mechanism 2 above) — e.g. a fixture file,
  reference data, or task-metadata file the agent isn't supposed to touch.
  Hash-checked FIRST, before the behavioral oracle ever runs (see below).
- **allowed_diff_paths** (list of strings): Paths the agent IS permitted to
  create or change. May reference paths not yet in `setup_repo` (new-file
  creation, e.g. `from-scratch`'s deliverable). Must not overlap
  `protected_paths` (loader raises `ValueError` if it does).
- **oracle** (mapping): `{type: command, argv: [...]}`. Only `type: command`
  is supported today — an argv run inside a throwaway, read-only,
  network-isolated container via `Sandbox.hidden_validate` (Task 2). By
  convention every manifest's `argv` first `cp -r /oracle /tmp/work`
  (writable tmpfs) before `cd`-ing in and running `oracle_test.sh`, since
  `/oracle` itself is mounted `:ro` and some oracles need scratch space
  (e.g. writing an extra test fixture file). `oracle_test.sh` here refers
  to a path re-injected from `oracle_files`, not `setup_repo`.

## Anti-gaming semantics (`run_oracle`)

`run_oracle(task, workspace) -> (completed: bool, detail: str)` runs, in
strict order, against the POST-RUN workspace (which only ever contained
`setup_repo` paths — never `oracle_files`):

1. **Protected-file hash check (hard cap).** Every `protected_paths` entry
   (an agent-VISIBLE file) is hashed and compared to the value computed
   from `setup_repo` at load time. Any mismatch (or missing file) ->
   `(False, "protected file tampered: <path>")` **immediately** — even if
   the behavioral oracle would otherwise pass. This is what stops a model
   from editing a fixture it isn't supposed to touch to force a pass.
   Never touches Docker.
2. **Diff constraint (hard cap).** Every file in the workspace that differs
   from `setup_repo` (or is new) and is not in `allowed_diff_paths` (nor
   `protected_paths`, already covered by step 1) -> `(False,
   "out-of-bounds edit: <path>")`. Never touches Docker.
3. **Behavioral oracle, with re-injection.** Only once (1) and (2) both
   pass: `task.oracle_files` are written into a PRIVATE copy of the
   (already-validated) workspace — the agent's own `workspace` is never
   touched — and `task.oracle` runs against THAT copy via
   `Sandbox.hidden_validate`, in a throwaway, read-only container outside
   the agent's reach. Its `(bool, detail)` is returned as-is.

Steps 1–2 need no Docker at all (pure hash/file comparisons); step 3 does.

## Example (abbreviated)

```yaml
id: bugfix-01
shape: bugfix
task_version: "1.1.0"
prompt: |
  ...
allowed_tools: [read_file, write_file, bash_exec]
budgets: {wall_clock_s: 180, tokens: 3000, steps: 8}
setup_repo:
  stats.sh: |
    #!/bin/bash
    average() { ... }   # missing closing brace -- planted bug
  NOTES.md: |
    # Task metadata
    Do not edit this file.
oracle_files:
  oracle_test.sh: |
    #!/bin/bash
    set -e
    bash -n stats.sh
    out="$(bash -c 'source stats.sh; summarize 2 4 6')"
    [ "$out" = "summary: avg=4" ] || exit 1
    echo PASS
protected_paths: [NOTES.md]
allowed_diff_paths: [stats.sh]
oracle:
  type: command
  argv: ["bash", "-c", "set -e; cp -r /oracle /tmp/work && cd /tmp/work && bash oracle_test.sh"]
```

Note `stats.sh` (the agent's editable target) and `NOTES.md` (agent-visible,
protected) both live in `setup_repo`; `oracle_test.sh` (never agent-visible)
lives in `oracle_files` and only ever reaches a real filesystem inside
`run_oracle`'s private re-injected copy.

## Loading Manifests

```python
from pathlib import Path
from llmtest.harness import tasks

all_tasks = tasks.load_b8_tasks(Path("."))
task = next(t for t in all_tasks if t.id == "bugfix-01")

ws = Path("/tmp/some-workspace")
tasks.materialize_repo(task, ws)   # writes setup_repo ONLY -- no oracle_files
# ... agent runs against ws; it never sees task.oracle_files ...
completed, detail = tasks.run_oracle(task, ws)   # re-injects oracle_files internally
```

## Validation

`load_b8_tasks` fails loud (raises `ValueError`, never silently skips) on:
missing required key, unknown `shape`, `setup_repo`/`oracle_files`/
`protected_paths`/`allowed_diff_paths`/`allowed_tools` not a non-empty
list/mapping of the right shape, a path claimed by both `setup_repo` and
`oracle_files`, a `protected_paths` entry absent from `setup_repo`, an
`allowed_diff_paths`/`protected_paths` overlap, `budgets` missing a
required sub-key, or an unsupported `oracle.type`.
