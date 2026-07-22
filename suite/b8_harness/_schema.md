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

**Language note:** the original 5 manifests (task-01..05.yaml) are bash, not
Python. The pinned sandbox image (`nvidia/cuda:12.6.2-base-ubuntu24.04`,
Task 2) has no Python/gcc/node interpreter — only bash/sh/perl/coreutils —
and the behavioral oracle must execute inside that same container (it runs
agent-produced code, per Design Decision #3). See `llmtest/harness/
tasks.py`'s module docstring.

**Update (task-b8local):** 3 real Python manifests (task-06..08.yaml,
ids `py-bugfix-01`/`py-fromscratch-01`/`py-edit-01`) were added for the
real local run. Their `oracle_files` run under `python3`, not `bash` --
`run_oracle`'s additive `oracle_image` param (threaded from suite.yaml's
`b8.sandbox.oracle_image`) overrides the pinned CUDA image with
`python:3.11-slim` (which has both `python3` AND `bash`, so it runs the
original bash oracles unchanged too) for these manifests' validation runs.
Their `oracle.argv` follows the exact same `cp -r /oracle /tmp/work && cd
/tmp/work && ...` convention as the bash manifests, just invoking
`python3 oracle_test.py` instead of `bash oracle_test.sh`. Oracle scripts
are stdlib-only (`sys.exit`, plain `assert`-shaped checks) — no pytest,
since `python:3.11-slim` doesn't have it installed.

**Update (task-b8expand):** 3 more real Python manifests were added
(task-09..11.yaml, ids `py-multifile-01`/`py-toolheavy-01`/
`py-fromscratch-02`), covering the `multi-file`/`tool-heavy` shapes in
Python for the first time plus a harder `from-scratch` task, so all 5 B8
shapes now have a Python counterpart. All 6 Python oracles (the original
3 plus these 3) were also HARDENED against a gaming vector the codex
review flagged (Important #1, "the hidden oracle can be read and
short-circuited by agent code" — the bash `source is_prime.sh; exit 0`
finding, which has an exact Python analog): the pre-hardening oracles
`import`ed the candidate solution directly into the oracle's own
process, so a solution with a module-level `sys.exit(0)` would raise
`SystemExit` (not caught by `except Exception`) and kill the whole
checker with exit code 0 before any check ran — `Sandbox.hidden_validate`
only looks at the process exit code, so this would wrongly register as a
pass. Every Python oracle now runs the candidate solution in a FRESH
`subprocess.run([sys.executable, ...])` per check and asserts on that
subprocess's stdout from the outside — a `sys.exit(0)` can only kill its
own throwaway subprocess, leaving no output to match, so the check still
correctly fails.

**Update (task-b8hard):** the 6 real Python manifests above (task-06..
11.yaml) are all solved 30/30 by gpt-oss-20b in a live run — they don't
discriminate. 5 more, HARDER Python manifests were added
(task-12..16.yaml, ids `py-hard-bugfix-01`/`py-hard-algo-01`/
`py-hard-edge-01`/`py-hard-multifile-01`/`py-hard-toolheavy-01`),
calibrated so a capable 20B model is genuinely expected to land roughly
40-80% completion on each — difficulty comes from CORRECTNESS/EDGE-CASE
reasoning, never obscurity or hidden syntax:
- `py-hard-bugfix-01` (shape `bugfix`) — a binary search whose matched-case
  branch continues right instead of left, so it silently returns the
  LAST occurrence of a duplicated target instead of the first; invisible
  on any input without duplicate target values.
- `py-hard-algo-01` (shape `from-scratch`) — an LRU cache; the trap is
  that `put()` on an ALREADY-PRESENT key must also refresh recency, not
  just `get()` (a very natural, wrong simplification only shows up right
  before an eviction decision).
- `py-hard-edge-01` (shape `from-scratch`) — second-largest-distinct-value;
  traps the classic `sorted(nums)[-2]` (no dedup) and sentinel-`0`
  (instead of `None`/`-inf`) naive approaches, both of which fail on
  duplicates-at-the-top and all-negative/all-duplicate/empty input
  respectively.
- `py-hard-multifile-01` (shape `multi-file`) — a checkout system spanning
  `cart.py`/`pricing.py` (both individually correct) and `checkout.py`
  (which applies tax before the discount instead of after) — the bug is
  an order-of-operations/interaction bug invisible from any single file.
- `py-hard-toolheavy-01` (shape `tool-heavy`) — a 5-file grade reporter
  where `curve.py` is an unusual-looking but fully CORRECT decoy, and the
  real bug (a uniform `>` vs `>=` boundary bug) is in `letter_grade.py`.

All 5 follow the exact same three-category schema and Python-oracle
subprocess-isolation convention as task-06..11.yaml (each was verified,
at authoring time, to pass a known-correct reference solution and fail a
known-flawed one on the specific documented edge). `config/suite.yaml`'s
`b8.tasks` allowlist (additive) now targets these 5 harder ids instead of
the original 6 — both the original 6 real Python manifests and the 5
bash placeholders (task-01..05.yaml) are left in place, not deleted, and
still load via `load_b8_tasks`, just excluded from a live `plan()` by
this allowlist.

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

`run_oracle(task, workspace) -> OracleResult` runs, in strict order, against
the POST-RUN workspace (which only ever contained `setup_repo` paths — never
`oracle_files`). `OracleResult` (Wave 3a) unpacks exactly like the original
`(completed: bool, detail: str)` 2-tuple via `__iter__` — every example below
that writes `(False, "...")` is that 2-tuple view; see "Machine-readable
oracle result (Wave 3a)" below for its additive structured fields.

1. **Protected-file hash check (hard cap).** Every `protected_paths` entry
   (an agent-VISIBLE file) is hashed and compared to the value computed
   from `setup_repo` at load time. Any mismatch (or missing file) ->
   `(False, "protected file tampered: <path>")` **immediately** — even if
   the behavioral oracle would otherwise pass. This is what stops a model
   from editing a fixture it isn't supposed to touch to force a pass.
   Never touches Docker.
2. **Tamper detection (hard cap, Wave 3a).** Three checks over
   `task.setup_repo`, each catching a way to dodge the diff-constraint
   below rather than satisfy it:
   - **Deletion.** Any `setup_repo` path that is MISSING from the
     workspace and not in `allowed_diff_paths` -> `(False,
     "protected/setup file deleted: <path>")`. The diff-constraint walk
     only ever inspects files that still exist, so an outright deletion
     was otherwise invisible to it.
   - **Type/mode change.** Any `setup_repo` path silently replaced by a
     DIRECTORY -> `(False, "setup file replaced with a directory:
     <path>")`. An *empty* replacement directory in particular produces
     neither a content diff nor a deletion hit, so it needs its own check.
   - **Disallowed symlink.** A symlink anywhere in the workspace at a path
     not in `allowed_diff_paths`/`protected_paths` -> `(False,
     "disallowed symlink: <path>")` — for both symlinked FILES and
     symlinked DIRECTORIES (a symlinked directory is still never
     traversed/descended into on the host, matching `Sandbox.
     snapshot_workspace`'s own symlink-safety precedent; it is now
     *reported*, not silently pruned). A symlink at an already-allowed
     path is left alone — no manifest opts a path into "symlinks
     permitted" today, so this is simply "not on the disallow list", not
     an explicit allow mechanism.
   Never touches Docker.
3. **Diff constraint (hard cap).** Every remaining file in the workspace
   that differs from `setup_repo` (or is new) and is not in
   `allowed_diff_paths` (nor `protected_paths`, already covered by step 1)
   -> `(False, "out-of-bounds edit: <path>")`. Never touches Docker.
4. **Behavioral oracle, with re-injection.** Only once (1)–(3) all pass:
   `task.oracle_files` are written into a PRIVATE copy of the
   (already-validated) workspace — the agent's own `workspace` is never
   touched — and `task.oracle` runs against THAT copy via
   `Sandbox.hidden_validate`, in a throwaway, read-only container outside
   the agent's reach. `run_oracle` then scans the oracle's own stdout for
   the machine-readable JSON result line (below); when present, it
   populates `OracleResult`'s structured fields and DERIVES `detail` from
   them (not from `hidden_validate`'s raw `exit=... stdout=...
   stderr=...` dump). When absent (an older/broken oracle — every one of
   the 5 bash placeholder manifests, task-01..05.yaml, included), `detail`
   falls back to `hidden_validate`'s own value, unchanged.

Steps 1–3 need no Docker at all (pure hash/file/type comparisons); step 4
does.

## Machine-readable oracle result (Wave 3a)

Codex review (B8 validity program): "Make oracle output machine-readable:
stage, reason code, failing case, expected, actual, exit status. Avoid
free-form stderr as the authoritative explanation." Every real Python
oracle (`task-06..16.yaml`, ids `py-*`/`py-hard-*`) now prints, in addition
to its original human-readable `PASS`/`FAIL: ...` line (unchanged, kept for
local debugging), exactly ONE JSON object as the LAST line of stdout:

```json
{"pass": false, "stage": "behavior", "reason_code": "wrong_output",
 "case": "letter_grade(90)", "expected": "A", "actual": "B",
 "exit_status": 1}
```

`run_oracle` (via `_parse_oracle_json_result`) scans stdout from the LAST
line backward for the first line that parses as a JSON object with a
boolean `pass` key — tolerating a trailing blank line or stray output after
it, and never mistaking an unrelated JSON-shaped debug line for the real
result (it must carry a boolean `pass`). When found, the structured fields
land on a row's `det_checks.oracle` as ADDITIVE keys alongside the original
`pass`/`detail` — the shape stays fully backward compatible with the Wave-1b
first-failure classifier (`llmtest.harness.failure_class`), which only ever
reads `det_checks.oracle.detail`.

**Fields:**

| Field         | Type          | Meaning                                                                 |
|---------------|---------------|--------------------------------------------------------------------------|
| `pass`        | bool          | Required. The oracle's own verdict for this run.                        |
| `stage`       | str \| null   | Which pipeline stage the result reflects (closed vocabulary below).     |
| `reason_code` | str \| null   | Why it failed (closed vocabulary below); `null` on a pass.              |
| `case`        | str \| null   | Human-readable description of the specific failing (or last) check.     |
| `expected`    | str \| null   | The expected value/output for `case`.                                  |
| `actual`      | str \| null   | The candidate's actual value/output for `case` — see BOUNDING below.    |
| `exit_status` | int           | The oracle process's own exit code (0 pass / 1 fail, by convention).    |

**Stage vocabulary (closed):** `compile` (the candidate failed to parse —
`SyntaxError`/`IndentationError`), `import` (failed to import —
`ModuleNotFoundError`/`ImportError`), `behavior` (ran fine; the check
itself passed or failed on ordinary/edge input alike — this suite does not
distinguish "edge" from "behavior" as a separate stage; see the note
below). `edge` is a RESERVED, not-yet-emitted stage value: a future
manifest's oracle is free to tag a mismatch on a specifically-designated
edge-case input as `stage: "edge"` instead of `"behavior"` if that
distinction is worth making for a given task — none of the 11 real Python
oracles do this today (deciding which of many cases counts as "the edge
one" would be an editorial judgment call per task, not a mechanical
translation of the existing FAIL text), but `run_oracle`'s parser accepts
any string here, not just this closed list.

**Reason-code vocabulary (closed, for the 4 the 11 oracles emit today):**
`syntax_error` (stage `compile`), `import_error` (stage `import`),
`runtime_error` (stage `behavior` — the candidate raised an uncaught
exception from inside an otherwise-normal call, e.g. an `IndexError`),
`wrong_output` (stage `behavior` — the candidate ran fine and printed
something, but it didn't match), `non_numeric_output` (stage `behavior`,
`py-hard-multifile-01`/task-15.yaml only — the candidate's output couldn't
even be parsed as a number where one was expected). Like `stage`, this is
a documented AUTHORING convention, not a runtime-enforced enum:
`run_oracle` passes through whatever string an oracle emits, so a Wave-4
task is free to introduce a new, more specific reason code without a
`llmtest/harness/tasks.py` change — keep any new code small, closed, and
documented here (or in the manifest's own `notes:`) rather than
proliferating one-off strings per task.

**`case`/`expected` vs `actual` — trust boundary.** `case` and `expected`
are always ORACLE-AUTHORED text (fixed at manifest-authoring time, drawn
from the oracle's own `CASES` list) — never candidate-controlled. `actual`
is the ONE field that can embed the CANDIDATE's own output (whatever the
solution under test printed). `run_oracle` truncates it to
`_ACTUAL_TRUNCATE_CHARS` (200) characters, with a trailing `...<TRUNCATED n
more chars>` marker, BEFORE deriving `detail` from it — so neither an
oversized nor a hostile-content `actual` can reach the Wave-1b classifier's
TRUSTED "Oracle rejection detail" evidence section unbounded. This
truncation happens in `run_oracle` itself (the trusted layer that builds
`det_checks.oracle`), not merely trusted to each oracle_test.py at the
source — one choke point that can't be forgotten per-task as Wave 4 adds
~40-50 more manifests.

**Fallback (malformed/absent JSON).** An oracle that prints no parseable
JSON line at all — every one of the 5 bash placeholder manifests
(task-01..05.yaml), which predate this convention, or any future broken
oracle — never crashes `run_oracle`: `detail` falls back to `hidden_
validate`'s original free-form `exit=... stdout=... stderr=...` string,
unchanged from pre-Wave-3a, and every structured field (`stage`,
`reason_code`, `case`, `expected`, `actual`) stays `None`/absent from
`det_checks.oracle`.

**Authoring pattern for a new oracle (Wave 4).** Every one of `task-
06..16.yaml`'s `oracle_test.py` shares the same small, stdlib-only,
duplicated-per-file helper pair (never imported from a shared module — the
oracle runs in an isolated `/oracle`-mounted container with nothing else
available):

```python
def _classify_crash(stderr):
    """(stage, reason_code, actual) for a candidate subprocess that
    exited nonzero before producing usable stdout."""
    ...

def _emit(passed, *, case=None, expected=None, actual=None,
          stage="behavior", reason_code=None):
    """Prints the human PASS/FAIL line, THEN the JSON result line, THEN
    sys.exit()s with the matching status."""
    ...
```

Each case loop calls `_run(...)` (a FRESH `subprocess.run([sys.executable,
...])`, per the SUBPROCESS-ISOLATED convention every Python oracle already
follows — never `import`ing the candidate into the checker's own process),
checks `r.returncode != 0` first (classify via `_classify_crash` and
`_emit` a `compile`/`import`/`behavior` failure), then compares
`r.stdout.strip()` against the expected value (`_emit` a `wrong_output`
failure on mismatch). A genuinely correct solution over every case reaches
`_emit(True)` at the end.

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

## `check_fixtures` (Wave 3b, B8 validity program) -- OPTIONAL, test-only

Codex review (B8 validity program, Wave 3b): "give every task a reference
solution, at least one alternate valid solution, and several plausible
incorrect or shortcut patches" -- this is the mechanism that PROVES a
task's oracle actually discriminates (rejects wrong/shortcut solutions,
accepts genuinely correct ones), rather than merely trusting that it does.
`check_fixtures` is an OPTIONAL top-level manifest key, structurally
identical in spirit to `setup_repo`/`oracle_files` (path -> content
mappings) but semantically completely different: **nothing in
`materialize_repo`, `run_oracle`, or `B8Harness` ever reads it.** It exists
purely so `tests/test_task_discrimination.py` can find a task's
proof-of-discrimination fixtures living alongside the oracle they exercise,
instead of hand-duplicating solutions in a test file far from the manifest
they target (the pre-Wave-3b pattern in `tests/test_harness_tasks.py`'s
`_CORRECT_SOLUTIONS`/`_HARD_FLAWED_SOLUTIONS` dicts).

**Shape:**

```yaml
check_fixtures:
  reference:            # REQUIRED when check_fixtures is present at all
    <path>: |
      <a genuinely correct solution -- the oracle must PASS this>
  alternate:             # OPTIONAL -- a second, DIFFERENT correct solution
    <path>: |
      <another genuinely correct solution, ideally a different approach>
  wrong:                 # REQUIRED, 1-3 entries
    - <path>: |
        <a plausible-but-flawed solution -- the oracle must FAIL this>
    - <path>: |
        <a second plausible-but-flawed solution>
```

Each of `reference`/`alternate`/one `wrong` entry is a mapping of
`relative/path -> full file content`, exactly like `setup_repo` -- but only
the file(s) that solution actually needs to WRITE, not the whole repo. A
discrimination test overlays these paths on top of an already-materialized
`setup_repo` workspace (so an omitted file in a `wrong` entry deliberately
LEAVES that file at its original `setup_repo` content -- itself sometimes
the most direct "wrong" fixture, e.g. "the agent fixed one of two files it
needed to fix").

**Rules a `check_fixtures` author must follow** (not mechanically enforced
by the loader beyond basic shape validation -- these are the properties
that make the discrimination test meaningful):

- Every solution (`reference`, `alternate`, every `wrong` entry) must only
  ever write paths in the task's own `allowed_diff_paths` -- never a
  `protected_paths` file. A fixture that touches a protected file gets
  rejected by the HARD-CAP tamper/hash check (`run_oracle` steps (a)/(b)),
  never even reaching the behavioral oracle -- that would prove the hard
  cap works, not that the oracle discriminates, and defeats the point of
  this convention.
- `reference` (and `alternate`, when present) must be a solution a human
  actually believes is CORRECT per the task's own prompt/contract -- verify
  it against the real oracle (`run_oracle`, or the faster hermetic
  host-subprocess path `tests/test_harness_tasks.py`'s
  `_run_oracle_locally` uses) before committing it, not merely written to
  "look right."
- Each `wrong` entry should be PLAUSIBLE -- a mistake a real agent might
  actually make (an off-by-one, a missed edge case, a copy-paste bug, an
  unfixed original bug) -- not an absurd/deliberately-nonsensical input;
  the discrimination proof is only meaningful against realistic failure
  modes.

**Excluded from row identity.** `check_fixtures` is deliberately EXCLUDED
from `fixture_sha` (see `llmtest.harness.tasks._strip_check_fixtures_for_
hash`) and never touches `setup_repo_sha` at all (that hash only ever
covers `setup_repo`). Adding a `check_fixtures` block to an existing
manifest, or later editing/fixing an existing one (e.g. because this
wave's own build found a fixture that didn't discriminate -- see below),
must never look like a fixture/oracle CONTENT change to row identity the
way an actual `setup_repo`/`oracle_files`/`oracle` edit legitimately does
(that case DOES bump `fixture_sha`/`task_version`, task-w3a precedent) --
`check_fixtures` is test scaffolding, not scored task content.

**Convention for Wave 4** (~40-50 new task manifests): every new manifest
should ship its own `check_fixtures` block from the start, verified against
the real oracle at authoring time exactly like the 11 in task-06..16.yaml
were for Wave 3b -- see `tests/test_task_discrimination.py`'s parametrized
test for the exact shape a new task's fixtures must satisfy (reference
[+alternate] PASS, every `wrong` entry FAILS).
