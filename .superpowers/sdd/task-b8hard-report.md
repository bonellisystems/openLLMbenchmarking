# B8-hard — 5 harder Python task manifests (task-b8hard)

## Why

The 6 existing real Python B8 tasks (`task-06..11.yaml`, ids `py-bugfix-01`/
`py-fromscratch-01`/`py-edit-01`/`py-multifile-01`/`py-toolheavy-01`/
`py-fromscratch-02`) are all solved **30/30** by gpt-oss-20b in a live run —
they don't discriminate, so B8's completion metric can't spread and the
first-failure classifier never sees a real task-logic ("c") failure to
classify. This task authors 5 genuinely harder manifests
(`task-12..16.yaml`, ids `py-hard-*`), mirrors the existing 3-category
schema exactly (`setup_repo` / `oracle_files` / `protected_paths` +
`allowed_diff_paths`), and re-targets `config/suite.yaml`'s `b8.tasks`
allowlist at them so the next live run measures something.

Target calibration: ~40-80% completion for a capable 20B model, coming from
CORRECTNESS/EDGE-CASE reasoning, not obscurity or hidden syntax errors. The
actual number can only be measured by the coordinator's live run — what
this task can and does prove is that each oracle **discriminates**: a
genuinely correct solution passes, and a specific, plausible, wrong solution
fails on a documented edge.

## The 5 tasks

### 1. `py-hard-bugfix-01` (shape `bugfix`, `suite/b8_harness/task-12.yaml`)

`search_utils.py` ships a binary search for `first_occurrence(nums, target)`
("return the index of the FIRST occurrence of `target` in a sorted,
possibly-duplicate-containing list, or -1"). The prompt gives the exact
contract and the current file (valid Python, no syntax error) but never
says what's wrong — only the contract.

**The trap:** on finding a match at `mid`, the planted code continues
searching the RIGHT half (`lo = mid + 1`) instead of the LEFT half (`hi =
mid - 1`). It keeps overwriting `result` with every further-right match
before the loop ends, so it actually converges on the **last** occurrence,
not the first — the opposite of the contract. This is invisible on any
input where `target` occurs at most once (first == last trivially) and only
diverges on inputs with **duplicate target values** — exactly what a model
that "smoke tests" its own fix against a couple of no-duplicate examples
would miss.

Oracle ground truth is computed **live** via stdlib `nums.index(target)`
(itself always the left-most match) — no hand-transcribed expected values.
20 cases including all-duplicates, duplicates at the head/middle/tail,
negative-number duplicate runs, an all-elements-equal-to-target case, and a
1000-element list with every value duplicated.

### 2. `py-hard-algo-01` (shape `from-scratch`, `task-13.yaml`)

Write `lru_cache.py`: an `LRUCache` class (`get`/`put`, fixed capacity,
least-recently-used eviction) from an exact, worked-example spec.

**The trap:** the contract explicitly states that `put()` on an
**already-present** key must ALSO refresh recency, not just `get()` — but
this is exactly the natural, wrong simplification to write (recency
tracking wired into `get()` only). It passes the standard textbook LRU
walkthrough (capacity exceeded by brand-new keys) and only fails once an
existing key is updated shortly before an eviction decision.

Oracle ground truth is computed **live** by a reference `_RefLRU` class
(stdlib `OrderedDict`) run against the same operation sequence, not
hand-computed literals — a stateful, multi-step scenario is exactly where
hand transcription is riskiest.

### 3. `py-hard-edge-01` (shape `from-scratch`, `task-14.yaml`)

Write `rankfind.py`: `second_largest_unique(nums)` — the second-largest
**distinct** value, or `None` if fewer than 2 distinct values exist.

**The traps** (the classic naive one-liners each pass a happy path and fail
a specific case):
- `sorted(nums)[-2]` (no dedup) fails `[7, 7, 7, 5]` → `7` (want `5`).
- A hand-rolled tracker seeded with sentinel `largest = second = 0` (instead
  of `None`) fails every all-negative input (nothing ever exceeds the `0`
  sentinel) AND every empty/single/all-duplicate/all-zero input (the
  sentinel is indistinguishable from "found nothing" — and `0` is also a
  legitimate data value).

21 cases: empty, single element, all-duplicate, duplicates at the top,
negatives, zero, two-distinct-heavy-duplication, ascending/descending,
`10**18`-scale integers, a 200-element list, and a large-magnitude negative
pair. Ground truth computed live via `sorted(set(nums), reverse=True)[1]`.

### 4. `py-hard-multifile-01` (shape `multi-file`, `task-15.yaml`)

Three files: `cart.py` (`Cart.subtotal()`, correct, protected),
`pricing.py` (`apply_discount`/`apply_tax`, both correct in isolation,
protected), `checkout.py` (`checkout_total`, editable). The prompt states
the business rule precisely: tax must be computed on the amount **after**
the flat discount, never before.

**The trap:** `checkout.py` calls `apply_tax` **then** `apply_discount` —
swapped. Neither `cart.py` nor `pricing.py` alone reveals anything wrong
(both are correct and internally consistent with their own docstrings); the
bug is only visible by holding the stated rule up against `checkout.py`'s
actual call order. Chosen so the bug is mathematically **order-sensitive**
(a percentage-discount version would make the two orders algebraically
identical by commutativity, hiding the bug entirely) — with a flat-dollar
discount and a percentage tax, `(subtotal - discount) * (1 + tax)` (correct)
and `(subtotal * (1 + tax)) - discount` (buggy) generally differ. The
sharpest case: a fully-discounted cart (`discount == subtotal`) gives
exactly `0.00` correctly vs. a nonzero leftover-tax charge under the bug —
qualitatively, not just numerically, different.

Oracle computes expected value **live** from the raw item list, independent
of `cart.py`/`pricing.py`, compared with float tolerance (`abs(got - want) <
0.005`) rather than brittle string equality (a rounded currency float can
print `"16.2"`, not `"16.20"`).

### 5. `py-hard-toolheavy-01` (shape `tool-heavy`, `task-16.yaml`)

Five files: `roster.py` (data), `stats.py` (`average`, trivially correct),
`curve.py` (`apply_curve`, an unusual-looking `/2`-adjustment-plus-clamp
formula that is **fully correct** — the deliberate decoy), `letter_grade.py`
(`letter_grade`, editable — the real bug), `report.py` (orchestration,
correct). The prompt states the observed symptom (a curved score of exactly
90.0 is graded B instead of A) and the exact intended scale, and explicitly
warns that "not every file that looks unusual is the source of the bug."

**The trap:** `letter_grade.py` uses `>` instead of `>=` on all four
boundaries (90/80/70/60) — its own docstring states the correct inclusive
scale while the code implements it wrong. The genuine difficulty is
locating the fix among 5 files without wasting the edit budget on the
plausible-looking `curve.py` decoy (which the prompt explicitly forbids
touching, "even if you notice something you'd otherwise want to change").

Oracle re-verifies `average`/`apply_curve` behaviorally (defense in depth on
top of their protected hash, mirroring `py-toolheavy-01`'s rationale) and
then hammers `letter_grade` at all 4 boundaries × {exact int, exact float,
just-below-float}, plus interior values so a solution can't special-case
just the 4 known boundary numbers.

## Correct-vs-flawed verification (per task)

Every oracle was run, at authoring time, directly against the **actual
YAML-loaded manifest content** (not a hand-copied draft) via a
`materialize_repo` + oracle-injection harness mirroring
`test_harness_tasks.py`'s own `_run_oracle_locally` — both hermetically
(host python, no Docker) and, for the pairs below, through the **real**
`run_oracle` → `Sandbox.hidden_validate` → `python:3.11-slim` container
path.

| Task | Correct reference | Flawed reference | Result |
|---|---|---|---|
| `py-hard-bugfix-01` | `hi = mid - 1` on match | shipped/unfixed (`lo = mid + 1` on match) | correct → `PASS`; flawed → `FAIL: first_occurrence([2, 2, 2], 2) -> '2' (want 0)` |
| `py-hard-algo-01` | `move_to_end` on both `get` and existing-key `put` | `move_to_end` only in `get` | correct → `PASS`; flawed → `FAIL: ...put(1,10)... -> ['2', '-1', '3'] (want ['-1', '10', '3'])` |
| `py-hard-edge-01` | seeded with `None` | seeded with sentinel `0` | correct → `PASS`; flawed → `FAIL: second_largest_unique([]) -> '0' (want None)` |
| `py-hard-multifile-01` | discount then tax | shipped/unfixed (tax then discount) | correct → `PASS`; flawed → `FAIL: checkout_total(items=[(10.0, 2)], discount=5.0, tax=0.08) -> 16.6 (want 16.2)` |
| `py-hard-toolheavy-01` | `>=` on all 4 boundaries | shipped/unfixed (`>` on all 4) | correct → `PASS`; flawed → `FAIL: letter_grade.letter_grade(90) -> 'B' (want 'A')` |

Each of the 5 also verified separately (both hermetically and through the
real container) against a **module-level `sys.exit(0)`-at-import** gaming
attempt on the correct reference solution — every one correctly still
fails, confirming the subprocess-isolation hardening (never `import` the
candidate into the checker's own process) carries over from the existing 6
tasks. Two additional real-container tests prove the "real protected fixture"
requirement: a correct `checkout.py` fix combined with a tampered
`pricing.py`, and a correct `letter_grade.py` fix combined with a tampered
`curve.py` decoy, are both rejected by the protected-hash hard cap before
the behavioral oracle ever runs.

## Wiring

`config/suite.yaml`'s `b8.tasks` allowlist is now:

```yaml
tasks: [py-hard-bugfix-01, py-hard-algo-01, py-hard-edge-01,
        py-hard-multifile-01, py-hard-toolheavy-01]
```

(previously the original 6 real-python ids). `replicates: 5` unchanged. The
original 6 real-Python manifests and the 5 bash placeholders are left in
place, not deleted — still returned by `load_b8_tasks`, still covered by
their own existing tests — just excluded from a live `plan()` by this
allowlist, exactly like the bash placeholders already were.

## Tests added/changed

- `tests/test_harness_tasks.py`: manifest count 11→16; `_ALL_PY_TASK_IDS`
  generic tests (`test_python_oracle_passes_for_a_genuinely_correct_solution`,
  `test_python_oracle_rejects_sys_exit_zero_at_import_solution`) now also
  parametrized over the 5 new ids (`_HARD_PY_TASK_IDS`) via 5 new
  `_CORRECT_SOLUTIONS` entries; new `_HARD_FLAWED_SOLUTIONS` map + a
  parametrized `test_hard_task_known_flawed_solution_is_rejected`; 16 new
  `@requires_docker` real-container tests (correct-passes/flawed-fails pairs
  for all 5 tasks, plus the 2 decoy/protected-tamper hard-cap tests above).
- `tests/test_b8.py`: renamed/rewrote
  `test_suite_yaml_b8_tasks_allowlist_is_the_six_real_python_task_ids` →
  `..._is_the_five_hard_python_task_ids`, asserting the new allowlist.
  (The two allowlist-override tests using `["py-bugfix-01", "py-edit-01"]`/
  `["py-toolheavy-01"]` needed no change — they explicitly override
  `cfg.suite["b8"]["tasks"]` with an arbitrary subset independent of
  whatever the real default is, and those ids still load fine.)
- `tests/test_b8_local.py`: updated a stale comment (no long claims this
  file's own `_ALL_PY_TASK_IDS` constant equals suite.yaml's allowlist,
  since it no longer does) — no behavioral/assertion changes; this file's
  own tests target the original 6 manifests directly by id, independent of
  suite.yaml's allowlist.
- `suite/b8_harness/_schema.md`: added a task-b8hard update note describing
  the 5 new manifests and the allowlist re-target, mirroring the existing
  task-b8local/task-b8expand update notes.

## RED / GREEN

- **RED** (proof the oracles discriminate, not just "some assertion always
  passes"): every one of the 5 flawed reference solutions above fails with
  a specific, informative `FAIL: ...` line naming the exact input and
  wrong-vs-expected values — this is the `det_checks.oracle.detail` text
  the first-failure classifier will read for real "task-logic" (c)
  failures.
- **GREEN**: `python -m pytest -q` → **599 passed, 1 failed** (the failure
  is the same pre-existing `test_adapters.py::
  test_file_delivery_adapter_embeds_packet_path_in_instruction_no_stdin`
  agy-CLI-full-path assertion documented as pre-existing in this
  environment before this task started — 572 passed + 1 pre-existing
  before this change, 599 passed + 1 pre-existing after; +27 new tests,
  all passing, 0 regressions). `tests/test_harness_tasks.py` alone: 70
  passed (up from 43).

## Concerns / calibration risk (flagged, not hidden)

- **A priori difficulty can't be exactly tuned to 40-80%** — that's a
  measured property of a live run against a real model, not something a
  manifest author can guarantee up front. Two of the 5 are more likely to
  land toward the easy end once actually run: `py-hard-algo-01` (LRU cache
  is a heavily-memorized interview problem) and `py-hard-toolheavy-01`
  (once `letter_grade.py` is actually read, the `>`-vs-`>=` mismatch against
  its own docstring is fairly visible — the difficulty is mostly in
  *locating* it among the decoy, not in the fix itself once found). The
  strongest genuine discriminators are expected to be `py-hard-bugfix-01`
  (the bug is invisible without a duplicate-target test case) and
  `py-hard-edge-01` (the sentinel-0/negative/all-duplicate traps are easy
  to miss even with careful review). If the live run shows the LRU/
  toolheavy tasks near 100%, that's a real signal to swap them for a
  harder variant in a future pass, not evidence the harness is broken.
- `py-hard-multifile-01`'s discriminating property (order-sensitivity)
  depends on using a FLAT discount + percentage tax, not two percentages
  (which would make the swap invisible by commutativity) — this is baked
  into the fixture correctly, but it's a fragile property worth remembering
  if this task is ever "simplified."
- The `py-hard-toolheavy-01` prompt states the observed symptom fairly
  directly ("a curved score of exactly 90.0 should be A but is currently
  B") to keep the task fair/solvable — this trades away some of the
  "needle in a haystack" difficulty py-toolheavy-01 (the original) has, in
  exchange for making the DECOY-avoidance the primary source of difficulty
  instead. This was a deliberate design choice (per advisor guidance during
  authoring) but is worth knowing if the observed pass rate is higher than
  expected.
- No changes were made to `llmtest/harness/tasks.py`, `b8_harness.py`, or
  any other harness code — this is fixture + config + test authoring only,
  per the build brief ("the coordinator drives the live run").

## Report

Full report: `.superpowers/sdd/task-b8hard-report.md` (this file).
