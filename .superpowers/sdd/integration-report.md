# P4 Battery Integration Report

**Branch:** `p4-all-batteries` (off `main` @ `999f603`) — NOT merged to main, left for review.
**Date:** 2026-07-17

## Status: SUCCESS — all 5 battery branches integrated cleanly

## Merge sequence

Stale battery worktrees (`.claude/worktrees/agent-*` for b3/b4/b6/b7) were removed
(`git worktree remove --force` + `git worktree prune`) before merging, since a
branch checked out in a worktree cannot be merged elsewhere.

```
999f603 main
  -> f1b4463 merge p4-b2-toolcalling   (clean, no conflicts)
  -> 6675bb6 merge p4-b3-hallucination (conflicts: suite.yaml, __init__.py)
  -> da028e5 merge p4-b4-longcontext   (conflicts: suite.yaml, __init__.py, validate_cmd.py)
  -> b432986 merge p4-b6-agenticcoding (conflicts: suite.yaml only; __init__.py auto-merged)
  -> 1fb8434 merge p4-b7-harnessmatrix (conflicts: suite.yaml, __init__.py)
```

## Final test count

**318 passed**, 1 pre-existing DeprecationWarning (regex flag position in a B3
fixture, not a failure), 0 errors — matches the expected 176 base + 35(B2) +
21(B3) + 24(B4) + 35(B6) + 27(B7) = 318.

`python -m llmtest validate` → exit 0, "71 rows checked, 0 errors".

## Batteries registered

All 7 IDs resolve cleanly via `llmtest.batteries.get(id)` with no collisions:

| id | class              | module                |
|----|--------------------|-----------------------| 
| 1  | B1Business         | b1_business           |
| 2  | B2ToolCalling      | b2_toolcalling         |
| 3  | B3Hallucination    | b3_hallucination      |
| 4  | B4LongContext      | b4_longcontext        |
| 5  | B5Serving          | b5_serving            |
| 6  | B6AgenticCoding    | b6_agenticcoding      |
| 7  | B7HarnessMatrix    | b7_harnessmatrix      |

`_REGISTRY` keys after registering all 7: `[1, 2, 3, 4, 5, 6, 7]`.

## Conflicts resolved (accumulation, no battery dropped)

- **`config/suite.yaml`** (conflicted in every merge after b2): `condition_vocab.cond`
  merged to the union `[PEAK, SUSTAINED32K, B1, B2, B3, B4, B6, B7, SELFTEST]`;
  `b2:`/`b3:`/`b4:`/`b6:`/`b7:` blocks all appended after `b1:`, each kept intact
  verbatim. B7's `condition_order` extension (`+ sysp, temp, toolfmt`) and its
  `condition_vocab.sysp/temp/toolfmt` additions auto-merged cleanly (no other
  branch touched those lines). `suite_version` left untouched at
  `suite-v2.0.0-shakedown`, as instructed. B4 introduced no `kv:` vocab change,
  so nothing to merge there.
- **`llmtest/batteries/__init__.py`**: all 6 lazy-import `elif` branches
  (ids 2,3,4,6,7) kept alongside id 1 and id 5; final chain is 1→2→3→4→5→6→7.
- **`llmtest/validate_cmd.py`** (conflicted only in the b4 merge): b4's branch
  refactored the B1 signal-lint loop into a shared `_lint_signal_values()`
  helper (used with `valid_types=b1_signal_types` for B1's original 3-type
  set, and with the default 4-type set — adds `not_contains` — for B4's own
  block); b3's branch had only added the module-level `import re` fix plus
  its own standalone B3 fixture-lint block using an inline loop. Git
  auto-merged the top-of-file refactor cleanly (b3 never touched those lines)
  but the two batteries' fixture-lint blocks, both appended at the same
  insertion point after B1's block, textually collided. Resolved by
  reconstructing both as separate, complete `if b3_config: / if b4_config:`
  loops (via a short Python script for byte-exact reproduction) — B3's block
  unchanged/inline as originally written, B4's block using the shared helper,
  B1's block using the shared helper as b4 refactored it. B3's `import re`
  fix is preserved (module-level, no more per-loop-iteration inline import).

## Frozen files — untouched

`git diff main..HEAD -- config/registry.yaml config/judges.yaml results/` is
empty. No writes into `results/`.

## Batteries that failed to integrate

None. All 5 branches (b2, b3, b4, b6, b7) merged successfully alongside the
existing b1/b5 on main.
