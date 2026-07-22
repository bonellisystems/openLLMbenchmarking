# W4B1 — B8 task-breadth expansion, Batch 1 (root-cause-localization + cross-module-contract)

## Status
Complete. 12 new B8 task manifests authored (`suite/b8_harness/task-17.yaml`
.. `task-28.yaml`), each with a verified `check_fixtures` block (reference
+ alternate + 1-3 wrong solutions). All 12 verified against the REAL
`run_oracle` path (Docker, `python:3.11-slim`) via
`pytest tests/test_task_discrimination.py -k "py-brk or py-xmod"`:
**54/54 passed** (12 reference PASS, 12 alternate PASS, 30 wrong-fixture
REJECTIONS, all rejected behaviorally — no hard-cap tamper short-circuits).
`config/suite.yaml`, `suite/b8_harness/_schema.md`, and
`tests/test_task_discrimination.py` were left untouched, per instructions
(coordinator-owned integration).

## Commit
`feat(b8): W4 batch 1 — root-cause-localization + cross-module-contract tasks (task-17..28)`
(see `git log` on `main` for the SHA — commit made after this report was
written, not pushed).

## Tasks authored

### Distant root-cause localization (`py-brk-*`, task-17..22)
| id | file | trap / difficulty |
|---|---|---|
| py-brk-01 | task-17.yaml | `strnorm.collapse_spaces` (leaf) silently fails its own "collapse ANY whitespace" contract (single-pass space replace, never touches tabs) → corrupts `tokenizer.tokenize` → wrong counts in `wordcount.count_words` (top), 3 files deep. Decoy: tokenizer/wordcount look like the "real" splitting/counting logic but are correct. |
| py-brk-02 | task-18.yaml | Digit-transposed constant `CM_PER_INCH = 2.45` (should be `2.54`) in `constants.py`, consumed 3 hops through `convert.py → garment.py → catalog.py`. Unused decoy constant `FEET_PER_YARD` in the same file. |
| py-brk-03 | task-19.yaml | `sensor.calibrate_reading` scales-then-clips internally instead of clip-then-scale (bug lives entirely inside one leaf function, not in how any orchestrator calls it) → `batch.py`/`report.py` (both clean) just average/format the wrong leaf output. |
| py-brk-04 | task-20.yaml | Classic mutable-default-argument aliasing: `basket.make_basket(initial_items=[])` shares one list across every order → later customers in the same `report.build_report` batch inherit earlier customers' items. Decoy: `orders.py` has its own (correctly-handled) `notes=None` default-arg pattern to bait pattern-matching. |
| py-brk-05 | task-21.yaml | `quote_cache.memoize_quote`'s cache key is `customer_id` only (drops `plan`) → a customer's 2nd quote for a different plan returns the 1st plan's cached price, 2 hops from `billing.py`. Decoy: `pricing.py`'s actual pricing formula is correct. |
| py-brk-06 | task-22.yaml | `scores.sorted_scores` sorts by `str(score)` (lexicographic, not numeric) with no alphabetical tie-break → wrong leaderboard order, 2 hops through `leaderboard.py`/`display.py` (both correct, no sort logic of their own). |

### Cross-module contracts / backward-compat (`py-xmod-*`, task-23..28)
| id | file | trap / difficulty |
|---|---|---|
| py-xmod-01 | task-23.yaml | Serializer/deserializer round-trip: `encoder.py` (editable) must match the escaping convention `decoder.py` (protected, correct) defines on its own side; `store.py` is the protected existing caller exercised. |
| py-xmod-02 | task-24.yaml | Producer/consumer key-name contract: `events.make_event` emits `"val"`, `aggregator.py` (protected) expects `"value"` → `KeyError` in the protected `pipeline.py` caller. |
| py-xmod-03 | task-25.yaml | Backward-compatible feature extension: add optional `bulk_qty` param to `discount.apply_member_discount` without breaking `legacy_caller.py` (protected, calls with exactly 2 positional args). |
| py-xmod-04 | task-26.yaml | Dual-schema (old/new) normalization: `normalize_profile` must satisfy TWO protected callers (`legacy_importer.py` old-schema, `current_importer.py` new-schema) plus extra-key stripping and empty-string-vs-missing precedence. |
| py-xmod-05 | task-27.yaml | Shared duck-typed backend interface: `ListBackend` (buggy, "first match wins" on repeated keys) must match `DictBackend` (correct) exactly, both driven polymorphically through the protected `repository.Repository`; oracle re-verifies `DictBackend` too so "fixing while breaking the decoy" is caught. |
| py-xmod-06 | task-28.yaml | Exception-TYPE contract: `parse_int_setting` must raise `ValueError` (not `AttributeError`) for `None`/non-str input, or the protected `settings_loader.load_setting`'s `except ValueError` doesn't catch it. |

## Discrimination verification
- Hermetic (host python, no Docker, fast iteration): all 12 tasks'
  reference + alternate PASS, every `wrong` entry FAILS — verified and
  fixed iteratively during authoring (one task, py-xmod-01, needed an
  added test case with `=` inside a KEY to actually discriminate its
  "forgot to escape `=`" wrong fixture; one alternate fixture, py-xmod-06,
  had an over-escaped regex caught the same way).
- **Real** (`run_oracle`, Docker, `python:3.11-slim`, via
  `pytest tests/test_task_discrimination.py -k "py-brk or py-xmod"`):
  **54 passed, 0 failed** — 12 reference PASS + 12 alternate PASS + 30
  wrong-fixture REJECTIONS, each confirmed to fail via the *behavioral*
  oracle (stage ∈ {compile,import,behavior}, truthy reason_code, no
  hard-cap tamper phrase in `detail`), not a hard-cap short-circuit.
- `python -c "from llmtest.harness.tasks import load_b8_tasks; ...`:
  **28** tasks load total (16 pre-existing + 12 new), all required keys
  present, all 12 new tasks carry `check_fixtures.reference`/`.wrong`.

## Full suite (`python -m pytest -q`)
`3 failed, 762 passed`. All 3 failures are pre-existing/expected, not
caused by any defect in the new tasks:
1. `tests/test_adapters.py::test_file_delivery_adapter_embeds_packet_path_in_instruction_no_stdin`
   — the pre-existing agy-path failure the brief told me to expect
   (unrelated to B8/this batch).
2. `tests/test_task_discrimination.py::test_at_least_the_11_real_python_tasks_carry_check_fixtures`
   — hardcodes `len(_TASK_IDS) == 11`; now 23 (11 + this batch's 12).
   This file was explicitly on the "do not touch" list — coordinator-owned.
3. `tests/test_harness_tasks.py::test_load_b8_tasks_returns_all_manifests_with_required_fields`
   — hardcodes `len(all_tasks) == 16`; now 28. **Not** on the explicit
   do-not-touch list, but the same class of shared/global manifest-count
   assertion (any parallel Wave-4 batch adding files would also trip it)
   — left untouched rather than risk a merge race with other batches;
   flagging here for the coordinator to bump alongside the other count
   test when integrating all Wave-4 batches.

## Concerns
- The two hardcoded-count test failures above (#2 known/expected, #3
  newly discovered) need a coordinator-side update once all Wave-4
  batches land (bump both counts to the final total, and expand
  `test_at_least_the_11...`'s hardcoded id-set/count).
- New task ids are NOT yet in `config/suite.yaml`'s `b8.tasks` allowlist
  (by design/instruction) — they load but won't run in a live `plan()`
  until the coordinator adds them.
- `py-xmod-05`'s `allowed_diff_paths` is the whole `backends.py` file
  (both `DictBackend` and `ListBackend` live in it, so `DictBackend`
  can't be hash-protected the way single-purpose files are elsewhere) —
  correctness of the untouched-but-editable `DictBackend` is enforced
  purely behaviorally (the oracle re-runs every op-sequence against it
  too), which `check_fixtures.wrong[2]` proves actually catches a
  regression there.

## Report path
`/d/BUILT-TOOLS/LLMtesting/llmtest-v2/.superpowers/sdd/task-w4b1-report.md`
