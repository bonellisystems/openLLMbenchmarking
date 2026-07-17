# Battery 4 (long-context + KV-quant sweep) — build report

Branch: `p4-b4-longcontext`. TDD build mirroring B1 (`b1_business.py`/`b1_fixtures.py`)
and B5 (`b5_serving.py`)'s patterns, per TESTPLAN §5.4.

## Files

- `llmtest/batteries/b4_fixtures.py` — `LongContextTask` dataclass, fail-loud loader
  (`load_longcontext_tasks`), document builder (`build_document`), deterministic
  scorer (`check_needle_signals`).
- `llmtest/batteries/b4_longcontext.py` — `@register class B4LongContext(Battery)`,
  `id = 4`. Sweep-composition helpers (`ctx_label`, `tiers_for_model`,
  `arm_fits_estimate`, `model_arms`) are module-level and independently unit-tested.
- `suite/b4_longcontext/task-{01..08}.yaml` — 8 tasks (2 single-needle, 2
  multi-needle, 2 multi-hop, 2 distractor-heavy).
- `config/suite.yaml` — new `b4:` block; `"B4"` added to `condition_vocab.cond`.
  `condition_vocab.kv` already covered `[f16, q8, q4]`, no change needed there.
- `llmtest/validate_cmd.py` — added a B4 fixture-lint block (id/kind/needles/
  depth_pct/signals), and extracted the B1 signal-value lint into a shared
  `_lint_signal_values()` helper (also used by B4; extended with a `not_contains`
  signal type B4 needed for distractor rejection).
- `llmtest/batteries/__init__.py` — lazy import for battery id 4.
- `tests/test_b4.py` — 24 tests (all green; see below).

## Design: two sweep dimensions, unioned per model

TESTPLAN 5.4 describes two different things under "Battery 4" that the Build spec's
one-line description ("non-quant-arm roster × long-context tasks × KV-quant settings
× ctx tiers → each combo a distinct WorkItem") compresses into a single grid:

1. **Capability sweep** ("Capability" bullet): NIAH/multi-hop retrieval quality as
   a function of context length, for *every* roster model, at one representative
   KV dtype (`standard_kv: q8`).
2. **KV-quant quality-cost sweep** ("KV-quant quality cost" bullet): f16 vs q8 vs
   q4 answer-quality comparison, explicitly scoped to **one designated model**
   (`qwen3.6-35b-a3b`) plus **one named spot-check** (`qwen3.6-27b-dense` at 32k,
   f16-vs-q4 only) — "primary sweep on qwen3.6-35b-a3b ... common context points
   that physically fit; 128k/256k points are q8/q4-only arms."

I implemented both, unioned per model in `model_arms()`:

- **Standard-sweep arms are physically-fit-pruned.** `arm_fits_estimate()`
  generalizes the existing `registry.fits()` (which always checks a fixed 128k
  floor regardless of the caller's ctx) to the *actual* requested ctx, reusing the
  same `kv_bytes_per_token` table + hybrid-linear-attention 0.25× discount. Arms
  that don't physically fit on T1 are silently dropped from the grid — this is
  what keeps an 11-model × 4-tier grid from planning doomed multi-minute launch
  attempts.
- **Designated-sweep arms (`kv_sweep_models`, `kv_spot_check`) are NOT pruned.**
  TESTPLAN names these points explicitly; the whole point of the sweep is to
  *empirically discover* whether e.g. q4 KV survives at 256k. An arm the fit
  estimate predicts won't boot is still planned, but tagged `fits-short-context`
  as an advisory (row lands in results either way — success or a loud
  `EXEC-ERROR` — rather than being silently absent from the plan).

Verified against the frozen registry: `arm_fits_estimate` reproduces TESTPLAN's
claim exactly for the primary model — q8/q4 fit at 128k, f16 doesn't; q4-only fits
at 256k (f16 AND q8 excluded). One concrete tension found and flagged below (spot
check).

## `claimed_ctx` handling (the flagged uncertainty)

Per the Build spec's own parenthetical — "ctx tiers ... where the model claims
support" — tiers above a model's `claimed_ctx` are **dropped for that model**, not
run. `tiers_for_model()` implements this. Separately, TESTPLAN 5.4 says
architecturally-capped models are "tested at their max and tagged
fits-short-context, not skipped" — so if *every* configured tier exceeds a model's
claim, its own max rides as one substituted row instead of the model getting zero
B4 coverage.

**With the current 11-model roster, the substitution branch never actually
triggers** — every `claimed_ctx` (131072 or 262144) is ≥ the smallest configured
tier (16384), so every model keeps at least the 16k/64k tiers naturally. I added
`test_tiers_for_model_substitutes_max_when_all_tiers_exceed_claim` against a
synthetic claim (8192) to exercise that branch directly, since no real roster
model reaches it today — it's there for a future short-context intake addition,
not dead code that will silently rot.

Every roster model does get at least one arm (`test_model_arms_every_roster_model_gets_at_least_one_arm`):
`agents-a1-35b`, `nemotron-3-nano-30b`, `ornith-1.0-35b`, `qwen3-coder-30b` are
each pruned down to a single (16384, q8) arm by `arm_fits_estimate` (they're
full-attention, no hybrid discount, and heavy enough that even 64k q8 KV doesn't
fit on T1) — real VRAM reality, not a bug.

## Scope divergence from a literal Build-spec reading (please confirm)

A literal "non-quant-arm roster × full kv_sweep × full ctx_tiers" cross product
(no designated-model restriction) would be **11 models × 8 tasks × up to 12
arms/model ≈ 1000+ rows** for B4 alone — inconsistent with TESTPLAN §8.1's own
"B4 ... 1.5–2h, dominated by prefill (+ KV sweep on **the one designated model**)"
estimate. I resolved this in favor of TESTPLAN's narrower, cost-consistent
intent (§5.4's explicit "primary sweep on qwen3.6-35b-a3b ... spot-check on
qwen3.6-27b-dense" language) rather than the Build bullet's compressed phrasing,
and made the restriction a `suite.yaml` knob (`kv_sweep_models`,
`kv_spot_check`) so it's a one-line config change, not a code change, if the
broader full-grid reading was actually intended. **Current full-roster plan()
total: 33 arms × 8 tasks × n_runs(1) = 264 rows** (verified in
`test_plan_full_grid_excludes_quant_arm_and_matches_summed_arms`, which computes
the expected count from `model_arms()` rather than a hardcoded magic number, so it
stays correct if the registry/suite.yaml grid changes).

## A physical-fit tension worth a second look

The spot-check's f16 arm (`qwen3.6-27b-dense` @ 32k) is named explicitly by
TESTPLAN for an f16-vs-q4 comparison. My physical-fit estimate says it's *tight*
on T1: weights(18.2) + f16-KV-at-32k(~6.0) + overhead(1.5) ≈ 25.7 GB vs 23.5 GB
usable — over budget by the generic `kv_bytes_per_token` table. I chose to still
**plan** this arm (trusting the named TESTPLAN point over my own estimate,
tagging it `fits-short-context` as an advisory) rather than silently drop it.
If it really doesn't fit, the real run will fail loudly (`EXEC-ERROR`, existing
row-level containment in `run_cmd.py`) — that failure is itself the empirical
answer TESTPLAN is asking for. Flagging in case Michael wants
`kv_bytes_per_token` overridden with a measured value for this model instead of
trusting the generic table.

## `preflight()` interpretation

TESTPLAN 7.4 lists `preflight(): corpora exist at declared lengths` as a B4-specific
example. Since B4 deliberately never stores a literal 256k-token fixture (the
Build spec says to store "the SEED/template + planted facts, not a 256k
literal"), there's no file whose existence to check. I translated this to: verify
the fixture directory loads (≥1 task), and self-test the **builder mechanism**
that stands in for stored corpora — for each configured ctx tier, build a document
from a representative task and assert it lands within 10% of the target token
count. All four tiers currently land within <1% (e.g. 128k tier: target 130048,
built ~130157).

## Test count: 24 (all green)

Grouped: fixture loader + lint (5, incl. a `validate_cmd` B4-lint integration
test) · document builder (3) · signal checker (1) · `ctx_label`/`tiers_for_model`/
`arm_fits_estimate` unit tests (5) · `model_arms` composition (4) · `plan()` (4,
incl. condition/row_id distinctness and `--force` run_n bumping) · `preflight()`
(2) · `execute()` (2, correct-needle-passes vs wrong-needle-fails-but-status-ok,
plus a spy check that the `kv` short-label → literal-dtype mapping (`q4`→`q4_0`)
and raw `ctx` token count actually reach `request_endpoint`).

## Contract verification

- `python -m pytest -q -m "not gpu"`: **200 passed** (176 pre-existing + 24 new).
- `python -m llmtest validate`: **exit 0**, `71 rows checked, 0 errors`.
- CLI wiring smoke-checked via `python -m llmtest run --battery 4 --task
  <nonexistent>` (0 planned after filter → confirms `batteries.get(4)`,
  `preflight()`, and `plan()` all execute cleanly through the real dispatch path
  without touching a GPU). This run appended 5 preflight selftest rows to
  `results/rows-suite-v2.0.0-shakedown.jsonl` as a side effect — reverted via
  `git checkout` immediately after, per the "do NOT touch results/" instruction;
  `git status` confirms that file is clean again.
- No edits to `results/`, `config/registry.yaml`, or `config/judges.yaml`.
