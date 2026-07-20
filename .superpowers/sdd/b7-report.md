# Battery 7 (harness matrix) — build report

Branch: `p4-b7-harnessmatrix`. Mirrors B1 (`b1_business.py`/`b1_fixtures.py`) and B5
(`b5_serving.py`) patterns per the build brief.

## Interpretation note (read this first — B7 was under-specified)

TESTPLAN §5.7 describes B7 as a comparison across **three real external
agentic-coding harnesses** — Hermes-agent (WSL2), OpenCode, and Claude Code via
a LiteLLM `/v1/messages` proxy — all hitting an identical pinned llama-server
endpoint, scored on completion rate / steps-tokens / first-failure diagnosis,
over a fixed 5-task agentic set (edit, multi-file, bugfix, tool-heavy,
from-scratch). TESTPLAN's own roadmap line explicitly slates that axis for
**P6** ("B7: WSL2 Hermes + OpenCode + LiteLLM-CC, pins recorded | 1 d") — it
needs WSL2 + OpenCode installed + a running LiteLLM proxy, none of which exist
at this build point (P4), and stat­nding those up is out of scope for this
ticket.

The build brief for this ticket asked for something different and buildable
**now**: a *sensitivity* matrix over harness-adjacent **config knobs** — system
prompt, temperature, tool-call format, n-gram spec-decode on/off — run against
the same llama-server backend already used by B1/B5, scored deterministically.

**Resolution:** I built the config-sensitivity interpretation from the brief,
not the external-harness-CLI interpretation from TESTPLAN §5.7. I reused the
`cond=B7` condition-vocabulary slot TESTPLAN reserves for the harness axis
(same battery, same slot) so the two interpretations aren't in conflict at the
schema level — they're two different *dimensions* of "harness sensitivity."
The matrix definition (`config/suite.yaml` → `b7.matrix.dimensions`) is fully
config-driven and generic (`_matrix_cells()` in `b7_harnessmatrix.py` builds
cells from whatever dimensions/values are declared, not hardcoded names), so
when the real external-harness axis lands at P6 it's a one-line addition:
`harness: {values: [hermes, opencode, claudecode-litellm], baseline: hermes}`
— no WorkItem/row-schema change needed. **Flagging this prominently for
review**: if the intent was actually to stand up the three real harnesses now
(pulling P6 forward), this build does not do that — say so and I'll re-scope.

## Design

**Matrix shape — one-factor-at-a-time (OFAT) from a fixed baseline cell**, not
a full factorial cross. TESTPLAN 5.7's Control principle ("harness is the only
variable") is applied per-dimension: each non-baseline cell flips exactly one
knob from the baseline, so any output shift in that cell is attributable to
that one variable. 4 dimensions × 2 values each → 1 baseline + 4 variants = **5
cells**:

| dimension | baseline | variant | what it isolates |
|---|---|---|---|
| `sysp` | `default` (business-assistant system prompt) | `minimal` (`"Be concise."`) | system-prompt sensitivity |
| `temp` | `t0` (temperature=0.0) | `tdef` (temperature omitted → runtime default) | sampling-determinism sensitivity |
| `toolfmt` | `native` (OpenAI `tools` API) | `prompted` (schema embedded as text, `TOOL_CALL: fn(...)` textual convention) | tool-calling-convention sensitivity |
| `spec` | `ngram32` (n-gram spec-decode on) | `off` | decode-path sensitivity — reuses the existing `spec` condition key/vocab from B1/B5, no new key |

Config lives in `config/suite.yaml` → `b7.matrix.dimensions`; `_matrix_cells()`
reads it generically (baseline dict + one variant per non-baseline value per
dimension), so adding a dimension or a 3rd value is a YAML edit, not a code
change.

**Scoring — deterministic throughout, `needs_judging=False` on every row.**
Per the brief's "prefer det_checks; needs_judging only where a judge is
required," I found no case here that strictly needs a judge:
- **Content signals** (`contains`/`regex`/`numeric`) — reused verbatim from
  `b1_fixtures.check_signals` (import, not duplication) against each probe's
  fixed expectations.
- **Format compliance** — `format_json` det_check for the one
  `response_format: json` probe (parses the whole response or a fenced
  ` ```json ` block).
- **Tool-call compliance** — `tool_call_compliance` det_check, branching on
  `toolfmt`: native checks `message.tool_calls[].function.name`; prompted
  checks a regex for the `TOOL_CALL: name(` textual convention.
- **Consistency vs. baseline** (non-baseline cells only, when the baseline
  row for the same model/probe/run_n has already been computed and is found
  in the store): `signal_agreement_vs_baseline` — fraction of shared
  content-signal pass/fail results that match the baseline's (pass ≥ 0.8,
  `agreement_threshold` in suite.yaml); plus informative (non-gating) drift
  metrics `length_ratio_vs_baseline` / `word_jaccard_vs_baseline`.
- **`byte_identical_vs_baseline`** (spec-off cell only, since it's the only
  variant that keeps temp=t0 while changing something the project's own root
  CLAUDE.md claims is lossless): exact string equality between the ngram-off
  response and the baseline response. This directly, empirically tests the
  "n-gram spec decode is lossless at temp=0 (byte-identical output)" claim
  through this exact harness — a good fit for a "harness sensitivity" battery.
- Baseline-lookup mechanics: `execute()` computes the baseline cell's
  deterministic `row_id` and scans `ctx.store.iter_rows()` for it (same O(n)
  scan pattern B1 already uses for `--force` bump logic); if not found (e.g. a
  filtered/partial run executed the variant before the baseline), the row
  still completes successfully with `status="ok"` — it just skips the
  vs-baseline checks rather than crashing. Tested explicitly
  (`test_execute_variant_missing_baseline_row_degrades_gracefully`).
- Plan-order guarantee: `_matrix_cells()` always emits baseline first, so a
  full unfiltered run computes it before its 4 sibling variants.

**Probes — 8 fixed tasks**, `suite/b7_harnessmatrix/probes/probe-01..08.yaml`,
loaded via `b7_fixtures.load_probe_tasks` (mirrors `b1_fixtures.load_unit_tasks`:
per-task `fixture_sha` = sha256 of the YAML bytes, fail-loud on malformed
fixtures). Diverse by design, one per axis that matters to the matrix:
1. `probe-01` helpdesk-ticket triage — general text/content-signal probe (also
   the one used in most execute()-level unit tests).
2. `probe-02` tool-call lookup — `expects_tool_call: true`, exercises `toolfmt`.
3. `probe-03` small code-edit (mirrors a B1 coding-style task).
4. `probe-04` structured JSON status report — exercises `format_json`.
5. `probe-05` arithmetic/cost estimate — `numeric` signal.
6. `probe-06` strict bullet-format instruction-following.
7. `probe-07` persona-sensitivity — purpose-built for `sysp`: facts (product,
   version, "zero downtime") must survive a system-prompt change even though
   tone should shift.
8. `probe-08` reasoning/summarization with an SLA threshold check.

Loader has its own lightweight lint (`lint_probe_tasks`: id format, duplicate
ids, signal shape/regex-compiles) tested against the real fixtures (clean) and
synthetic bad ones. I did **not** wire a B7 block into `validate_cmd.py`
(unlike B1, which has one) — kept the change footprint smaller since the
contract only requires `llmtest validate` exit 0, which already holds (B7
fixtures pass the generic mojibake/UTF-8 scan). Wiring a full validate_cmd
lint block mirroring B1's is a reasonable P5+ follow-up if the review wants
parity there.

**Condition/schema wiring:**
- `condition_order` extended with `sysp, temp, toolfmt` (appended at the end —
  doesn't reorder or affect existing B1/B5 condition strings, since
  `canonical_condition` only renders keys present in a given battery's
  `parts` dict).
- `condition_vocab.cond` gains `B7`; new `sysp`/`temp`/`toolfmt` vocab entries.
- **Found and fixed a real YAML gotcha while wiring the `spec` dimension**:
  `values: [ngram32, off]` in `suite.yaml` silently parsed the bare `off` as
  Python `False` (YAML 1.1 boolean-literal folding, PyYAML's default
  resolver) — same class of bug B5 avoids by hardcoding the Python string
  `"off"` rather than sourcing it from YAML. Fixed by quoting: `"off"`, with a
  comment flagging the gotcha for future dimension additions.
- `battery=7` throughout; `row_id`/`fixture_sha`/`condition` composite key
  matches schema.py's idempotency contract unchanged.

## Scale

Roster: 11 non-quant-arm models (12 registry models, `gemma-4-26b-a4b-mxfp4`
excluded via `role: quant-arm`, same rule as B1). 8 probes × 5 cells × 2 runs
(`b7.n_runs`, matching TESTPLAN 5.7's reproducibility line "N≥2 for harness
cells (B7)") = 80 work-units per model → **880 planned WorkItems** total.

## Verification

- `python -m pytest -q -m "not gpu"` → **203 passed** (27 new in `tests/test_b7.py`,
  176 pre-existing unchanged, including all of B1/B5).
- `python -m llmtest validate` → **exit 0**, `71 rows checked, 0 errors`
  (row count unchanged from the frozen baseline — see caution below).
- Manual end-to-end smoke: `python -m llmtest run --battery 7 --model
  gpt-oss-20b --task b7.probe-01` actually launched a real llama-server and
  ran 10 real B7 rows against gpt-oss-20b on the box — confirms the
  plan→execute→row pipeline works end-to-end, not just against stubs.
  **This wrote real rows into `results/rows-suite-v2.0.0-shakedown.jsonl` and
  `results/sessions.jsonl`, which I reverted with `git checkout --` (confirmed
  clean via `git status`) since the brief says not to touch `results/`.**
  Flagging this so the reviewer knows a real (harmless, reverted) B7 run
  happened during the build — worth knowing if any B7-tagged artifacts ended
  up in `artifacts/` on the box (gitignored; I deleted the stray
  `artifacts/b7/` + `server-*.log` files this run produced, and confirmed no
  `llama-server.exe` process was left running).

## Files touched

- `llmtest/batteries/b7_harnessmatrix.py` (new) — `B7HarnessMatrix(Battery)`, id=7.
- `llmtest/batteries/b7_fixtures.py` (new) — probe loader + lint.
- `suite/b7_harnessmatrix/probes/probe-01..08.yaml` (new) — the 8 probes.
- `llmtest/batteries/__init__.py` — lazy import for battery id 7.
- `config/suite.yaml` — `condition_order`/`condition_vocab` additions, new
  `b7:` block (n_runs, ctx, max_tokens, agreement_threshold, probes_dir,
  matrix.dimensions).
- `tests/test_b7.py` (new) — 27 tests: registry, fixture loader + lint, matrix
  cell generation, `plan()` count/condition-distinctness/force-bump, all
  `execute()` det_check paths (signals, json format, native/prompted tool
  calls, signal-agreement-vs-baseline, byte-identical-vs-baseline both ways,
  missing-baseline graceful degradation), suite.yaml wiring.

## Known gaps / follow-ups (not done, flagging rather than silently skipping)

1. **The real external-harness axis** (Hermes-agent/OpenCode/Claude-Code-LiteLLM)
   is not implemented — by design, per the interpretation above; still P6 scope.
2. No `validate_cmd.py` fixture-lint wiring for B7 (loader-level lint only,
   tested in `test_b7.py`) — B1-parity follow-up if wanted.
3. Only 2 values per dimension (OFAT, no partial factorial) — brief allowed
   "2-3 values"; kept to 2 to hold the matrix at 5 cells for a first pass.
   Growing any dimension to 3 values is a YAML-only change.
4. `chat-template options`, listed as an example dimension in the brief, was
   deliberately dropped from the 4 chosen dimensions (kept to "3-4
   dimensions" as instructed) — real chat-template variation needs a
   `--chat-template-file`/`--jinja` flags_overlay at the ServerManager level,
   which felt like unnecessary server.py surface area for a first-pass matrix
   already covering 4 orthogonal knobs at the request-construction level.
