# Agentic-Quality v2.1 — Design Spec

**Goal:** Close LLMtest's agentic blind spot — where a model can pass B2's single-call tool tests yet fail as a real agent (e.g. qwen3.6-35b-a3b's silent subagent failures vs gpt-oss's agentic strength). Two components, one suite version bump.

**Architecture:** Additive extension of the existing framework. (1) Wire B2's two *judged* axes into the existing B1 judge pipeline. (2) Add a new battery **B8** that runs models through real external agent harnesses and measures agentic reliability. No changes to the frozen B1–B7 v2.0.0 data.

**Tech stack:** Python `llmtest` package; prism llama.cpp fork (ServerManager); existing judge panel (claude/codex/gemini, median-of-3); new external harnesses (OpenCode, Claude-Code-via-LiteLLM, Hermes-agent/WSL2).

## Global Constraints

- **Fully autonomous — ZERO human-in-the-loop.** No sign-off gates anywhere. Every calibration/validation step is automated (see §Part 1 calibration gate). This is a hard requirement — a human gate defeats the benchmark's purpose.
- **Version boundary: `suite-v2.1.0`.** The `suite-v2.0.0` results stay frozen and comparable; all new work is the version bump. Harness/anchor/pin changes land only at this boundary.
- **Battery identity:** B7 stays as the built config-sensitivity matrix (fix its stale TESTPLAN §5.7 cross-reference in comments only). The harness matrix is a **new B8** — never renumber B7 (would break frozen data).
- **Row-schema-as-interface preserved:** append-only sharded store, per-row pins, content-hashed identity. B8 adds harness/LiteLLM/system-prompt pins per row.
- **Local-git-only.** No remote, no push.

---

## Part 1 — Wire B2 judged axes 5 & 8 into the judge pipeline

**Why:** Axes 5 (error-recovery) and 8 (faithfulness-to-tool-results) can't be scored deterministically — they need a judge reading *how* a model handled a tool error / *whether* its answer stayed faithful to tool output. Today they carry `needs_judging=True` + a `fabrication_guard` floor but are NOT wired (`JUDGED_BATTERIES={1}`).

### Components

1. **Per-axis packets (never blended).** Axis 5 and axis 8 each get their own packet + own rubric. A packet cohort = every non-quant-arm model's response to one `(B2 scenario, axis, run_n)`, blinded behind per-judge letter permutations + CAL-strong/CAL-weak — identical machinery to B1.

2. **Richer packet body (the one real design point).** Unlike a B1 text answer, a B2 axis packet shows the *tool interaction* so the judge can score it:
   - **axis 5:** scenario prompt + tool schemas + **the injected tool error** + the model's recovery (follow-up calls + final answer).
   - **axis 8:** scenario prompt + **the tool results handed to the model** + its final answer (faithful vs fabricated beyond results).
   The builder assembles this from the **fixture** (prompt/tools/injected-error/results) + the **model artifact** (calls + text), both already present in B2 rows/fixtures.

3. **Battery-aware anchor resolution.** `packets.py` currently derives the anchor from a `b1.<unit>-NN` task_id. Generalize to: B1 → business unit (unchanged); B2 → the **axis** (`grading/anchors/b2-axis5-error-recovery.md`, `grading/anchors/b2-axis8-faithfulness.md`). No change to B1 behavior.

4. **Pipeline switch + aggregation.** `JUDGED_BATTERIES = {1, 2}`; the runner already filters `needs_judging` rows so B2 axis-5/8 rows flow in. `aggregate()` yields per-model axis-5 and axis-8 medians; the report's B2 section replaces its "not yet wired" caveat with real scores beside the deterministic axes.

5. **Autonomous calibration gate (replaces human sign-off).** Anchors + CAL pairs are LLM-authored from a fixed meta-prompt, then committed + content-hashed (`rubric_sha`). Every run, the panel judges the CAL anchors; acceptance is automatic iff **CAL-strong median ≈ 9, CAL-weak ≈ 2, drift within tolerance** (the existing `drift_flags` logic becomes the gate). Failing calibration auto-flags for regeneration. The scale self-validates by judge behavior, not human approval.

### Caveat (in-scope to state, follow-on to fix)
Judged-axis quality is only as good as the axis-5/8 scenarios. If current fixtures' error-injection is thin, authoring richer error-recovery scenarios is a **follow-on task-authoring pass**; Part 1 wires the pipeline.

---

## Part 2 — B8 Agentic Harness Compatibility

**Why:** The real agentic test — run each model through real harnesses on multi-turn tasks and measure whether it functions as an agent. This is the un-built TESTPLAN §5.7.

### Components

1. **`HarnessAdapter` ABC** (mirrors the judge-adapter pattern): `setup(task, endpoint, workspace) → run() → Trace → teardown()`, version auto-captured. Three adapters:

   | Harness | Model transport | Pins captured |
   |---|---|---|
   | **OpenCode** | llama-server directly (OpenAI-compatible) | opencode version |
   | **Claude-Code-via-LiteLLM** | LiteLLM proxy: Anthropic `/v1/messages` ↔ OpenAI → llama-server | claude + LiteLLM versions |
   | **Hermes-agent (WSL2)** | llama-server (OpenAI-compatible), driven in WSL2 | hermes commit |

   `ServerManager` serves the model once; adapters point at it. Every row pins **harness version + LiteLLM version + system-prompt hash** (§5.7).

2. **Five tasks, sandboxed, deterministic completion.** Reuse B6 shapes where they exist (from-scratch = snake/tetris; bugfix = planted bug) + add **edit**, **multi-file change**, **tool-heavy**. Each runs in a fresh temp git workspace; **completion is a deterministic check on final state** (compiles / tests pass / bug fixed / artifact runs).

3. **Scoring — deterministic first, judged only where required:**
   - **Completion rate**, **steps**, **tokens** — deterministic (trace + endpoint usage).
   - **First-failure diagnosis (a/b/c/d):** (a) schema-never-parsed and (d) harness-bug read **deterministically from logs**; (b) parsed-but-misused and (c) task-logic use a **judged classifier** (an LLM labels the first failure — autonomous, no human).
   - **Subagent canary (unscored):** on the delegation-required task, did the model spawn a **working** subagent (a subagent process ran + returned a usable subtask result)? Deterministic; a headline finding per §5.7, not a score.

4. **Reproducibility break (stated plainly).** B1–B7 are byte-deterministic; **B8 cannot be** — real agentic loops vary run-to-run even at temp-0. B8 is **variance-tolerant by design**: N≥2 runs/cell, report **distributions** (completion %, median steps), not exact values. This is B8's distinguishing property, documented in the report.

5. **Cost + scoping knob.** Full matrix = 16 models × 3 harnesses × 5 tasks × 2 runs = **480 multi-turn runs** (~1–2 days GPU; many local 24–30B models fail outright). Config knob **`b8.models`** (subset) + designed to run on a **rented Blackwell**, like the assessment.

### Data model
B8 rows extend the standard row schema: `battery=8`, `condition` encodes `(harness, task, run)`, plus `harness_version` / `litellm_version` / `system_prompt_sha` pins in the session/row provenance; `metrics` = `{completion, steps, tokens, first_failure, subagent_spawned}`. `needs_judging=True` only for the (b)/(c) first-failure classification rows.

## Testing
- Part 1: unit tests for battery-aware anchor resolution + B2 packet assembly (fixture+artifact → packet body); a `--fake` judge dry-run over B2 axis packets confirms end-to-end wiring without burning quota.
- Part 2: a HarnessAdapter contract test (mock harness → Trace), a per-harness smoke test (one trivial task, one small model), and a deterministic-completion-check unit test per task.

## Open items / follow-ons (out of this spec)
- Richer axis-5/8 error-injection scenarios (task authoring).
- OpenCode install/version pin (new harness in this environment; Hermes + LiteLLM already run here).
- Whether B8 subagent canary extends to OpenCode/Hermes native delegation vs only Claude-Code Task.
