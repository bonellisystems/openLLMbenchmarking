# Agentic-Quality v2.1 — Design Spec

**Goal:** Close LLMtest's agentic blind spot — where a model can pass B2's single-call tool tests yet fail as a real agent (e.g. qwen3.6-35b-a3b's silent subagent failures vs gpt-oss's agentic strength). Two components, one suite-version bump.

**Architecture:** Additive extension of the existing framework. (1) Wire B2's two *judged* axes into the existing B1 judge pipeline. (2) Add a new battery **B8** that runs models through real external agent harnesses. No changes to the frozen B1–B7 v2.0.0 data.

**Tech stack:** Python `llmtest` package; prism llama.cpp fork (ServerManager); existing judge panel (median-of-3); new external harnesses (OpenCode, Claude-Code-via-LiteLLM, Hermes/WSL2).

> **Reviewed by codex (gpt-5.6-sol) 2026-07-19** — 27 findings; the substantive ones are folded in below and tagged `[Rn]`.

## Global Constraints

- **Fully autonomous — ZERO human-in-the-loop.** No sign-off gates. All calibration/validation is automated (§Part 1). Hard requirement.
- **Version boundary: `suite-v2.1.0`.** **`[R24]` Boundary policy:** frozen B1–B7 v2.0.0 rows are **imported by reference, not copied** — the v2.1 report reads both shards and labels every battery's `source_suite`. New/changed rows (B2 judged axes, B8) are minted under `suite-v2.1.0`. B2's *deterministic* axes 1-4/6-7 are **not re-run**; they carry `source_suite=v2.0.0`; only the new *judged* axis-5/8 rows are v2.1. The report is a clearly-labeled composite, never a silent merge.
- **Battery identity:** B7 stays the built config-sensitivity matrix (fix its stale §5.7 cross-ref in comments only). Harness matrix is a **new B8**. Never renumber B7.
- **Row-schema-as-interface preserved:** append-only sharded store, per-row pins, content-hashed identity.
- **Local-git-only.** No remote, no push.

---

## Part 1 — Wire B2 judged axes 5 & 8 into the judge pipeline

**Why:** Axes 5 (error-recovery) and 8 (faithfulness-to-tool-results) need a judge reading *how* a model handled a tool error / *whether* it stayed faithful to tool output. Today they carry `needs_judging=True` + a `fabrication_guard` floor but aren't wired (`JUDGED_BATTERIES={1}`).

### 1.1 Per-axis packets, axis in the identity `[R1]`
The judged cohort key is **`(battery, task_id, axis, run_n)`** — axis is a first-class part of packet identity and validation, so axes 5 and 8 can never blend. `packet_id` gains `axis` in its hash preimage. A cohort = every eligible model's response to one `(scenario, axis, run)`.

### 1.2 Richer packet body, reproducibly `[R10]`
Body shows the tool interaction: **axis 5** = scenario prompt + tools + injected tool error + the model's recovery; **axis 8** = scenario prompt + the tool results handed to the model + its final answer. Built from the **fixture** (prompt/tools/injected-error/results) + the **model artifact** (calls+text). **Reproducibility:** packet construction verifies the fixture's `fixture_sha` matches the row's stored `fixture_sha`; a mismatch **rejects** the build (no silent rebuild from a mutated fixture). The judge-visible interaction is assembled from the immutable artifact + the hash-verified fixture.

### 1.3 Battery-aware resolution (anchors, CAL, aggregation) `[R2][R4]`
Generalize the B1-hardcoded `_unit_from_task_id` into a **battery-aware dimension resolver**: B1 → business unit; B2 → **axis**. This one resolver drives *all four* lookups together — rubric anchor, CAL-strong, CAL-weak, and the aggregation/report grouping — with explicit B2 paths (`grading/anchors/b2-axis5-error-recovery.md`, `…axis8-faithfulness.md`; `grading/calibration/b2/axis5.yaml`, `…axis8.yaml`) and hash verification. The B1 `b1.<unit>-NN` parser is guarded to **reject B2 task_ids** (fail loud, never mis-parse). Aggregation reports per `(model, axis)` for B2, per `(model, unit)` for B1.

### 1.4 Deterministic-floor × judged-score combination rule `[R3]`
Explicit precedence: the deterministic `fabrication_guard` is a **hard cap**, not an annotation. A **failed** fabrication check (a trap value appears) **caps the axis score at 2** regardless of the judge's number; a passed/NA check leaves the judged median untouched. This rule lives in aggregation, is unit-tested, and is shown in the report.

### 1.5 Autonomous, *non-circular* calibration `[R5][R6]`
The naive "regenerate anchors until the panel scores them right" is circular — it overfits anchors to the judges. Instead:
- **Author once, freeze, pin.** Anchors + CAL pairs are LLM-authored from a fixed meta-prompt, then **frozen + content-hashed** with their `author_model` / `author_prompt_sha` / `author_params` pinned in the artifact. They are versioned inputs, not run-time-regenerated.
- **Validate against invariants + a holdout, not the panel's own scores.** Acceptance checks are **judge-independent**: (a) CAL-strong must out-score CAL-weak on *every* judge (ordinal invariant), (b) a small **holdout** of hand-diverse trap/no-trap responses must land on the correct side of the fabrication guard, (c) drift within tolerance. Regeneration is **bounded (≤2 attempts)** and any still-failing axis is **quarantined**, not silently re-tuned.
- **Quarantine gate is explicit `[R6]`.** A new pre-publication state: an axis whose calibration fails is **excluded from model scores** and flagged in the report — it does not leak partially-calibrated numbers.

### 1.6 Cohort completeness / quorum `[R7]`
The current "build only if *every* cohort model has an ok row" means one missing model suppresses everyone's score — **this exact failure produced our 15 incomplete B1 packets.** Fix: a per-run **frozen cohort manifest** + a **quorum rule** — a packet scores with ≥ Q models present (default Q = full roster; configurable floor), **reports the missing members explicitly**, and never silently rebuilds a smaller anonymous cohort (which would shift letter permutations). Missing-member provenance is recorded on the packet.

### 1.7 Scope note
`role=quant-arm` (existing registry field) remains the exclusion mechanism `[R8]`. Blinding already mixes packet content into `base_seed`, so cross-packet letter correlation is already mitigated; we additionally fold `packet_id` into the permutation seed for new v2.1 packets as cheap hardening `[R9]`.

---

## Part 2 — B8 Agentic Harness Compatibility

**Why:** The real agentic test — run models through real harnesses on multi-turn tasks. The un-built TESTPLAN §5.7.

### 2.0 Feasibility spikes FIRST (gate the build) `[R15][R16]`
Before the matrix is built, two prerequisite spikes, each with a documented pass/fail:
- **LiteLLM protocol spike:** confirm Anthropic `/v1/messages` ↔ local OpenAI translation survives tool-result blocks, streaming, stop reasons, and Claude Code's assumptions — or define the "unsupported" terminal result and drop that harness gracefully.
- **Server-profile spike:** confirm whether one llama-server config serves all harnesses, or whether each `(model, harness)` needs its own **server profile** (chat template / tool-call parser / stop tokens / ctx). Output: a validated **server-profile matrix**; controlled restarts per profile are allowed, every server flag + template hash pinned.

### 2.1 `HarnessAdapter` ABC
`setup(task, endpoint, workspace) → run() → Trace → teardown()`, version auto-captured. Adapters: **OpenCode** (llama-server direct), **Claude-Code-via-LiteLLM** (proxy), **Hermes/WSL2**. `ServerManager` serves the model per the profile matrix.

### 2.2 Full execution provenance in identity `[R11][R12][R26]`
B8 rows carry an **`execution_provenance_sha`** over *every* behavior-affecting input: harness version, LiteLLM version, server profile (flags + template hash), and the **complete rendered system/tool prompt** (not just a static system-prompt string — harnesses assemble prompts dynamically from defaults + project files + tool schemas). This sha is part of `condition`, so different harness/profile executions never collide.
**Non-determinism vs append-only identity:** B8 replaces `run_n` with **`replicate_n` (logical) + `attempt_id` (execution)**. Same cell + replicate can be *attempted* multiple times with distinct `attempt_id`s; row identity includes `attempt_id`, so replicates never dedup-collide. Analysis eligibility rules define which attempts count (e.g. first N non-infra-error attempts).

### 2.3 Sandboxing + anti-gaming `[R13][R14]`
Agentic harnesses execute arbitrary commands — this is a **security boundary**, not a temp dir. Each run executes in a **disposable container/VM**: secret-free environment, no host credential mounts, **network egress policy** (only the model endpoint), CPU/GPU/wall/token quotas, **process-tree kill** on teardown, and post-run cleanup verification. **Completion is gamed-proofed:** hidden validators live **outside the writable workspace**, protected files are hash-checked, a diff-constraint bounds what the agent may touch, and independent behavioral tests run **after** the harness exits.

### 2.4 Tasks — immutable, versioned manifests `[R25]`
Five task *shapes* (edit, multi-file, bugfix, tool-heavy, from-scratch), each a **versioned immutable manifest**: setup-repo hash, allowed tools, budgets, hidden oracle/validator, expected outcome, task version. Reuse B6 content where it exists (snake/tetris, planted bug) but promote to full manifests with precise completion oracles.

### 2.5 Metrics — normalized, comparable `[R20]`
A **normalized trace-event schema** every adapter maps into (turn / tool-call / tool-result / subagent-spawn / terminal). Authoritative **prompt/completion tokens come from the server side** (llama-server usage), not harness proxies (which omit/double-count); harness-native metrics are retained separately for reference. `steps` = normalized agent turns.

### 2.6 First-failure classification — its own pipeline `[R18][R19]`
A categorical label does **not** fit the numeric median-of-3 judge pipeline. B8 gets a **separate blinded classification pipeline**: deterministic detectors run first with stated precedence — (a) schema-never-parsed and (d) harness-bug where log-inferable — and **every unresolved failed trace** goes to a **panel classifier** (blinded, per-trace votes, majority + tie handling, an explicit `unknown/ambiguous` label, confusion/abstention reporting). It shares the adapter/blinding infrastructure but has its own label schema and aggregation.

### 2.7 Subagent canary — fully specified `[R21]`
The delegation-required task is **task #5's explicit variant** (a subtask a competent agent would delegate). Per harness, define the **observable spawn event + "usable result" criterion**. Emit **`not_applicable`** (never false) when a harness has no delegation primitive, so the metric is honest per-harness.

### 2.8 Replicates + statistics `[R22]`
N=2 yields only 0/50/100% — no distribution. Default **N ≥ 5 replicates/cell**; the report presents **raw per-replicate outcomes + a completion proportion with a Wilson interval**, never a smooth "distribution" claim at small N. `b8.replicates` is configurable.

### 2.9 Budgets + terminal statuses `[R23]`
Every B8 run has explicit **per-run wall-clock / token / step budgets**, standardized **terminal statuses** (completed / failed-task / budget-exceeded / infra-error / killed), kill-escalation, and partial-trace preservation. Early-stopping rules are unbiased (never stop only on success).

### 2.10 Cost + scoping
Matrix = models × 3 harnesses × 5 tasks × N replicates → large (many local 24–30B models fail outright). `b8.models` + `b8.harnesses` + `b8.replicates` config knobs; runs on a rented Blackwell like the assessment.

## Testing `[R27]`
- **Part 1:** unit tests for battery-aware resolution, axis-keyed packet assembly, the fabrication-cap combination rule, and the non-circular calibration invariants; `--fake` judge dry-run over B2 axis packets.
- **Part 2:** `HarnessAdapter` contract test (mock harness → Trace); **fault-injection tests per adapter** — hangs, malformed tool calls, proxy disconnect, missing usage, child-process leak, partial writes, teardown failure — asserting correct terminal status, trace durability, resource cleanup, safe retry; per-task deterministic-completion + anti-gaming (protected-file tamper) tests; the two §2.0 spikes are themselves gated deliverables.

## Phasing
Part 1 (self-contained, reuses the judge pipeline) ships first. Part 2 starts with the §2.0 spikes; the full matrix is built only if they pass.

## Out of scope / follow-ons
Richer axis-5/8 error-injection scenarios (authoring); extending the subagent canary to OpenCode/Hermes native delegation; a shared container-runtime for other batteries.
