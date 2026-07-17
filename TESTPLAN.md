# TESTPLAN — LLMtest v2.0

**Status:** APPROVED 2026-07-16 (amendments 25–30 folded). Task authoring gated on P0 exit (§9); runner implementation proceeds per §9 phasing.
**Spec version:** 2.0.0 · 2026-07-16
**Decided via:** brainstorming session 2026-07-16 (8 seed questions + architecture review §1/§2, amendments 1–30 folded).

---

## 1. Purpose & scope

LLMtest v2 is a versioned, repeatable, templated evaluation framework: every future model drop receives the identical battery, results append to one schema, and all tables regenerate from data. Workflow target: `llmtest intake` → `llmtest run` → `llmtest judge` → `llmtest tables`.

**v2.0 execution scope:** full 7-battery baseline on **Tier 1** (24 GB laptop) for six models, plus one budgeted rented-GPU session for the vLLM leg. **v2.0 platform scope:** the tier system, registry, intake pipeline, and row schema support T1–T8 from day one so upper tiers append without migration.

Out of scope for v2.0: 120B-class models (legacy PRO-6000 data imported as annotated appendix), Tier-2 business units as scored runs (defined + seeded only), harness investigations beyond the pinned three (post-baseline timeboxes), Tier 4+ hardware.

---

## 2. Environment invariants & serving standard

- **Primary rig (T1):** RTX 5090 Laptop, 24 GB GDDR7, 896 GB/s, Windows 11. Decode is memory-bandwidth-bound; PP is compute-bound (power/thermal state is a real confound for PP — see §7.3 session fields).
- **Serving standard (inherited from root `CLAUDE.md`, restated as ground truth):** prism-ml llama.cpp fork (`D:\BUILT-TOOLS\LLMtesting\bonsai\bin\llama-server.exe`, pinned build recorded per session), `-ngl 99 --jinja -fa on --spec-type ngram-mod --spec-ngram-mod-n-match 32 --cache-ram 0`. Never `--spec-ngram-mod-n-match < 16`. Never `draft-mtp` on GGUF (measured 0.20×). Ollama is for pulls and tool-call sanity only; it never mints authoritative speed rows except as a sanctioned B5 shootout arm.
- **"Turbocache" is retired as a term.** The mechanism behind the v1 256k configs is named precisely: **KV-cache quantization + flash attention + extended `num_ctx`** (v1: Ollama `OLLAMA_KV_CACHE_TYPE=q8_0` + `OLLAMA_FLASH_ATTENTION=1`; v2 fork flags: `-fa on -ctk q8_0 -ctv q8_0 -c N`), cheap on qwen3.6 because its attention is ~75% linear. Battery 4's f16/q8/q4 sweep (§5.4) is the empirical backing for the KV-quant standard.
- **Format rules:** MXFP4 = the local FP4 (GGUF/llama.cpp/Ollama). NVFP4 = vLLM/TensorRT only; **no GGUF path exists** — never assume an NVFP4 GGUF. "Blackwell-optimized" for local llama.cpp legs means MXFP4 or QAT-Q4 checkpoints; NVFP4 checkpoints only on vLLM legs.
- **Naming:** every model in every table and every result row is named by **full HF repo path + exact quant filename**.
- **Local WSL2 vLLM hard rule:** never produces runtime verdicts for models >14 GB (CUDA-graph capture fails → eager fallback ≈ 12 t/s junk). Such sessions are refused `timing_authoritative` by the ServerManager.

---

## 3. Tier system, registry, intake

### 3.1 Tiers (VRAM as primary axis)

A tier is a property of the **deployment artifact** (HF repo + exact quant file), never the model family. Placement is computed:

```
fits(tier) := weights_bytes
            + KV @ 128k-context floor (per KV-dtype standard)
            + runtime overhead (CUDA graphs / activations)
            ≤ MEASURED usable VRAM for the tier's SKU
```

Rules: MoE weights count **total** params resident, not active. Artifacts that fit weights but miss the 128k KV floor are tagged `fits-short-context` (the Qwen3-30B/48k case) — placed, not hidden. `tiers.yaml` stores **measured** usable VRAM per SKU (marketed ≠ allocatable; B300 markets 288 GB, plan against ~268 usable).

| Tier | VRAM | SKU | Engine policy |
|---|---|---|---|
| T1 | 24 GB | 5090 Laptop | llama.cpp fork + ngram (home tier) |
| T2 | 32 GB | 5090 Desktop | same engine; content = quant-fidelity deltas (Q4 vs Q6/Q8 same repo) + KV headroom; doubles as desktop buy/no-buy data |
| T3 | 96 GB | RTX PRO 6000 | dual engine (fork + vLLM); deployment-decision tier (gpt-oss-120b router class); seeded day one by legacy Section-1 data |
| T4 | 141 GB | H200 | vLLM/SGLang primary; NVFP4/FP8 first-class; ~230B-class Q4 |
| T5 | ~268 GB | B300 | ~355–480B-class Q4 |
| T6 | 2×B300 ~536 GB | tensor-parallel vLLM/SGLang/TRT-LLM | ~670–744B-class Q4 |
| T7 | 4×B300 ~1.07 TB | — | trillion-class Q4 / 670B FP8-native |
| T8 | 8×B300 ~2.14 TB | — | FP8/BF16 giants + high-concurrency serving characterization |

T4–T8 rows must log `tp_degree` + `nvidia-smi topo -m` output (NVLink vs PCIe changes results). **Cross-tier lineage study:** Qwen3-Coder-30B (T1) → Coder-Next → Coder-480B (T5) — same family, tier-scaled, isolates what VRAM buys.

**Per-tier battery scope:** T1–T3 full 7-battery for deployment candidates. T4–T5 full only if a live deployment/inference-economics question exists, else deterministic + serving subset. T6–T8 deterministic + serving + capability spot-check by default; full runs require explicit promotion + budget sign-off. Budget gates per tier live in `budgets.yaml` with a pre-flight spot-price check — rates are never hardcoded.

**Hard rule:** tier N opens only after the suite runs clean on tier N−1. The pipeline is never debugged on rented multi-GPU.

### 3.2 Intake pipeline (every new model drop)

New drop → `registry.yaml` entry (full HF repo path, quant file, license, arch, total/active params, claimed ctx, chat template, **plus artifact provenance `{source_repo, download_date, sha256, v1_continuity: bool}`** so tables can mark which v2 rows are same-artifact-as-v1) → auto tier placement via `fits()` → triage gates **in order**:

1. **LICENSE:** deployable commercially for clients? CC-BY-NC-class → **park immediately, $0 judge spend** (the Command-A rule).
2. **Template + tool-call smoke** (T-battery 4/4-class).
3. **SMOKE SUITE head-to-head vs the tier's current champion** (challenger packets, §6.2).

→ promote to full tier baseline, or **park with a documented, queryable reason** (Nemotron precedent: parked ≠ forgotten). MoA lineup generalizes to champion-per-role-**per-tier**; tier tables regenerate like everything else.

### 3.3 External anchoring

Public benchmarks are never rebuilt. The registry carries public reference scores per model (SWE-bench Verified, Terminal-Bench 2.0, NL2Repo, Claw-Eval, …) as **imported columns with source + date**. Signature metric: **DELTA between public coding scores and our private business-unit scores** — benchmark-tuned model detection. Vendors pin harness+sampling per benchmark row; our Battery 7 pinning matches that practice.

### 3.4 Signature output

Quality-vs-tier curves per business unit ("what does T3 buy over T1 for cyber / coding / runbooks") — the chart that prices local vs rented vs purchased hardware, including the PRO 6000 buy decision.

### 3.5 v2.0 baseline roster (T1) — artifact selection rule

**Selection rule (applies to ALL SIX baseline artifacts, frozen at P0 — Coder-30B included; baseline artifacts freeze at P0, not intake):**
1. **v1/session-5 measured data exists → pin the EXACT artifact:** SHA256 the on-disk files from those runs; registry names whatever repo mirrors them.
2. **No local history → house ladder** (CLAUDE.md: unsloth UD-Q4_K_XL / Q4_K_M class).
3. **Deviation only if `fits()` at the 128k floor forces it** — reason recorded on the registry entry.

Pin = **SHA256 + download date** (HF repos requantize in place; a disk-vs-repo mismatch at P0 is a **recorded finding, never auto-refetched**). Speed tables annotate **quant FAMILY** (IQ vs K-quant vs MXFP4 — dequant cost differs on CUDA; families are never mixed silently).

| # | Registry id | Rule | Artifact |
|---|---|---|---|
| 1 | qwen3.6-35b-a3b | 1 | on-disk `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` IQ4_XS (the v1 scorecard artifact — resolves bartowski-vs-unsloth by rule, not preference) |
| 2 | gemma-4-26b-a4b | 1 | on-disk `unsloth/gemma-4-26B-A4B-it-qat-GGUF` UD-Q4_K_XL |
| 3 | ornith-1.0-35b | 1 | **the on-disk 18.4 GB MXFP4 hybrid** (session-5/6 measured artifact; jashepp `Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix` mirrors it). Registry notes the v1 scorecard row used bartowski IQ4_XS — different artifact, `v1_continuity` recorded per data source |
| 4 | gpt-oss-20b | 1 | on-disk `unsloth/gpt-oss-20b-GGUF` F16 (MXFP4 experts) — **shakedown model** |
| 5 | qwen3.6-27b-dense | 1 | on-disk `unsloth/Qwen3.6-27B-GGUF` Q5_K_M (session-5/6 games + n-gram artifact; v1 scorecard's NVFP4-GGUF conversion noted in registry) |
| 6 | qwen3-coder-30b | 2 | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` UD-Q4_K_XL (no local history → house ladder) |

**Roster expansion (Michael, 2026-07-16):**

| # | Registry id | Rule | Artifact |
|---|---|---|---|
| 7 | nemotron-3-nano-30b | 1 | on-disk `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` UD-Q4_K_XL (session-6 artifact; 21.3 GB — expect `fits-short-context` on T1) |
| 8 | granite-4.1-30b | 1 | on-disk `unsloth/granite-4.1-30b-GGUF` UD-Q4_K_XL (session-6 artifact; hybrid Mamba; the 12.08× n-gram case) |
| 9 | bonsai-ternary-27b | 1 | on-disk `prism-ml/Ternary-Bonsai-27B-gguf` Q2_0 (6.7 GB; needs the prism fork's custom kernels — the T1 standard binary) |

**Quantization floor policy (Michael, 2026-07-16):** no artifact below 4-bit enters the suite — quality-loss/hallucination risk — with exactly ONE designated exception: `prism-ml/Ternary-Bonsai-27B` (1.71 bpw) rides as the **sub-4-bit exhibit**, included specifically to pressure-test its "95% intelligence retained" marketing against the full battery. Its rows carry tag `sub4bit-exhibit`; intake auto-parks any other <4-bit candidate with reason `below-quant-floor`.

**Roster expansion 2 (Michael, 2026-07-16, from live HF search):**

| # | Registry id | Rule | Artifact |
|---|---|---|---|
| 10 | agents-a1-35b | 2 | `jashepp/Agents-A1-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF` MXFP4_MOE_Q8_0 (18.4 GB) — InternScience Agents-A1, agentic+VLM |
| 11 | ornith-1.0-9b | 2 | `jashepp/Ornith-1.0-9B-MXFP4_Hybrid-Imatrix-GGUF` MXFP4_Q8_0 (8.9 GB, MIT) |
| — | gemma-4-26b-a4b-mxfp4 | quant-arm | `FreedomAISVR/Gemma-4-26B-A4B-it-QAT-MXFP4-GGUF` mxfp4_moe (13.7 GB) — **not a scorecard roster member**; runs B5+B2+B6 arms only |

**The controlled-comparison grid these purchases:**
- **Training delta:** agents-a1-35b vs ornith-1.0-35b — same qwen3.5-MoE backbone, same jashepp MXFP4 recipe, same size class; only the training differs.
- **Scale delta:** ornith-1.0-9b vs ornith-1.0-35b — same family, same quant recipe.
- **Quant-format delta:** gemma-4-26b-a4b (QAT-UD-Q4_K_XL) vs gemma-4-26b-a4b-mxfp4 (QAT-MXFP4_MOE) — same google QAT base weights; only the container differs.

Baseline = **11 full-battery models + 1 quant-arm**; est. ≈ 15–16 laptop-nights. T3 lineage note: `unsloth/Qwen3-Coder-Next-GGUF` (80B-A3B, ~43 GB @ Q4) confirmed available — the Coder-30B → Next → 480B middle rung for the rented tier.

Any substitution is a registry diff, not a table footnote.

---

## 4. Methodology bar (applies to everything)

- **Reproducibility:** pinned fork build hash + full launch flags per session; model SHA256 + exact quant filename per row; `suite_version` + `fixture_sha` + `code_sha` per row; temp 0 + fixed seed for deterministic checks; per-model default temp with **N≥3 runs** (mean±sd) for judged tasks; **N≥2** for harness cells (B7).
- **Deterministic checkers over LLM judges wherever possible** — JSON validity, needle recall, package/CVE verification, does-the-game-load.
- **Two suite sizes:** smoke (~44 min/model, §8.2) and full (~8–12 h/model, §8.1); wall-clock + cost estimated before any run (§8).
- **Judging:** panel of 3, blind, content-hashed packets, median-of-3 — full spec §6.
- **Decisions live in config, not constants:** `tiers.yaml`, `registry.yaml`, `judges.yaml`, `budgets.yaml`, `suite.yaml` — all covered by the freeze tag.

---

## 5. The battery — seven dimensions

### 5.1 Battery 1 — Business-unit scorecard (0–10 anchored)

- **Scale:** 0–10 with **written anchors per category** defining 0 / 3 / 5 / 7 / 10; a 10 is indistinguishable from a strong frontier answer. Old 1–5 data is not convertible → the roster re-baselines under v2.
- **Tier 1 units (15, always run):** coding/software, IT infrastructure, cybersecurity, marketing, SEO, sales, outreach, finance (spanning strategic finance AND transactional accounting: AP/AR, intercompany), operations, customer support/helpdesk triage, knowledge management/SOP & runbook generation, legal & compliance, data analytics/BI (incl. NL→SQL), project management/PMO, HR/people ops.
- **Density:** 8 tasks/unit = 120 tasks, difficulty split **2 easy / 3 medium / 3 hard** (hard-density chosen deliberately: v1's failure mode was ceiling compression — top 5 within 0.22 on 1–5).
- **Tier 2 units (9, defined + 2 seed tasks each, NOT run in v2.0):** procurement/vendor management (incl. third-party security questionnaires), risk management/audit, account management (QBR prep, renewal risk), product management, executive strategy, training/L&D (incl. security-awareness content), R&D/research synthesis, PR/communications, partnerships/bizdev. Promote to full runs in v2.1 only after one complete baseline validates rubric anchors and judge calibration.
- **Saturation check (post-run, automated):** any Tier-1 unit where **all models score ≥8** gets its easiest task swapped for a hard one before v2.1.
- **Task flavor (amended 2026-07-17, Michael):** realistic *work shapes* (ticket triage, runbook generation, vendor security questionnaires, security-awareness material, QBR decks, contract/document review, data analysis) — but **NOT MSP-specific**: scenario contexts must span the full client-industry range. Every task carries a required `industry` tag from the controlled vocabulary in `suite.yaml`: `life_sciences` (incl. animal/veterinary), `oil_gas_energy`, `legal`, `financial_services`, `aec` (architecture/engineering/construction), `manufacturing_ag`, `healthcare_compliance`, `generic_smb`, `public_sector`, `msp_internal` (the provider's own ops). **Distribution rule (lint-enforced):** each unit's 8 tasks span ≥5 distinct industries, and no single industry appears in >2 tasks per unit. This also enables a scorecard-by-industry slice at table time. (Deep vertical testing remains Tier-3 pluggable modules — B1 industry tags are breadth, not depth; no double-build.)
- Run params: per-model default temp, `num_predict`-equivalent cap per task class, N=3.

### 5.2 Battery 2 — Tool calling (8 axes, scored separately, never blended)

1. Schema adherence — strict JSON validity rate (deterministic).
2. Correct tool selection among distractor tools.
3. Parallel calls in one turn.
4. Chained/dependent calls (A's output feeds B).
5. Error recovery — tool returns error/garbage; retry/adapt vs fabricate.
6. Abstention — no suitable tool exists; does it invent one.
7. Tool calls at long context (32k+ of history, then a call).
8. Faithfulness to tool results — final answer vs what the tool returned (shared with B3).

~40 scenarios × N3. Axes 1–4, 6–7 deterministic; 5 and 8 judged. `preflight()`: all tool schemas parse.

### 5.3 Battery 3 — Hallucination (measured as a curve)

Categories: closed-domain factual QA (deterministic ground truth) · abstention calibration (unanswerable + false-premise; reward "I don't know", penalize confident fabrication) · fabricated artifacts (citations, URLs, CVE numbers, PowerShell cmdlets/Graph endpoints, npm/pip slopsquatting — all machine-verifiable) · document faithfulness (every claim grounded in the provided doc; judged with deterministic claim-extraction assist) · tool-result faithfulness (shared with B2) · multi-turn consistency (self-contradiction across turns).

**Degradation curves — the headline output.** Hallucination/faithfulness rate as a function of: (a) **context fill:** 8k → 32k → 64k → 128k → 256k; (b) conversation turn count; (c) cumulative output length. Output = per-model curve, not a single number (operationalizes the "qwen3.6 hallucinates after a while" claim).

### 5.4 Battery 4 — Long context (256k target, 128k floor)

- **Capability:** NIAH single + multi-needle; RULER-style aggregation + variable tracking; QA-at-depth with position sweep (lost-in-the-middle); long-document summarization fidelity.
- **Serving mechanism documentation:** the exact flags that reach ≥128k–256k on 24 GB per model — `-ctk/-ctv` dtype, `-fa on`, SWA behavior, RoPE/YaRN settings — named in this document per model at P4, verified empirically, never folklore.
- **KV-quant quality cost (fit-aware):** primary sweep on **qwen3.6-35b-a3b** — f16/q8/q4 compared at common context points that physically fit; 128k/256k points are q8/q4-only arms. **Plus a full-attention spot-check** on qwen3.6-27b-dense (32k, f16-vs-q4, NIAH slice, ~15 min) — the primary model's ~75%-linear attention makes it the *least* KV-quant-sensitive architecture on the roster; the q8 standard must not be generalized from the easy case. If q4 KV tanks needle recall, the caveat enters §2's serving standard.
- Models architecturally capped below 128k are tested at their max and tagged `fits-short-context`, not skipped.
- `preflight()`: corpora exist at declared lengths.
- **Cache overlay caveat (amendment 21):** B3/B4 run with a bounded prompt cache (`--cache-ram N`, RSS watched) — validated in P4 shakedown against the documented OOM-reap failure mode; on reap, fall back to ascending-context slot-reuse ordering instead of the RAM prompt cache.

### 5.5 Battery 5 — Throughput & serving (standardized conditions)

- **Named conditions (mandatory on every speed row):** **PEAK** = short-context, first ~1k generated tokens. **SUSTAINED-32k** = decode measured with 32k already in cache. The gemma 163-vs-101 discrepancy is the founding case; `timing_authoritative=true` exists only on B5-minted sessions and **all speed tables filter on it**.
- Per model × runtime: decode t/s (both conditions), PP t/s, TTFT, concurrency scaling — single-stream plus 1/2/4/8/16 locally, extended ladders on rented hardware; aggregate + per-stream (RTX PRO 6000 table style).
- **Runtime shootout:** llama.cpp fork (ngram ON and OFF) vs Ollama vs vLLM where feasible. Output: per-model verdict — best runtime + exact flags for max speed.
- **Spec-decode statistics** recorded whenever the server reports them: `n_drafted`, `n_accepted`, `accept_rate` (the 8.87×-vs-1.95× spread is an acceptance story).
- **Rented leg (Verda PRO-6000, one batched session per baseline, HARD CAP $20,** priority-ordered so budget exhaustion drops the tail):
  - **P1:** three-arm spec-decode A/B on qwen3.6-35b under vLLM — baseline vs **MTP** (`speculative_config`) vs **vLLM-ngram** — same prompts. Settles MTP-vs-ngram with both mechanisms actually working (MTP was inert in Ollama, 0.20× via GGUF `draft-mtp`).
  - **P2:** NVFP4-vs-local-quant speed deltas + PEAK/SUSTAINED-32k/PP/TTFT per model.
  - **P3:** concurrency — full ladder to 128+ only for gpt-oss-20b (find the plateau the legacy table flagged "still climbing") and qwen3.6; trimmed ladder (1/8/32/128) for the rest.
  - **Pre-flight before renting:** (a) verify a vLLM-servable checkpoint exists per model (no NVFP4 Ornith exists; confirm the Qwen3.6 checkpoint ships the MTP head) — models without one get the verdict **"llama.cpp-only (no vLLM checkpoint)"**, a documented reason, not a gap; (b) smoke-test the runner's remote-endpoint mode against local WSL2 vLLM on gpt-oss-20b (fits with CUDA graphs) — rented hours measure, never debug — and keep that run as a same-vLLM-version **calibration anchor** between laptop and PRO 6000.
  - One session = one pinned environment (card, vLLM build, driver — on every row). 120B-class out of v2.0; legacy Section-1 data imported as `v1-legacy` appendix.

### 5.6 Battery 6 — Agentic coding (one-shot roster + self-correction)

- **One-shot roster** (bare one-liner prompts, blind review per session-5 method, N=3): Snake, Tetris, Arkanoid, Flappy Bird, Doodle Jump (procgen platforms), Asteroids (procgen field, splitting rocks), tiny roguelike (procgen dungeon; ASCII @-on-grid acceptable), and the flagship: **fly.pieter-style 3D browser flight sim** (Three.js, procgen skyscrapers/terrain, single HTML file). Graded on playability, feature depth, bug count (fixed rubric; judged) + the deterministic gate.
- **The gate (deterministic "green"):** headless Playwright — load (console-error-free) → motion (canvas frame-diff) → input (scripted keys → state/pixel delta) → game-specific probe. **Playability is a separate judged score and NEVER gates self-correction metrics.**
  - **Two probe profiles (amendment 18):** **FIXTURE** profile (authored known-goods + planted bugs): `window.__SEED__` / `window.__score__` hooks allowed — we own the code. **ONE-SHOT** profile (model-authored games): black-box probes only — one-liner prompts stay bare, no hook demands. Flight sim = procgen + model-authored → seed-invariant black-box by definition.
  - **Gate self-test lives in `preflight()`:** every probe must pass on the known-good version before any bugged variant counts; probe failure = harness bug, not model failure — battery refuses to execute.
  - **Gate runs decoupled from generation** (server idle/torn down); WebGL/3D probes get SwiftShader-aware extended timeouts (headless chromium = software GL).
- **Planted-bug track:** 3–4 games with known-good versions; bugs injected at three difficulties (crash/syntax · logic e.g. collision off-by-one · subtle e.g. render/state desync). Hint escalation **H0** ("something's wrong, find it") → **H1** (symptom) → **H2** (console error / failing behavior).
- **Loop protocol:** cap **N=6 structured 2/2/2** (iters 1–2 H0, 3–4 H1, 5–6 H2). Thresholds: fixed-unaided (≤2), fixed-with-symptom (≤4), fixed-with-error (≤6), plus raw steps-to-green. **Early stops:** green; two consecutive identical/no-op patches → **DNF-loop** (loop rate reported per model). The model's **stated diagnosis is logged every iteration** (detection rate and fix rate stay separable). Regressions logged per iteration.
- **Self-debug track:** model's own broken one-shot game fed back with symptom only (H1-equivalent), escalate to console error after 2 failed iters, same gate, same early stops, cap 6.
- **Report per model:** found-it-unaided %, mean steps-to-fix, regression rate, DNF-loop rate.
- Human eyeball only on judge-flagged cases (spread >2) — same rule as B1.
- Note: fix iterations are edit-tasks → n-gram accelerated (3–9×) — the cheapest decode in the suite.

### 5.7 Battery 7 — Harness compatibility matrix

- **v2.0 roster (all pinned):** Hermes-agent (WSL2; in-house), OpenCode, Claude Code via LiteLLM `/v1/messages` proxy. Pins = harness version + LiteLLM version + system-prompt hash **on every row**; harness upgrades only at suite-version boundaries (same rule as judge pins).
- **Control:** all harnesses hit the identical pinned llama-server endpoint/config — harness is the only variable. Server-side logging records the **sampling params each harness actually sends** (they inject their own temp/top_p; the confound is recorded, not discovered later).
- **Fixed 5-task set:** one edit, one multi-file change, one bugfix, one tool-heavy task, one from-scratch build. **N≥2 runs per cell** (agentic runs are high-variance). Scored: completion rate, steps/tokens consumed, plus a **first-failure diagnosis per failed run:** (a) tool-call schema never parsed, (b) parsed but misused, (c) task-logic failure, (d) harness-side bug. "Model X fails in harness Y" must be a diagnosis, not a mystery.
- **Claude Code Task/subagent axis:** excluded from scoring (documented 0% with local models — non-differentiating) but kept as **one unscored canary probe**; a model that ever spawns a working subagent is a headline finding.
- **Investigations (explicitly post-baseline; never creep into P6):** Continue-CLI / Pi / Maki, 1-hr timebox each. Promotion criteria: headless + custom OpenAI endpoint + pinnable version + non-interactive scriptability. **Cline:** dropped for v2.0 with documented reason (no headless mode) — on the v2.1 re-check list, not the permanent-drop list.

---

## 6. Judging — panel of 3, blind, content-addressed

### 6.1 Panel

Three judges — **Claude, Codex, Gemini** — each grades every packet independently and blind: same anchored rubric, same calibration anchors, randomized identities, no judge sees another's scores. **Final score = median of 3; pairwise = majority vote.** Judges are invoked **headless/scripted with pinned model strings from `judges.yaml` — never graded in-session by the orchestrator model.**

- Per-task spread (max−min): ≤2 → accept median; >2 → auto-flag for human review sample (GitHub Issues workflow). A unit with systematic flags = rubric bug → fix before v2.1.
- **Suite health metrics on every table run:** inter-judge agreement (% within 1 pt, mean spread), anchor drift, kin-delta.
- **Self-preference guard:** kin map = gpt-oss↔OpenAI judge, gemma↔Google judge (Claude judge has no kin in the roster — the neutral reference). Each judge's kin-vs-non-kin delta is logged and reported; median neutralizes it, visibility keeps it honest. Packet build **scrubs explicit self-identification strings** from answers (best-effort; kin-delta is the backstop).
- **Pins:** `judges.yaml` pins **CLI version + model string**; both recorded on every judgment row (CLI releases change judge-side prompting — same confound class as harness pins). Proposed pins (frozen at P3 after live enumeration, Michael signs off): Anthropic `claude-fable-5` (dated snapshot if available) via `claude -p`; OpenAI = Codex CLI current flagship reasoning model; Google `gemini-3-pro`-class via gemini CLI (auth = setup prerequisite).
- **Quota dry-run at P3 exit:** ~20 real packets end-to-end, per-judge burn logged, baseline judging cost extrapolated **before** P8 commits to ~900 packets.

### 6.2 Packets (content-hashed, two modes)

- **COHORT (baseline):** all cohort answers + calibration anchors, forced ranking. `packet_id = sha256(task_id | run_n | sorted member row_ids | rubric_sha | anchor_shas | blinding_base_seed)` — scoped to (task, run); **requires cohort completeness**, surfaced by `llmtest status`.
- **CHALLENGER (intake):** challenger + tier champion + anchors only. Roster growth costs O(challenger); cohort packets are never re-minted.
- Anchored 0–10 is canonical in both modes; pairwise = within-packet rankings (cohort) or champion head-to-head (challenger).
- Rubric fix → new `rubric_sha` → new packet_ids → natural re-judge; old judgments retained for audit; aggregation at table time selects judgments matching the checked-out `rubric_sha`.
- **Blinding:** seed per **(packet, judge)** — per-judge letter permutations so position bias decorrelates across the panel; permutations recorded in the **committed map** (`results/packets/`). Judgment rows store **letter AND resolved model_id at write time** — interpretability survives any single file loss; the committed map is audit/recovery, not a dependency.
- **Judgment row:** `(packet_id, judge_id, judge_model_pin, judge_cli_version, letter, model_id, score_0_10, rank, reason ← inline plain one-liner, ts)` — idempotent on `(packet_id, judge_id, letter)`. Spread>2 triage is self-contained in-repo.
- Invocation: one packet per headless call; JSON-schema-constrained response (per-letter score + one-line reason + ranking). **The schema requires the full letter set — a parseable-but-partial response is invalid → one retry → else `status=error`; no partial packets.** The calibration pair rides as blinded letters **indistinguishable from cohort answers** (anchor scores are recovered at aggregation via the map, not requested separately).
- Aggregates (median/spread/kin-delta/agreement) are **computed at table time, never stored** — a judging re-run cannot desync aggregates.

---

## 7. Architecture

### 7.1 Repo (`D:\BUILT-TOOLS\LLMtesting\llmtest-v2\`, standalone git — **LOCAL-ONLY by decision 2026-07-16**; no remote. Public-from-birth hygiene retained so a remote can be added later without history rewrites. Single-NVMe backup risk accepted by Michael; optional cold-copy of the repo folder is a manual chore, not framework scope.)

```
llmtest-v2/
├─ TESTPLAN.md                    # this document
├─ CLAUDE.md                      # repo-scoped conventions; inherits root serving standard
├─ pyproject.toml
├─ llmtest/                       # the package (mission's "runner/" deliverable)
│  ├─ cli.py                      # intake | run | judge | tables | validate | status | import-legacy
│  ├─ schema.py                   # row dataclasses + validator (same code CI runs, used at write time)
│  ├─ server.py                   # ServerManager
│  ├─ registry.py                 # registry + fits() (one code path for placement AND preflight)
│  ├─ intake.py                   # license gate → smoke → promote/park
│  ├─ judging/                    # packets, blinding, panel invocation, aggregation
│  ├─ tables.py                   # byte-deterministic regeneration
│  └─ batteries/                  # b1…b7 plugins
├─ config/                        # tiers.yaml registry.yaml judges.yaml budgets.yaml suite.yaml
├─ suite/                         # b1_business/ … b7_harness/ + modules/{agmfg,energy}/
├─ grading/                       # rubric anchors, judge prompt templates, calibration pair
├─ results/
│  ├─ rows-<suite_version>.jsonl  # SHARDED append-only results
│  ├─ judgments.jsonl · sessions.jsonl
│  ├─ packets/                    # committed blinding maps (tiny, no content)
│  └─ tables/*.md                 # generated, canonical
├─ artifacts/                     # GITIGNORED (transcripts, traces, screenshots, packet texts)
└─ .github/workflows/ci.yml
```

Client packs: `..\client-packs\` — physically outside the repo (§10). **Public-from-birth hygiene:** no secrets in tree ever; configs committed as templates; gitignored `.env` for keys/endpoints; judge CLIs keep their own auth; pre-commit + CI secret scan (gitleaks-class). Repo gets its own scoped CLAUDE.md so sessions inside it don't depend on root context.

### 7.2 Row schema (the real interface — designed hardest)

**Idempotency key** = `(suite_version, model_id, quant_sha256, battery, task_id, fixture_sha, condition, run_n)`; `row_id = sha256(canonical join)`. `llmtest run` skips existing keys by default; `--force` re-runs. Effects: fixture edits invalidate exactly the edited task; code refactors invalidate nothing; multi-hour interrupted suites resume free.

`result` row fields:

| Group | Fields |
|---|---|
| Versioning | `schema_version` · `suite_version` (declared string in suite.yaml, pinned by freeze tag) · `fixture_sha` (task's full fixture bundle, **rubrics excluded**) · `code_sha` (git — provenance only, NOT in key) |
| Identity | `row_id` · `parent_id` (B6 iterations→episode, B7 steps→episode) · `battery` · `task_id` · `condition` · `run_n` |
| Condition | canonically-ordered `key=value` composite; vocabulary AND order fixed in `suite.yaml` (it is inside the hash key — unordered composites fork row_ids) |
| Artifact | `model_id` · `hf_repo` · `quant_file` · `quant_sha256` · `tier` (denormalized for publication) |
| Session | `session_id` → sessions.jsonl |
| Sampling | `{temp, top_p, seed, max_tokens}` **as observed server-side** (B7 harness-injection confound lives here) |
| Request/response | `request{fixture_id, prompt_sha256}` · `response_meta{tokens_in, tokens_out, ttft_ms, decode_tps, pp_tps, finish_reason, truncated, n_drafted, n_accepted, accept_rate}` |
| Scoring | `det_checks{…}` inline · `needs_judging` (routes to phase-separated judge — never per-row inline) · `metrics{…}` (battery-specific, validated per-battery sub-schema) |
| Authority | `timing_authoritative: bool` — true only on B5-minted serving sessions (incl. sanctioned Ollama arms); **all speed tables filter on it** |
| Evidence | `artifacts{name → {sha256, relpath}}` |
| Status | `status (ok/error/dnf/excluded)` · `error_detail` · `tags[]` (`seed`, `non-reportable`, `legacy-v1`, `calibration-anchor`, `fits-short-context`, `selftest`, …) |

`session` row (sessions.jsonl): runtime · runtime_build · **normalized config** `{ctx, kv_dtype, flash_attn, spec_type, spec_params, parallel}` · raw invocation string · hardware SKU · measured_usable_vram · tp_degree · topology · driver/CUDA env · **power_mode · ac_state** · optional periodic clock/temp samples → artifacts (B5 sessions).

Write-time validation uses the identical `schema.py` validator CI runs. **Results sharded** as `rows-<suite_version>.jsonl`; CI enforces: shards belonging to tagged suite versions are **byte-identical to the tag** (append-only enforced, not promised). Tables glob all shards.

### 7.3 ServerManager

`request_endpoint(model_id, runtime, flags_overlay={}, parallel=1, ctx=…, kv=…) → EndpointHandle{base_url, session_id, provenance}`

- **Per-runtime translation layer** (llama-server flags / Ollama env+options / vLLM args) producing the **normalized config** recorded on the session + raw invocation — cross-runtime B5 comparability lives in data, not prose. The sanctioned Ollama arm's env (`OLLAMA_KV_CACHE_TYPE`, flash-attn) lives **here**, in the translation layer; fork build hash + Ollama version are pinned in config alongside the judge pins.
- **Startup orphan sweep:** detect/kill stray llama-server/ollama processes + VRAM audit before any launch (crashed runs are guaranteed by the resume story).
- **VRAM preflight calls `registry.fits()`** — one code path for placement and preflight.
- Standard-flag composition (CLAUDE.md defaults) + battery overlay; config-match reuse; teardown by PID with VRAM-drain verification; port allocation; logs → artifacts.
- Remote attach mode (Verda): declared descriptor + `/v1/models` probe. Policy enforcement: WSL2-vLLM >14 GB refused `timing_authoritative`; Ollama sessions auto-tagged non-authoritative outside sanctioned B5 arms.

### 7.4 Plugin ABC (minimal, refactorable)

```python
class Battery(ABC):
    id: int
    def fixtures(self, suite_cfg) -> list[Task]
    def plan(self, models, suite_cfg) -> list[WorkItem]      # key tuples; diff vs rows = free resume/status
    def preflight(self, ctx) -> list[Row]                    # OPTIONAL; selftest-tagged rows;
                                                             # battery refuses to execute on failure
                                                             # (B6 known-good gate · B2 schemas parse · B4 corpora exist)
    def execute(self, item, ctx) -> list[Row]                # rows PLURAL; det checks inline; needs_judging flags
    def build_judge_packets(self, rows) -> list[Packet]      # optional
```

Batteries talk to the **ServerManager**, never a bare endpoint; B5 owns server lifecycle. ABC validated by its two most dissimilar clients: **build order = ServerManager → B5 → B1**; interface survives both unmodified → it survives the rest.

### 7.5 CLI

```
llmtest intake <hf-repo> --quant <file>
llmtest run    --suite smoke|full [--model M] [--battery N] [--task ID] [--condition C]
               [--force] [--keep-server] [--debug]      # one task, server up, transcript → artifacts/
llmtest judge  --pending [--judge claude|codex|gemini] [--packets-only]
llmtest tables                                          # byte-deterministic (stable sort, fixed float fmt)
llmtest validate [--serving]                            # schema+fixture lint (==CI) | re-runnable serving canary
llmtest status                                          # done/pending matrix from resume keys; cohort completeness
llmtest import-legacy                                   # v1 scorecard + Section-1 PRO-6000 → v1-legacy rows
```

**Serving canary (`validate --serving`):** re-runs the Ornith ngram edit A/B and compares against a stored reference band (Table-4-derived, thermal tolerance recorded in config) — re-runnable health check, not a one-off P1 gate.

### 7.6 CI (integrity only, never GPU)

Schema validation (all shards) · fixture lint incl. **no-client-path lint** · **non-ASCII/mojibake lint on docs AND fixtures** (in a fixture, an invisible mojibake character silently alters tokenization for every model) · tables-regenerate-byte-clean · secret scan · append-only shard check vs tags. **Local-only amendment:** these run as `llmtest validate` + pytest locally (the P0 exit gate); `ci.yml` is authored but dormant until a remote ever exists. Flagged-disagreement → rubric-bug workflow tracks in `results/FLAGS.md`.

---

## 8. Sizing & cost (estimated before any run, per mission)

### 8.1 FULL suite per model (T1 local, fork+ngram standard)

| Battery | Est. wall-clock | Dominated by |
|---|---|---|
| B1 (120×3) | 1.5–2.5 h | decode (thinking models high end) |
| B2 (~40×3) | 0.5 h | deterministic |
| B3 (~80×3 + curves) | 1.5–2 h | prefill at 128k/256k (bounded cache overlay, §5.4) |
| B4 (grid + sweeps) | 1.5–2 h | prefill (+KV sweep on the one designated model) |
| B5 local shootout | 0.75–1.25 h | server relaunches |
| B6 (one-shot + planted + self-debug) | 1.5–2 h | cheap — edit iterations are ngram-accelerated 3–9× |
| B7 (3×5×2) | 1.5–2.5 h | agentic loops |
| **Total** | **≈ 8–12 h/model** | overnight-batched + resumable |

**Baseline (6 models): ≈ 60–70 h ≈ 8–9 laptop-nights** + Verda ≤ $20 + judging ≈ 800–900 packets × 3 judges ≈ 10–15 M judge input tokens (Claude via subscription; Codex/Gemini quota confirmed by the P3 dry-run before P8 commits).

### 8.2 SMOKE per model ≈ 44 min

B1 15×1 (8′) · B2 8 (4′) · B3 12 + one 32k curve point (8′) · B4 32k NIAH slice (5′) · B5 PEAK ngram-A/B fork-only (6′) · B6 snake+tetris one-shot + 1 planted bug (8′) · B7 OpenCode ×2 tasks (5′). Intake smoke = this subset, challenger-vs-champion.

---

## 9. Build phasing (every battery gates on a clean gpt-oss-20b shakedown)

| Phase | Contents | Est. |
|---|---|---|
| P0 | repo init, schema+validator, configs, integrity checks, registry SHAs frozen. **EXIT CRITERION (amended 2026-07-16, local-only): full local integrity pass green — pytest + `llmtest validate` + tables byte-clean — BEFORE any task authoring.** GitHub remote dropped by decision; `ci.yml` authored but dormant | 0.5–1 d |
| P1 | ServerManager + debug CLI; validated by Ornith ngram A/B canary (→ §7.5, re-runnable) | 0.5–1 d |
| P2 | B5 plugin (ABC client #1 — owns servers) | 1 d |
| P3 | B1 + judging end-to-end; **judge pins frozen (Michael signs off) + quota dry-run (~20 packets)** | 1.5–2 d |
| P4 | B2/B3/B4; cache-overlay validation vs OOM-reap (amendment 21) | 1.5 d |
| P5 | B6: gate (two profiles) + known-good fixtures + planted bugs | 1–1.5 d |
| P6 | B7: WSL2 Hermes + OpenCode + LiteLLM-CC, pins recorded | 1 d |
| P7 | Tier-3 seeds (shakedown-only) + intake pipeline + import-legacy | 0.5–1 d |
| P8 | **freeze tag `suite-v2.0.0`** → full T1 baseline (8–9 nights) → Verda P1–P3 → tables → report | — |

**Task authoring runs parallel to P1–P6** (the long pole: 120 B1 tasks + rubrics + B2/B3 fixtures). **Review split (amendment 23):** Michael signs off rubric **ANCHORS for all units + the calibration pair** (anchors are the scale itself) and spot-reviews MSP-core tasks (cyber, IT infra, helpdesk, KM/runbooks); the remainder is delegated. Harness investigations (Continue/Pi/Maki) are explicitly post-baseline.

---

## 10. Tier-3 module contract (two layers)

- **(a) PUBLIC vertical module** — synthetic-but-realistic tasks, committable/shippable. **(b) PRIVATE client task packs** — identical schema, separate uncommitted storage (`..\client-packs\`), run per engagement. **HARD RULE:** no client-identifiable material ever enters shared suite fixtures — synthetic analogs modeled on the work *shape* only. The suite is versioned and may be published; client documents are NDA'd. Private packs are the billable "local-LLM readiness assessment" shape — schema kept clean enough to sell.
- **Unit definitions authored in v2.0** (from the two Copilot outputs — authoring input, drop-off location TBD at P7): ag/manufacturing = production, supply chain/logistics, QA/food safety, facilities/EAM, health & safety; energy = ops module per the Caturus list. Definitions ≠ tasks; only definitions ship.
- **Seed = integration test:** each module's one seed task must exercise **every** vertical-specific schema slot — domain fixture file, vertical rubric anchors, module-tagged results row, round-trip into regenerated tables. Seeds run in **shakedown only** (gpt-oss-20b pass), tagged `seed`/`non-reportable`, never in baseline tables (n=1 module scores are noise).
- **Build-out triggers:** a module graduates to full authoring when a concrete client deployment decision needs it — ag/manufacturing likely fires first; author against the real engagement scope at that point, not speculatively.

---

## 11. Setup prerequisites & open items

1. ~~GitHub private remote~~ **DROPPED (Michael, 2026-07-16): repo stays local-git-only.** No `gh` dependency. P0 exit criterion = local integrity pass (§9). Flagged-disagreement workflow (§6.1/§7.6) tracks in `results/FLAGS.md` instead of GitHub Issues.
2. **Gemini judge access** — confirm gemini CLI (or API key) on this machine before P3.
3. **Judge pins** — live enumeration + Michael sign-off at P3 (candidates in §6.1); frozen in `judges.yaml` with CLI versions.
4. **Runtime pins in config** — fork binary build hash + Ollama version pinned alongside judge pins; sanctioned Ollama arm env in the ServerManager translation layer (§7.3).
5. **Baseline artifact freeze (all six)** — selection rule §3.5 applied at P0: on-disk SHA256 + download date recorded; disk-vs-repo mismatches recorded as findings.
6. **Node + Playwright + Chromium** — P5 dependency; SwiftShader flags for WebGL/3D probes noted in the gate spec.
7. **WSL2 vLLM env** — built and **VERSION-MATCHED to the chosen Verda image** (the laptop↔PRO-6000 calibration anchor requires same-version); confirmed before the B5 rented session.
8. **Verda account readiness** — SSH keys, quota, image choice, spot-price check (budgets.yaml) before P8's rented session.
9. **Disk audit** — ≥150 GB headroom including artifacts growth; retention/rotation policy noted (artifacts are gitignored; cold-storage sync optional).
10. **Bench-night profile** — AC power + performance plan, sleep/hibernate off, **Windows Update PAUSED for the baseline window**, no other GPU consumers; profile recorded on session rows (ties to §7.2 power_mode/ac_state fields).
11. **WSL2 harness state** — Hermes install + LiteLLM versions confirmed at P6; pins recorded then.
12. **Two Copilot outputs** (Tier-3 unit definitions) — Michael drops them at an agreed path before P7.

## 12. Decision log

Amendments 1–24 from the §1/§2 design review plus the eight seed-question resolutions (scope, judging panel, B6 gate/N, Verda budget, harness roster, tiered registry, Tier-3 layers, repo hardening) are folded inline above; the brainstorm transcript is the authoritative record.

**Approval:** TESTPLAN approved by Michael 2026-07-16, conditional on amendments **25–30** (artifact selection rule · judgment completeness/blinded anchors · §11 prerequisite additions · P0 exit criterion · mojibake lint · registry provenance) — folded in this revision. Task authoring gated on P0 exit (§9). `suite_version` for the first freeze: **suite-v2.0.0**.

**Spec version:** 2.0.0 (approved) · supersedes 2.0.0-draft1.
