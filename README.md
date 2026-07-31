# openLLMbenchmarking

An honest, reproducible benchmark of **locally-servable LLMs** — 20 models across 11
evaluation batteries, ~12,000 recorded runs, every row traceable back to the exact prompt,
the exact answer, and the serving configuration that produced it.

The point of this project is not another leaderboard. It is a harness that tries very hard
**not to lie to you** — including about itself. Blank cells mean *not run*, never a zero.
Every gap is listed with its cause and what it would cost to close. Several findings below
exist only because the harness caught itself producing numbers that looked fine and weren't.

**Start here:** open [`dashboard/index.html`](dashboard/index.html) in a browser — a
single self-contained page, no build step, no server, no network calls. To check any score
against the raw record, open [`dashboard/explorer/index.html`](dashboard/explorer/index.html)
and read what the model actually wrote.

---

## Coverage

**204 of 220 cells (92.7%)**, 12,134 rows, schema-validated clean.

| Battery | Measures | Unit | Kind |
|---|---|---|---|
| **B1** | Business scorecard — 15 units × 8 tasks × 3 reps, blinded 3-judge panel | /10 | judged |
| **B2** | Tool calling — schema, selection, argument shape | % | deterministic |
| **B3** | Hallucination resistance — unanswerable / false-premise prompts | % | deterministic |
| **B4** | Long-context retrieval — needle recall over a 16k→256k sweep | % | deterministic |
| **B5** | Serving throughput — decode t/s, spec-decode on/off arm | t/s | timing |
| **B6** | Agentic coding — 5 from-scratch + 5 planted-bug tasks | % | deterministic |
| **B7** | Reproducibility — signal agreement across a config matrix | % | deterministic |
| **B8** | Agentic harness — a real OpenCode agent in a container, 23 sealed tasks | % | deterministic |
| **B9** | Game builds — one-shot browser games, scored by *driving them* in headless Chrome | % | deterministic |
| **B10** | Security review — vulnerable/patched pairs, decoys, multi-defect chains | score | deterministic |
| **B11** | Tool loop — can the model actually drive a harness end to end | % | deterministic |

## Five things that are easy to misread

1. **Most rankings are ties.** At n=30 (B2/B6) and n=69 (B8) the Wilson intervals overlap
   heavily — the top five B8 models are one statistical tier. The B1 judge panel agrees
   within a point on only **35%** of answers (mean spread 2.45). Read tiers, not positions.

2. **B2 is a formation floor, not agentic skill.** It asks whether a well-formed tool call
   comes out. Nearly everything scores ~100%, and it cannot detect a model that confidently
   narrates work it never performed. **B11** exists because of that gap: it advertises real
   tools, executes them, and scores the **filesystem afterwards**, so narrating a command
   you never ran scores zero.

3. **B5 under-reports speculative decoding by design of its workload.** Its spec arm
   generates fresh text, where n-gram drafting almost never hits (~1.00×). On edit/rewrite
   work — what agentic coding actually is — the same flag is worth **1.95× to 12.08×**.

4. **In B10, specificity is the discriminator, not sensitivity.** Sensitivity runs near
   ceiling for nearly every model. What separates them is whether they also declare
   *already-patched* code vulnerable. A model that shouts VULNERABLE at everything scores
   100% sensitivity and is useless on a real engagement.

5. **B9 measures `runs_clean`, not "is it a good game".** A visual oracle cannot prove game
   logic advanced: a frozen snake with an animated particle layer changes more of the board
   (0.0107) than a working one (0.0049). So the gate is load / paint / animate / keys-wired /
   input-safe. Playability is human-graded in the explorer.

## Hardware is not interchangeable

Re-running one model on an A100 instead of a Blackwell card moved its deterministic scores
by up to **13 points at temperature 0** — batching and GPU numerics shift borderline
outputs. Every run in this repo records its serving configuration
(`results/sessions.jsonl`), and the roster is pinned to RTX PRO 6000 Blackwell for
comparability. Local development targets an RTX 5090 Laptop (24 GB).

## Known gaps — and why they exist

All 16 open cells have a stated cause. None is a hidden failure.

| Gap | Cells | Cause |
|---|---|---|
| **B8 on six models** | 6 | B8's completion oracle validates inside a **container**, and the rented boxes have no Docker. Running with the sandbox disabled lets the *agent* run but the *oracle* fails setup, so completion is never credited. Those rows record `oracle.detail = "hidden_validate setup failed"` and are excluded as **missing** measurements — scoring them naively produced a flat 0% for five models including `gpt-oss-120b`, which every other signal says is among the strongest here. |
| **B1 on three models** | 3 | Needs a judging pass, not GPU time. The matrix keys B1 on a *judged score*, so generating rows leaves the cell grey. |
| **`bonsai-ternary-27b` B10/B11** | 2 | Its `Q2_0` is a prism-ml custom quantization; the official llama.cpp image cannot load the file at all. Needs the prism fork built on-box. |
| **`qwen3-235b` B8–B11** | 4 | Deliberately held out pending a large-model pass. At 134 GB it needs `--cpu-moe`, which streams experts over PCIe at ~8× the cost per row. Recorded in `scripts/build_run_manifest.py` (`EXCLUDED`) and reversible by deleting one entry. |

**B4 has only ever run 7 of its 8 tasks** — for every model, including the frozen roster.
`build_document()` sizes filler with a fixed 4-chars-per-token heuristic, and one task's
filler is dense log text that really tokenizes at ~2.97, so it overflows its own tier at
every arm and the server rejects it. No row is written, so the loss is invisible unless you
check completeness against the *expected* task list. The missing task is the canonical
depth-50% "lost in the middle" needle probe.

## Repository layout

```
llmtest/            the harness: batteries, judging, scoring, schema, validation
suite/              task definitions per battery (b1_business … b11 tool loop)
config/             registry.yaml (models), suite.yaml, tiers.yaml (VRAM budgets)
scripts/            runners, run planning, provisioning, reporting
results/            append-only row store + judgments + generated tables
results_games/      B9 rows and the actual playable HTML builds
results_security/   B10 rows      results_tools/  B11 rows
results_b8_<model>/ B8 per-model sweeps
dashboard/          the coverage page and the run explorer
deploy/             box-side guards used by rented-instance runs
docs/, TESTPLAN.md  the plan of record
```

## Running it

```bash
pip install -e ".[dev]"
python -m pytest -q -m "not gpu"    # unit tests, no GPU needed
python -m llmtest validate          # schema-check every row in the store
python -m llmtest tables            # regenerate the derived tables
```

Serving standard is llama.cpp with n-gram speculative decoding on
(`--spec-type ngram-mod --spec-ngram-mod-n-match 32`) — lossless at temperature 0 and free
on edit-heavy work. Rented runs are planned and provisioned by:

```bash
python scripts/build_run_manifest.py     # what is missing, resolved against the HF API
python scripts/emit_run_plan.py          # box-side scripts, cheapest-model-first
python scripts/remaining_cost.py         # what is left and what it would cost
python scripts/rent_and_run.py --check   # verify offers and plan; rent nothing
```

The provisioner refuses any card that is not the pinned SKU, verifies it again *on the
box* before work starts, fetches each model immediately before running it and deletes it
after, pulls results every few minutes, and destroys the instance on completion. Each of
those rules exists because its absence cost something real.

## Licence / provenance

Model names are upstream Hugging Face repo identifiers. Results were produced by this
harness on an RTX 5090 Laptop (24 GB) and rented RTX PRO 6000 Blackwell instances.
