#!/usr/bin/env python3
"""Regenerate data.json (and re-embed it into index.html) for the LLM Test Bench page.

Everything with a machine-readable source is RECOMPUTED from the eval shards, so the
page can never drift from the data. Numbers that came from earlier ad-hoc laptop
sessions (before the versioned suite existed) are carried as explicitly-labelled
constants with their source file recorded, never silently mixed with suite results.

    python build_data.py            # rewrite data.json + re-embed into index.html
    python build_data.py --check    # print what would change, write nothing

Sources
  suite shards : ../llmtest-v2/results/rows-suite-v2.*.jsonl      (B2/B3/B5/B6 det + serving)
  B8 dirs      : ../llmtest-v2/results_b8_<model>/                (agentic completion)
  B1 judged    : constants below (panel-judged 0-10; see JUDGED_B1 provenance)
  ngram / MTP  : ../vast-5090-qwen36/RESULTS5-local-lowbit-codegen.md
  games        : ../benchmark-tables.md  (session-5/6 one-shot builds)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "llmtest-v2"
OUT = ROOT / "data.json"
HTML = ROOT / "index.html"

# ---------------------------------------------------------------------------
# Battery definitions (what each column actually measures)
# ---------------------------------------------------------------------------
PHASES = [
    {"id": "B1", "name": "Business Scorecard", "unit": "/10", "kind": "judged",
     "blurb": "15 business units x 8 tasks x 3 reps, scored 0-10 by a blinded 3-judge panel "
              "(Claude Fable-5 / GPT-5.6-sol / Gemini 3.1 Pro) against per-unit rubrics with "
              "CAL-strong/CAL-weak calibration anchors in every packet.",
     "sub": "by business unit"},
    {"id": "B2", "name": "Tool Calling", "unit": "%", "kind": "deterministic",
     "blurb": "Can the model emit a well-formed tool call: correct schema, right tool selected, "
              "argument shapes valid. This is a FORMATION floor, not agentic skill - most models "
              "score ~100% and it cannot detect delegation failures.",
     "sub": "by axis"},
    {"id": "B3", "name": "Hallucination Resistance", "unit": "%", "kind": "deterministic",
     "blurb": "Unanswerable / trick / false-premise prompts. Scores the 'correct' signal: did it "
              "refuse or hedge instead of fabricating. Note the 300-token answer budget starves "
              "reasoning models, which spend it on hidden thinking.",
     "sub": None},
    {"id": "B4", "name": "Long-Context Retrieval", "unit": "%", "kind": "deterministic",
     "blurb": "Needle-in-a-haystack recall across a context-length sweep (16k -> 256k). Arms are "
              "pruned per model by VRAM fit, so 100B+ models legitimately get zero arms on a "
              "single card.",
     "sub": None},
    {"id": "B5", "name": "Serving Throughput", "unit": "t/s", "kind": "deterministic",
     "blurb": "Decode tokens/sec on the datacenter box, with a speculative-decoding on/off arm. "
              "IMPORTANT: this arm generates fresh text, where n-gram spec-decode cannot help - "
              "see the Speculative Decoding panel for the edit-workload numbers.",
     "sub": None},
    {"id": "B6", "name": "Agentic Coding", "unit": "%", "kind": "deterministic",
     "blurb": "10 tasks: 5 from-scratch (is_prime, word-count CLI, backup.sh, debounce, SQL) and "
              "5 planted-bug fixes. Deterministic checks only - the judged quality axis is built "
              "but not yet run. Does NOT include the game builds (see Game Builds panel).",
     "sub": "by coding track"},
    {"id": "B7", "name": "Reproducibility", "unit": "%", "kind": "deterministic",
     "blurb": "Same prompt across a config matrix (system prompt / temperature / tool format / "
              "spec-decode). Reports how often the deterministic signals agree with the baseline "
              "cell - i.e. how much the answer moves when harness settings move.",
     "sub": None},
    {"id": "B8", "name": "Agentic Harness", "unit": "%", "kind": "deterministic",
     "blurb": "Real OpenCode agent in a disposable container: 23 sealed Python tasks across "
              "break-fix / cross-module / feature / stateful / build / robustness, scored by a "
              "hidden oracle. SINGLE-AGENT ONLY - no task requires spawning a subagent.",
     "sub": "by task category"},
    {"id": "B9", "name": "Game Builds", "unit": "%", "kind": "deterministic",
     "blurb": "One-shot browser games from a bare one-line prompt (snake, tetris, arkanoid, "
              "flappy, doodle jump, asteroids, roguelike, and a fly.pieter-style 3D flight sim), "
              "scored by DRIVING each build in headless Chrome: does it load, paint, animate, "
              "wire up keys, survive a key burst. Gameplay quality is human-graded in the "
              "explorer - a browser cannot tell 'the snake moved' from 'a particle blinked'.",
     "sub": "by game"},
    {"id": "B10", "name": "Security Review", "unit": "score", "kind": "deterministic",
     "blurb": "Authorised-pentest code review on vulnerable/patched PAIRS plus safe-but-alarming "
              "decoys. Headline is a usable-finding score = whole-chain recall x specificity, "
              "because sensitivity is ~100% for every model and the real discriminator is not "
              "inventing defects in already-fixed code. Includes a hard tier of multi-defect "
              "chains graded on how much of the chain is found.",
     "sub": "by measure"},
    {"id": "B11", "name": "Tool Loop", "unit": "%", "kind": "deterministic",
     "blurb": "Can the model actually DRIVE a harness: emit a structured tool call, read the "
              "result, act on it. The client advertises schemas and owns execution, because "
              "llama.cpp's --tools never tells the model the tools exist. Scored from the "
              "filesystem, so narrating a command you never ran scores zero.",
     "sub": "by task"},
]

# ---------------------------------------------------------------------------
# B1 panel-judged scores. Frozen 16 from results/REPORT.md (18-letter full-roster
# packets). laguna-s-2.1 was judged in a 3-letter incremental wave and CAL-RESCALED
# onto that same scale (raw 6.99 -> 6.13; judges run ~0.9pt lenient in small packets).
# ---------------------------------------------------------------------------
JUDGED_B1 = {
    "qwen3.6-27b-dense": 7.6, "ornith-1.0-35b": 7.4, "gemma-4-31b-dense": 7.4,
    "qwen3.6-35b-a3b": 7.3, "gemma-4-26b-a4b": 7.2, "qwen3-235b": 7.2,
    "bonsai-ternary-27b": 6.8, "ornith-1.0-9b": 6.7, "agents-a1-35b": 6.7,
    "gpt-oss-120b": 6.6, "granite-4.1-30b": 6.4, "glm-4.5-air": 6.3,
    "gpt-oss-20b": 6.0, "nemotron-3-nano-30b": 5.9, "llama-4-scout": 5.6,
    "qwen3-coder-30b": 5.0, "laguna-s-2.1": 6.1,
}

# ---------------------------------------------------------------------------
# n-gram speculative decoding, EDIT-HEAVY workload (rewrite a file with ~full
# context overlap), prism llama.cpp fork, temp 0 => byte-identical output.
# Source: vast-5090-qwen36/RESULTS5-local-lowbit-codegen.md (RTX 5090 Laptop 24GB).
# This is the workload the suite's B5 arm does NOT exercise.
# ---------------------------------------------------------------------------
NGRAM_EDIT = [
    {"model": "granite-4.1-30b",     "base": 31.9,  "ngram": 385.8, "note": "hybrid Mamba ~29B near-dense - biggest measured"},
    {"model": "qwen3.6-27b-dense",   "base": 31.0,  "ngram": 273.0, "note": "slow dense model gains the most per accepted draft"},
    {"model": "gemma-4-26b-a4b",     "base": 125.0, "ngram": 682.0, "note": "MoE A4B - fastest absolute edit throughput"},
    {"model": "gpt-oss-20b",         "base": 153.0, "ngram": 655.0, "note": ""},
    {"model": "ornith-1.0-35b",      "base": 153.0, "ngram": 596.0, "note": "MXFP4 A3B"},
    {"model": "nemotron-3-nano-30b", "base": 194.5, "ngram": 621.6, "note": "hybrid Mamba MoE"},
    {"model": "qwen3.6-35b-a3b",     "base": 174.0, "ngram": 340.0, "note": "gains least - matches fewer exact n-grams"},
]
NGRAM_CONTROL = {"model": "gemma-4 (from-scratch control)", "base": 158.0, "ngram": 258.0,
                 "note": "NOT an edit task - novel code still self-repeats enough for ~1.6x"}
NGRAM_NMATCH = [{"n": 8, "speedup": 0.0, "note": "SLOWER than no spec-decode - never go below 16"},
                {"n": 16, "speedup": 3.9, "note": "floor"},
                {"n": 24, "speedup": 4.98, "note": "default"},
                {"n": 32, "speedup": 5.46, "note": "best for edit-heavy work"},
                {"n": 48, "speedup": 5.4, "note": "plateau"}]

# MTP (multi-token prediction) draft heads.
MTP = [
    {"config": "gemma-4-31b-dense + separate mtp-*.gguf draft (prism fork)",
     "base": 68.0, "with": 173.0, "verdict": "works", "note": "2.55x lossless - use --spec-draft-model, NOT the embedded head"},
    {"config": "qwen3.6-27b-MTP embedded head, --spec-type draft-mtp",
     "base": 30.1, "with": 6.0, "verdict": "dead-end", "note": "0.20x = 5x SLOWER; head loads but has ~zero acceptance"},
    {"config": "qwen3.6-27b-MTP under Ollama",
     "base": 31.0, "with": 31.3, "verdict": "inert", "note": "draft head never engaged - Ollama exposes no --spec-type"},
]

# ---------------------------------------------------------------------------
# One-shot GAME builds. Ad-hoc laptop sessions 5/6 (source: benchmark-tables.md),
# graded by blind code review - NOT a scored battery, NOT all models, no replicates.
# ---------------------------------------------------------------------------
GAMES_HISTORICAL = [
    {"model": "ornith-1.0-35b",    "snake": "pass", "snake_note": "clean",
     "tetris": "partial", "tetris_note": "richest build (hold/ghost/7-bag); 1 edge bug - forces invalid rotation when no wall-kick fits"},
    {"model": "gemma-4-26b-a4b",   "snake": "pass", "snake_note": "clean",
     "tetris": "partial", "tetris_note": "plays correctly but game-over does not halt - re-arms the rAF loop it just cancelled"},
    {"model": "bonsai-ternary-27b", "snake": "pass", "snake_note": "clean, most polished (particles, screen-shake, mobile d-pad)",
     "tetris": "partial", "tetris_note": "great logic (7-bag, wall-kicks); 2 render bugs - falling piece never drawn, bevel covers board"},
    {"model": "qwen3.6-27b-dense", "snake": "pass", "snake_note": "clean",
     "tetris": "pass", "tetris_note": "1 harmless issue"},
    {"model": "qwen3.6-27b-mtp",   "snake": "pass", "snake_note": "clean",
     "tetris": "partial", "tetris_note": "buggy sample"},
    {"model": "bonsai-1bit-27b",   "snake": "fail", "snake_note": "dead loop - per-frame dt vs 150ms tick, tickAccum unused, snake never advances",
     "tetris": "fail", "tetris_note": "crashes on start - SHAPES_keys typo for SHAPES_KEYS; drop timer also dead"},
]
GAMES_PLANNED = [
    {"game": "Snake", "state": "ad-hoc only"},
    {"game": "Tetris", "state": "ad-hoc only"},
    {"game": "Arkanoid", "state": "not built"},
    {"game": "Flappy Bird", "state": "not built"},
    {"game": "Doodle Jump (procgen platforms)", "state": "not built"},
    {"game": "Asteroids (procgen field, splitting rocks)", "state": "not built"},
    {"game": "Tiny roguelike (procgen dungeon)", "state": "not built"},
    {"game": "fly.pieter-style 3D flight sim (Three.js, procgen city, single HTML file)", "state": "not built"},
]

# ---------------------------------------------------------------------------
# Known gaps - what the numbers on this page do NOT cover.
# ---------------------------------------------------------------------------
GAPS = [
    {"sev": "high", "title": "B4 has only ever run 7 of its 8 tasks - the classic single-needle probe is missing",
     "detail": "b4.single-needle-01 has ZERO rows for all 16 roster models; the other seven B4 "
               "tasks have 49 each. build_document() sizes the filler with a 4-chars-per-token "
               "heuristic, and that task's filler is dense operational log text (timestamps, asset "
               "IDs, digit groups) that really tokenizes at ~2.97 chars/token - a 1.35x overshoot. "
               "Every arm therefore overflows its own tier and the server rejects the request: "
               "20716 vs 16384, 86644 vs 65536, 174563 vs 131072, 350408 vs 262144. No row is "
               "written, so the loss is invisible unless task-level completeness is checked. The "
               "missing task is the canonical 'lost in the middle' needle-in-a-haystack probe at "
               "depth 50%, which is the single most standard thing B4 claims to measure.",
     "fix": "Size the document by real tokenization (or measure-and-trim) instead of a fixed "
            "chars/token ratio. Note this re-bases B4: existing rows cover 7 tasks, so a fixed "
            "run is not directly comparable to the frozen roster until every model is re-run."},
    {"sev": "high", "title": "Coverage is ragged across the three newest batteries",
     "detail": "B9 (games) ran for 12 models but 4 of those have partial rows, and the four "
               "largest models plus laguna have none at all - 96 completed rows were lost when a "
               "rented box was left idle, ran out of credit and could not be restarted. B10 "
               "(security) covers 6 models of 20. B11 (tool loop) covers 1. Every blank cell in "
               "the matrix is genuinely unrun, never a zero - but the newer columns are far "
               "thinner than B1-B7 and should not be read as a roster-wide ranking yet.",
     "fix": "One rented session per battery to fill the roster; pull results per phase, not at "
            "the end."},
    {"sev": "high", "title": "Subagent delegation is deliberately unscored - and the canary never fires",
     "detail": "TESTPLAN 5.7 excludes the subagent axis from scoring ON PURPOSE - 'documented 0% "
               "with local models - non-differentiating' - keeping it as one unscored canary. The "
               "consequence still matters: every B8 row records subagent_spawned = 'no', so B8 is "
               "SINGLE-AGENT completion only. B11 now covers the related question (can the model "
               "drive a tool loop at all) and the answer is yes, but that is a client-owned loop, "
               "not model-initiated delegation.",
     "fix": "Re-test the axis periodically; the 0% result predates the current model generation."},
    {"sev": "high", "title": "B10's hard tier reverses its own base tier - so neither alone is safe to quote",
     "detail": "On single-defect textbook cases every model scores at or near 100% and the base "
               "tier cannot separate them. On multi-defect chains the ordering changes outright: "
               "abl-gemma-4-31b is perfect on the base tier and worst on chains (25% whole-chain), "
               "while abl-qwen3.6-27b leads. Quoting the base tier alone would have produced - and "
               "did produce - the wrong recommendation.",
     "fix": "Always report the hard tier alongside; treat base-tier scores as a floor, not a rank."},
    {"sev": "medium", "title": "The abliteration A/Bs are confounded by quantisation",
     "detail": "Abliterated builds beat their bases in both families (qwen3.6-27b 25%->75% "
               "whole-chain, gemma-4-31b 66%->100% base specificity), which is the opposite of the "
               "usual assumption. But the abliterated files are Q4_K while the bases are Q5_K_M, so "
               "part of that delta could be quantisation rather than abliteration.",
     "fix": "Re-run one pair at matched quant before treating the abliteration gain as established."},
    {"sev": "medium", "title": "Small samples - most rankings are statistical ties",
     "detail": "B2/B6 run n=30 per model, B8's sweep n=69, B10's hard tier n=12 per model. Wilson "
               "intervals on whole-chain recall span roughly +/-25 points, so abl-qwen3.6-27b vs "
               "gpt-oss-120b is NOT a separated result. The matrix marks ties with a tilde.",
     "fix": "More replicates on the tasks that actually discriminate, rather than more tasks."},
    {"sev": "medium", "title": "Judged axes built and never run",
     "detail": "B6's 0-10 code-quality axis has 510 generated rows waiting and is not wired into "
               "JUDGED_BATTERIES; B2's error-recovery and faithfulness axes are wired but have "
               "never been executed. Both would add discrimination to batteries currently sitting "
               "at a ceiling, and both need judge quota rather than GPU.",
     "fix": "One judging pass each, when judge budget allows."},
    {"sev": "medium", "title": "Games are scored for 'runs clean', not for being good games",
     "detail": "The browser oracle can prove a build loads, paints, animates, wires keys and "
               "survives input. It CANNOT tell that the snake advanced - validated the hard way: a "
               "frozen snake whose particle layer animates changes more of the board than a working "
               "one. Gameplay quality is therefore human-graded in the explorer.",
     "fix": "Per-game probes exist for the authored snake fixture; extend them to tetris and the "
            "rest if scored gameplay is wanted."},
    {"sev": "medium", "title": "The suite under-reports speculative decoding",
     "detail": "B5's spec-decode arm measures ~1.00x for every model because it generates fresh "
               "text, where n-gram drafting almost never hits. On edit/rewrite work - what agentic "
               "coding actually does - the same feature is worth 1.95x to 12.08x. Laguna also ran "
               "with no acceleration at all, and its purpose-built DFlash draft could not be loaded "
               "by upstream llama.cpp (wrong tensor count - it needs poolside's fork).",
     "fix": "Give B5 an edit-workload arm; test DFlash on poolside's fork."},
    {"sev": "low", "title": "The quant-format A/B was never actually run",
     "detail": "gemma-4-26b-a4b-mxfp4 exists as a controlled quant arm ('runs B5 + B2 + B6') but "
               "produced B8 rows only. Worse, that B8 data is what the roster model's agentic score "
               "uses, so gemma-4-26b-a4b's B8 is the MXFP4 quant while its B1-B7 are UD-Q4_K_XL - "
               "one row mixing two quants.",
     "fix": "Run the mxfp4 arm through B2/B5/B6, or drop the arm and re-run B8 on UD-Q4_K_XL."},
    {"sev": "low", "title": "Judges agree with each other only 35% of the time",
     "detail": "Across 6,120 judged answers the 3-seat panel lands within 1 point of itself on just "
               "35.1%; mean spread is 2.45 points. A B1 gap of a few tenths is inside judge noise. "
               "Gemini scores its own family +0.53 higher than others; codex scores its own -0.67.",
     "fix": "Publish per-model score intervals from the judge spread; read B1 as tiers."},
    {"sev": "low", "title": "Laguna has no B5 / B7 / B8, and its B1 is a rescaled incremental wave",
     "detail": "B5 and B7 were skipped (box-specific / needs the fork's spec arm) and B8 postdates "
               "its peer group. Its B1 6.1 comes from a 3-letter incremental packet rescaled through "
               "the CAL anchors rather than a full-roster packet - defensible, one step further from "
               "the frozen sixteen.",
     "fix": "Fold Laguna into the next full-roster judging wave."},
]

CAVEATS = {
    "hardware": "B1-B7 for the original 16 models ran on 2x RTX PRO 6000 (Blackwell). Laguna ran "
                "on a single rented RTX PRO 6000 Blackwell. The B8 agentic sweep ran locally on an "
                "RTX 5090 Laptop 24GB. Hardware is NOT interchangeable: re-running Laguna on an "
                "A100 (Ampere) moved its deterministic scores by up to 13 points (B6 87 -> 100, "
                "B2 97 -> 90) at temperature 0, because batching and GPU numerics shift borderline "
                "outputs.",
    "ngram_workload": "n-gram speculative decoding is lossless (temp-0 output is byte-identical) "
                      "and costs no VRAM, but its speedup depends entirely on how much the output "
                      "repeats the context. Edit/rewrite: 2-12x. Fresh generation: ~1.0-1.6x.",
    "b1_incremental": "Laguna's B1 was judged incrementally (only Laguna + calibration anchors, "
                      "leaving the frozen 16 untouched) and then rescaled: the same fixed anchors "
                      "score 9.0/1.5 in a small packet vs 8.0/1.0 in the full-roster packet, so "
                      "judges are ~0.9pt more lenient without the comparison set. Raw 6.99 -> 6.13.",
}


def wilson(k, n, z=1.96):
    """Wilson score interval as (lo%, hi%). The suite's per-model n is small
    (30 for B2/B6, 69 for the B8 sweep), so a bare percentage badly overstates
    precision - every rate on the page carries this."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0.0, c - m), 1), round(100 * min(1.0, c + m), 1))


def mean_ci(vals, z=1.96):
    """Normal-approx CI for a mean of per-row rates (B4 recall, B7 agreement)."""
    n = len(vals)
    if n < 2:
        return None
    mu = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(n)
    scale = 100 if max(vals, default=0) <= 1.0 else 1
    return (round(scale * max(0.0, mu - z * se), 1), round(scale * min(1.0 if scale == 100 else 1e9,
                                                                      mu + z * se), 1))


def mark_ties(models, matrix):
    """Per battery, flag every model whose CI overlaps the leader's CI. Those
    models are NOT distinguishable at 95% and must not be read as ranked."""
    for p in PHASES:
        pid = p["id"]
        scored = [(m, matrix[m][pid]) for m in models
                  if matrix[m].get(pid, {}).get("score") is not None]
        if not scored:
            continue
        lead_m, lead = max(scored, key=lambda t: t[1]["score"])
        lead_ci = lead.get("ci")
        for m, c in scored:
            ci = c.get("ci")
            if lead_ci and ci:
                c["tied_with_leader"] = not (ci[1] < lead_ci[0] or ci[0] > lead_ci[1])
            else:
                c["tied_with_leader"] = None
        matrix[lead_m][pid]["is_leader"] = True


def compute_b2_axes(rows):
    """B2 per-axis pass rates. The aggregate hides the real discriminators:
    a model can sit at 90% overall while scoring 0/3 on parallel calls."""
    ax = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        if r.get("battery") != 2:
            continue
        axis = r.get("task_id", "").split(".")[-1].rsplit("-", 1)[0]
        ok = det_pass(r)
        if ok is None:
            continue
        d = ax[r["model_id"]][axis]
        d[1] += 1
        d[0] += 1 if ok else 0
    return {m: [{"name": a, "score": round(100 * v[0] / v[1]),
                 "display": f"{v[0]}/{v[1]}"} for a, v in sorted(axes.items())]
            for m, axes in ax.items()}


def compute_prefill(rows):
    """Prompt-processing throughput. Agentic prompts run ~13k tokens, so prefill
    often dominates wall-clock even though only decode t/s gets quoted."""
    pp = collections.defaultdict(list)
    for r in rows:
        if r.get("battery") != 5:
            continue
        v = (r.get("metrics") or {}).get("pp_tps")
        if v:
            pp[r["model_id"]].append(float(v))
    return {m: round(statistics.median(v)) for m, v in pp.items()}


def compute_b7_drift(rows):
    """Which harness knob actually moves the answer. Lower agreement = that
    setting destabilises output more."""
    dims = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r.get("battery") != 7:
            continue
        parts = dict(p.split("=", 1) for p in r.get("condition", "").split(";") if "=" in p)
        sig = (r.get("det_checks") or {}).get("signal_agreement_vs_baseline")
        v = sig.get("value") if isinstance(sig, dict) else None
        if v is None and isinstance(sig, dict) and "pass" in sig:
            v = 1.0 if sig["pass"] else 0.0
        if v is None:
            continue
        for k in ("sysp", "temp", "toolfmt", "spec"):
            if k in parts:
                d = dims[f"{k}={parts[k]}"]
                d[0] += float(v)
                d[1] += 1
    out = [{"cell": k, "agreement": round(100 * s / n, 1), "n": n}
           for k, (s, n) in dims.items() if n]
    return sorted(out, key=lambda x: x["agreement"])


def compute_failure_classes():
    """B8 first-failure classes: WHY a run failed, which is more actionable than
    the completion rate. a=schema-never-parsed b=parsed-but-tool-misused
    c=task-logic-wrong d=harness-bug e=budget/step-exhausted."""
    LABEL = {"a": "schema never parsed", "b": "tool parsed but misused",
             "c": "task logic wrong", "d": "harness bug", "e": "budget/steps exhausted",
             "unknown": "unclassified"}
    per = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for p in (REPO / "results").glob("b8_classifications*.jsonl"):
        for line in p.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            lab = r.get("label") or "unknown"
            total[lab] += 1
            mid = r.get("model_id") or "(unattributed)"
            per[mid][lab] += 1
    return {"labels": LABEL,
            "total": [{"label": k, "name": LABEL.get(k, k), "n": v}
                      for k, v in total.most_common()],
            "per_model": {m: [{"label": k, "name": LABEL.get(k, k), "n": v}
                              for k, v in c.most_common()] for m, c in per.items()}}


def compute_judge_reliability():
    """How trustworthy the judged B1 numbers are: panel agreement, score spread,
    and kin_delta (does a judge score its own family higher?)."""
    maps = {}
    for p in (REPO / "results" / "packets").glob("*.map.json"):
        try:
            maps[p.name.split(".")[0]] = json.load(p.open(encoding="utf-8"))
        except Exception:
            continue
    groups = collections.defaultdict(dict)
    for p in (REPO / "results").glob("judgments*.jsonl"):
        for line in p.open(encoding="utf-8"):
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("status") != "ok":
                continue
            groups[(j["packet_id"], j.get("model_id"))][j["judge_id"]] = j.get("score")
    spreads, per_model = [], collections.defaultdict(list)
    kin = collections.defaultdict(lambda: collections.defaultdict(list))
    try:
        kin_map = (yaml.safe_load((REPO / "config" / "judges.yaml").read_text(encoding="utf-8"))
                   or {}).get("kin_map") or {}
    except Exception:
        kin_map = {}
    for (pid, mid), sc in groups.items():
        vals = [v for v in sc.values() if isinstance(v, (int, float))]
        is_cal = str(mid).startswith("CAL-")
        if len(vals) >= 2 and not is_cal:
            spreads.append(max(vals) - min(vals))   # real answers only; CAL anchors
                                                    # are deliberately extreme and
                                                    # would inflate apparent agreement
        if mid and not str(mid).startswith("CAL-"):
            per_model[mid].append(statistics.median(vals) if vals else None)
        for jid, v in sc.items():
            if isinstance(v, (int, float)) and mid:
                kin[jid]["kin" if kin_map.get(mid) == jid else "other"].append(v)
    agree = 100 * sum(1 for s in spreads if s <= 1) / len(spreads) if spreads else None
    return {
        "packets_with_panel": len(spreads),
        "agreement_pct": round(agree, 1) if agree is not None else None,
        "mean_spread": round(statistics.mean(spreads), 2) if spreads else None,
        "spread_gt2_pct": round(100 * sum(1 for s in spreads if s > 2) / len(spreads), 1)
                          if spreads else None,
        "kin_delta": [{"judge": j,
                       "kin_mean": round(statistics.mean(v["kin"]), 2) if v.get("kin") else None,
                       "other_mean": round(statistics.mean(v["other"]), 2) if v.get("other") else None,
                       "delta": round(statistics.mean(v["kin"]) - statistics.mean(v["other"]), 2)
                                if v.get("kin") and v.get("other") else None}
                      for j, v in sorted(kin.items())],
    }


def compute_efficiency(models, matrix):
    """Quality per GB of weights - the axis that decides what you actually run on
    a 24GB card. A 6.7GB model at 6.8/10 is a different proposition to an 18GB
    model at 7.4/10."""
    try:
        reg = (yaml.safe_load((REPO / "config" / "registry.yaml").read_text(encoding="utf-8"))
               or {}).get("models") or {}
    except Exception:
        return []
    out = []
    for m in models:
        gb = reg.get(m, {}).get("weights_gb")
        b1 = matrix[m].get("B1", {}).get("score")
        b8 = matrix[m].get("B8", {}).get("score")
        if not gb:
            continue
        out.append({"model": m, "gb": gb,
                    "b1": b1, "b8": b8,
                    "b1_per_gb": round(b1 / gb, 3) if b1 else None,
                    "fits_24gb": gb + 2.5 <= 23.5,
                    "arch": "MoE" if reg.get(m, {}).get("arch", {}).get("moe") else "dense",
                    "quant": reg.get(m, {}).get("quant_family", "")})
    return sorted(out, key=lambda r: -(r["b1_per_gb"] or 0))


def compute_quant_ab(rows):
    """The controlled quant-format experiment (same google QAT base, UD-Q4_K_XL vs
    MXFP4_MOE container). The registry says the arm 'runs B5 + B2 + B6', but only
    B8 rows were ever produced - so the comparison CANNOT be made, and worse, the
    only mxfp4 data (B8) is what the roster model's agentic score is drawn from,
    mixing quants inside one model's row. Report that state rather than a fake delta."""
    pair = ("gemma-4-26b-a4b", "gemma-4-26b-a4b-mxfp4")
    have = {mid: sorted({r.get("battery") for r in rows
                         if r.get("model_id") == mid and r.get("battery")})
            for mid in pair}
    comparable = sorted(set(have[pair[0]]) & set(have[pair[1]]))
    return {"pair": list(pair), "batteries_run": have, "comparable": comparable,
            "note": ("Only B8 exists for the MXFP4 arm and nothing else, so no "
                     "quant-format delta can be computed. Note the roster model's "
                     "B8 figure is the MXFP4 quant while its B1-B7 figures are "
                     "UD-Q4_K_XL - the one row mixes two quants.")}


def _load(path):
    out = []
    p = REPO / path
    if p.exists():
        for line in p.open(encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def compute_b9(matrix):
    """Games: share of builds that run clean when driven in a real browser."""
    rows = _load("results_games/rows-games.jsonl")
    tally = collections.defaultdict(lambda: [0, 0])
    per_game = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        m = r["model_id"]
        ok = bool((r.get("metrics") or {}).get("runs_clean"))
        tally[m][1] += 1
        tally[m][0] += ok
        g = r.get("task_id", "").split(".")[-1]
        per_game[m][g][1] += 1
        per_game[m][g][0] += ok
    for m, kn in tally.items():
        k, n = kn
        if m in matrix and n:
            matrix[m]["B9"] = {"tested": True, "n": n, "score": round(100 * k / n),
                               "display": "%d%%" % round(100 * k / n), "k": k,
                               "ci": wilson(k, n),
                               "sub": [{"name": g, "score": round(100 * v[0] / v[1]),
                                        "display": "%d/%d" % (v[0], v[1])}
                                       for g, v in sorted(per_game[m].items())]}


def compute_b10(matrix):
    """Security: headline is whole-chain recall x specificity - the two axes that decide
    whether a finding is usable. Sensitivity is a sub-score because it is ~100% for
    everything and therefore discriminates nothing."""
    rows = _load("results_security/rows-security.jsonl")
    agg = collections.defaultdict(lambda: {"sens": [0, 0], "spec": [0, 0], "dec": [0, 0],
                                           "chain": [0, 0], "whole": [0, 0], "ref": 0, "n": 0})
    for r in rows:
        a = agg[r["model_id"]]
        met = r.get("metrics") or {}
        det = r.get("det_checks") or {}
        ok = bool(det.get("correct_verdict", {}).get("pass"))
        a["n"] += 1
        if met.get("refused"):
            a["ref"] += 1
        if met.get("tier") == "hard":
            if met.get("expect_vulnerable"):
                a["chain"][0] += met.get("found_n", 0) or 0
                a["chain"][1] += met.get("chain_total", 0) or 0
                a["whole"][1] += 1
                a["whole"][0] += 1 if det.get("found_whole_chain", {}).get("pass") else 0
            else:
                a["spec"][1] += 1
                a["spec"][0] += 1 if ok else 0
        else:
            if met.get("expect_vulnerable"):
                a["sens"][1] += 1
                a["sens"][0] += 1 if ok else 0
            elif "decoy" in r.get("task_id", ""):
                a["dec"][1] += 1
                a["dec"][0] += 1 if ok else 0
            else:
                a["spec"][1] += 1
                a["spec"][0] += 1 if ok else 0

    def pct(x):
        return "%d%%" % round(100 * x[0] / x[1]) if x[1] else "-"

    for m, a in agg.items():
        if m not in matrix or not a["n"]:
            continue
        wc = (a["whole"][0] / a["whole"][1]) if a["whole"][1] else None
        sp_n = a["spec"][1] + a["dec"][1]
        sp = ((a["spec"][0] + a["dec"][0]) / sp_n) if sp_n else None
        score = round(100 * wc * sp) if (wc is not None and sp is not None) else None
        matrix[m]["B10"] = {
            "tested": True, "n": a["n"], "score": score,
            "display": (str(score) if score is not None else "-"),
            "ci": wilson(a["whole"][0], a["whole"][1]) if a["whole"][1] else None,
            "sub": [
                {"name": "whole-chain", "score": round(100 * wc) if wc is not None else 0,
                 "display": pct(a["whole"])},
                {"name": "specificity", "score": round(100 * sp) if sp is not None else 0,
                 "display": ("%d%%" % round(100 * sp)) if sp is not None else "-"},
                {"name": "chain recall",
                 "score": round(100 * a["chain"][0] / a["chain"][1]) if a["chain"][1] else 0,
                 "display": pct(a["chain"])},
                {"name": "sensitivity",
                 "score": round(100 * a["sens"][0] / a["sens"][1]) if a["sens"][1] else 0,
                 "display": pct(a["sens"])},
                {"name": "decoys",
                 "score": round(100 * a["dec"][0] / a["dec"][1]) if a["dec"][1] else 0,
                 "display": pct(a["dec"])},
                {"name": "refusals", "score": 100 if a["ref"] == 0 else 0,
                 "display": str(a["ref"])},
            ]}


def compute_b11(matrix):
    """Tool loop: task completion, verified from the filesystem rather than narration."""
    rows = _load("results_tools/rows-tools.jsonl")
    tally = collections.defaultdict(lambda: [0, 0])
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    calls = collections.defaultdict(list)
    for r in rows:
        m = r["model_id"]
        ok = bool((r.get("metrics") or {}).get("completed"))
        tally[m][1] += 1
        tally[m][0] += ok
        t = r.get("task_id", "").split(".")[-1]
        per[m][t][1] += 1
        per[m][t][0] += ok
        calls[m].append((r.get("metrics") or {}).get("n_tool_calls", 0) or 0)
    for m, kn in tally.items():
        k, n = kn
        if m in matrix and n:
            cs = sorted(calls[m])
            med = cs[len(cs) // 2] if cs else 0
            matrix[m]["B11"] = {"tested": True, "n": n, "score": round(100 * k / n),
                                "display": "%d%%" % round(100 * k / n), "k": k,
                                "ci": wilson(k, n), "median_tool_calls": med,
                                "sub": [{"name": t, "score": round(100 * v[0] / v[1]),
                                         "display": "%d/%d" % (v[0], v[1])}
                                        for t, v in sorted(per[m].items())]}


def rows_from_shards():
    out = []
    for p in sorted((REPO / "results").glob("rows-suite-v2.*.jsonl")):
        if "shakedown" in p.name:
            continue
        for line in p.open(encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def det_pass(row):
    """True when every deterministic check on the row passed."""
    checks = [v["pass"] for v in (row.get("det_checks") or {}).values()
              if isinstance(v, dict) and "pass" in v]
    return all(checks) if checks else None


def compute_matrix():
    rows = rows_from_shards()
    models = sorted({r["model_id"] for r in rows if r.get("battery")})
    agg = {m: {} for m in models}

    # --- B2 / B6: all-checks-pass rate; B3: the 'correct' signal ---
    for bat, key in ((2, None), (6, None), (3, "correct")):
        tally = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r.get("battery") != bat:
                continue
            if key:
                c = (r.get("det_checks") or {}).get(key)
                ok = c.get("pass") if isinstance(c, dict) else None
            else:
                ok = det_pass(r)
            if ok is None:
                continue
            t = tally[r["model_id"]]
            t[1] += 1
            t[0] += 1 if ok else 0
        for m, (p, n) in tally.items():
            agg[m][f"B{bat}"] = {"tested": True, "n": n, "score": round(100 * p / n),
                                 "display": f"{round(100*p/n)}%", "sub": [],
                                 "k": p, "ci": wilson(p, n)}

    # --- B4: needle recall ---
    b4 = collections.defaultdict(list)
    for r in rows:
        if r.get("battery") == 4:
            v = (r.get("metrics") or {}).get("needle_recall")
            if v is not None:
                b4[r["model_id"]].append(float(v))
    for m, v in b4.items():
        sc = 100 * statistics.mean(v) if max(v) <= 1.0 else statistics.mean(v)
        agg[m]["B4"] = {"tested": True, "n": len(v), "score": round(sc),
                        "display": f"{round(sc)}%", "sub": [], "ci": mean_ci(v)}

    # --- B5: decode t/s, split by spec arm (this is where ngram shows ~1.0x) ---
    b5 = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("battery") != 5:
            continue
        met = r.get("metrics") or {}
        tps = met.get("decode_tps") or met.get("predicted_per_second") or met.get("tps")
        if not tps:
            continue
        arm = "ngram" if "spec=ngram" in r.get("condition", "") else \
              ("off" if "spec=off" in r.get("condition", "") else "other")
        b5[r["model_id"]][arm].append(float(tps))
    b5_arms = {}
    for m, arms in b5.items():
        ng = arms.get("ngram") or []
        off = arms.get("off") or []
        best = statistics.median(ng or off)
        agg[m]["B5"] = {"tested": True, "n": len(ng) + len(off), "score": round(best),
                        "display": f"{round(best)} t/s", "sub": [],
                        "ci": None}   # t/s median over n=8; CI not meaningful, see gaps
        if ng and off:
            b5_arms[m] = {"off": round(statistics.median(off), 1),
                          "ngram": round(statistics.median(ng), 1),
                          "speedup": round(statistics.median(ng) / statistics.median(off), 2)}

    # --- B7: agreement vs baseline ---
    b7 = collections.defaultdict(list)
    for r in rows:
        if r.get("battery") != 7:
            continue
        sig = (r.get("det_checks") or {}).get("signal_agreement_vs_baseline")
        val = sig.get("value") if isinstance(sig, dict) else None
        if val is None and isinstance(sig, dict) and "pass" in sig:
            val = 1.0 if sig["pass"] else 0.0
        if val is not None:
            b7[r["model_id"]].append(float(val))
    for m, v in b7.items():
        sc = 100 * statistics.mean(v) if max(v) <= 1.0 else statistics.mean(v)
        agg[m]["B7"] = {"tested": True, "n": len(v), "score": round(sc),
                        "display": f"{round(sc)}%", "sub": [], "ci": mean_ci(v)}

    # --- B1: judged panel scores + per-unit breakdown from the shard's units ---
    unit_of = {}
    for r in rows:
        if r.get("battery") == 1:
            tid = r.get("task_id", "")
            unit_of[tid] = tid.split(".")[1].rsplit("-", 1)[0] if "." in tid else tid
    for m, sc in JUDGED_B1.items():
        if m in agg:
            agg[m]["B1"] = {"tested": True, "n": sum(1 for r in rows if r.get("battery") == 1
                                                     and r["model_id"] == m),
                            "score": sc, "display": f"{sc}", "sub": []}

    # --- B6 sub-scores by track (scratch vs bugfix) ---
    for m in models:
        tracks = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r.get("battery") != 6 or r["model_id"] != m:
                continue
            ok = det_pass(r)
            if ok is None:
                continue
            tr = "from-scratch" if "scratch" in r.get("task_id", "") else "bug-fix"
            tracks[tr][1] += 1
            tracks[tr][0] += 1 if ok else 0
        if tracks and "B6" in agg[m]:
            agg[m]["B6"]["sub"] = [{"name": k, "score": round(100 * v[0] / v[1]),
                                    "display": f"{round(100*v[0]/v[1])}%"}
                                   for k, v in sorted(tracks.items())]

    # --- B8: completion from the per-model sweep dirs ---
    CATS = {"brk": "break-fix", "xmod": "cross-module", "feat": "feature",
            "state": "stateful", "build": "build/migration", "robust": "robustness",
            "hard": "hard"}
    # B8 rows live in two places: the per-model sweep dirs (results_b8_<model>/) and
    # the committed shard (the first 2-model run + the dev/confirmatory waves).
    # The mxfp4 quant-arm id carries gemma's agentic result -> fold onto the roster id.
    ARM_TO_ROSTER = {"gemma-4-26b-a4b-mxfp4": "gemma-4-26b-a4b"}
    b8_tally = collections.defaultdict(lambda: [0, 0])
    b8_cats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    b8_seen = collections.defaultdict(set)          # dedupe by row_id across sources

    def take_b8(r):
        if r.get("battery") != 8:
            return
        met = r.get("metrics") or {}
        if met.get("terminal_status") == "infra-error":
            return              # eligibility rule: infra errors are not model failures
        mid = ARM_TO_ROSTER.get(r.get("model_id"), r.get("model_id"))
        rid = r.get("row_id")
        if rid and rid in b8_seen[mid]:
            return
        if rid:
            b8_seen[mid].add(rid)
        done = bool(met.get("completion"))
        b8_tally[mid][1] += 1
        b8_tally[mid][0] += 1 if done else 0
        mt = re.search(r"py-([a-z]+)-", r.get("task_id", ""))
        cat = CATS.get(mt.group(1), mt.group(1)) if mt else "other"
        b8_cats[mid][cat][1] += 1
        b8_cats[mid][cat][0] += 1 if done else 0

    for f in sorted(REPO.glob("results_b8_*/*.jsonl")):
        for line in f.open(encoding="utf-8"):
            try:
                take_b8(json.loads(line))
            except Exception:
                continue
    for r in rows:                                   # committed shard
        take_b8(r)

    for mid, (p, n) in b8_tally.items():
        if not n or mid not in agg:
            continue
        agg[mid]["B8"] = {
            "tested": True, "n": n, "score": round(100 * p / n),
            "display": f"{round(100*p/n)}%", "k": p, "ci": wilson(p, n),
            "sub": [{"name": k, "score": round(100 * v[0] / v[1]),
                     "display": f"{round(100*v[0]/v[1])}%"}
                    for k, v in sorted(b8_cats[mid].items())]}

    # fill blanks
    for m in models:
        for p in PHASES:
            agg[m].setdefault(p["id"], {"tested": False, "n": None, "score": None,
                                        "display": None, "sub": []})
    return models, agg, b5_arms


def b1_units_for(model, rows=None):
    """Per-unit B1 detail is only published for Laguna (computed during its judging
    wave); the frozen 16 keep the report's overall score."""
    if model != "laguna-s-2.1":
        return []
    vals = [("operations", 7.2), ("data_analytics", 7.1), ("hr_people_ops", 6.9),
            ("finance", 6.7), ("sales", 6.5), ("marketing", 6.4), ("seo", 6.3),
            ("helpdesk", 6.3), ("outreach", 6.1), ("knowledge_mgmt", 6.1),
            ("coding", 6.1), ("project_mgmt", 5.7), ("cybersecurity", 5.6),
            ("legal_compliance", 5.0), ("it_infra", 3.9)]
    return [{"name": u, "score": s, "display": f"{s}"} for u, s in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    models, matrix, b5_arms = compute_matrix()
    # Models that appear ONLY in the newer batteries (the abliterated security arms
    # ran B10/B11 but none of B1-B8) would otherwise be silently dropped from the
    # roster - the exact "gap" this report is meant to surface.
    extra = set()
    for shard in ("results_games/rows-games.jsonl", "results_security/rows-security.jsonl",
                  "results_tools/rows-tools.jsonl"):
        for r in _load(shard):
            extra.add(r.get("model_id"))
    for m in sorted(x for x in extra if x and x not in matrix):
        matrix[m] = {}
        models.append(m)
    # drop the quant-arm pseudo-model from the roster view
    models = [m for m in models if m != "gemma-4-26b-a4b-mxfp4"]
    for m in models:
        if matrix[m].get("B1", {}).get("tested"):
            matrix[m]["B1"]["sub"] = b1_units_for(m)

    rows_all = rows_from_shards()
    b2ax = compute_b2_axes(rows_all)
    for m in models:
        if matrix[m].get("B2", {}).get("tested") and m in b2ax:
            matrix[m]["B2"]["sub"] = b2ax[m]
    compute_b9(matrix)
    compute_b10(matrix)
    compute_b11(matrix)
    mark_ties(models, matrix)
    prefill = compute_prefill(rows_all)
    for m, v in prefill.items():
        if m in matrix and matrix[m].get("B5", {}).get("tested"):
            matrix[m]["B5"]["prefill_tps"] = v

    data = {
        "generated_from": "llmtest-v2 results shards + labelled session constants",
        "phases": PHASES,
        "models": models,
        "matrix": {m: matrix[m] for m in models},
        "ngram_edit": [dict(r, speedup=round(r["ngram"] / r["base"], 2)) for r in
                       sorted(NGRAM_EDIT, key=lambda r: -(r["ngram"] / r["base"]))],
        "ngram_control": dict(NGRAM_CONTROL,
                              speedup=round(NGRAM_CONTROL["ngram"] / NGRAM_CONTROL["base"], 2)),
        "ngram_nmatch": NGRAM_NMATCH,
        "b5_spec_arms": b5_arms,
        "mtp": MTP,
        "games_historical": GAMES_HISTORICAL,
        "games_planned": GAMES_PLANNED,
        "gaps": GAPS,
        "caveats": CAVEATS,
        "prefill": prefill,
        "b7_drift": compute_b7_drift(rows_all),
        "failure_classes": compute_failure_classes(),
        "judge_reliability": compute_judge_reliability(),
        "efficiency": compute_efficiency(models, matrix),
        "quant_ab": compute_quant_ab(rows_all),
    }

    payload = json.dumps(data, indent=1, ensure_ascii=True)
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        print(f"models={len(models)} b5_arms={len(b5_arms)} "
              f"changed={'yes' if old != payload else 'no'}")
        return 0

    OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(models)} models, {len(b5_arms)} B5 spec arms")

    if HTML.exists():
        html = HTML.read_text(encoding="utf-8")
        new, n = re.subn(r"const DATA = /\* build \*/.*?;\n",
                         "const DATA = /* build */ " + json.dumps(data, ensure_ascii=True) + ";\n",
                         html, count=1, flags=re.S)
        if n:
            HTML.write_text(new, encoding="utf-8")
            print(f"re-embedded DATA into {HTML.name}")
        else:
            print("WARNING: DATA marker not found in index.html - not embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
