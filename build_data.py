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
import re
import statistics
from pathlib import Path

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
     "sub": None},
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
    {"sev": "critical", "title": "Subagent delegation is measured by NOTHING",
     "detail": "Every B8 row for every model records subagent_spawned = 'no' (verified across all "
               "1,000+ rows): none of the 23 tasks requires delegating to a subagent, so the "
               "subagent canary never fires. qwen3.6-35b-a3b tops B8 at 94% on SINGLE-AGENT "
               "coding while being separately known to FAIL delegation - it makes zero Task calls "
               "and confabulates having delegated. gpt-oss-* and gemma-4-* handle delegation. "
               "None of that behaviour is captured in any battery. Do not read B8 as a "
               "delegation ranking.",
     "fix": "Add B8 tasks that only pass via a spawned subagent, and score the canary."},
    {"sev": "high", "title": "Game / graphical builds are not a battery",
     "detail": "B6 is is_prime, a word-count CLI, backup.sh, debounce, one SQL query and 5 planted "
               "bugs. The game roster (Snake, Tetris, Arkanoid, Flappy Bird, Doodle Jump, "
               "Asteroids, roguelike, fly.pieter-style 3D sim) is written into the v2.1 design "
               "spec but was never implemented, so no model has a scored game result. Snake and "
               "Tetris have ad-hoc session-5/6 verdicts for 6 models only.",
     "fix": "Implement the spec'd one-shot roster with the playability/feature/bug rubric."},
    {"sev": "high", "title": "The suite under-reports speculative decoding",
     "detail": "B5's spec-decode arm measures ~1.00x for every model because it generates fresh "
               "text, where n-gram drafting has almost no hit rate. On edit/rewrite work - what "
               "agentic coding actually does - the same feature is worth 1.95x to 12.08x. The "
               "big numbers are real but live outside the suite.",
     "fix": "Give B5 an edit-workload arm so the headline serving number reflects real use."},
    {"sev": "high", "title": "Laguna ran with NO speculative decoding - and a purpose-built draft model exists",
     "detail": "poolside ships laguna-s-2.1-DFlash-BF16.gguf in its own GGUF repo: a DFlash "
               "speculator (6 sliding-attention layers, shares embeddings with the target, tagged "
               "speculative-decoding) built specifically to draft for Laguna. The Blackwell run "
               "used none of it. It also could not use n-gram spec-decode, because --spec-type "
               "ngram-mod is a prism llama.cpp FORK feature while Laguna's custom arch needs "
               "official llama.cpp b10087+. So every Laguna throughput figure here is "
               "un-accelerated, and the cheapest available speedup was left on the table.",
     "fix": "Re-serve Laguna with the draft: llama-server -m laguna-UD-IQ4_XS.gguf "
            "-md laguna-s-2.1-DFlash-BF16.gguf, and record decode t/s with vs without."},
    {"sev": "medium", "title": "MTP was never tried on the big card",
     "detail": "MTP with a SEPARATE draft GGUF is proven locally at 2.55x lossless on dense "
               "gemma-4-31b (68 -> 173 t/s) - the old 0.20x result was the embedded-head trap, not "
               "MTP itself. No datacenter session tested it, so none of the B5 numbers reflect "
               "what these models can actually do when drafted.",
     "fix": "Add a draft-model arm to B5 for every model that has a published draft/MTP GGUF."},
    {"sev": "medium", "title": "B6 judged quality never ran",
     "detail": "B6 reports only deterministic pass/fail. The 0-10 judged quality axis (does the "
               "code read well, handle edges, avoid gratuitous complexity) is built but has not "
               "been run, so a model that squeaks past the checks scores the same as one that "
               "writes genuinely good code.",
     "fix": "Run the B6 judged axis through the same 3-seat panel as B1."},
    {"sev": "medium", "title": "B8 missing for the four largest models",
     "detail": "glm-4.5-air, gpt-oss-120b, llama-4-scout and qwen3-235b have no B8 rows - they "
               "don't fit the 24GB laptop that ran the agentic sweep, and the rented Blackwell "
               "sessions were spent on Laguna.",
     "fix": "One Blackwell session running the B8 container sweep for the big four."},
    {"sev": "low", "title": "Laguna has no B5 / B7, and its B1 is a rescaled incremental wave",
     "detail": "B5 and B7 were skipped for Laguna (B5 is box-specific and B7 needs the fork's "
               "spec arm). Its B1 6.1 comes from a 3-letter incremental packet rescaled through "
               "the CAL anchors rather than a full-roster packet - defensible, but one step "
               "further from the frozen 16.",
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
                                 "display": f"{round(100*p/n)}%", "sub": []}

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
                        "display": f"{round(sc)}%", "sub": []}

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
                        "display": f"{round(best)} t/s", "sub": []}
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
                        "display": f"{round(sc)}%", "sub": []}

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
            "display": f"{round(100*p/n)}%",
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
    # drop the quant-arm pseudo-model from the roster view
    models = [m for m in models if m != "gemma-4-26b-a4b-mxfp4"]
    for m in models:
        if matrix[m].get("B1", {}).get("tested"):
            matrix[m]["B1"]["sub"] = b1_units_for(m)

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
