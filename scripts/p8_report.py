"""P8 assessment report generator (read-only, idempotent, re-runnable).

Reads all battery result rows (results/rows-<suite_version>.jsonl) plus the
B1 judge panel (results/judgments.jsonl + results/packets/*.map.json) and
emits:
  - results/REPORT.md   -- the comprehensive report (all 5 sections below)
  - stdout               -- a condensed chat-ready summary of the same data

Sections: (1) overview/run metadata, (2) B1 business scorecard (flagship,
reuses llmtest/judging/aggregate.py's exact P3 aggregation), (3) per-battery
deterministic summaries (B2/B3/B4/B6/B7), (4) B5 serving, (5) data-quality
caveats.

Contract (per P8 build instructions): pure read + generate. This script
NEVER calls a GPU endpoint or a judge CLI, and NEVER writes to config/*,
llmtest/batteries/*, or any frozen artifact -- it only reads existing
results/* files and writes results/REPORT.md. Safe to re-run at any point
while batteries are still generating or judging is still in progress: every
section degrades to an explicit "(no data yet)" / "(in progress)" note
instead of crashing on missing/partial data.

Usage:
    python scripts/p8_report.py [--root PATH]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llmtest.judging.aggregate import AggResult, aggregate, load_maps, load_refscores  # noqa: E402
from llmtest.judging.runner import resolve_cohort_models  # noqa: E402
from llmtest.registry import load_config  # noqa: E402
from llmtest.store import Store  # noqa: E402
from llmtest.tables import _current_rubric_sha, render_scorecard  # noqa: E402

BATTERY_NAMES = {
    1: "B1 Business (judged)",
    2: "B2 Tool Calling (deterministic + 2 judged axes not yet wired)",
    3: "B3 Hallucination Curve (deterministic)",
    4: "B4 Long Context (deterministic)",
    5: "B5 Serving / Throughput (timing)",
    6: "B6 Agentic Coding (deterministic + judged correctness not yet wired)",
    7: "B7 Harness/Config Sensitivity Matrix (deterministic)",
}
ALL_BATTERIES = [1, 2, 3, 4, 5, 6, 7]
# Full-roster baseline letter count is derived dynamically in load_baseline_maps()
# from the packets on disk (current roster of N models + CAL-strong + CAL-weak), so the
# scorecard never goes stale when the roster changes (was 13 at 11 models, 17 at 15, 18 at 16).
TOTAL_BASELINE_PACKETS = 360  # 120 B1 tasks x 3 runs
FULL_JUDGE_PANEL = ("claude", "codex", "gemini")


# --------------------------------------------------------------------------
# Loading (pure reads; every loader degrades to empty/None on missing files)
# --------------------------------------------------------------------------

def load_rows(root: Path, suite_version: str, caveats: list[str]) -> list[dict]:
    """Reads ONLY results/rows-<suite_version>.jsonl directly (NOT via
    Store.iter_rows(), which globs rows-*.jsonl and would also slurp in
    results/rows-<suite_version>-shakedown.jsonl -- a separate, frozen,
    smaller P0-P2 shakedown run with its own suite_version string that must
    never be mixed into the P8 suite counts)."""
    path = root / "results" / f"rows-{suite_version}.jsonl"
    if not path.exists():
        caveats.append(f"rows file not found: {path.name} -- battery rows section will be empty")
        return []
    rows, bad = [], 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        caveats.append(f"{bad} malformed line(s) in {path.name} skipped")
    return rows


def load_baseline_maps(root: Path) -> dict[str, dict]:
    """{packet_id: map} restricted to the full-roster B1 baseline packets --
    those whose letters_by_judge has the MAXIMUM letter count present on disk
    (current roster of N models + CAL-strong + CAL-weak). Packet maps with a
    smaller letter count are partial-roster / dry-run / quota-probe packets from
    earlier waves and are out of scope for the baseline scorecard. Deriving the
    target from the packets (rather than a hardcoded 13/17/18) keeps the
    scorecard correct across roster changes."""
    maps_all = load_maps(root / "results" / "packets")
    counts = [len(next(iter(m["letters_by_judge"].values())))
              for m in maps_all.values() if m.get("letters_by_judge")]
    if not counts:
        return {}
    target = max(counts)
    return {
        pid: m for pid, m in maps_all.items()
        if m.get("letters_by_judge")
        and all(len(lm) == target for lm in m["letters_by_judge"].values())
    }


def fully_judged_count(baseline_maps: dict[str, dict], judgments: list[dict],
                        judge_ids: list[str]) -> tuple[int, int]:
    """(fully_judged, total) over baseline_maps. A packet counts as fully
    judged when EVERY configured judge has an ok row for EVERY letter in
    that packet's map -- mirrors llmtest/judging/runner.py's
    summarize_judging() per-(packet,judge) "done" rule, computed here
    directly from ok judgment rows (no run_pending()/build_cohort_packets()
    call -- this script never writes packet artifacts, only reads them)."""
    ok_letters: dict[tuple[str, str], set] = defaultdict(set)
    for j in judgments:
        if j.get("status") == "ok":
            ok_letters[(j["packet_id"], j["judge_id"])].add(j["letter"])

    done = 0
    for pid, m in baseline_maps.items():
        letters_by_judge = m.get("letters_by_judge", {})
        complete = True
        for jid in judge_ids:
            letter_map = letters_by_judge.get(jid)
            if letter_map is None:
                continue
            if not ok_letters.get((pid, jid), set()).issuperset(letter_map):
                complete = False
                break
        if complete:
            done += 1
    return done, len(baseline_maps)


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------

def fmt1(x) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else "-"


def pct(frac, n=None) -> str:
    if frac is None:
        return "-"
    s = f"{frac * 100:.0f}%"
    return f"{s} (n={n})" if n is not None else s


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def det_pass_values(det_checks: dict, exclude: tuple[str, ...] = ()) -> list[bool]:
    """Every boolean 'pass' value across a row's det_checks dict, generic
    across every battery's key naming (axisN_*, contains-N, required.*,
    signal_agreement_vs_baseline, ...). Robust to rows lacking det_checks
    entirely (returns [])."""
    out = []
    for k, v in (det_checks or {}).items():
        if k in exclude:
            continue
        if isinstance(v, dict) and isinstance(v.get("pass"), bool):
            out.append(v["pass"])
    return out


def is_empty_output(row: dict) -> bool:
    """True generation-empty proxy: zero response chars (or zero predicted
    tokens) AND not a legitimate native tool-call response (B7's
    tool_call_compliance check passing means the answer correctly landed in
    `tool_calls`, not `content` -- empty `content` there is expected
    behavior, not a failure, and must not be double-counted as empty)."""
    dc = row.get("det_checks") or {}
    tcc = dc.get("tool_call_compliance")
    if isinstance(tcc, dict) and tcc.get("pass") is True:
        return False
    m = row.get("metrics", {})
    chars = m.get("chars", m.get("code_chars"))
    predicted_n = row.get("response_meta", {}).get("predicted_n")
    return (chars is not None and chars == 0) or predicted_n == 0


def condition_parts(condition: str) -> dict:
    if not condition:
        return {}
    return dict(p.split("=", 1) for p in condition.split(";") if "=" in p)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Section 1: Overview
# --------------------------------------------------------------------------

def build_overview(root, cfg, rows, roster, baseline_maps, judgments, judge_ids,
                    hardware_label, caveats) -> str:
    suite_version = cfg.suite["suite_version"]
    by_battery = Counter(r["battery"] for r in rows)
    models_seen_by_battery = defaultdict(set)
    for r in rows:
        models_seen_by_battery[r["battery"]].add(r["model_id"])

    fully_judged, total_baseline = fully_judged_count(baseline_maps, judgments, judge_ids)
    ok_judgments = sum(1 for j in judgments if j.get("status") == "ok")
    err_judgments = sum(1 for j in judgments if j.get("status") != "ok")

    lines = [
        "## 1. Overview", "",
        f"- Suite version: `{suite_version}`",
        f"- Roster: {len(roster)} models (+ CAL-strong / CAL-weak calibration identities in B1)",
        f"- Batteries: {len(ALL_BATTERIES)} (B1-B7)",
        f"- Hardware: {hardware_label}",
        "",
        "### Row generation status (per battery)", "",
    ]
    gen_rows = []
    for b in ALL_BATTERIES:
        n = by_battery.get(b, 0)
        seen = models_seen_by_battery.get(b, set())
        missing = sorted(set(roster) - seen) if n else sorted(roster)
        status = "not started" if n == 0 else (
            "complete (all roster models present)" if not missing
            else f"in progress ({len(seen)}/{len(roster)} models seen)")
        gen_rows.append([f"B{b}", BATTERY_NAMES[b], str(n), status])
    lines.append(md_table(["Battery", "Name", "Rows", "Status"], gen_rows))
    lines.append("")
    lines.append(f"Total rows loaded: **{len(rows)}**")
    lines.append("")

    lines += [
        "### B1 judging progress", "",
        f"- Baseline packets (full-roster cohorts): **{total_baseline}** "
        f"(target {TOTAL_BASELINE_PACKETS})",
        f"- Fully judged (all {len(judge_ids) or len(FULL_JUDGE_PANEL)} judges complete): "
        f"**{fully_judged} / {total_baseline}** "
        f"({(100.0 * fully_judged / total_baseline):.1f}%)" if total_baseline else
        f"- Fully judged: 0 / 0 (no baseline packets built yet)",
        f"- Judgment rows: {ok_judgments} ok, {err_judgments} error/other "
        f"(all packets, not just the 360-baseline scope)",
    ]
    if total_baseline != TOTAL_BASELINE_PACKETS:
        caveats.append(f"expected {TOTAL_BASELINE_PACKETS} baseline (full-roster) B1 packets, "
                        f"found {total_baseline} on disk")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Section 2: B1 Business Scorecard (flagship -- reuses aggregate.py exactly)
# --------------------------------------------------------------------------

def build_b1_section(root, cfg, rows, judgments, baseline_maps) -> tuple[str, AggResult]:
    judge_ids = sorted(cfg.judges.get("judges", {}))
    kin_map = cfg.judges.get("kin_map", {})
    refscores_path = root / "grading" / "calibration" / "refscores.yaml"
    refscores = load_refscores(refscores_path) if refscores_path.exists() else None
    current_rubric_sha = _current_rubric_sha(root)
    roster_filter = set(resolve_cohort_models(cfg))
    units = cfg.suite.get("b1", {}).get("units_tier1", [])

    agg = aggregate(rows, judgments, baseline_maps, kin_map=kin_map, refscores=refscores,
                     judge_ids=judge_ids, current_rubric_sha=current_rubric_sha,
                     roster_filter=roster_filter)

    lines = ["## 2. B1 Business Scorecard (flagship)", "",
              f"Median-of-3-judges aggregation over the {len(baseline_maps)} baseline "
              f"(full-roster) B1 packets, {len(units)} business units. Reused as-is from "
              "`llmtest/judging/aggregate.py::aggregate()` / `llmtest/tables.py::render_scorecard()` "
              "(the P3 scorecard aggregation) -- table-time only, nothing stored.", ""]

    if not agg.model_overall:
        lines.append("**(judging too early -- no scored packets yet; scorecard columns will "
                      "populate as B1 judgments accumulate)**")
    else:
        ranked = sorted(agg.model_overall.items(), key=lambda kv: -kv[1])
        lines.append("### Ranked (overall = mean of per-unit means)")
        lines.append("")
        rank_rows = [[str(i + 1), m, fmt1(score)] for i, (m, score) in enumerate(ranked)]
        # widen with unscored roster models trailing, for visibility
        unscored = [m for m in agg.model_roster if m not in agg.model_overall]
        for m in unscored:
            rank_rows.append(["-", m, "no judgments yet"])
        lines.append(md_table(["Rank", "Model", "Overall"], rank_rows))
        lines.append("")

    lines.append("### Full scorecard (units x models)")
    lines.append("")
    lines.append(render_scorecard(agg, units))
    return "\n".join(lines) + "\n", agg


# --------------------------------------------------------------------------
# Section 3: Per-battery deterministic summaries
# --------------------------------------------------------------------------

def summarize_b2(rows: list[dict], cfg) -> str:
    b2 = [r for r in rows if r["battery"] == 2]
    if not b2:
        return "(no B2 rows yet)"
    axes_cfg = cfg.suite.get("b2", {}).get("axes", {})
    by_model: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in b2:
        for k, v in (r.get("det_checks") or {}).items():
            if not k.startswith("axis"):
                continue
            if isinstance(v, dict) and isinstance(v.get("pass"), bool):
                by_model[r["model_id"]][k].append(v["pass"])

    axis_keys = sorted({k for m in by_model.values() for k in m},
                        key=lambda k: int(k[4:].split("_", 1)[0]) if k[4:].split("_", 1)[0].isdigit() else 99)
    headers = ["Model"] + [f"ax{k[4:].split('_', 1)[0]} {axes_cfg.get(int(k[4:].split('_', 1)[0]), '')}".strip()
                            for k in axis_keys]
    rows_out = []
    for model in sorted(by_model):
        cells = [model]
        for k in axis_keys:
            vals = by_model[model].get(k)
            cells.append(pct(mean(vals), len(vals)) if vals else "-")
        rows_out.append(cells)

    judged_axes = cfg.suite.get("b2", {}).get("judged_axes", [5, 8])
    note = (f"\n\nAxes {judged_axes} (error recovery / faithfulness) get a best-effort "
            "deterministic floor here (`needs_judging=True`) but are NOT yet wired into "
            "the judging pipeline (`JUDGED_BATTERIES = {1}` in `llmtest/judging/runner.py`) "
            "-- treat their pass-rate as a floor, not a final score.")
    return md_table(headers, rows_out) + note


def summarize_b3(rows: list[dict]) -> str:
    b3 = [r for r in rows if r["battery"] == 3]
    if not b3:
        return "(no B3 rows yet)"
    by_model: dict[str, dict] = defaultdict(lambda: {"hedged": [], "fabricated": [], "correct": [], "n": 0})
    for r in b3:
        m = r.get("metrics", {})
        bm = by_model[r["model_id"]]
        bm["n"] += 1
        for k in ("hedged", "fabricated", "correct"):
            if k in m:
                bm[k].append(bool(m[k]))

    rows_out = []
    for model in sorted(by_model):
        bm = by_model[model]
        rows_out.append([model, str(bm["n"]), pct(mean(bm["hedged"])),
                          pct(mean(bm["fabricated"])), pct(mean(bm["correct"]))])

    note = ("\n\nInterpretation: `fabricated` = model asserted a false/invented fact under "
            "pressure (lower is better); `hedged` = model declined/flagged uncertainty "
            "(context-dependent -- good on unanswerable probes, bad on the closed-domain "
            "control); `correct` = matched the expected answer signal. **A low `correct` "
            "or high `fabricated` rate here may reflect over-strict first-pass deterministic "
            "signal checks (regex/keyword matching), not necessarily worse model behavior "
            "-- treat as a signal to spot-check transcripts before drawing conclusions.**")
    return md_table(["Model", "N", "Hedge rate", "Fabricate rate", "Correct rate"], rows_out) + note


def summarize_b4(rows: list[dict], cfg) -> str:
    b4 = [r for r in rows if r["battery"] == 4]
    if not b4:
        return "(no B4 rows yet)"
    b4cfg = cfg.suite.get("b4", {})
    standard_kv = b4cfg.get("standard_kv", "q8")
    kv_sweep_models = set(b4cfg.get("kv_sweep_models") or [])
    spot_model = (b4cfg.get("kv_spot_check") or {}).get("model")

    def ctx_sort_key(label: str) -> int:
        try:
            return int(label[:-1]) if label.endswith("k") else int(label)
        except ValueError:
            return 0

    # Table 1: models x ctx-tier, standard kv only
    grid: dict[tuple, list[float]] = defaultdict(list)
    ctx_labels = set()
    for r in b4:
        parts = condition_parts(r["condition"])
        kv, ctx = parts.get("kv"), parts.get("ctx")
        recall = r.get("metrics", {}).get("needle_recall")
        if recall is None or kv != standard_kv:
            continue
        grid[(r["model_id"], ctx)].append(recall)
        ctx_labels.add(ctx)
    ctx_sorted = sorted(ctx_labels, key=ctx_sort_key)
    models_here = sorted({r["model_id"] for r in b4})
    rows1 = []
    for model in models_here:
        cells = [model]
        for ctx in ctx_sorted:
            vals = grid.get((model, ctx))
            cells.append(pct(mean(vals)) if vals else "-")
        rows1.append(cells)
    table1 = md_table(["Model"] + ctx_sorted, rows1) if ctx_sorted else "(no standard-kv rows yet)"

    # Table 2: kv-quant sweep -- models x kv dtype (mean recall across their swept ctx tiers)
    sweep_targets = sorted(kv_sweep_models | ({spot_model} if spot_model else set()))
    grid2: dict[tuple, list[float]] = defaultdict(list)
    kvs_seen = set()
    for r in b4:
        if r["model_id"] not in sweep_targets:
            continue
        parts = condition_parts(r["condition"])
        kv, recall = parts.get("kv"), r.get("metrics", {}).get("needle_recall")
        if recall is None:
            continue
        grid2[(r["model_id"], kv)].append(recall)
        kvs_seen.add(kv)
    kvs_sorted = sorted(kvs_seen)
    rows2 = []
    for model in sweep_targets:
        cells = [model]
        for kv in kvs_sorted:
            vals = grid2.get((model, kv))
            cells.append(pct(mean(vals)) if vals else "-")
        rows2.append(cells)
    table2 = md_table(["Model"] + kvs_sorted, rows2) if kvs_sorted else "(no kv-quant-sweep rows yet)"

    return (f"**Retrieval accuracy by ctx-tier** (standard kv={standard_kv})\n\n{table1}\n\n"
            f"**KV-quant quality sweep** (kv_sweep_models + spot-check model)\n\n{table2}")


def summarize_b6(rows: list[dict]) -> str:
    b6 = [r for r in rows if r["battery"] == 6]
    if not b6:
        return "(no B6 rows yet)"
    by_model: dict[str, dict] = defaultdict(lambda: {
        "det": [], "compile": [], "empty": 0, "n": 0})
    for r in b6:
        bm = by_model[r["model_id"]]
        bm["n"] += 1
        dc = r.get("det_checks") or {}
        vals = det_pass_values(dc)
        if vals:
            bm["det"].append(sum(vals) / len(vals))
        compile_ok = dc.get("compile_ok")
        if isinstance(compile_ok, dict) and isinstance(compile_ok.get("pass"), bool):
            bm["compile"].append(compile_ok["pass"])
        # B6-specific "empty" signal: no fenced code block extracted at all
        # (code_chars==0), the literal thing det_checks['code_extracted']
        # checks -- more precise here than the generic chars==0 proxy, since
        # a model can produce prose (chars>0) while still emitting no code.
        code_chars = r.get("metrics", {}).get("code_chars")
        if code_chars == 0:
            bm["empty"] += 1

    rows_out = []
    total_empty = 0
    for model in sorted(by_model):
        bm = by_model[model]
        total_empty += bm["empty"]
        rows_out.append([model, str(bm["n"]), pct(mean(bm["det"])),
                          pct(mean(bm["compile"])) if bm["compile"] else "-",
                          str(bm["empty"])])
    note = (f"\n\nEmpty-output rows (zero generated code / zero predicted tokens) across "
            f"all models: **{total_empty}**. `compile_ok` only applies to Python tasks. "
            "Correctness/completeness is `needs_judging=True` on every row (not yet wired "
            "into the judging pipeline) -- these det-pass rates are static-signal floors, "
            "not full correctness scores.")
    return md_table(["Model", "N", "Det-pass rate", "compile_ok rate", "Empty outputs"],
                     rows_out) + note


def summarize_b7(rows: list[dict], cfg) -> str:
    b7 = [r for r in rows if r["battery"] == 7]
    if not b7:
        return "(no B7 rows yet)"
    dims_cfg = cfg.suite.get("b7", {}).get("matrix", {}).get("dimensions", {})
    baseline_dims = {k: v["baseline"] for k, v in dims_cfg.items()}

    def cell_name(parts: dict) -> str:
        diffs = [f"{d}={parts.get(d)}" for d in dims_cfg if parts.get(d) not in (None, baseline_dims[d])]
        return "baseline" if not diffs else ",".join(diffs)

    grid: dict[str, dict] = defaultdict(lambda: {"content": [], "agree": [], "byte_id": [], "n": 0})
    for r in b7:
        parts = condition_parts(r["condition"])
        cell = cell_name(parts)
        dc = r.get("det_checks") or {}
        g = grid[cell]
        g["n"] += 1
        content_vals = det_pass_values(
            dc, exclude=("signal_agreement_vs_baseline", "byte_identical_vs_baseline"))
        if content_vals:
            g["content"].append(sum(content_vals) / len(content_vals))
        sab = dc.get("signal_agreement_vs_baseline")
        if isinstance(sab, dict) and sab.get("agreement_rate") is not None:
            g["agree"].append(sab["agreement_rate"])
        bib = dc.get("byte_identical_vs_baseline")
        if isinstance(bib, dict) and isinstance(bib.get("pass"), bool):
            g["byte_id"].append(bib["pass"])

    order = ["baseline"] + sorted(k for k in grid if k != "baseline")
    rows_out = []
    for cell in order:
        g = grid.get(cell)
        if not g:
            continue
        rows_out.append([cell, str(g["n"]), pct(mean(g["content"])),
                          pct(mean(g["agree"])) if g["agree"] else "-",
                          pct(mean(g["byte_id"])) if g["byte_id"] else "-"])
    note = ("\n\n`Agreement vs baseline` = fraction of shared content-signal checks that "
            "match the baseline cell's pass/fail (threshold "
            f"{cfg.suite.get('b7', {}).get('agreement_threshold', 0.8)}). "
            "`Byte-identical` only applies to the spec=off,temp=t0 cell -- a direct "
            "empirical check of the project's own \"n-gram spec-decode is lossless at "
            "temp=0\" claim (CLAUDE.md).")
    return md_table(["Cell (vs baseline)", "N", "Content det-pass", "Mean agreement",
                      "Byte-identical rate"], rows_out) + note


def build_battery_section(rows: list[dict], cfg) -> str:
    lines = ["## 3. Per-battery deterministic summaries", ""]
    lines += ["### B2 Tool Calling -- per-axis pass rate by model", "",
              summarize_b2(rows, cfg), ""]
    lines += ["### B3 Hallucination Curve -- fabricate/hedge/correct rates by model", "",
              summarize_b3(rows), ""]
    lines += ["### B4 Long Context -- retrieval accuracy by ctx-tier & kv-quant", "",
              summarize_b4(rows, cfg), ""]
    lines += ["### B6 Agentic Coding -- code det-pass by model + empty-output count", "",
              summarize_b6(rows), ""]
    lines += ["### B7 Harness Matrix -- config-sensitivity vs baseline by dimension", "",
              summarize_b7(rows, cfg), ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Section 4: B5 Serving
# --------------------------------------------------------------------------

def build_b5_section(rows: list[dict]) -> str:
    b5 = [r for r in rows if r["battery"] == 5]
    lines = ["## 4. B5 Serving / Throughput", ""]
    if not b5:
        lines.append("(no B5 rows yet -- serving battery has not started generating. "
                      "NOTE: these numbers, once present, are from the RTX PRO 6000 "
                      "rented run, not the local RTX 5090 laptop canonical numbers in "
                      "the project reference docs.)")
        return "\n".join(lines) + "\n"

    lines.append("_RTX PRO 6000 (rented) numbers -- not the local RTX 5090 canonical "
                  "figures in `vast-5090-qwen36/RESULTS5-local-lowbit-codegen.md`._")
    lines.append("")

    def get_tps(r):
        return r.get("metrics", {}).get("decode_tps", r.get("response_meta", {}).get("decode_tps"))

    # PEAK single-stream: ngram32 vs off, + speedup
    peak: dict[tuple, float] = {}
    for r in b5:
        parts = condition_parts(r["condition"])
        if parts.get("cond") != "PEAK" or "conc" in parts:
            continue
        tps = get_tps(r)
        if tps is not None:
            peak[(r["model_id"], parts.get("spec", "?"))] = tps
    models_here = sorted({r["model_id"] for r in b5})
    rows1 = []
    for model in models_here:
        ngram = peak.get((model, "ngram32"))
        off = peak.get((model, "off"))
        speedup = (ngram / off) if (ngram and off) else None
        rows1.append([model, fmt1(ngram) if ngram else "-", fmt1(off) if off else "-",
                      f"{speedup:.2f}x" if speedup else "-"])
    lines += ["### PEAK decode t/s (single-stream)", "",
              md_table(["Model", "ngram32 t/s", "off t/s", "speedup"], rows1), ""]

    # SUSTAINED32K
    sustained: dict[tuple, float] = {}
    for r in b5:
        parts = condition_parts(r["condition"])
        if parts.get("cond") != "SUSTAINED32K" or "conc" in parts:
            continue
        tps = get_tps(r)
        if tps is not None:
            sustained[(r["model_id"], parts.get("spec", "?"))] = tps
    rows2 = []
    for model in models_here:
        ngram = sustained.get((model, "ngram32"))
        off = sustained.get((model, "off"))
        rows2.append([model, fmt1(ngram) if ngram else "-", fmt1(off) if off else "-"])
    lines += ["### SUSTAINED-32K decode t/s", "",
              md_table(["Model", "ngram32 t/s", "off t/s"], rows2), ""]

    # concurrency ladder
    conc_grid: dict[tuple, float] = {}
    concs = set()
    for r in b5:
        parts = condition_parts(r["condition"])
        if parts.get("cond") != "PEAK" or "conc" not in parts:
            continue
        agg_tps = r.get("metrics", {}).get("aggregate_tps")
        if agg_tps is not None:
            conc_grid[(r["model_id"], parts["conc"])] = agg_tps
            concs.add(parts["conc"])
    concs_sorted = sorted(concs, key=lambda c: int(c))
    rows3 = []
    for model in models_here:
        cells = [model]
        for c in concs_sorted:
            v = conc_grid.get((model, c))
            cells.append(fmt1(v) if v is not None else "-")
        rows3.append(cells)
    lines += ["### Concurrency ladder (aggregate t/s, ngram32)", "",
              md_table(["Model"] + [f"conc={c}" for c in concs_sorted], rows3) if concs_sorted
              else "(no concurrency-ladder rows yet)", ""]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Section 5: Data-quality caveats
# --------------------------------------------------------------------------

def build_caveats_section(rows: list[dict], agg: AggResult, judgments: list[dict],
                           baseline_maps: dict, extra_caveats: list[str]) -> str:
    lines = ["## 5. Data-quality caveats", ""]

    lines.append("### Empty-output counts by battery")
    lines.append("")
    empty_rows = []
    for b in ALL_BATTERIES:
        brows = [r for r in rows if r["battery"] == b]
        if not brows:
            continue
        n_empty = sum(1 for r in brows if is_empty_output(r))
        empty_rows.append([f"B{b}", str(len(brows)), str(n_empty),
                            pct(n_empty / len(brows) if brows else None)])
    lines.append(md_table(["Battery", "N rows", "Empty outputs", "Empty rate"], empty_rows)
                 if empty_rows else "(no rows loaded)")
    lines.append("")
    lines.append("_\"Empty\" = zero response chars or zero predicted tokens, EXCLUDING B7 rows "
                 "where a native tool call correctly landed the answer in `tool_calls` instead "
                 "of `content` (that's expected empty content, not a failure). B6 uses its own "
                 "code_chars==0 signal in section 3 instead (more precise: catches prose-only "
                 "responses with no code block, which still have chars>0)._")
    lines.append("")

    lines.append("### First-pass / recalibration flags")
    lines.append("")
    lines += [
        "- B3 hallucination: fabricate/hedge/correct rates use deterministic keyword/regex "
        "signal checks -- these are first-pass proxies, not judged; a low `correct` rate can "
        "mean the model was wrong OR that the signal check is too strict. Spot-check "
        "transcripts under `artifacts/b3/` before treating this as a ranking.",
        "- B2 axes 5 (error recovery) and 8 (faithfulness), and all of B6's correctness axis, "
        "carry `needs_judging=True` but are NOT wired into the judging pipeline yet "
        "(`JUDGED_BATTERIES = {1}` in `llmtest/judging/runner.py`) -- their det-pass numbers "
        "in section 3 are best-effort deterministic floors only.",
        "- B7 is the project's own least-specified battery (see "
        "`.superpowers/sdd/b7-report.md`): it measures config-knob sensitivity "
        "(system prompt / temp / tool-call format / n-gram spec), not the TESTPLAN-5.7 "
        "external-harness axis (Hermes/OpenCode/LiteLLM-CC), which is P6 scope.",
        "- `config/suite.yaml` records that b6's `max_tokens_by_track` and b7's `max_tokens` "
        "were both raised mid-P8 after truncation caused many empty rows on reasoning models "
        "(hidden-thinking budget burn) -- the empty-output counts above reflect the CURRENT "
        "data only; historically-truncated rows from before that config bump may still be "
        "mixed in if they were never re-run with `force`.",
    ]
    lines.append("")

    lines.append("### Judging completeness")
    lines.append("")
    ok_j = sum(1 for j in judgments if j.get("status") == "ok")
    err_j = sum(1 for j in judgments if j.get("status") != "ok")
    lines += [
        f"- {len(baseline_maps)} baseline B1 packets on disk (target {TOTAL_BASELINE_PACKETS}).",
        f"- {ok_j} ok / {err_j} error judgment rows total (includes packets outside the "
        "full-roster baseline scope and 9 leftover `judge_cli_version='fake'` dry-run rows "
        "from an earlier smoke test, which the baseline-packet filter already excludes).",
        f"- Incomplete panels (< {3} judges) among scored baseline packets: "
        f"{agg.incomplete_panel_count}.",
        f"- Errored judge-packet pairs counted in aggregation: {agg.error_rows_count}.",
        f"- Spread>2 flags: {len(agg.spread_flags)}; calibration-drift flags: "
        f"{len(agg.drift_flags)} (see `results/FLAGS.md` if `llmtest tables` has been run "
        "separately -- this report does not regenerate it to stay pure-read).",
    ]
    lines.append("")

    if extra_caveats:
        lines.append("### Other notes")
        lines.append("")
        lines += [f"- {c}" for c in extra_caveats]
        lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Condensed chat-ready summary
# --------------------------------------------------------------------------

def build_condensed(rows, roster, agg, baseline_maps, judgments, judge_ids, hardware_label) -> str:
    by_battery = Counter(r["battery"] for r in rows)
    fully_judged, total_baseline = fully_judged_count(baseline_maps, judgments, judge_ids)
    lines = [
        "=" * 72,
        "P8 REPORT -- condensed summary",
        "=" * 72,
        f"Roster: {len(roster)} models | Hardware: {hardware_label}",
        "Row counts: " + ", ".join(f"B{b}={by_battery.get(b, 0)}" for b in ALL_BATTERIES),
        f"B1 judging: {fully_judged}/{total_baseline} baseline packets fully judged "
        f"({(100.0 * fully_judged / total_baseline):.0f}%)" if total_baseline else
        "B1 judging: no baseline packets built yet",
    ]
    if agg.model_overall:
        ranked = sorted(agg.model_overall.items(), key=lambda kv: -kv[1])
        lines.append("B1 scorecard (top 5): " + ", ".join(
            f"{i+1}.{m}={fmt1(s)}" for i, (m, s) in enumerate(ranked[:5])))
        lines.append(f"  agreement={agg.agreement_pct:.0f}% mean_spread={agg.mean_spread:.2f} "
                      f"incomplete_panels={agg.incomplete_panel_count} "
                      f"errors={agg.error_rows_count}")
    else:
        lines.append("B1 scorecard: judging too early -- no scored packets yet")
    lines.append("=" * 72)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_report(root: Path) -> tuple[str, str]:
    cfg = load_config(root)
    suite_version = cfg.suite["suite_version"]
    roster = resolve_cohort_models(cfg)
    hardware_label = "2x RTX PRO 6000 (rented)"

    caveats: list[str] = []
    rows = load_rows(root, suite_version, caveats)

    store = Store(root / "results")
    judgments = list(store.iter_judgments())
    baseline_maps = load_baseline_maps(root)
    judge_ids = sorted(cfg.judges.get("judges", {}))

    overview = build_overview(root, cfg, rows, roster, baseline_maps, judgments,
                               judge_ids, hardware_label, caveats)
    b1_section, agg = build_b1_section(root, cfg, rows, judgments, baseline_maps)
    battery_section = build_battery_section(rows, cfg)
    b5_section = build_b5_section(rows)
    caveats_section = build_caveats_section(rows, agg, judgments, baseline_maps, caveats)

    header = (
        "# LLMtest v2 -- P8 Assessment Report\n\n"
        f"_Generated by `scripts/p8_report.py`. Suite `{suite_version}`. "
        "Re-run this script any time to refresh -- it is pure read + generate "
        "and safe to run on partial/in-progress data._\n"
    )
    full_md = "\n".join([header, overview, b1_section, battery_section, b5_section,
                          caveats_section])

    condensed = build_condensed(rows, roster, agg, baseline_maps, judgments,
                                 judge_ids, hardware_label)
    return full_md, condensed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="repo root (default: parent of scripts/)")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else _REPO_ROOT

    full_md, condensed = build_report(root)

    out_path = root / "results" / "REPORT.md"
    out_path.write_text(full_md, encoding="utf-8", newline="\n")

    print(condensed)
    print(f"\n[p8_report] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
