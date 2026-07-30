#!/usr/bin/env python3
"""B9/B10/B11 report sections, for p8_report.py.

These three batteries were added after p8_report.py was written, and it still covers
B1-B8 only - so results/REPORT.md, the canonical report, contained no mention of the
game builds, the security review, or the tool loop at all. Their rows live outside the
suite shards (results_games/, results_security/, results_tools/) because they are not
suite-schema rows, which is exactly why the existing loader never picked them up.

Kept in its own module so p8_report.py's structure is untouched: it imports and calls
build_b9_b11_section(root).
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

SHARDS = {
    9: ("results_games", "rows-games.jsonl"),
    10: ("results_security", "rows-security.jsonl"),
    11: ("results_tools", "rows-tools.jsonl"),
}


def _read(root: Path, battery: int) -> list[dict]:
    d, f = SHARDS[battery]
    p = root / d / f
    if not p.exists():
        return []
    out = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval. Quoted because at these n the intervals are wide enough that
    two models can look ranked while overlapping completely."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _tbl(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_no rows_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _pct(k: int, n: int) -> str:
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% ({k}/{n}) [{100 * lo:.0f}-{100 * hi:.0f}]"


def build_b9(rows: list[dict]) -> str:
    """B9 measures RUNS_CLEAN, not 'is it a fun game'.

    A visual oracle cannot prove game logic advanced: a frozen snake with an animated
    particle layer changes more of the board (0.0107) than a working one (0.0049). So the
    gate is the conjunction of load/surface/paint/loop/keys-wired/input-safe, and
    `runs_clean` is what gets reported.
    """
    if not rows:
        return "_no B9 rows_\n"
    by = collections.defaultdict(lambda: [0, 0])
    rung = collections.defaultdict(collections.Counter)
    games = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        m = r["model_id"]
        ok = bool((r.get("det_checks") or {}).get("runs_clean", {}).get("pass"))
        by[m][0] += ok
        by[m][1] += 1
        g = r["task_id"].split(".", 1)[-1]
        games[g][0] += ok
        games[g][1] += 1
        rung[m][(r.get("metrics") or {}).get("attempt_rung", "base")] += 1

    out = ["#### Per model - runs_clean (95% Wilson)", ""]
    body = []
    for m, (k, n) in sorted(by.items(), key=lambda x: -(x[1][0] / max(1, x[1][1]))):
        ladder = ", ".join(f"{a}:{c}" for a, c in sorted(rung[m].items()))
        body.append([m, _pct(k, n), ladder])
    out.append(_tbl(["Model", "runs_clean", "attempt ladder"], body))
    out += ["", "#### Per game - how hard was each build?", ""]
    gb = [[g, _pct(k, n)] for g, (k, n) in sorted(games.items(), key=lambda x: x[1][0] / max(1, x[1][1]))]
    out.append(_tbl(["Game", "runs_clean"], gb))
    out += ["", "`attempt ladder` records which rung produced the answer: `base`, then a "
                "larger token budget, then thinking disabled. Reasoning models burn a small "
                "`max_tokens` entirely on hidden thinking and return EMPTY content at "
                "`finish=length` - gemma-4-26b scored 4% before the ladder and 58% after, so a "
                "row at a higher rung is a budget artefact, not a quality signal.", ""]
    return "\n".join(out)


def build_b10(rows: list[dict]) -> str:
    """B10 separates SENSITIVITY from SPECIFICITY, because the Mythos result turned on it:
    on the OpenBSD SACK bug essentially every model flagged the vulnerable code, and what
    separated them was whether they also declared the PATCHED code vulnerable. A model
    that calls everything vulnerable scores 100% sensitivity and is useless."""
    if not rows:
        return "_no B10 rows_\n"
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    refus = collections.Counter()
    total = collections.Counter()
    recall = collections.defaultdict(list)
    for r in rows:
        m = r["model_id"]
        d = r.get("det_checks") or {}
        met = r.get("metrics") or {}
        exp = met.get("expect_vulnerable")
        correct = bool(d.get("correct_verdict", {}).get("pass"))
        tid = r["task_id"]
        hard = tid.startswith("b10hard.")
        # Decoys are `b10.decoy_*` - clean code that LOOKS alarming. They are broken out
        # from plain specificity because they are the deliberate trap, not just an
        # ordinary patched file.
        decoy = "decoy" in tid
        if hard:
            bucket = "hard_vuln" if exp is True else "hard_clean"
        elif decoy:
            bucket = "decoy"
        elif exp is True:
            bucket = "sensitivity"
        else:
            bucket = "specificity"
        agg[m][bucket][0] += correct
        agg[m][bucket][1] += 1
        if exp is True and not hard:
            agg[m]["cwe"][1] += 1
            if bool(d.get("cwe_correct", {}).get("pass")):
                agg[m]["cwe"][0] += 1
        if hard and exp is True:
            recall[m].append(float(met.get("chain_recall") or 0.0))
            agg[m]["whole"][1] += 1
            if bool(d.get("found_whole_chain", {}).get("pass")):
                agg[m]["whole"][0] += 1
        total[m] += 1
        if met.get("refused"):
            refus[m] += 1

    def spec_rate(m):
        a = agg[m]["specificity"]
        return a[0] / a[1] if a[1] else 0.0

    body = []
    for m in sorted(agg, key=lambda x: -spec_rate(x)):
        a = agg[m]
        rc = recall[m]
        body.append([
            m,
            _pct(*a["sensitivity"]) if a["sensitivity"][1] else "-",
            _pct(*a["specificity"]) if a["specificity"][1] else "-",
            _pct(*a["decoy"]) if a["decoy"][1] else "-",
            f"{100 * sum(rc) / len(rc):.0f}%" if rc else "-",
            _pct(*a["whole"]) if a["whole"][1] else "-",
            _pct(*a["cwe"]) if a["cwe"][1] else "-",
            f"{refus[m]}/{total[m]}",
        ])
    out = ["#### Per model (95% Wilson)", "",
           _tbl(["Model", "Sensitivity", "Specificity", "Decoys",
                 "Chain recall", "Whole chain", "CWE named", "Refusals"], body),
           "",
           "**Specificity is the discriminator, not sensitivity.** Sensitivity runs near "
           "ceiling for almost everything; a model that shouts VULNERABLE at every input "
           "scores 100% there and is worthless on an engagement. Refusals are counted "
           "separately and never scored as a wrong answer - refusing is a different "
           "failure from being wrong.",
           "",
           "**Base and hard tiers can disagree, so neither alone is safe to quote.** The "
           "base tier is one defect per file; the hard tier is multi-defect chains where "
           "`Chain recall` is the fraction of planted defects found and `Whole chain` is "
           "the stricter question of whether a single answer got ALL of them - which is "
           "what an engagement actually needs.",
           ""]
    return "\n".join(out)


def build_b11(rows: list[dict]) -> str:
    """B11 is the agentic floor B2 cannot measure: B2 scores tool-call FORMATION (is the
    JSON well-formed, is the right tool selected), which is ~100% for most models. B11
    makes the client advertise real tools, execute them, and feed results back, then
    scores the FILESYSTEM afterwards - so a model that narrates having edited a file
    without calling the tool fails, which is precisely the qwen3.6-35b-a3b subagent
    behaviour that B2 rated as passing."""
    if not rows:
        return "_no B11 rows_\n"
    by = collections.defaultdict(lambda: [0, 0])
    calls = collections.defaultdict(list)
    confab = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        m = r["model_id"]
        d = r.get("det_checks") or {}
        by[m][0] += bool(d.get("completed", {}).get("pass"))
        by[m][1] += 1
        confab[m][0] += bool(d.get("no_confabulation", {}).get("pass"))
        confab[m][1] += 1
        calls[m].append((r.get("metrics") or {}).get("n_tool_calls", 0))

    body = []
    for m, (k, n) in sorted(by.items(), key=lambda x: -(x[1][0] / max(1, x[1][1]))):
        cs = sorted(calls[m])
        med = cs[len(cs) // 2] if cs else 0
        body.append([m, _pct(k, n), _pct(*confab[m]), str(med)])
    return "\n".join([
        "#### Per model (95% Wilson)", "",
        _tbl(["Model", "Task completed", "No confabulation", "Median tool calls"], body),
        "",
        "Scored from the FILESYSTEM after the loop ends, not from what the model said it "
        "did. `no confabulation` fails a model that claims work it never performed.",
        ""])


def build_b9_b11_section(root: Path) -> str:
    r9, r10, r11 = _read(root, 9), _read(root, 10), _read(root, 11)
    parts = ["## B9-B11 - Game Builds, Security Review, Tool Loop", "",
             f"_B9 {len(r9)} rows, B10 {len(r10)} rows, B11 {len(r11)} rows. These "
             "batteries keep their own shards (`results_games/`, `results_security/`, "
             "`results_tools/`) because they are not suite-schema rows._", "",
             "### B9 Game Builds", "", build_b9(r9), "",
             "### B10 Security Review", "", build_b10(r10), "",
             "### B11 Tool Loop", "", build_b11(r11), ""]
    covered = {b: len({r["model_id"] for r in rr})
               for b, rr in ((9, r9), (10, r10), (11, r11))}
    parts += [f"**Coverage:** B9 {covered[9]} models, B10 {covered[10]}, B11 {covered[11]}. "
              "A blank is 'not yet run', never 'scored zero'.", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    print(build_b9_b11_section(Path(__file__).resolve().parents[1]))
