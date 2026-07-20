"""Non-circular B2 calibration gate + quarantine (Task 6).

The frozen CAL-strong/CAL-weak anchors (Task 2 -- grading/calibration/b2/
<axis>.yaml, authored once, pinned, never re-tuned) are NOT adjusted to
please the judge panel. This module only CHECKS whether the panel's
already-produced judgments on those anchors satisfy two judge-independent
invariants, per B2 axis dimension, and QUARANTINES an axis when they don't.
There is no panel-driven regeneration here -- authoring stays frozen; this
gate is purely a read of judgments that already exist.

A quarantined axis is meant to be excluded from `b2_axis_scores` at report
time (Task 7): its real-model scores aren't trustworthy because the panel
didn't demonstrably score the frozen anchors correctly.

Data shapes mirror aggregate.py (CAL_IDENTITIES, refscores {strong, weak,
tolerance}): judgment rows are dicts with packet_id/judge_id/model_id/score/
status; a packet is a B2 axis packet iff maps[packet_id].get("dim") starts
with "axis" (that dim IS the axis key); CAL answers appear as
model_id == "CAL-strong" / "CAL-weak". Only status=="ok" rows count.
"""
from __future__ import annotations

import statistics

CAL_STRONG = "CAL-strong"
CAL_WEAK = "CAL-weak"


def calibration_status(
    judgments: list[dict],
    maps: dict,
    *,
    refscores: dict | None = None,
) -> dict[str, str]:
    """{axis_dim: "accepted" | "quarantined"} for every B2 axis dimension
    present in `maps` (i.e. every distinct maps[pid]["dim"] starting with
    "axis"), regardless of whether any CAL rows were judged for it yet.

    Both invariants must hold for an axis to be "accepted":
      1. Ordinal, per judge: aggregating that judge's CAL-strong scores (via
         median, across every packet of this axis) must exceed the median
         of that judge's CAL-weak scores -- for EVERY judge who scored this
         axis's CAL answers. A judge missing either side (can't form both
         medians) fails this invariant -- partial coverage can't be
         confirmed non-inverted, and this gate is autonomous (no human
         fallback), so it defaults to quarantine rather than a silent pass.
      2. Drift: |median(all CAL-strong scores for this axis) -
         refscores["strong"]| <= tolerance, AND the same for CAL-weak vs
         refscores["weak"]. "All" pools every ok CAL-strong/-weak score for
         the axis across every judge and packet -- NOT a median of
         per-judge medians.

    An axis with zero ok CAL rows can't be validated and is quarantined.
    """
    refscores = {"strong": 9, "weak": 2, "tolerance": 1, **(refscores or {})}
    tolerance = refscores["tolerance"]

    axes: set[str] = {
        m["dim"] for m in maps.values()
        if isinstance(m.get("dim"), str) and m["dim"].startswith("axis")
    }

    # axis -> judge_id -> {"strong": [scores], "weak": [scores]}
    per_axis_judge: dict[str, dict[str, dict[str, list]]] = {axis: {} for axis in axes}

    for j in judgments:
        if j.get("status") != "ok":
            continue
        model_id = j.get("model_id")
        if model_id not in (CAL_STRONG, CAL_WEAK):
            continue
        m = maps.get(j["packet_id"])
        if m is None:
            continue
        dim = m.get("dim")
        if dim not in per_axis_judge:
            continue   # not a (known) B2 axis packet
        cal_type = "strong" if model_id == CAL_STRONG else "weak"
        judge_scores = per_axis_judge[dim].setdefault(
            j["judge_id"], {"strong": [], "weak": []})
        judge_scores[cal_type].append(j["score"])

    status: dict[str, str] = {}
    for axis in sorted(axes):
        judge_map = per_axis_judge[axis]

        ordinal_ok = bool(judge_map)   # no judges scored -> can't confirm ordinal
        for scores in judge_map.values():
            if not scores["strong"] or not scores["weak"]:
                ordinal_ok = False
                break
            if statistics.median(scores["strong"]) <= statistics.median(scores["weak"]):
                ordinal_ok = False
                break

        all_strong = [s for scores in judge_map.values() for s in scores["strong"]]
        all_weak = [s for scores in judge_map.values() for s in scores["weak"]]
        drift_ok = (
            bool(all_strong) and bool(all_weak)
            and abs(statistics.median(all_strong) - refscores["strong"]) <= tolerance
            and abs(statistics.median(all_weak) - refscores["weak"]) <= tolerance
        )

        status[axis] = "accepted" if (ordinal_ok and drift_ok) else "quarantined"

    return status
