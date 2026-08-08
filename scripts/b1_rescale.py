#!/usr/bin/env python3
"""CAL-rescale B1 scores from a small incremental wave onto the frozen scale.

WHY THIS EXISTS
---------------
B1 is judged in cohort packets: one judge call scores every model's answer to one
task, side by side, letter-blinded. The frozen 16-model scorecard was judged in
18-letter packets (16 models + the two CAL anchors). A model added later is judged
in its own small packet -- laguna alone (3 letters), the abliterated pair (4) --
because re-judging the whole roster to add one model costs the entire panel again.

Small packets are NOT the same measuring instrument. With no strong peers to
compare against, the panel drifts lenient, and the CAL anchors prove it: the SAME
two fixed answers, with the same rubric, score

    18-letter packets   CAL-strong 7.62   CAL-weak 0.89
     4-letter packets   CAL-strong 8.43   CAL-weak 1.36
     3-letter packets   CAL-strong 8.74   CAL-weak 1.36

That is up to ~1.1 points of pure instrument offset, and it shrinks as the packet
widens -- which is exactly why the correction has to be MEASURED per wave rather
than carried over from the last one. Publishing a raw small-packet score next to
the frozen 16 would have put the newcomer roughly six ranks too high.

THE CORRECTION
--------------
The anchors are fixed answers with fixed reference scores, so they are the only
points on the scale whose true value is known to be identical across waves. Map
the wave's scale onto the frozen scale through those two points:

    frozen = weak_f + (raw - weak_s) * (strong_f - weak_f) / (strong_s - weak_s)

Applied to the overall score AND to every per-unit score, so the department
breakdown on the dashboard can never contradict the headline above it.

Anchors are computed on the SAME statistic as the scores they correct -- mean of
per-packet medians within a unit, then mean of unit means. aggregate() drops CAL
identities from model_overall on purpose (they are not models), so this recomputes
them from packet_answers rather than reaching for a different average.

ON THE PUBLISHED LAGUNA NUMBER
------------------------------
laguna-s-2.1 shipped at 6.1, documented as "raw 6.99 -> 6.13". That used the
ROUNDED anchor constants quoted in the prose (9.0/1.5 -> 8.0/1.0). Measured
anchors give 6.03. --verify reproduces both, so the 0.10 gap is attributable and
not a method difference. The measured path is what this tool applies: the rounded
constants describe the 3-letter wave only, and are visibly wrong for the 4-letter
one (8.43, not 9.0).

    python scripts/b1_rescale.py --verify           # reproduce laguna, prove the method
    python scripts/b1_rescale.py --width 4          # rescale the abliterated wave
    python scripts/b1_rescale.py --width 4 --emit-python   # paste-ready build_data blocks
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.judging.aggregate import aggregate, load_maps  # noqa: E402
from llmtest.store import Store  # noqa: E402

# The frozen scorecard's cohort width: 16 models + CAL-strong + CAL-weak. Every
# other wave is corrected ONTO this one, because this is the scale the published
# 16-model ranking already lives on.
FROZEN_WIDTH = 18

# The rounded constants the original laguna rescale used, kept only so --verify can
# demonstrate that the published 6.13 is reproducible and the delta is rounding.
PROSE_SMALL = (9.0, 1.5)
PROSE_FROZEN = (8.0, 1.0)
LAGUNA_PUBLISHED = 6.13


def packet_widths(maps: dict) -> dict:
    """{packet_id: letters in the packet}. The width IS the wave identity: a map
    deliberately does not record which model each letter is (that is the blinding),
    so letter count is the only wave marker available without joining judgments."""
    out = {}
    for pid, m in maps.items():
        lb = m.get("letters_by_judge") or {}
        out[pid] = len(next(iter(lb.values()), {})) if lb else 0
    return out


def aggregate_width(width: int, maps: dict, widths: dict, judgments: list):
    """Aggregate ONE wave in isolation.

    Each wave is aggregated alone rather than all-at-once because a wave's anchors
    describe that wave's instrument. Pooling waves would average the offsets
    together and correct nobody correctly.
    """
    submaps = {p: m for p, m in maps.items() if widths.get(p) == width}
    subj = [j for j in judgments if widths.get(j["packet_id"]) == width]
    return aggregate(rows=[], judgments=subj, maps=submaps), len(submaps), len(subj)


def cal_anchors(res) -> tuple:
    """(CAL-strong, CAL-weak) on the model_overall statistic: mean of per-packet
    medians within each unit, then mean of unit means."""
    def stat(ident):
        per_unit = collections.defaultdict(list)
        for pa in res.packet_answers:
            if pa.model_id == ident:
                per_unit[pa.unit].append(pa.median)
        if not per_unit:
            return None
        unit_means = [sum(v) / len(v) for v in per_unit.values()]
        return sum(unit_means) / len(unit_means)
    return stat("CAL-strong"), stat("CAL-weak")


def make_map(small: tuple, frozen: tuple):
    """Two-point linear map from a wave's scale onto the frozen scale."""
    s_s, w_s = small
    s_f, w_f = frozen
    if None in (s_s, w_s, s_f, w_f):
        raise SystemExit("missing CAL anchors -- cannot rescale without both ends")
    span = s_s - w_s
    if span <= 0:
        raise SystemExit(f"degenerate anchors (strong {s_s} <= weak {w_s})")
    gain = (s_f - w_f) / span
    return lambda raw: w_f + (raw - w_s) * gain, gain


def per_unit_raw(res, model_id) -> list:
    """[(unit, raw mean)] for one model, ordered high to low like the dashboard."""
    out = [(unit, st["mean"]) for (m, unit), st in res.model_unit_stats.items()
           if m == model_id]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CAL-rescale a small B1 wave onto the frozen scale")
    ap.add_argument("--width", type=int, default=None,
                    help="cohort width of the wave to rescale (3 = laguna, 4 = abliterated pair)")
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the published laguna rescale and report the method delta")
    ap.add_argument("--emit-python", action="store_true",
                    help="print paste-ready dashboard/build_data.py blocks")
    args = ap.parse_args(argv)

    maps = load_maps(ROOT / "results" / "packets")
    widths = packet_widths(maps)
    judgments = list(Store(ROOT / "results").iter_judgments())

    frozen_res, fp, fj = aggregate_width(FROZEN_WIDTH, maps, widths, judgments)
    frozen_anchors = cal_anchors(frozen_res)
    print(f"frozen scale (width {FROZEN_WIDTH}): {fp} packets, {fj} judgment rows")
    print(f"  CAL-strong {frozen_anchors[0]:.4f}   CAL-weak {frozen_anchors[1]:.4f}")

    if args.verify:
        res, _, _ = aggregate_width(3, maps, widths, judgments)
        raw = res.model_overall.get("laguna-s-2.1")
        small = cal_anchors(res)
        prose_fn, _ = make_map(PROSE_SMALL, PROSE_FROZEN)
        meas_fn, _ = make_map(small, frozen_anchors)
        print(f"\nlaguna-s-2.1 raw {raw:.4f}")
        print(f"  rounded prose constants {PROSE_SMALL} -> {PROSE_FROZEN}: {prose_fn(raw):.4f} "
              f"(published {LAGUNA_PUBLISHED})")
        print(f"  measured anchors ({small[0]:.4f},{small[1]:.4f}) -> "
              f"({frozen_anchors[0]:.4f},{frozen_anchors[1]:.4f}): {meas_fn(raw):.4f}")
        ok = abs(prose_fn(raw) - LAGUNA_PUBLISHED) < 0.005
        print(f"  reproduction of the published value: {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
        if args.width is None:
            return 0

    if args.width is None:
        ap.error("--width is required unless --verify is used alone")

    res, np_, nj = aggregate_width(args.width, maps, widths, judgments)
    small = cal_anchors(res)
    fn, gain = make_map(small, frozen_anchors)
    print(f"\nwave (width {args.width}): {np_} packets, {nj} judgment rows")
    print(f"  CAL-strong {small[0]:.4f}   CAL-weak {small[1]:.4f}   gain {gain:.4f}")
    print(f"  error rows in this wave: {res.error_rows_count}")

    overall, units = {}, {}
    for model_id, raw in sorted(res.model_overall.items(), key=lambda x: -x[1]):
        overall[model_id] = fn(raw)
        units[model_id] = [(u, fn(v)) for u, v in per_unit_raw(res, model_id)]
        print(f"\n  {model_id}: raw {raw:.4f} -> rescaled {fn(raw):.4f} "
              f"(published as {round(fn(raw), 1)})")
        for u, v in units[model_id]:
            print(f"      {u:20s} {v:5.1f}")

    if args.emit_python:
        print("\n# --- paste into dashboard/build_data.py ---")
        print("# JUDGED_B1 entries:")
        for m, v in overall.items():
            print(f'    "{m}": {round(v, 1)},')
        print("\n# _RESCALED_UNITS entries:")
        for m, us in units.items():
            body = ", ".join(f'("{u}", {round(v, 1)})' for u, v in us)
            print(f'    "{m}": [{body}],')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
