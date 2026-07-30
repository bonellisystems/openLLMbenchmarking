#!/usr/bin/env python3
"""What is still missing, and what would it cost to finish?

Reads the same two sources the run planner does - the dashboard's coverage matrix and
the registry - and prices the remaining cells with emit_run_plan's est_seconds(), so the
number quoted here is the same model that orders the run rather than a fresh guess.

    python scripts/remaining_cost.py
    python scripts/remaining_cost.py --rate 1.20 --credit 12.43
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT.parent / "llm-eval-dashboard"
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from emit_run_plan import ROWS, est_seconds  # noqa: E402

# Cells that no amount of GPU time closes, with the reason. Quoting a GPU cost for these
# would be wrong: they are blocked on something else entirely.
NON_GPU = {
    "B1": "needs a judging pass (judge quota), not GPU time - generation alone leaves "
          "the cell grey because the matrix gates B1 on a judged score",
}
NEEDS_OTHER_BINARY = {
    "bonsai-ternary-27b": "Q2_0 needs the prism fork's custom kernels; the official "
                          "ggml image cannot load it, so its cells fail at serve time",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=1.20, help="$/hour for an RTX PRO 6000")
    ap.add_argument("--credit", type=float, default=None)
    args = ap.parse_args()

    data = json.loads((DASH / "data.json").read_text(encoding="utf-8"))
    reg = yaml.safe_load((ROOT / "config" / "registry.yaml").read_text(encoding="utf-8"))["models"]
    phases = [p["id"] for p in data["phases"]]
    matrix = data["matrix"]

    total = len(data["models"]) * len(phases)
    tested = sum(1 for m in data["models"] for p in phases
                 if matrix.get(m, {}).get(p, {}).get("tested"))
    print(f"coverage: {tested}/{total} = {100 * tested / total:.1f}%")

    rows = []
    for mid in data["models"]:
        missing = [p for p in phases if not matrix.get(mid, {}).get(p, {}).get("tested")]
        if not missing:
            continue
        e = reg.get(mid, {}) or {}
        gpu_cells = [b for b in missing if b not in NON_GPU]
        blocked = [b for b in missing if b in NON_GPU]
        gb = float(e.get("weights_gb") or 0)
        fake = {"id": mid, "batteries": gpu_cells, "gb": gb,
                "fits_card": (gb + 6) <= 96 if gb else True}
        secs = est_seconds(fake, reg) if gpu_cells else 0.0
        rows.append((mid, missing, gpu_cells, blocked, gb, secs))

    rows.sort(key=lambda r: r[5])
    print(f"\n{'model':<22} {'missing':<26} {'GB':>6} {'est h':>6} {'est $':>7}  note")
    tot_h = 0.0
    for mid, missing, gpu_cells, blocked, gb, secs in rows:
        h = secs / 3600
        tot_h += h
        note = ""
        if mid in NEEDS_OTHER_BINARY:
            note = "WILL FAIL: " + NEEDS_OTHER_BINARY[mid].split(";")[0]
        elif blocked:
            note = f"{'+'.join(blocked)} not GPU-bound"
        print(f"{mid:<22} {','.join(missing):<26} {gb:6.1f} {h:6.1f} "
              f"{h * args.rate:7.2f}  {note}")

    n_gpu = sum(len(r[2]) for r in rows)
    n_blocked = sum(len(r[3]) for r in rows)
    print(f"\nGPU-closable cells : {n_gpu}  ->  ~{tot_h:.1f} h  ~${tot_h * args.rate:.2f}")
    print(f"not GPU-bound      : {n_blocked}")
    for b, why in NON_GPU.items():
        print(f"    {b}: {why}")
    for m, why in NEEDS_OTHER_BINARY.items():
        print(f"    {m}: {why}")
    if args.credit is not None:
        print(f"\ncredit ${args.credit:.2f} buys ~{args.credit / args.rate:.1f} h", end="")
        run = 0.0
        got = 0
        for mid, _m, gpu_cells, _b, _gb, secs in rows:
            if run + secs / 3600 > args.credit / args.rate:
                break
            run += secs / 3600
            got += len(gpu_cells)
        print(f" -> about {got} of {n_gpu} remaining GPU cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
