#!/usr/bin/env python3
"""Author the planted-bug fixtures for the B9 self-correction track.

Takes games that PASS the gate cleanly, injects one known bug, and keeps the result
only if it satisfies both halves of the TESTPLAN 5.6 preflight rule:

    the known-good build must pass every probe, AND the bugged build must fail

A planted bug the gate cannot see is worthless - "did the model fix it?" would be
unanswerable - so a variant that still passes is discarded and reported, never
shipped. That is the same discipline as `preflight()`: probe failure is a harness
bug, not a model failure.

Bug kinds are limited to what the current gate can actually detect:

    crash_load    a typo'd identifier at top level      -> throws before anything runs
    crash_input   a typo'd call inside the key handler  -> throws the moment you play
    frozen        the per-frame advance is neutered     -> draws forever, never moves

The plan's third tier ("subtle render/state desync") is deliberately NOT planted
here: catching it needs the per-game probes (snake grows on eat, tetris clears a
line) that are not built yet, so it could not be scored.

    python scripts/plant_game_bugs.py --src ../michael --out suite/b9_games/planted
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.harness.game_oracle import run_game_checks  # noqa: E402

MARK = "<!-- planted-bug fixture: do not ship as a known-good -->"

# Builds documented as broken in benchmark-tables.md. The gate passes some of them
# (a frozen snake whose particle layer animates), which is exactly the limitation
# recorded in game_oracle: gate-clean is NOT the same as known-good. A fixture must
# start from a build we have actually confirmed by eye, so these are excluded by
# name rather than trusted to the gate.
DOCUMENTED_BROKEN = {"snake_bonsai-1bit.html", "tetris_bonsai-1bit.html",
                     "tetris.html", "tetris_27b.html", "snake_granite.html"}


def crash_load(html: str):
    """Reference an undefined identifier at top level: dies before the game starts."""
    m = re.search(r"<script[^>]*>", html)
    if not m:
        return None
    i = m.end()
    return html[:i] + "\nconst __cfg = __GAME_SETTINGS__.initial;\n" + html[i:]


def crash_input(html: str):
    """Call a typo'd function inside the key handler: dies the moment you press a key.
    This is the real-world shape of the SHAPES_keys / SHAPES_KEYS failure."""
    m = re.search(r"(addEventListener\(\s*['\"]keydown['\"]\s*,\s*(?:function\s*\([^)]*\)|\([^)]*\)\s*=>)\s*\{)",
                  html)
    if not m:
        return None
    i = m.end()
    return html[:i] + "\n  handleKeyPress(e.key);\n" + html[i:]


def frozen_v2(html: str):
    """Simpler, more reliable freeze: make the tick interval never elapse."""
    if "setInterval(" in html:
        return re.sub(r"setInterval\(([^,]+),\s*[0-9]+\s*\)",
                      r"setInterval(\1, 99999999)", html, count=1)
    m = re.search(r"(function\s+(update|move|tick|step)\s*\([^)]*\)\s*\{)", html)
    if m:
        return html[:m.end()] + "\n  return;\n" + html[m.end():]
    m = re.search(r"((?:const|let|var)\s+(update|move|tick|step)\s*=\s*(?:function\s*\([^)]*\)|\([^)]*\)\s*=>)\s*\{)",
                  html)
    if m:
        return html[:m.end()] + "\n  return;\n" + html[m.end():]
    return None


BUGS = {"crash_load": crash_load, "crash_input": crash_input, "frozen": frozen_v2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../michael", help="directory of candidate known-good games")
    ap.add_argument("--out", default="suite/b9_games/planted")
    ap.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    ap.add_argument("--max-goods", type=int, default=4)
    args = ap.parse_args()

    src = Path(args.src)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    good_dir = out / "known_good"
    good_dir.mkdir(exist_ok=True)
    chrome = args.chrome if Path(args.chrome).exists() else None
    tmp = out / "_tmp.html"

    print("== preflight: finding builds that pass the gate cleanly ==")
    goods = []
    for p in sorted(src.glob("*.html")):
        if p.name == "index.html" or p.name in DOCUMENTED_BROKEN:
            if p.name in DOCUMENTED_BROKEN:
                print(f"  {p.name:32s} excluded - documented broken (gate cannot see logic bugs)")
            continue
        r = run_game_checks(p, chrome_path=chrome)
        print(f"  {p.name:32s} {r.score}/6 {'CLEAN' if r.runs_clean else 'fails'}")
        if r.runs_clean:
            goods.append(p)
        if len(goods) >= args.max_goods:
            break
    if not goods:
        print("no clean known-good build found - cannot plant bugs")
        return 1

    print(f"\n== planting into {len(goods)} known-good build(s) ==")
    kept = dropped = 0
    for p in goods:
        html = p.read_text(encoding="utf-8", errors="replace")
        (good_dir / p.name).write_text(html, encoding="utf-8")
        for kind, fn in BUGS.items():
            try:
                bugged = fn(html)
            except Exception as e:                                # noqa: BLE001
                bugged = None
                print(f"  {p.stem}__{kind}: injector error {e}")
            if not bugged or bugged == html:
                print(f"  {p.stem}__{kind}: no injection point - skipped")
                continue
            tmp.write_text(MARK + "\n" + bugged, encoding="utf-8")
            res = run_game_checks(tmp, chrome_path=chrome)
            if res.runs_clean:
                dropped += 1
                print(f"  {p.stem}__{kind}: DISCARDED - gate still passes, bug is invisible")
                continue
            broke = [k for k, v in res.checks.items() if not v]
            dst = out / f"{p.stem}__{kind}.html"
            dst.write_text(MARK + "\n" + bugged, encoding="utf-8")
            kept += 1
            print(f"  {p.stem}__{kind}: kept - breaks {', '.join(broke)}")
    if tmp.exists():
        tmp.unlink()
    print(f"\nplanted {kept} fixture(s); {dropped} discarded as undetectable")
    print(f"known-goods copied to {good_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
