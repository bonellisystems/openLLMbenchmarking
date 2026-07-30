#!/usr/bin/env python3
"""Merge a gap-closing run's pulled output into the canonical results tree.

The box writes everything under /root/out; watch_run.py pulls that to
results_gapclose/out/. This maps each piece to where the dashboard actually reads it:

    out/suite/rows-suite-*.jsonl   -> results/rows-suite-*.jsonl     (B1-B7)
    out/games/rows-games.jsonl     -> results_games/rows-games.jsonl (B9)
    out/security/rows-security...  -> results_security/...           (B10)
    out/tools/rows-tools.jsonl     -> results_tools/...              (B11)
    out/b8_<model>/*.jsonl         -> results_b8_<model>/            (B8)
    out/games/builds/*             -> results_games/builds/          (explorer artifacts)
    out/suite/sessions.jsonl       -> results/sessions.jsonl

EVERYTHING IS DEDUPED AND IDEMPOTENT. Runs get pulled repeatedly while in progress, so
this must be safe to re-run against a partially-merged tree: suite rows go through
Store.append (row_id dedupe + schema validation), and the B9/B10/B11 shards dedupe on
(battery, model_id, task_id, run_n, condition).

    python scripts/merge_gapclose.py --src results_gapclose/out --dry-run
    python scripts/merge_gapclose.py --src results_gapclose/out
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.store import Store  # noqa: E402

# (battery, model_id, task_id, run_n, condition) identifies one B9/B10/B11 row.
KEY = ("battery", "model_id", "task_id", "run_n", "condition")

# (model_id, battery) pairs whose rows must NOT enter the store, with the reason. For
# rows that were produced successfully but describe a serving config that did not run -
# worse than a missing row, because they populate a cell with a mislabelled measurement.
#
# Currently empty. abl-opus-35b-a3b B4 was quarantined here on a WRONG diagnosis: I read
# a "request (20716 tokens) exceeds the available context size (16384 tokens)" error as
# proof that `-c` was being split across 4 slots. The server log says otherwise -
# n_ctx_slot was 16384/65536/131072/262144 for the four arms with kv_unified='true', i.e.
# every arm got exactly the context its row is labelled with. The 16384 in that message
# was the 16k arm's own budget, not a quartered 64k. Those rows are valid.
QUARANTINE: dict[tuple[str, int], str] = {}

CUSTOM = [
    ("games", "rows-games.jsonl", "results_games"),
    ("security", "rows-security.jsonl", "results_security"),
    ("tools", "rows-tools.jsonl", "results_tools"),
]


def rowkey(r: dict) -> tuple:
    return tuple(r.get(k) for k in KEY)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a pull mid-write can leave one truncated last line
    return out


def merge_custom(src_dir: Path, shard: str, dest_dir: Path, dry: bool) -> tuple[int, int]:
    incoming = read_jsonl(src_dir / shard)
    if not incoming:
        return 0, 0
    dest = ROOT / dest_dir / shard
    have = {rowkey(r) for r in read_jsonl(dest)}
    new = [r for r in incoming if rowkey(r) not in have]
    # An identical row arriving twice within one pull is also a duplicate.
    seen, fresh = set(have), []
    for r in new:
        k = rowkey(r)
        if k not in seen:
            seen.add(k)
            fresh.append(r)
    if fresh and not dry:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8", newline="\n") as f:
            for r in fresh:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(incoming), len(fresh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results_gapclose/out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = ROOT / args.src
    if not src.exists():
        print(f"nothing to merge: {src} does not exist")
        return 1
    dry = args.dry_run
    tag = "[dry-run] " if dry else ""
    print(f"{tag}merging {src}")

    # --- suite rows (B1-B7): Store.append validates AND dedupes by row_id -----------
    # SEARCHED RECURSIVELY ON PURPOSE. The two suite drivers write to different places:
    # bigmodel_gen is invoked with --results-dir $OUT/suite, while p8_gen_serving.py and
    # p8_gen_b5.py build their own Store(LLMTEST_OUT) and write to /root/out directly.
    # Globbing only out/suite/ therefore found B1/B2/B3/B6 and silently dropped every
    # B4, B5 and B7 row - the exact batteries this run exists to produce. Store.append
    # dedupes by row_id, so a shard matched twice costs nothing.
    store = Store(ROOT / "results")
    suite_seen = suite_new = suite_bad = 0
    quarantined = 0
    for shard in sorted(src.rglob("rows-suite-*.jsonl")):
        for r in read_jsonl(shard):
            suite_seen += 1
            if (r.get("model_id"), r.get("battery")) in QUARANTINE:
                quarantined += 1
                continue
            if dry:
                continue
            try:
                if store.append(r):
                    suite_new += 1
            except Exception as e:
                suite_bad += 1
                if suite_bad <= 5:
                    print(f"  REJECTED {r.get('row_id', '?')[:16]}: {e}")
    print(f"  suite B1-B7 : {suite_seen} read, {suite_new} new"
          + (f", {suite_bad} REJECTED BY SCHEMA" if suite_bad else ""))
    if quarantined:
        print(f"  QUARANTINED : {quarantined} rows held back:")
        for (mid, bat), why in QUARANTINE.items():
            print(f"    {mid} B{bat} - {why}")

    # --- per-battery custom shards --------------------------------------------------
    for sub, shard, dest in CUSTOM:
        seen, new = merge_custom(src / sub, shard, dest, dry)
        label = {"games": "B9 games", "security": "B10 security", "tools": "B11 tools"}[sub]
        print(f"  {label:<12}: {seen} read, {new} new")

    # --- B8 per-model dirs ----------------------------------------------------------
    b8_files = b8_new = 0
    for d in sorted(src.glob("b8_*")):
        if not d.is_dir():
            continue
        dest = ROOT / f"results_{d.name}"
        for f in sorted(d.rglob("*")):
            if not f.is_file():
                continue
            target = dest / f.relative_to(d)
            b8_files += 1
            if target.exists() and target.stat().st_size == f.stat().st_size:
                continue
            b8_new += 1
            if not dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
    print(f"  B8 dirs     : {b8_files} files, {b8_new} new/changed")

    # --- game build artifacts (the explorer shows these) ----------------------------
    builds = src / "games" / "builds"
    n_b = 0
    if builds.exists():
        dest = ROOT / "results_games" / "builds"
        for f in sorted(builds.rglob("*")):
            if f.is_file() and not (dest / f.name).exists():
                n_b += 1
                if not dry:
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest / f.name)
    print(f"  game builds : {n_b} new artifacts")

    # --- sessions (serving provenance; dedupe on session_id) ------------------------
    sess_new = 0
    for p in sorted(src.rglob("sessions.jsonl")):
        have = {s.get("session_id") for s in read_jsonl(ROOT / "results" / "sessions.jsonl")}
        for s in read_jsonl(p):
            if s.get("session_id") in have:
                continue
            have.add(s.get("session_id"))
            sess_new += 1
            if not dry:
                store.append_session(s)
    print(f"  sessions    : {sess_new} new")

    # --- report which (model, battery) steps the box says failed --------------------
    steps = ROOT / args.src.split("/")[0] / "steps"
    if not steps.exists():
        steps = src.parent / "steps"
    if steps.exists():
        bad = [ln.strip() for ln in steps.read_text(encoding="utf-8").splitlines()
               if " ok" not in ln and ln.strip()]
        print(f"\n  steps file: {len(bad)} non-ok entries"
              + ("" if not bad else ":\n    " + "\n    ".join(bad[:25])))

    if dry:
        print("\ndry run - nothing written")
    else:
        print("\nnext: python -m llmtest validate  &&  "
              "cd ../llm-eval-dashboard && python build_data.py && python build_explorer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
