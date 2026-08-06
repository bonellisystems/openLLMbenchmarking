#!/usr/bin/env python3
"""CONTROL: does judging via API+batch score the same as judging via `claude -p`?

Before any real judging moves to the Batch API, measure the delivery change on data
where the answer is already known. This re-judges a sample of packets that ALREADY have
committed `claude` judgments and compares score-for-score.

Why this specific design:

* Only 18/19-letter full-roster packets are eligible. Re-judging a small incremental
  packet would confound the delivery change with the packet-size leniency already
  documented (judges run ~0.9pt lenient in small packets), and we would learn nothing
  about either.
* The CAL-strong / CAL-weak anchors are in every packet with known reference scores, so
  the control reads the calibration drift directly rather than inferring it.
* `claude -p` wraps the packet in Claude Code's own system prompt and tool surface; the
  API sends the packet alone. That is the whole hypothesis under test - identical packet
  bytes, different surrounding context.

Nothing is written to results/. This spends money (a sample of 18-letter packets is
~20k input tokens each), so it defaults to a small n and prints the measured cost.

    python scripts/judge_batch_control.py --n 12          # submit, wait, compare
    python scripts/judge_batch_control.py --estimate-only # price it, submit nothing
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.judging import batch_api  # noqa: E402
from llmtest.judging.adapters import parse_reply  # noqa: E402

# Opus-class list price; the run prints ACTUAL usage so this is only for the estimate.
USD_IN_PER_M, USD_OUT_PER_M = 15.0, 75.0
BATCH_MULT = 0.5


def eligible_packets(limit: int):
    """Packets with committed claude judgments, biggest cohort first (full-roster)."""
    # committed[packet_id][letter] = (score, model_id). model_id is what identifies the
    # CAL anchors - the packet map deliberately does NOT record which letter is which
    # model (that is the blinding), so the judgment rows are the only place the mapping
    # exists after the fact.
    judged = collections.defaultdict(dict)
    jf = ROOT / "results" / "judgments.jsonl"
    for line in jf.open(encoding="utf-8"):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("judge_id") == "claude" and j.get("score") is not None:
            judged[j["packet_id"]][j["letter"]] = (j["score"], j.get("model_id"))
    out = []
    for pid, letters in judged.items():
        body = ROOT / "artifacts" / "packets" / f"{pid}.claude.txt"
        mp = ROOT / "results" / "packets" / f"{pid}.map.json"
        if not body.exists() or not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        # letters_by_judge[judge] is {letter: answer_sha} - keys are the letters the
        # judge must score, and parse_reply wants them as a list.
        exp = sorted((m.get("letters_by_judge", {}).get("claude") or {}).keys())
        if len(exp) < 10:            # full-roster cohorts only
            continue
        out.append((pid, body, exp, letters, m))
    out.sort(key=lambda x: x[0])     # deterministic sample, not "whatever dict order"
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="pin the SAME model the committed judgments used, so the only "
                         "variable is delivery")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    sample = eligible_packets(args.n)
    if not sample:
        print("no eligible packets (need committed claude judgments + packet bodies)")
        return 2
    chars = sum(b.stat().st_size for _, b, _, _, _ in sample)
    est_in = chars / 4
    est_out = 2500 * len(sample)
    est = (est_in / 1e6 * USD_IN_PER_M + est_out / 1e6 * USD_OUT_PER_M) * BATCH_MULT
    print(f"sample      : {len(sample)} full-roster packets, {chars/1e6:.2f} MB "
          f"(~{est_in/1e3:.0f}k input tokens)")
    print(f"model pin   : {args.model}  (matches the committed judgments)")
    print(f"est cost    : ~${est:.2f} at batch rates (actual usage printed after)")
    if args.estimate_only:
        print("\n--estimate-only: nothing submitted.")
        return 0

    key = batch_api.load_api_key(ROOT)
    reqs = [batch_api.build_request(pid, b.read_text(encoding="utf-8"), args.model)
            for pid, b, _, _, _ in sample]
    bid = batch_api.submit(key, reqs)
    print(f"\nbatch id    : {bid}  (results retrievable for 29 days)")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "batch_control_id.txt").write_text(bid, encoding="utf-8")

    seen = {}
    def tick(b):
        c = b.get("request_counts") or {}
        s = tuple(sorted(c.items()))
        if s != seen.get("last"):
            seen["last"] = s
            print(f"   {b.get('processing_status')}: {dict(c)}", flush=True)
    batch = batch_api.poll(key, bid, interval=15, on_tick=tick)
    results = batch_api.fetch_results(key, batch)

    deltas, cal_api, cal_cli, bad = [], collections.defaultdict(list), collections.defaultdict(list), 0
    tin = tout = 0
    for pid, body, exp, committed, m in sample:
        res = results.get(pid)
        if not res:
            bad += 1
            continue
        u = batch_api.usage_of(res)
        tin += u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) \
            + u.get("cache_read_input_tokens", 0)
        tout += u.get("output_tokens", 0)
        text, err = batch_api.reply_text(res)
        if err:
            print(f"   {pid[:12]}: {err}")
            bad += 1
            continue
        parsed, perr = parse_reply(text, exp)
        if perr:
            print(f"   {pid[:12]}: unparseable ({perr})")
            bad += 1
            continue
        for L in exp:
            if L not in parsed["scores"] or L not in committed:
                continue
            cli_score, model_id = committed[L]
            deltas.append(parsed["scores"][L] - cli_score)
            if model_id in ("CAL-strong", "CAL-weak"):
                tag = model_id.split("-", 1)[1]
                cal_api[tag].append(parsed["scores"][L])
                cal_cli[tag].append(cli_score)

    print("\n" + "=" * 62)
    print(f"compared    : {len(deltas)} letter-scores across {len(sample)-bad} packets"
          f"{f' ({bad} unusable)' if bad else ''}")
    if deltas:
        print(f"mean delta  : {statistics.mean(deltas):+.2f} pts (API minus CLI)")
        print(f"median      : {statistics.median(deltas):+.1f}   "
              f"stdev {statistics.pstdev(deltas):.2f}   "
              f"range {min(deltas):+d}..{max(deltas):+d}")
        agree = sum(1 for d in deltas if d == 0) / len(deltas)
        within1 = sum(1 for d in deltas if abs(d) <= 1) / len(deltas)
        print(f"exact match : {agree:.0%}   within 1pt: {within1:.0%}")
    for tag in ("strong", "weak"):
        if cal_api[tag]:
            print(f"CAL-{tag:6}: API {statistics.mean(cal_api[tag]):.2f} vs "
                  f"CLI {statistics.mean(cal_cli[tag]):.2f}")
    print(f"tokens      : {tin/1e3:.0f}k in, {tout/1e3:.0f}k out")
    actual = (tin / 1e6 * USD_IN_PER_M + tout / 1e6 * USD_OUT_PER_M) * BATCH_MULT
    print(f"actual cost : ~${actual:.2f} (batch rates, list prices)")
    u0 = batch_api.usage_of(next(iter(results.values())))
    print(f"cache usage : creation={u0.get('cache_creation_input_tokens', 0)} "
          f"read={u0.get('cache_read_input_tokens', 0)}  "
          f"(0/0 confirms the prefix is under the cacheable floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
