#!/usr/bin/env python3
"""Judge the claude seat through the Anthropic Batch API instead of `claude -p`.

WHY THIS EXISTS, AND WHAT IT COSTS YOU TO USE IT
------------------------------------------------
The claude seat normally runs `claude -p` on subscription quota. Michael asked
for the API path on 2026-08-10 because the Fable weekly allotment was spent and
the week had just reset. This is a DELIBERATE, RECORDED deviation from the
2026-08-06 ruling that judging stays on the CLI.

That ruling was made on measured evidence, and it still applies: re-judging 12
full-roster packets through the API under the same pin scored -0.24 pt lower on
average (CAL-strong 8.56 vs 8.78, CAL-weak 0.78 vs 1.22), with 3/12 replies
unparseable. `claude -p` wraps the packet in Claude Code's own system prompt and
tool surface; the API sends the packet alone. Identical bytes, different context.

Three things keep that tolerable HERE and none of them are hand-waving:

  1. It is one seat of three. The panel takes a median, so a systematic shift in
     one seat moves the median far less than it moves that seat.
  2. The CAL anchors ride in every packet, and this wave is rescaled onto the
     frozen scale through them (scripts/b1_rescale.py). A delivery offset that
     shifts the anchors the same way it shifts the model is absorbed by that map
     rather than passed through to the published number.
  3. Every judgment written here records delivery="api-batch". A row that cannot
     say how it was produced cannot be retired later, which is exactly the
     mistake the hardware campaign spent a rental undoing.

Batch, not /v1/messages: same prices at 50%, and this is offline work with no
interactivity - the shape the Batch API exists for. Results stay retrievable for
29 days, so a lost local copy is recoverable from the batch id alone.

    python scripts/judge_api_wave.py --estimate-only    # price it, submit nothing
    python scripts/judge_api_wave.py --limit 6          # small probe first
    python scripts/judge_api_wave.py                    # the full wave
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.judging.runner import (  # noqa: E402
    JUDGED_BATTERIES, resolve_cohort_models, run_pending)
from llmtest.registry import load_config  # noqa: E402
from llmtest.judging import batch_api  # noqa: E402
from llmtest.judging.adapters import parse_reply  # noqa: E402
from llmtest.store import Store  # noqa: E402
from llmtest import schema  # noqa: E402

USD_IN_PER_M, USD_OUT_PER_M, BATCH_MULT = 15.0, 75.0, 0.5
JUDGE_ID = "claude"


def pending_packets(cfg, limit=None):
    """(packet_id, body_path, expected_letters, map) for packets of the CURRENT
    cohort that this seat has not judged.

    The packet set comes from run_pending(packets_only=True) - the runner's own
    enumeration, driven by suite.yaml's b1.cohort_models - and NOT from globbing
    results/packets. Globbing was the first attempt and it was wrong: that
    directory accumulates every wave ever built, including a 360-packet
    19-letter cohort that was minted and never judged. The glob offered 720
    packets and an $90 estimate for a job that is 360 packets and ~$20. Asking
    the runner keeps this seat judging exactly what the other two seats judge.
    """
    rows = list(Store(ROOT / "results").iter_rows())
    res = run_pending(
        rows=rows, root=ROOT, store=Store(ROOT / "results"),
        rubric_dir=ROOT / "grading" / "anchors",
        calibration_dir=ROOT / "grading" / "calibration",
        out_artifacts=ROOT / "artifacts" / "packets",
        out_maps=ROOT / "results" / "packets",
        judge_prompt_path=ROOT / "grading" / "judge_prompt.md",
        judges_cfg=cfg.judges["judges"],
        cohort_models=resolve_cohort_models(cfg),
        judge_filter=JUDGE_ID, packets_only=True,
    )
    have = {}
    for j in Store(ROOT / "results").iter_judgments():
        if j.get("judge_id") != JUDGE_ID:
            continue
        have.setdefault(j["packet_id"], set())
        have[j["packet_id"]].add(j["letter"] if j.get("status") == "ok" else "-")
    out = []
    for pk in res.packets:
        pid = pk.packet_id
        mp = ROOT / "results" / "packets" / f"{pid}.map.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        letters = sorted((m.get("letters_by_judge", {}).get(JUDGE_ID) or {}).keys())
        if not letters:
            continue
        done = have.get(pid, set())
        if "-" in done or (done and set(letters).issubset(done)):
            continue
        body = ROOT / "artifacts" / "packets" / f"{pid}.{JUDGE_ID}.txt"
        if not body.exists():
            continue
        out.append((pid, body, letters, m))
        if limit and len(out) >= limit:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Judge the claude seat via the Batch API")
    # NOT judges.yaml's `model`. That key holds "claude-fable-5", a CLI alias the
    # `claude` binary resolves; the API needs a real model id and would 404 on it.
    # opus-4-8 is the pin the 2026-08-06 API control measured, and it already
    # appears in the frozen scorecard's claude rows via the CLI's own fallback,
    # so it is the pin that keeps this seat closest to what it replaces.
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="API model id (default claude-opus-4-8)")
    ap.add_argument("--limit", type=int, default=None, help="probe with N packets first")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args(argv)

    cfg = load_config(ROOT)
    model = args.model

    sample = pending_packets(cfg, args.limit)
    if not sample:
        print("nothing to do - every packet already has claude judgments")
        return 0
    chars = sum(b.stat().st_size for _, b, _, _ in sample)
    est_in = chars / 4
    est_out = 900 * len(sample)
    est = (est_in / 1e6 * USD_IN_PER_M + est_out / 1e6 * USD_OUT_PER_M) * BATCH_MULT
    print(f"packets     : {len(sample)} pending for the {JUDGE_ID} seat")
    print(f"model pin   : {model}")
    print(f"est tokens  : ~{est_in/1e3:.0f}k in, ~{est_out/1e3:.0f}k out")
    print(f"est cost    : ~${est:.2f} at batch rates (actual usage printed after)")
    if args.estimate_only:
        print("\n--estimate-only: nothing submitted.")
        return 0

    key = batch_api.load_api_key(ROOT)
    reqs = [batch_api.build_request(pid, b.read_text(encoding="utf-8"), model,
                                    max_tokens=args.max_tokens)
            for pid, b, _, _ in sample]
    bid = batch_api.submit(key, reqs)
    print(f"\nbatch id    : {bid}  (results retrievable for 29 days)")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "batch_wave_id.txt").write_text(bid, encoding="utf-8")

    seen = {}

    def tick(b):
        c = b.get("request_counts") or {}
        s = tuple(sorted(c.items()))
        if s != seen.get("last"):
            seen["last"] = s
            print(f"   {b.get('processing_status')}: {dict(c)}", flush=True)

    batch = batch_api.poll(key, bid, interval=20, on_tick=tick)
    results = batch_api.fetch_results(key, batch)

    store = Store(ROOT / "results")
    battery_rows = [r for r in Store(ROOT / 'results').iter_rows()
                    if r.get('needs_judging') and r.get('battery') in JUDGED_BATTERIES]
    row_id_to_model_id = {r['row_id']: r['model_id'] for r in battery_rows}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written = unparseable = errored = 0
    tin = tout = 0
    for pid, body, letters, m in sample:
        res = results.get(pid)
        if not res:
            errored += 1
            continue
        u = batch_api.usage_of(res)
        tin += (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                + u.get("cache_read_input_tokens", 0))
        tout += u.get("output_tokens", 0)
        text, err = batch_api.reply_text(res)
        if err:
            print(f"   {pid[:12]}: {err}")
            errored += 1
            continue
        parsed, perr = parse_reply(text, letters)
        if perr:
            print(f"   {pid[:12]}: unparseable ({perr})")
            unparseable += 1
            continue
        # Mirrors p8_judge.judge_one_pair exactly. The map does NOT carry model
        # identities - that is the blinding - so letters_by_judge maps a letter to
        # a ROW ID, and the row id resolves to a model through the result rows.
        # "ranking" is a LIST in rank order, so rank is its 1-based index; reading
        # it as a dict was the first attempt and it crashed on the probe.
        letter_map = (m.get("letters_by_judge") or {}).get(JUDGE_ID) or {}
        for letter in letters:
            score = parsed["scores"].get(letter)
            if score is None:
                continue
            identity = letter_map.get(letter)
            model_id = identity if identity in ("CAL-strong", "CAL-weak")                 else row_id_to_model_id.get(identity, identity)
            try:
                rank = parsed["ranking"].index(letter) + 1
            except (ValueError, AttributeError, KeyError):
                rank = None
            row = {
                "schema_version": schema.SCHEMA_VERSION,
                "packet_id": pid,
                "judge_id": JUDGE_ID,
                "judge_model_pin": model,
                # No CLI in this path; record the API surface instead so the row
                # still says what produced it.
                "judge_cli_version": f"anthropic-batch-{batch_api.API_VERSION}",
                "delivery": "api-batch",
                "letter": letter,
                "model_id": model_id,
                "score": score,
                "reason": (parsed.get("reasons") or {}).get(letter),
                "rank": rank,
                "ts": ts,
                "status": "ok",
            }
            if store.append_judgment(row):
                written += 1

    print("\n" + "=" * 60)
    print(f"written     : {written} judgment rows")
    print(f"unparseable : {unparseable}   errored: {errored}")
    print(f"tokens      : {tin/1e3:.0f}k in, {tout/1e3:.0f}k out")
    actual = (tin / 1e6 * USD_IN_PER_M + tout / 1e6 * USD_OUT_PER_M) * BATCH_MULT
    print(f"actual cost : ~${actual:.2f} (batch rates, list prices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
