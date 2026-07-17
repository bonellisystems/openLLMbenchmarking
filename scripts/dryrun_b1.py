"""P3 Task 12 -- QUOTA DRY-RUN driver: real GPU generation for the 3-model dry-run
cohort (gpt-oss-20b, qwen3.6-35b-a3b, gemma-4-26b-a4b) x {cybersecurity, it_infra}
x run_n=1, plus a post-hoc measurement report over the resulting judgments.

Evolved from scripts/slice_b1.py (Task 10's single-task vertical-slice driver):
`gen` batches ALL pending WorkItems for one model (units in DRY-RUN_UNITS,
run_n==1) into ONE process invocation, reusing a single ServerManager/server
launch across all 16 tasks for that model -- the dominant cost per model is the
server launch (weights load), not per-task inference, so one launch beats 16.
RESUMABLE: rows already in the store are skipped via row_id dedupe (pre-filtered
against existing_row_ids(); Store.append() would also no-op on a collision, this
just avoids re-executing a completed row's GPU call at all). Re-invoke the same
command and it picks up wherever it left off.

`report` does NOT invoke any judge or generate anything -- it's pure post-hoc
measurement over results/judgments.jsonl + the packet body files already on
disk. Real per-call token counts are NOT recoverable: judgments.jsonl stores the
PARSED reply (scores/reasons/ranking), not the judge CLI's raw stdout, so both
input and output are ESTIMATED via a chars/4 heuristic --
  - input: the actual packet body file bytes (exact chars, real; this is
    genuinely what was sent on stdin/via file to the judge).
  - output: the parsed {scores, reasons, ranking} reconstituted as compact JSON
    (a FLOOR estimate -- it excludes any prose the CLI wrapped around the JSON
    that parse_reply() discarded, and excludes invisible reasoning-token spend
    for CLIs that burn a hidden reasoning budget, e.g. codex's ultra effort).
This limitation is stated explicitly in the report's own output, not just here.

Usage:
    python scripts/dryrun_b1.py gen --model gpt-oss-20b
    python scripts/dryrun_b1.py gen --model qwen3.6-35b-a3b
    python scripts/dryrun_b1.py gen --model gemma-4-26b-a4b
    python scripts/dryrun_b1.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.batteries.b1_business import B1Business          # noqa: E402
from llmtest.registry import load_config                      # noqa: E402
from llmtest.run_cmd import RunContext                         # noqa: E402
from llmtest.store import Store                                # noqa: E402

DRYRUN_UNITS = ("cybersecurity", "it_infra")


def _unit_of(task_id: str) -> str:
    """'b1.cybersecurity-01' -> 'cybersecurity' (same convention as
    judging/packets.py's _unit_from_task_id)."""
    suffix = task_id.split(".", 1)[1] if "." in task_id else task_id
    unit, sep, _num = suffix.rpartition("-")
    return unit if sep else suffix


def do_gen(args) -> int:
    cfg = load_config(ROOT)
    store = Store(ROOT / "results")
    battery = B1Business()

    items = battery.plan(cfg, store, model_filter=args.model, force=False)
    items = [i for i in items if _unit_of(i.task_id) in DRYRUN_UNITS and i.run_n == 1]
    if not items:
        print(f"dryrun_b1 gen: no planned items for model={args.model!r} "
              f"(units={DRYRUN_UNITS}, run_n=1) -- check registry.yaml / unit dirs")
        return 1

    done = store.existing_row_ids()
    pending = [i for i in items if i.row_id not in done]
    print(f"dryrun_b1 gen: model={args.model} planned={len(items)} "
          f"already-done={len(items) - len(pending)} pending={len(pending)}")
    if not pending:
        print("dryrun_b1 gen: 0 pending -- nothing to do")
        return 0

    ctx = RunContext(cfg=cfg, store=store, root=ROOT, keep_server=False, debug=False)
    failures = 0
    t0 = time.time()
    try:
        for i, item in enumerate(pending, 1):
            item_t0 = time.time()
            try:
                for row in battery.execute(item, ctx):
                    appended = store.append(row)
                    dt = time.time() - item_t0
                    print(f"  [{i}/{len(pending)}] {item.task_id} "
                          f"row_id={row['row_id'][:12]} status={row['status']} "
                          f"appended={appended} ({dt:.1f}s)")
            except Exception as e:                       # row-level containment
                failures += 1
                print(f"  [{i}/{len(pending)}] EXEC-ERROR {item.task_id}: {e}")
    finally:
        # VRAM drain: always teardown at the end of this invocation, mirroring
        # run_cmd.run_run()'s finally block -- next model's `gen` starts clean.
        if ctx.server is not None:
            ctx.server.teardown()
            print("dryrun_b1 gen: server torn down (VRAM released)")
    print(f"dryrun_b1 gen: done in {time.time() - t0:.1f}s, {failures} failures")
    return 1 if failures else 0


def _reconstruct_output_text(rows: list[dict]) -> str:
    """Rebuild the compact-JSON reply body a judge's ok letters imply, from
    the PARSED fields judgments.jsonl actually stores (score, reason, rank).
    Used only to floor-estimate output chars -- see module docstring."""
    scores = {r["letter"]: r["score"] for r in rows}
    reasons = {r["letter"]: r["reason"] for r in rows}
    ranking = [r["letter"] for r in sorted(rows, key=lambda r: r["rank"])]
    return json.dumps({"scores": scores, "reasons": reasons, "ranking": ranking})


def _packet_id_from_map_path(map_path: Path) -> str:
    name = map_path.name
    suffix = ".map.json"
    return name[: -len(suffix)] if name.endswith(suffix) else map_path.stem


def do_report(args) -> int:
    store = Store(ROOT / "results")
    judgments = list(store.iter_judgments())
    if not judgments:
        print("dryrun_b1 report: no judgments.jsonl rows yet")
        return 0

    packets_dir = ROOT / "artifacts" / "packets"
    maps_dir = ROOT / "results" / "packets"

    # Scope to the Task 12 dry-run packets ONLY: 3-model cohort + 2
    # calibration letters = 5 expected letters per judge. This deliberately
    # excludes any other *.map.json on disk (e.g. an earlier single-model
    # vertical-slice packet with a different letter count) so this burn
    # summary measures ONLY the dry-run's real judging.
    dryrun_expected: dict[str, dict[str, set]] = {}   # packet_id -> judge_id -> {letters}
    for map_path in sorted(maps_dir.glob("*.map.json")):
        m = json.loads(map_path.read_text(encoding="utf-8"))
        letters_by_judge = m.get("letters_by_judge", {})
        if any(len(v) != 5 for v in letters_by_judge.values()):
            continue
        dryrun_expected[_packet_id_from_map_path(map_path)] = {
            jid: set(letters) for jid, letters in letters_by_judge.items()}

    by_judge_packet: dict[tuple, list[dict]] = {}
    for j in judgments:
        if j["packet_id"] not in dryrun_expected:
            continue
        by_judge_packet.setdefault((j["judge_id"], j["packet_id"]), []).append(j)

    judge_ids = sorted({jid for jid, _pid in by_judge_packet})
    print(f"dryrun_b1 report: {len(dryrun_expected)} dry-run packets in scope "
          "(other packet.map.json files on disk, e.g. earlier single-model "
          "slices, are excluded)")
    print("NOTE: input chars are REAL packet-body bytes; output chars are a "
          "FLOOR estimate reconstructed from the parsed scores/reasons/ranking "
          "-- raw judge CLI stdout is not persisted, so true output including "
          "any wrapper prose or hidden reasoning-token spend is not recoverable "
          "from this log; token counts below are chars/4 estimates throughout, "
          "not exact tokenizer counts, for ALL three judges including claude. "
          "A pair is classified 'ok' when its ok-status letters cover the "
          "packet's full expected letter set (regardless of any earlier "
          "superseded error row for the same pair -- append-only history), "
          "and 'error' only when it has zero ok letters and at least one "
          "terminal error row (the runner's own completeness rule).")
    print()

    grand_ok = grand_err = grand_in = grand_out = 0
    for judge_id in judge_ids:
        pairs = {pid: rows for (jid, pid), rows in by_judge_packet.items() if jid == judge_id}
        n_ok_pairs = n_err_pairs = 0
        in_chars_total = out_chars_total = 0
        ts_values = []
        for packet_id, rows in pairs.items():
            expected = dryrun_expected[packet_id].get(judge_id, set())
            ok_rows = [r for r in rows if r["status"] == "ok"]
            ok_letters = {r["letter"] for r in ok_rows}
            body_path = packets_dir / f"{packet_id}.{judge_id}.txt"
            if body_path.exists():
                in_chars_total += len(body_path.read_text(encoding="utf-8"))
            if ok_letters and ok_letters.issuperset(expected):
                n_ok_pairs += 1
                out_chars_total += len(_reconstruct_output_text(ok_rows))
                ts_values.append(max(r["ts"] for r in ok_rows))
            elif not ok_letters and any(r["status"] == "error" for r in rows):
                n_err_pairs += 1
                ts_values.append(max(r["ts"] for r in rows if r["status"] == "error"))
            # else: genuinely partial (some ok letters, not all) -- doesn't
            # occur in a finished run; deliberately uncounted either way
            # rather than silently misclassified.
        n_calls = n_ok_pairs + n_err_pairs
        mean_in = in_chars_total / n_calls if n_calls else 0.0
        mean_out = out_chars_total / n_calls if n_calls else 0.0
        span = f"{min(ts_values)} .. {max(ts_values)}" if ts_values else "n/a"

        print(f"judge={judge_id}")
        print(f"  pairs judged: {n_ok_pairs} ok, {n_err_pairs} error, {n_calls} total "
              f"(of {len(dryrun_expected)} dry-run packets)")
        print(f"  input:  total {in_chars_total} chars (mean {mean_in:.0f}/call) "
              f"~= {in_chars_total / 4:.0f} tokens (mean ~{mean_in / 4:.0f}/call)")
        print(f"  output: total {out_chars_total} chars (mean {mean_out:.0f}/call, "
              f"FLOOR estimate) ~= {out_chars_total / 4:.0f} tokens "
              f"(mean ~{mean_out / 4:.0f}/call)")
        print(f"  judgment ts span (log-recorded, second precision): {span}")
        print()

        grand_ok += n_ok_pairs
        grand_err += n_err_pairs
        grand_in += in_chars_total
        grand_out += out_chars_total

    print(f"TOTAL: {grand_ok} ok + {grand_err} error = {grand_ok + grand_err} "
          f"(packet x judge) pairs judged, "
          f"{grand_in} input chars (~{grand_in / 4:.0f} tok), "
          f"{grand_out} output chars (~{grand_out / 4:.0f} tok, floor estimate)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)

    g = sub.add_parser("gen", help="plan()+execute() ALL pending dry-run WorkItems "
                                    "for one model (cybersecurity+it_infra, run_n=1), "
                                    "one server launch, resumable")
    g.add_argument("--model", required=True)
    g.set_defaults(func=do_gen)

    r = sub.add_parser("report", help="post-hoc per-judge burn summary from "
                                       "results/judgments.jsonl + packet bodies "
                                       "(no judge invocation, no GPU)")
    r.set_defaults(func=do_report)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
