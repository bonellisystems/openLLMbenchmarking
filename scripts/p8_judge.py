"""P8: CONCURRENT judge driver -- runs pending B1 (packet, judge) pairs
through a thread pool instead of llmtest/judging/runner.py's sequential
run_pending() loop (~40h serial for 360 packets x 3 judges -> a fraction of
that at concurrency=N).

This script REUSES llmtest/judging/runner.py + packets.py + adapters.py's
real internals for every actual judging-logic step; it adds ONLY:
  - concurrency orchestration (ThreadPoolExecutor over pending pairs)
  - a shared threading.Lock serializing Store.append_judgment (NOT
    thread-safe on its own -- it does a full read-then-append of
    judgments.jsonl: reads existing_judgment_keys() via a fresh full-file
    scan, then appends, with no atomicity between the two)
  - a throwaway judgments store for --fake so testing never touches the
    real results/judgments.jsonl

Frozen / untouched: config/judges.yaml, config/registry.yaml,
llmtest/judging/*.py (imported, never edited, never duplicated in spirit --
see the two exceptions below, both trivial and explicitly justified).

Reused pieces, and why each is safe to call from many threads:
  - llmtest.judging.runner.run_pending(..., packets_only=True): builds (or
    idempotently re-confirms, content-hashed) every packet's map.json +
    per-judge body file ONCE, single-threaded, BEFORE the thread pool
    starts. This sidesteps a real race: build_cohort_packets() rewrites
    those files unconditionally every call, so nesting run_pending calls
    inside worker threads (one per pending pair) would let two threads
    truncate/rewrite the SAME body file concurrently -- e.g. while a
    FileDeliveryAdapter (gemini/agy) is reading it from disk via --add-dir.
    Building once up front means workers only ever READ already-written
    body files, never write packet files.
  - llmtest.judging.runner._existing_index(store): the runner's own
    (packet_id, judge_id) -> {ok letters, "-"} indexer, imported and called
    directly (not reimplemented) so "what's already judged" always matches
    the sequential runner's own bookkeeping byte-for-byte.
  - llmtest.judging.runner.resolve_cohort_models / JUDGED_BATTERIES:
    imported as-is (public names).
  - llmtest.judging.adapters.make_adapter / FakeJudgeAdapter / .invoke():
    each call builds a fresh adapter instance and spawns its own subprocess
    (or, for fake, does pure in-memory work) -- no shared mutable state
    across calls (BaseAdapter.argv is built once in __init__ and only ever
    READ inside invoke()), so concurrent invocation from N threads is safe.
    The one module-level mutable state in adapters.py, make_adapter's
    `_file_delivery_variants` memo cache, is pre-warmed single-threaded
    below (one call per active judge id before the pool starts) so no two
    threads race to populate it on the first gemini call; even unwarmed
    this is harmless (worst case two equivalent classes get minted).
  - Reply parsing is 100% inside adapter.invoke() (-> parse_reply); this
    script never parses a judge's raw reply itself.

Two small exceptions, both control-flow glue rather than judging logic,
copied because run_pending doesn't expose them as standalone callables:
  (a) the pending-pair predicate (ok-letters superset check + the
      has_error_row/--retry-errors skip), mirrored line-for-line from
      run_pending's inner loop (runner.py ~186-200).
  (b) the per-pair invoke-once/retry-once/row-shaping glue (build the
      adapter, read the body, invoke, retry on invalid reply, shape the
      judgment/error dict, decide ok/warn/error), mirrored from the same
      loop body (runner.py ~202-275). Every field name/value and the
      retry-once-then-error/never-strand-partial-ok-behind-error-row rule
      match runner.py exactly -- see compute_pending_pairs() and
      judge_one_pair() below, each annotated with the runner.py lines they
      mirror.

Usage:
    python scripts/p8_judge.py --fake --concurrency 8 --limit 20
    python scripts/p8_judge.py --concurrency 12
    python scripts/p8_judge.py --concurrency 12 --judge codex
    python scripts/p8_judge.py --concurrency 12 --retry-errors
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest import schema                                            # noqa: E402
from llmtest.judging.adapters import FakeJudgeAdapter, make_adapter    # noqa: E402
from llmtest.judging.runner import (                                   # noqa: E402
    JUDGED_BATTERIES,
    _existing_index,
    resolve_cohort_models,
    run_pending,
)
from llmtest.registry import load_config                               # noqa: E402
from llmtest.store import Store                                        # noqa: E402


def _now() -> str:
    # Mirrors runner.py's private _now() (a one-liner) -- kept local
    # rather than importing a leading-underscore symbol across modules
    # for something this trivial (same convention runner.py itself uses
    # for packets.py's _unit_from_task_id).
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_fake_scores(letters: list[str]) -> dict[str, int]:
    return {letter: 5 for letter in letters}


class PendingPair:
    __slots__ = ("packet", "judge_id", "letter_map", "expected_letters", "ok_letters_before")

    def __init__(self, packet, judge_id, letter_map, expected_letters, ok_letters_before):
        self.packet = packet
        self.judge_id = judge_id
        self.letter_map = letter_map
        self.expected_letters = expected_letters
        self.ok_letters_before = ok_letters_before  # snapshot at enumeration time


@dataclass
class PairResult:
    judge_id: str
    written: int
    outcome: str  # "ok" | "warn" | "error"


def compute_pending_pairs(packets, judge_ids: list[str], store: Store,
                           retry_errors: bool) -> list[PendingPair]:
    """Mirrors run_pending()'s inner pending predicate EXACTLY
    (runner.py run_pending ~186-200): a pair is skipped when its ok
    letters already cover the packet's full expected set, or when it has
    a terminal "-" row with zero ok letters and --retry-errors wasn't
    passed. Uses the runner's own _existing_index() helper (imported, not
    reimplemented) so "what's already recorded" can't drift from what the
    sequential runner would see for the same store."""
    existing_index = _existing_index(store)
    pairs: list[PendingPair] = []
    for packet in packets:
        map_data = json.loads(Path(packet.map_path).read_text(encoding="utf-8"))
        letters_by_judge = map_data["letters_by_judge"]
        for judge_id in judge_ids:
            letter_map = letters_by_judge.get(judge_id)
            if letter_map is None:
                continue
            expected_letters = sorted(letter_map)
            key = (packet.packet_id, judge_id)
            existing_letters = existing_index.get(key, set())
            ok_letters = existing_letters - {"-"}
            has_error_row = "-" in existing_letters

            if ok_letters.issuperset(expected_letters):
                continue  # fully judged
            if has_error_row and not ok_letters and not retry_errors:
                continue  # terminal error, not retrying this run

            pairs.append(PendingPair(packet, judge_id, letter_map, expected_letters, ok_letters))
    return pairs


def judge_one_pair(pair: PendingPair, *, judges_cfg: dict, fake: bool, fake_scores_fn,
                    timeout: int, row_id_to_model_id: dict, store: Store,
                    write_lock: threading.Lock) -> PairResult:
    """Mirrors run_pending()'s per-(packet, judge) invocation body
    (runner.py run_pending ~202-275): read the body, invoke, retry once on
    an invalid reply, then either write one ok row per letter, write a
    single terminal "-" error row (only when ZERO ok letters exist for
    this pair), or -- if some ok letters already exist but the retry still
    failed -- warn loudly and write nothing (never strand real scores
    behind a terminal error marker). Runs entirely in this worker thread
    except the store.append_judgment() call(s), which are serialized under
    `write_lock` (Store's own read-then-append is not atomic)."""
    packet = pair.packet
    judge_id = pair.judge_id
    expected_letters = pair.expected_letters

    body_path = Path(packet.bodies[judge_id])
    packet_text = body_path.read_text(encoding="utf-8")

    if fake:
        adapter = FakeJudgeAdapter(fake_scores_fn)
        judge_model_pin = "fake"
        judge_cli_version = "fake"
    else:
        cfg_entry = judges_cfg[judge_id]
        adapter = make_adapter(judge_id, cfg_entry)
        judge_model_pin = cfg_entry["model"]
        judge_cli_version = cfg_entry.get("cli_version")

    reply = adapter.invoke(packet_text, expected_letters, timeout=timeout, packet_path=body_path)
    if reply.parsed is None:
        reply = adapter.invoke(packet_text, expected_letters, timeout=timeout,
                                packet_path=body_path)  # retry once on invalid reply
    # Fallback model (e.g. claude fable-5 -> opus-4-8): if the primary model failed
    # both attempts and a fallback_model is configured, try it once; on success switch
    # the recorded pin so provenance reflects which model actually judged the packet.
    if reply.parsed is None and not fake:
        fb = judges_cfg[judge_id].get("fallback_model")
        if fb:
            fb_entry = dict(judges_cfg[judge_id]); fb_entry["model"] = fb
            fb_reply = make_adapter(judge_id, fb_entry).invoke(
                packet_text, expected_letters, timeout=timeout, packet_path=body_path)
            if fb_reply.parsed is not None:
                reply = fb_reply
                judge_model_pin = fb

    ts = _now()
    written = 0

    if reply.parsed is not None:
        parsed = reply.parsed
        for letter in expected_letters:
            identity = pair.letter_map[letter]
            model_id = identity if identity in ("CAL-strong", "CAL-weak") \
                else row_id_to_model_id.get(identity, identity)
            judgment_row = {
                "schema_version": schema.SCHEMA_VERSION,
                "packet_id": packet.packet_id,
                "judge_id": judge_id,
                "judge_model_pin": judge_model_pin,
                "judge_cli_version": judge_cli_version,
                "letter": letter,
                "model_id": model_id,
                "score": parsed["scores"][letter],
                "reason": parsed["reasons"][letter],
                "rank": parsed["ranking"].index(letter) + 1,
                "ts": ts,
                "status": "ok",
            }
            with write_lock:
                wrote = store.append_judgment(judgment_row)
            if wrote:
                written += 1
        return PairResult(judge_id=judge_id, written=written, outcome="ok")

    if pair.ok_letters_before:
        # Partial ok letters already exist -- writing a "-" row here would
        # permanently strand those real scores behind a terminal marker.
        # Warn instead and leave the pair pending for the next run.
        missing = sorted(set(expected_letters) - pair.ok_letters_before)
        print(
            f"WARNING: judge {judge_id!r} failed for packet {packet.packet_id!r} "
            f"({reply.error or 'invalid reply after retry'}) -- "
            f"{sorted(pair.ok_letters_before)} already recorded ok; missing letters "
            f"{missing} remain PENDING, no error row written",
            file=sys.stderr,
        )
        return PairResult(judge_id=judge_id, written=0, outcome="warn")

    error_row = {
        "schema_version": schema.SCHEMA_VERSION,
        "packet_id": packet.packet_id,
        "judge_id": judge_id,
        "judge_model_pin": judge_model_pin,
        "judge_cli_version": judge_cli_version,
        "letter": "-",
        "model_id": None,
        "score": None,
        "reason": reply.error or "invalid reply after retry",
        "rank": None,
        "ts": ts,
        "status": "error",
    }
    with write_lock:
        wrote = store.append_judgment(error_row)
    return PairResult(judge_id=judge_id, written=0, outcome="error" if wrote else "warn")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Concurrent judge driver (P8)")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--judge", choices=["claude", "codex", "gemini"], default=None,
                     help="restrict INVOCATION to one judge (packets always cover the full panel)")
    ap.add_argument("--retry-errors", action="store_true",
                     help="re-invoke pairs whose only existing row is a terminal '-' error")
    ap.add_argument("--fake", action="store_true",
                     help="use FakeJudgeAdapter (no real CLIs); judgments go to a throwaway "
                          "store, never results/judgments.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap number of pairs processed")
    ap.add_argument("--timeout", type=int, default=300, help="per-invoke subprocess timeout (s)")
    ap.add_argument("--fake-store-dir", default=None,
                     help="override the throwaway --fake results dir (default: a fresh dir "
                          "under the system temp dir, via tempfile.mkdtemp)")
    args = ap.parse_args(argv)

    root = ROOT
    cfg = load_config(root)
    judges_cfg = cfg.judges["judges"]

    if args.judge is not None and args.judge not in judges_cfg:
        print(f"unknown judge id {args.judge!r}; configured: {sorted(judges_cfg)}")
        return 2

    rows_store = Store(root / "results")  # ALWAYS the real store: rows are read-only here
    rows = list(rows_store.iter_rows())
    cohort_models = resolve_cohort_models(cfg)

    if args.fake:
        fake_dir = Path(args.fake_store_dir) if args.fake_store_dir \
            else Path(tempfile.mkdtemp(prefix="llmtest_fake_judge_"))
        judge_store = Store(fake_dir)
        print(f"[fake] judgments will be written to throwaway store: {judge_store.dir} "
              f"(real results/judgments.jsonl is never opened in --fake mode)")
    else:
        judge_store = rows_store

    # Build/confirm every packet ONCE, single-threaded, at real repo paths
    # (idempotent + content-hashed -- safe even when nothing changed).
    # packets_only=True returns before touching `judge_store` at all, so
    # which Store instance is passed here is immaterial.
    packets_result = run_pending(
        rows=rows, root=root, store=judge_store,
        rubric_dir=root / "grading" / "anchors",
        calibration_dir=root / "grading" / "calibration",
        out_artifacts=root / "artifacts" / "packets",
        out_maps=root / "results" / "packets",
        judge_prompt_path=root / "grading" / "judge_prompt.md",
        judges_cfg=judges_cfg, cohort_models=cohort_models,
        judge_filter=args.judge, packets_only=True,
    )
    packets = packets_result.packets
    print(f"packets: {len(packets)} built/confirmed, {len(packets_result.skipped)} cohorts skipped")

    judge_ids = sorted(judges_cfg)
    active_judge_ids = [args.judge] if args.judge else judge_ids

    if not args.fake:
        # Pre-warm make_adapter's file-delivery memo cache single-threaded
        # so no two worker threads race to populate it on first use.
        for jid in active_judge_ids:
            make_adapter(jid, judges_cfg[jid])

    pairs = compute_pending_pairs(packets, active_judge_ids, judge_store, args.retry_errors)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    total = len(pairs)
    print(f"pending pairs: {total} (concurrency={args.concurrency}, "
          f"judge_filter={args.judge or 'ALL'}, retry_errors={args.retry_errors}, fake={args.fake})")
    if total == 0:
        print("nothing to do")
        return 0

    battery_rows = [r for r in rows if r.get("needs_judging") and r.get("battery") in JUDGED_BATTERIES]
    row_id_to_model_id = {r["row_id"]: r["model_id"] for r in battery_rows}

    write_lock = threading.Lock()
    counters_lock = threading.Lock()
    done = 0
    per_judge_done = {jid: 0 for jid in active_judge_ids}
    error_count = 0
    warn_count = 0
    written_total = 0

    def _run(pair: PendingPair) -> PairResult:
        return judge_one_pair(
            pair, judges_cfg=judges_cfg, fake=args.fake, fake_scores_fn=_default_fake_scores,
            timeout=args.timeout, row_id_to_model_id=row_id_to_model_id,
            store=judge_store, write_lock=write_lock,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_run, pair): pair for pair in pairs}
        for fut in as_completed(futures):
            pair = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001 -- surface, keep the pool alive
                with counters_lock:
                    done += 1
                    error_count += 1
                print(f"EXCEPTION pair packet={pair.packet.packet_id} judge={pair.judge_id}: "
                      f"{exc!r}", file=sys.stderr)
                continue

            with counters_lock:
                done += 1
                per_judge_done[result.judge_id] = per_judge_done.get(result.judge_id, 0) + 1
                written_total += result.written
                if result.outcome == "error":
                    error_count += 1
                elif result.outcome == "warn":
                    warn_count += 1
                if done % 10 == 0 or done == total:
                    per_judge_str = ", ".join(f"{j}={c}" for j, c in sorted(per_judge_done.items()))
                    print(f"progress: {done}/{total} pairs done | per-judge: {per_judge_str} | "
                          f"errors={error_count} warns={warn_count} | "
                          f"judgments_written={written_total}")

    print(f"DONE: {done}/{total} pairs processed, {written_total} judgment rows written, "
          f"{error_count} error pairs, {warn_count} warn(pending-retained) pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
