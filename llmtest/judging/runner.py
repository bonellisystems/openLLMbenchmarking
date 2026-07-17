"""Judge runner (TESTPLAN 6.1/7.5) -- pending orchestration: packets -> adapter invoke -> judgments.

Ties together Task 5's `build_cohort_packets`, Task 6's adapters, and
Store's append-only judgments log. `run_pending` is the pure orchestration
core -- all paths/config are passed in explicitly (mirroring
`build_cohort_packets`'s own style) so it's testable against a tmp root
without needing a full on-disk `config/*.yaml` fixture set.
`llmtest/judge_cmd.py` is the thin CLI layer that resolves `cfg`/`root`/
`Store` from the real repo and calls this.

Cohort completeness always packetizes for the FULL judge panel
(`sorted(judges_cfg)`), regardless of `--judge` filtering -- the committed
map's `letters_by_judge` must stay stable across runs that only invoke one
judge at a time. `--judge` only narrows which (packet, judge) pairs get
INVOKED this run.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from llmtest import schema
from llmtest.batteries.b1_fixtures import load_unit_tasks
from llmtest.judging.adapters import FakeJudgeAdapter, make_adapter
from llmtest.judging.packets import PacketRecord, build_cohort_packets
from llmtest.store import Store

# Only Battery 1 rows carry needs_judging today; kept as a set (not a bare
# `== 1`) so a future battery can opt into judging without touching the
# filter's shape.
JUDGED_BATTERIES = {1}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unit_from_task_id(task_id: str) -> str:
    """'b1.cybersecurity-01' -> 'cybersecurity'. Duplicated from
    packets.py's private helper (3 lines) rather than importing a
    leading-underscore symbol across modules."""
    after_battery = task_id.split(".", 1)[1] if "." in task_id else task_id
    unit, sep, _num = after_battery.rpartition("-")
    return unit if sep else after_battery


def build_signals_by_task(root: Path | str, units: list[str]) -> dict[str, list]:
    """{task_id: [signal dicts]} for every task across `units`, keyed the
    same way row/task_id fields are ("b1.<unit>-NN"). Feeds
    `build_cohort_packets`'s `signals_by_task` so every packetized cohort's
    evidence tables render real det-signal results for cohort AND
    calibration answers alike (Task-5 review Finding 1: when a task_id is
    absent, the WHOLE packet degrades to blank evidence to avoid a
    structural tell -- this must not happen for tasks actually being judged)."""
    root = Path(root)
    out: dict[str, list] = {}
    for unit in units:
        for task in load_unit_tasks(root, unit):
            out[f"b1.{task.id}"] = task.signals
    return out


def resolve_cohort_models(cfg) -> list[str]:
    """suite.yaml `b1.cohort_models` override when present (Task 12 knob for
    partial-roster judging waves / the quota dry-run), else every registry
    model without role=quant-arm and with a real local_path -- same filter
    `B1Business.plan()` uses to decide which models actually get B1 rows."""
    override = cfg.suite.get("b1", {}).get("cohort_models")
    if override:
        return list(override)
    return sorted(
        model_id for model_id, m in cfg.registry["models"].items()
        if m.get("role") != "quant-arm"
        and not str(m.get("local_path", "")).startswith("TO-")
    )


def _default_fake_scores(letters: list[str]) -> dict[str, int]:
    return {letter: 5 for letter in letters}


@dataclass
class RunResult:
    packets: list[PacketRecord] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    judgments_written: int = 0
    errors_written: int = 0


def _existing_index(store: Store) -> dict[tuple[str, str], set[str]]:
    """{(packet_id, judge_id): {letters already recorded}} -- "-" in the set
    means a terminal error row is already there for that pair."""
    idx: dict[tuple[str, str], set[str]] = {}
    for j in store.iter_judgments():
        idx.setdefault((j["packet_id"], j["judge_id"]), set()).add(j["letter"])
    return idx


def run_pending(
    *,
    rows: list[dict],
    root: Path | str,
    store: Store,
    rubric_dir: Path | str,
    calibration_dir: Path | str,
    out_artifacts: Path | str,
    out_maps: Path | str,
    judge_prompt_path: Path | str,
    judges_cfg: dict,
    cohort_models: list[str],
    judge_filter: str | None = None,
    packets_only: bool = False,
    fake: bool = False,
    fake_scores_fn=None,
    timeout: int = 300,
    retry_errors: bool = False,
) -> RunResult:
    """Build packets for every complete needs_judging cohort, then invoke
    (packet x judge) pairs lacking a judgment set -- one judgment row per
    letter, idempotent on (packet_id, judge_id, letter); a reply that's
    still invalid after one retry becomes a single letter="-" error row --
    but ONLY when the pair has zero ok letters so far (see below).

    `judges_cfg` is `config/judges.yaml`'s `judges:` dict ({judge_id:
    {model, cli, cli_version, delivery, invoke, ...}}) -- the full panel is
    `sorted(judges_cfg)`, always used for packet-building; `judge_filter`
    (the CLI's `--judge`) only restricts which judges get INVOKED.

    A (packet, judge) pair is "fully judged" (and skipped) only when its
    existing OK letters cover the packet's FULL letter set from the map --
    NOT merely "some row exists for this pair". A pair with partial ok
    letters is pending: it gets re-invoked normally, and only the still-
    missing letters get appended (Store's own (packet_id, judge_id, letter)
    dedup makes re-writing an already-present letter a no-op). If that
    re-invoke still fails while partial ok letters already exist, NO "-"
    error row is written -- doing so would permanently strand the real
    scores behind a terminal marker that blocks all future re-invocation.
    Instead a loud warning is printed (pair stays pending for the next run).
    A "-" error row is only ever written when the pair has ZERO ok letters.

    `retry_errors` (the CLI's `--retry-errors`): a pair whose only existing
    row is a terminal "-" error (zero ok letters) is normally skipped on
    every subsequent run. With `retry_errors=True`, such pairs are treated
    as pending again and re-invoked; the old "-" row is left in place as
    append-only history (a fresh success writes new ok-letter rows, which
    never collide with it since their letters differ).
    """
    root = Path(root)
    judge_ids = sorted(judges_cfg)
    if judge_filter is not None:
        if judge_filter not in judge_ids:
            raise ValueError(f"unknown judge id: {judge_filter!r} (configured: {judge_ids})")
        active_judge_ids = [judge_filter]
    else:
        active_judge_ids = judge_ids

    battery_rows = [r for r in rows
                    if r.get("needs_judging") and r.get("battery") in JUDGED_BATTERIES]
    units = sorted({_unit_from_task_id(r["task_id"]) for r in battery_rows})
    signals_by_task = build_signals_by_task(root, units)

    packets, skipped = build_cohort_packets(
        battery_rows,
        rubric_dir=Path(rubric_dir), calibration_dir=Path(calibration_dir),
        out_artifacts=Path(out_artifacts), out_maps=Path(out_maps), root=root,
        judge_ids=judge_ids, cohort_models=cohort_models,
        judge_prompt_path=Path(judge_prompt_path),
        signals_by_task=signals_by_task,
    )

    result = RunResult(packets=packets, skipped=skipped)
    if packets_only:
        return result

    row_id_to_model_id = {r["row_id"]: r["model_id"] for r in battery_rows}
    existing_index = _existing_index(store)
    scores_fn = fake_scores_fn or _default_fake_scores

    for packet in packets:
        map_data = json.loads(Path(packet.map_path).read_text(encoding="utf-8"))
        letters_by_judge = map_data["letters_by_judge"]

        for judge_id in active_judge_ids:
            letter_map = letters_by_judge.get(judge_id)
            if letter_map is None:
                continue  # defensive: map doesn't carry this judge
            expected_letters = sorted(letter_map)
            key = (packet.packet_id, judge_id)
            existing_letters = existing_index.get(key, set())
            ok_letters = existing_letters - {"-"}
            has_error_row = "-" in existing_letters

            if ok_letters.issuperset(expected_letters):
                continue  # fully judged: every expected letter already has an ok row

            if has_error_row and not ok_letters and not retry_errors:
                continue  # terminal error, zero real progress -- not retrying this run

            body_path = Path(packet.bodies[judge_id])
            packet_text = body_path.read_text(encoding="utf-8")

            if fake:
                adapter = FakeJudgeAdapter(scores_fn)
                judge_model_pin = "fake"
                judge_cli_version = "fake"
            else:
                cfg_entry = judges_cfg[judge_id]
                adapter = make_adapter(judge_id, cfg_entry)
                judge_model_pin = cfg_entry["model"]
                judge_cli_version = cfg_entry.get("cli_version")

            reply = adapter.invoke(packet_text, expected_letters, timeout=timeout,
                                    packet_path=body_path)
            if reply.parsed is None:
                reply = adapter.invoke(packet_text, expected_letters, timeout=timeout,
                                        packet_path=body_path)  # retry once on invalid reply

            ts = _now()
            if reply.parsed is not None:
                parsed = reply.parsed
                for letter in expected_letters:
                    identity = letter_map[letter]
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
                    if store.append_judgment(judgment_row):
                        result.judgments_written += 1
                        existing_index.setdefault(key, set()).add(letter)
            elif ok_letters:
                # Partial ok letters already exist for this pair -- writing a
                # "-" error row here would permanently strand those real
                # scores behind a terminal marker (Finding 1). Warn loudly
                # instead and leave the pair pending for the next run.
                missing = sorted(set(expected_letters) - ok_letters)
                print(
                    f"WARNING: judge {judge_id!r} failed for packet "
                    f"{packet.packet_id!r} ({reply.error or 'invalid reply after retry'}) "
                    f"-- {sorted(ok_letters)} already recorded ok; missing letters "
                    f"{missing} remain PENDING, no error row written",
                    file=sys.stderr,
                )
            else:
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
                if store.append_judgment(error_row):
                    result.errors_written += 1
                    existing_index.setdefault(key, set()).add("-")

    return result


def summarize_judging(store: Store, packets: list[PacketRecord],
                       judge_ids: list[str]) -> dict[str, int]:
    """{'done', 'pending', 'error'} counts over every (packet, judge) pair
    for `judge_ids` across `packets` -- the same done/pending/error
    classification `run_pending` uses to decide whether to (re-)invoke.
    Used by `llmtest status --judging`."""
    existing = _existing_index(store)
    counts = {"done": 0, "pending": 0, "error": 0}
    for packet in packets:
        map_data = json.loads(Path(packet.map_path).read_text(encoding="utf-8"))
        letters_by_judge = map_data["letters_by_judge"]
        for judge_id in judge_ids:
            letter_map = letters_by_judge.get(judge_id)
            if letter_map is None:
                continue
            expected = set(letter_map)
            got = existing.get((packet.packet_id, judge_id), set())
            # Ok-letter completeness is checked BEFORE the "-" presence
            # check: a pair recovered via --retry-errors has a full ok
            # letter set plus a kept historical "-" row (Finding 2 keeps it
            # as append-only history), and must report "done", not "error".
            if got.issuperset(expected):
                counts["done"] += 1
            elif "-" in got:
                counts["error"] += 1
            else:
                counts["pending"] += 1
    return counts
