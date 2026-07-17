"""Tests for the judge runner (TESTPLAN 6.1/7.5) -- pending orchestration,
delivery-agnostic invocation (via FakeJudgeAdapter, never a real CLI),
idempotency, retry-then-error, and status --judging counts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from llmtest import schema
from llmtest.judging.runner import (
    build_signals_by_task,
    resolve_cohort_models,
    run_pending,
    summarize_judging,
)
from llmtest.store import Store

ROOT = Path(__file__).resolve().parents[1]  # real repo root -> real grading/judge_prompt.md

UNIT = "cybersecurity"
COHORT = ["model-a", "model-b"]
JUDGES_CFG = {
    "j1": {"model": "pin-1", "cli_version": "v1", "invoke": "unused {model}"},
    "j2": {"model": "pin-2", "cli_version": "v2", "invoke": "unused {model}"},
}


def _write_task(root: Path, unit: str, num: int, prompt: str, signals=None) -> str:
    unit_dir = root / "suite" / "b1_business" / unit
    unit_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{unit}-{num:02d}"
    data = {"id": suffix, "unit": unit, "difficulty": "easy", "class": "short",
            "prompt": prompt, "signals": signals or []}
    (unit_dir / f"task-{num:02d}.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return f"b1.{suffix}"


def _write_anchor(rubric_dir: Path, unit: str, text="anchor v1 -- 0/3/5/7/10 definitions"):
    rubric_dir.mkdir(parents=True, exist_ok=True)
    (rubric_dir / f"{unit}.md").write_text(text, encoding="utf-8")


def _write_calibration(cal_dir: Path):
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "strong.md").write_text("A frontier-quality reference answer.", encoding="utf-8")
    (cal_dir / "weak.md").write_text("A vague, partly-wrong reference answer.", encoding="utf-8")


def _make_row(root: Path, store: Store, model_id: str, task_id: str, run_n=1,
              text: str | None = None, det_checks: dict | None = None) -> dict:
    text = text or f"Answer from {model_id} for {task_id}: MFA enforced, patched."
    quant_sha = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
    fixture_sha = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    row = schema.ResultRow.new(
        suite_version="suite-test", model_id=model_id, hf_repo="o/r",
        quant_file="q.gguf", quant_sha256=quant_sha, tier="T1", battery=1,
        task_id=task_id, fixture_sha=fixture_sha,
        condition="runtime=fork;spec=ngram32;kv=q8;ctx=16k;cond=B1",
        run_n=run_n, session_id="s", needs_judging=True, status="ok",
        det_checks=det_checks if det_checks is not None else {})
    art_dir = root / "artifacts" / "b1"
    art_dir.mkdir(parents=True, exist_ok=True)
    path = art_dir / f"{row.row_id}.txt"
    path.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    row.artifacts = {"response": {"sha256": sha, "relpath": f"b1/{row.row_id}.txt"}}
    d = row.to_dict()
    store.append(d)
    return d


def _base_kwargs(tmp_path: Path) -> dict:
    rubric_dir = tmp_path / "grading" / "anchors"
    calibration_dir = tmp_path / "grading" / "calibration"
    _write_anchor(rubric_dir, UNIT)
    _write_calibration(calibration_dir)
    return dict(
        root=tmp_path, rubric_dir=rubric_dir, calibration_dir=calibration_dir,
        out_artifacts=tmp_path / "artifacts" / "packets",
        out_maps=tmp_path / "results" / "packets",
        judge_prompt_path=ROOT / "grading" / "judge_prompt.md",
        cohort_models=COHORT,
    )


def _seed_two_tasks(tmp_path: Path, store: Store) -> list[str]:
    """2 tasks in one unit x 2 cohort models -> 2 complete cohorts (packets)."""
    task_ids = [_write_task(tmp_path, UNIT, 1, "Draft an incident summary."),
                _write_task(tmp_path, UNIT, 2, "Triage this alert.")]
    for task_id in task_ids:
        for model_id in COHORT:
            _make_row(tmp_path, store, model_id, task_id)
    return task_ids


# --- end-to-end fake run: judgment count = packets x judges x letters ---


def test_end_to_end_fake_run_writes_expected_judgment_count(tmp_path):
    store = Store(tmp_path / "results")
    _seed_two_tasks(tmp_path, store)
    rows = list(store.iter_rows())

    result = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True,
                          **_base_kwargs(tmp_path))

    assert result.skipped == []
    assert len(result.packets) == 2                       # 2 tasks -> 2 cohorts
    # letters per packet = 2 cohort models + CAL-strong + CAL-weak = 4
    assert result.judgments_written == 2 * 2 * 4           # packets x judges x letters
    assert result.errors_written == 0
    assert len(list(store.iter_judgments())) == 16
    assert all(j["status"] == "ok" for j in store.iter_judgments())


def test_idempotent_rerun_adds_no_new_judgments(tmp_path):
    store = Store(tmp_path / "results")
    _seed_two_tasks(tmp_path, store)
    kwargs = _base_kwargs(tmp_path)
    rows = list(store.iter_rows())

    first = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True, **kwargs)
    assert first.judgments_written == 16

    second = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True, **kwargs)
    assert second.judgments_written == 0
    assert second.errors_written == 0
    assert len(list(store.iter_judgments())) == 16          # unchanged


def test_packets_only_skips_invocation_entirely(tmp_path):
    store = Store(tmp_path / "results")
    _seed_two_tasks(tmp_path, store)
    rows = list(store.iter_rows())

    result = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True,
                          packets_only=True, **_base_kwargs(tmp_path))

    assert len(result.packets) == 2
    assert result.judgments_written == 0
    assert result.errors_written == 0
    assert list(store.iter_judgments()) == []


# --- retry-once-then-error ---


def _garbage_scores(letters):
    # float scores -> parse_reply rejects (bool/int-only) -> JudgeReply.error set.
    return {letter: 7.5 for letter in letters}


def _seed_ok_judgment(store: Store, packet_id: str, judge_id: str, letter: str,
                       model_id: str, score: int = 7) -> None:
    store.append_judgment({
        "schema_version": schema.SCHEMA_VERSION,
        "packet_id": packet_id,
        "judge_id": judge_id,
        "judge_model_pin": "pin-1",
        "judge_cli_version": "v1",
        "letter": letter,
        "model_id": model_id,
        "score": score,
        "reason": "seeded ok",
        "rank": 1,
        "ts": "2026-07-17T00:00:00+00:00",
        "status": "ok",
    })


def _seed_error_row(store: Store, packet_id: str, judge_id: str) -> None:
    store.append_judgment({
        "schema_version": schema.SCHEMA_VERSION,
        "packet_id": packet_id,
        "judge_id": judge_id,
        "judge_model_pin": "pin-1",
        "judge_cli_version": "v1",
        "letter": "-",
        "model_id": None,
        "score": None,
        "reason": "seeded terminal error",
        "rank": None,
        "ts": "2026-07-17T00:00:00+00:00",
        "status": "error",
    })


def _letter_model_id(letter_map: dict, letter: str, rows: list[dict]) -> str:
    """Mirror the runner's own identity -> model_id resolution so seeded
    judgment rows in tests look exactly like ones the runner itself would
    have written."""
    identity = letter_map[letter]
    if identity in ("CAL-strong", "CAL-weak"):
        return identity
    row_id_to_model_id = {r["row_id"]: r["model_id"] for r in rows}
    return row_id_to_model_id[identity]


def test_adapter_returning_garbage_twice_writes_one_error_row(tmp_path):
    """A judge whose replies never validate must be retried exactly once,
    then produce a SINGLE letter="-" error row per (packet, judge) -- not
    one row per retry attempt, not one per expected letter."""
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())

    result = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                          fake=True, fake_scores_fn=_garbage_scores, **_base_kwargs(tmp_path))

    assert len(result.packets) == 1
    assert result.judgments_written == 0
    assert result.errors_written == 1
    judgments = list(store.iter_judgments())
    assert len(judgments) == 1
    err = judgments[0]
    assert err["letter"] == "-"
    assert err["score"] is None
    assert err["status"] == "error"
    assert err["model_id"] is None

    # Idempotent: a terminal error row blocks re-invocation, not just re-write.
    result2 = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                           fake=True, fake_scores_fn=_garbage_scores, **_base_kwargs(tmp_path))
    assert result2.errors_written == 0
    assert len(list(store.iter_judgments())) == 1


# --- mixed-state pairs: partial ok letters are pending, never stranded (Finding 1) ---


def test_partial_ok_pair_is_pending_and_completes(tmp_path):
    """A pair with SOME ok letters already recorded (but not the packet's
    full letter set) must be treated as pending, not "fully judged". A
    later successful run appends only the missing letters -- Store dedup
    no-ops the already-present one -- and writes no "-" error row."""
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())
    kwargs = _base_kwargs(tmp_path)

    packets_result = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                                  fake=True, packets_only=True, **kwargs)
    packet = packets_result.packets[0]
    map_data = json.loads(Path(packet.map_path).read_text(encoding="utf-8"))
    letter_map = map_data["letters_by_judge"]["j1"]
    expected_letters = sorted(letter_map)
    assert len(expected_letters) == 4                # 2 cohort + CAL-strong + CAL-weak

    seed_letter = expected_letters[0]
    seed_model_id = _letter_model_id(letter_map, seed_letter, rows)
    _seed_ok_judgment(store, packet.packet_id, "j1", seed_letter, seed_model_id)

    result = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                          fake=True, **kwargs)

    assert result.judgments_written == 3             # the 3 still-missing letters
    assert result.errors_written == 0
    judgments = list(store.iter_judgments())
    assert len(judgments) == 4
    assert all(j["letter"] != "-" for j in judgments)
    assert {j["letter"] for j in judgments} == set(expected_letters)


def test_error_row_only_when_zero_ok_letters(tmp_path, capsys):
    """A retry that still fails while PARTIAL ok letters already exist must
    NOT write a "-" error row -- that would permanently strand the real
    scores behind a terminal marker. Instead: print a loud warning naming
    the pair and the still-missing letters, and leave the pair pending."""
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())
    kwargs = _base_kwargs(tmp_path)

    packets_result = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                                  fake=True, packets_only=True, **kwargs)
    packet = packets_result.packets[0]
    map_data = json.loads(Path(packet.map_path).read_text(encoding="utf-8"))
    letter_map = map_data["letters_by_judge"]["j1"]
    expected_letters = sorted(letter_map)
    seed_letter = expected_letters[0]
    seed_model_id = _letter_model_id(letter_map, seed_letter, rows)
    _seed_ok_judgment(store, packet.packet_id, "j1", seed_letter, seed_model_id)

    result = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                          fake=True, fake_scores_fn=_garbage_scores, **kwargs)

    assert result.errors_written == 0
    assert result.judgments_written == 0
    judgments = list(store.iter_judgments())
    assert len(judgments) == 1                        # only the pre-seeded ok row
    assert judgments[0]["letter"] == seed_letter

    captured = capsys.readouterr()
    noise = captured.out + captured.err
    assert "j1" in noise
    assert packet.packet_id in noise
    for letter in sorted(set(expected_letters) - {seed_letter}):
        assert letter in noise

    # The pair stays pending -- proven by a later run with a WORKING judge
    # still completing it (a "-" row would have blocked re-invocation).
    result2 = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                           fake=True, **kwargs)
    assert result2.judgments_written == 3
    final = list(store.iter_judgments())
    assert len(final) == 4
    assert all(j["letter"] != "-" for j in final)


# --- --retry-errors: terminal error rows become pending again (Finding 2) ---


def test_retry_errors_flag_rejudges_terminal_errors(tmp_path):
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())
    kwargs = _base_kwargs(tmp_path)

    packets_result = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                                  fake=True, packets_only=True, **kwargs)
    packet = packets_result.packets[0]
    map_data = json.loads(Path(packet.map_path).read_text(encoding="utf-8"))
    expected_letters = sorted(map_data["letters_by_judge"]["j1"])

    _seed_error_row(store, packet.packet_id, "j1")

    without_flag = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                                fake=True, **kwargs)
    assert without_flag.judgments_written == 0
    assert without_flag.errors_written == 0
    assert len(list(store.iter_judgments())) == 1     # untouched: just the seeded "-" row

    with_flag = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                             fake=True, retry_errors=True, **kwargs)
    assert with_flag.judgments_written == 4           # full letter set re-judged
    assert with_flag.errors_written == 0

    judgments = list(store.iter_judgments())
    assert len(judgments) == 5                        # old "-" row + 4 new ok rows
    ok_letters = {j["letter"] for j in judgments if j["status"] == "ok"}
    assert ok_letters == set(expected_letters)
    error_rows = [j for j in judgments if j["status"] == "error"]
    assert len(error_rows) == 1
    assert error_rows[0]["reason"] == "seeded terminal error"   # old row kept as history


# --- model_id resolution (cohort row_id -> real model_id; CAL letters literal) ---


def test_model_id_resolved_via_map_for_cohort_and_calibration_letters(tmp_path):
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())

    run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]}, fake=True,
                **_base_kwargs(tmp_path))

    model_ids = {j["model_id"] for j in store.iter_judgments()}
    assert model_ids == set(COHORT) | {"CAL-strong", "CAL-weak"}


# --- --judge filter restricts invocation, not packet building ---


def test_judge_filter_restricts_invocation_to_one_judge(tmp_path):
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())
    kwargs = _base_kwargs(tmp_path)

    result = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True,
                          judge_filter="j1", **kwargs)
    assert result.judgments_written == 4                  # 1 packet x 1 judge x 4 letters
    judge_ids_seen = {j["judge_id"] for j in store.iter_judgments()}
    assert judge_ids_seen == {"j1"}

    # A follow-up run without the filter picks up the OTHER judge only.
    result2 = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True, **kwargs)
    assert result2.judgments_written == 4
    judge_ids_seen2 = {j["judge_id"] for j in store.iter_judgments()}
    assert judge_ids_seen2 == {"j1", "j2"}
    assert len(list(store.iter_judgments())) == 8


def test_unknown_judge_filter_raises(tmp_path):
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())

    with pytest.raises(ValueError):
        run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True,
                     judge_filter="nope", **_base_kwargs(tmp_path))


# --- signals_by_task populated for every packetized cohort -> evidence tables render ---


def test_signals_by_task_populated_so_evidence_tables_render(tmp_path):
    """Task-5 review Finding 1 handoff: when a task_id is absent from
    signals_by_task, packets.py degrades the WHOLE packet (cohort AND
    calibration letters) to a blank evidence table to avoid a structural
    tell. The runner must populate signals_by_task for every unit actually
    being packetized so real evidence renders instead of the blank
    fallback."""
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.",
                           signals=[{"type": "contains", "value": "MFA"}])
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id,
                   det_checks={"contains-0": {"pass": True}})
    rows = list(store.iter_rows())

    result = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True,
                          packets_only=True, **_base_kwargs(tmp_path))

    assert len(result.packets) == 1
    body_path = next(iter(result.packets[0].bodies.values()))
    body_text = Path(body_path).read_text(encoding="utf-8")
    assert "(no det-signal evidence)" not in body_text
    assert "contains-0" in body_text
    assert "PASS" in body_text


def test_build_signals_by_task_keys_full_task_id(tmp_path):
    _write_task(tmp_path, UNIT, 1, "Draft an incident summary.",
                signals=[{"type": "contains", "value": "MFA"}])
    out = build_signals_by_task(tmp_path, [UNIT])
    assert out == {f"b1.{UNIT}-01": [{"type": "contains", "value": "MFA"}]}


# --- status --judging: summarize_judging counts match actual state ---


def test_summarize_judging_counts_done_pending_error(tmp_path):
    store = Store(tmp_path / "results")
    # 2 tasks -> 2 packets; judge fully with j1, leave j2 pending on both.
    _seed_two_tasks(tmp_path, store)
    rows = list(store.iter_rows())
    kwargs = _base_kwargs(tmp_path)

    result = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True,
                          judge_filter="j1", **kwargs)
    assert result.judgments_written == 8                  # 2 packets x 1 judge x 4 letters

    counts = summarize_judging(store, result.packets, sorted(JUDGES_CFG))
    assert counts == {"done": 2, "pending": 2, "error": 0}  # j1 done x2, j2 pending x2


def test_summarize_judging_counts_error_pair_as_error_not_pending(tmp_path):
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    for model_id in COHORT:
        _make_row(tmp_path, store, model_id, task_id)
    rows = list(store.iter_rows())
    kwargs = _base_kwargs(tmp_path)

    result = run_pending(rows=rows, store=store, judges_cfg={"j1": JUDGES_CFG["j1"]},
                          fake=True, fake_scores_fn=_garbage_scores, **kwargs)
    assert result.errors_written == 1

    counts = summarize_judging(store, result.packets, ["j1"])
    assert counts == {"done": 0, "pending": 0, "error": 1}


# --- resolve_cohort_models ---


def test_resolve_cohort_models_suite_override():
    class _Cfg:
        suite = {"b1": {"cohort_models": ["a", "b"]}}
        registry = {"models": {}}
    assert resolve_cohort_models(_Cfg()) == ["a", "b"]


def test_resolve_cohort_models_default_excludes_quant_arm_and_stub_paths():
    class _Cfg:
        suite = {}
        registry = {"models": {
            "real-1": {"role": "shakedown", "local_path": "D:\\models\\real1"},
            "real-2": {"local_path": "D:\\models\\real2"},
            "quant": {"role": "quant-arm", "local_path": "D:\\models\\q"},
            "stub": {"local_path": "TO-DOWNLOAD"},
        }}
    assert resolve_cohort_models(_Cfg()) == ["real-1", "real-2"]


def test_incomplete_cohort_is_skipped_not_invoked(tmp_path):
    store = Store(tmp_path / "results")
    task_id = _write_task(tmp_path, UNIT, 1, "Draft an incident summary.")
    _make_row(tmp_path, store, "model-a", task_id)          # model-b never answers
    rows = list(store.iter_rows())

    result = run_pending(rows=rows, store=store, judges_cfg=JUDGES_CFG, fake=True,
                          **_base_kwargs(tmp_path))

    assert result.packets == []
    assert len(result.skipped) == 1
    assert "model-b" in result.skipped[0]["reason"]
    assert result.judgments_written == 0
    assert list(store.iter_judgments()) == []
