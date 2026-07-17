"""Tests for judging packet builder (TESTPLAN 6.1) — cohort build, blinding, maps, scrubbing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llmtest import schema
from llmtest.judging.packets import build_cohort_packets, scrub

ROOT = Path(__file__).resolve().parents[1]  # real repo root -> real grading/judge_prompt.md
JUDGE_IDS = ["claude", "codex", "gemini"]
COHORT = ["model-a", "model-b"]

UNIT = "cybersecurity"
TASK_SUFFIX = f"{UNIT}-01"
TASK_ID = f"b1.{TASK_SUFFIX}"
TASK_PROMPT = "Draft an incident summary for CVE-2026-9999 affecting 10 endpoints."


def _write_fixture(root: Path):
    unit_dir = root / "suite" / "b1_business" / UNIT
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "task-01.yaml").write_text(
        "id: {}\nunit: {}\ndifficulty: easy\nclass: short\nprompt: |\n  {}\nsignals: []\n"
        .format(TASK_SUFFIX, UNIT, TASK_PROMPT),
        encoding="utf-8",
    )


def _write_anchor(rubric_dir: Path, text="anchor v1 -- 0/3/5/7/10 definitions"):
    rubric_dir.mkdir(parents=True, exist_ok=True)
    (rubric_dir / f"{UNIT}.md").write_text(text, encoding="utf-8")


def _write_calibration(cal_dir: Path):
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "strong.md").write_text("A frontier-quality reference answer.", encoding="utf-8")
    (cal_dir / "weak.md").write_text("A vague, partly-wrong reference answer.", encoding="utf-8")


def _make_row(root: Path, model_id: str, run_n=1, text=None, status="ok"):
    text = text or f"Answer from {model_id}: MFA enforced, CVE-2026-9999 patched on all endpoints."
    quant_sha = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
    row = schema.ResultRow.new(
        suite_version="suite-test", model_id=model_id, hf_repo="o/r",
        quant_file="q.gguf", quant_sha256=quant_sha, tier="T1", battery=1,
        task_id=TASK_ID, fixture_sha="f" * 64,
        condition="runtime=fork;spec=ngram32;kv=q8;ctx=16k;cond=B1",
        run_n=run_n, session_id="s", needs_judging=True, status=status,
        det_checks={"contains-0": {"pass": True}})
    if status == "ok":
        art_dir = root / "artifacts" / "b1"
        art_dir.mkdir(parents=True, exist_ok=True)
        path = art_dir / f"{row.row_id}.txt"
        path.write_text(text, encoding="utf-8")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        row.artifacts = {"b1": {"sha256": sha, "relpath": f"b1/{row.row_id}.txt"}}
    return row.to_dict()


def _base_kwargs(tmp_path):
    _write_fixture(tmp_path)
    rubric_dir = tmp_path / "grading" / "anchors"
    calibration_dir = tmp_path / "grading" / "calibration"
    _write_anchor(rubric_dir)
    _write_calibration(calibration_dir)
    return dict(
        rubric_dir=rubric_dir, calibration_dir=calibration_dir,
        out_artifacts=tmp_path / "artifacts" / "packets",
        out_maps=tmp_path / "results" / "packets",
        root=tmp_path, judge_ids=JUDGE_IDS, cohort_models=COHORT,
        judge_prompt_path=ROOT / "grading" / "judge_prompt.md",
    )


def test_complete_cohort_builds_one_packet_with_blinded_map(tmp_path):
    """(a) two fake models' rows + calibration files -> one packet; map has 4 letters
    (2 models + 2 calibration) with at least one differing per-judge permutation."""
    rows = [_make_row(tmp_path, "model-a"), _make_row(tmp_path, "model-b")]
    kwargs = _base_kwargs(tmp_path)
    packets, skipped = build_cohort_packets(rows, **kwargs)

    assert skipped == []
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.task_id == TASK_ID
    assert pkt.run_n == 1
    assert pkt.unit == UNIT
    assert set(pkt.bodies) == set(JUDGE_IDS)
    for path in pkt.bodies.values():
        assert path.exists()
        body_text = path.read_text(encoding="utf-8")
        assert "CVE-2026-9999" in body_text  # answer content made it into the packet body
    assert pkt.map_path.exists()

    map_data = json.loads(pkt.map_path.read_text(encoding="utf-8"))
    assert map_data["task_id"] == TASK_ID
    assert map_data["run_n"] == 1
    assert map_data["rubric_sha"] == pkt.rubric_sha

    letters_by_judge = map_data["letters_by_judge"]
    assert set(letters_by_judge) == set(JUDGE_IDS)
    for judge_letters in letters_by_judge.values():
        assert set(judge_letters) == {"A", "B", "C", "D"}
        values = set(judge_letters.values())
        assert "CAL-strong" in values
        assert "CAL-weak" in values

    orderings = {tuple(judge_letters[l] for l in "ABCD")
                 for judge_letters in letters_by_judge.values()}
    assert len(orderings) > 1  # at least one judge's permutation differs from the others


def test_packet_id_deterministic_and_changes_with_rubric(tmp_path):
    """(b) same inputs -> same packet_id; changing the rubric (anchor) file changes packet_id."""
    rows = [_make_row(tmp_path, "model-a"), _make_row(tmp_path, "model-b")]
    kwargs = _base_kwargs(tmp_path)

    packets1, _ = build_cohort_packets(rows, **kwargs)
    packets2, _ = build_cohort_packets(rows, **kwargs)
    assert packets1[0].packet_id == packets2[0].packet_id

    (kwargs["rubric_dir"] / f"{UNIT}.md").write_text(
        "anchor v2 -- revised definitions", encoding="utf-8")
    packets3, _ = build_cohort_packets(rows, **kwargs)
    assert packets3[0].packet_id != packets1[0].packet_id


def test_incomplete_cohort_is_skipped_and_reported(tmp_path):
    """(c) incomplete cohort (one model missing an ok row) -> no packet, reported in skipped."""
    rows = [_make_row(tmp_path, "model-a")]  # model-b never answered
    kwargs = _base_kwargs(tmp_path)
    packets, skipped = build_cohort_packets(rows, **kwargs)

    assert packets == []
    assert len(skipped) == 1
    assert skipped[0]["task_id"] == TASK_ID
    assert "model-b" in skipped[0]["reason"]


def test_missing_artifact_file_skipped_defensively(tmp_path):
    """A complete cohort whose answer artifact file is missing on disk is
    skipped (defensively), not a crash -- the row claims an artifact but the
    file underneath it is gone."""
    rows = [_make_row(tmp_path, "model-a"), _make_row(tmp_path, "model-b")]
    kwargs = _base_kwargs(tmp_path)

    # Delete model-b's artifact file after the row was written, simulating a
    # missing artifact (disk cleanup, partial sync, etc).
    relpath = rows[1]["artifacts"]["b1"]["relpath"]
    (tmp_path / "artifacts" / relpath).unlink()

    packets, skipped = build_cohort_packets(rows, **kwargs)
    assert packets == []
    assert len(skipped) == 1
    assert "model-b" in skipped[0]["reason"]
    assert "artifact" in skipped[0]["reason"]


def test_scrub_removes_self_identification_and_vendor_names():
    """(d) scrub() strips self-id phrasing and vendor/model names."""
    out = scrub("I am Gemma, made by Google")
    assert "Gemma" not in out
    assert "Google" not in out
