"""Tests for the P8 report's B2 judged-axis section + source_suite labels
(P1-T7 / agentic-quality v2.1 Part 1 finale).

Exercises `scripts/p8_report.py` directly (no prior test coverage existed
for this script): `load_rows` reading BOTH results/rows-suite-v2.0.0.jsonl
and results/rows-suite-v2.1.0.jsonl (tagging every row `source_suite`, never
silently blending them), and a new B2-judged section that surfaces
`aggregate(...).b2_axis_scores` (Task 5, fabrication hard-capped) next to
the existing deterministic per-axis pass-rate table, excludes any axis whose
`calibration_status(...)` (Task 6) is "quarantined" from the printed scores,
and lists sub-quorum B2 packets from the committed maps' `missing_models`
(Task 3).
"""
from __future__ import annotations

import importlib.util
import json
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# scripts/ isn't a package (no __init__.py) -- import p8_report.py directly
# by file path, same trick the script itself uses to reach llmtest/.
_SPEC = importlib.util.spec_from_file_location(
    "p8_report", REPO_ROOT / "scripts" / "p8_report.py")
p8_report = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("p8_report", p8_report)
_SPEC.loader.exec_module(p8_report)


def _write_row(root: Path, suite_version: str, *, battery=1, model_id="model-a",
                task_id="b1.finance-01", run_n=1) -> None:
    path = root / "results" / f"rows-{suite_version}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1, "row_id": f"{suite_version}-{model_id}-{task_id}-{run_n}",
        "parent_id": None, "suite_version": suite_version, "fixture_sha": "sha",
        "code_sha": "unknown", "battery": battery, "task_id": task_id,
        "condition": "cond=B1", "run_n": run_n, "model_id": model_id,
        "hf_repo": "o/r", "quant_file": "q.gguf", "quant_sha256": "qsha",
        "tier": "T1", "session_id": "s", "sampling": {}, "ts": "2026-07-19T00:00:00+00:00",
        "request": {}, "response_meta": {}, "det_checks": {}, "needs_judging": False,
        "metrics": {}, "timing_authoritative": False, "artifacts": {}, "status": "ok",
        "error_detail": None, "tags": [],
    }
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


# --- load_rows: multi-shard read + source_suite tagging ---


def test_load_rows_tags_source_suite_from_primary_shard(tmp_path):
    _write_row(tmp_path, "suite-v2.0.0", model_id="model-a")
    caveats: list[str] = []
    rows = p8_report.load_rows(tmp_path, "suite-v2.0.0", caveats)
    assert len(rows) == 1
    assert rows[0]["source_suite"] == "suite-v2.0.0"


def test_load_rows_degrades_cleanly_when_v21_shard_absent(tmp_path):
    """Only v2.0.0 exists today (brief's stated real-world state) -- no
    v2.1.0 file at all must not crash and must not add a spurious caveat
    (that shard's absence is the expected, normal state)."""
    _write_row(tmp_path, "suite-v2.0.0")
    caveats: list[str] = []
    rows = p8_report.load_rows(tmp_path, "suite-v2.0.0", caveats)
    assert len(rows) == 1
    assert not any("v2.1.0" in c for c in caveats)


def test_load_rows_reads_both_shards_and_never_blends_source_suite(tmp_path):
    """A synthetic v2.1.0 shard alongside the real v2.0.0 shard: both are
    read, and every row keeps its OWN shard's source_suite -- proving rows
    from the two suite versions are never silently merged into one
    unlabeled pool."""
    _write_row(tmp_path, "suite-v2.0.0", battery=2, model_id="model-a",
                task_id="b2.error-recovery-01")
    _write_row(tmp_path, "suite-v2.1.0", battery=2, model_id="model-a",
                task_id="b2.error-recovery-01", run_n=2)
    caveats: list[str] = []
    rows = p8_report.load_rows(tmp_path, "suite-v2.0.0", caveats)

    by_suite = {r["source_suite"] for r in rows}
    assert by_suite == {"suite-v2.0.0", "suite-v2.1.0"}
    assert len(rows) == 2


# --- load_baseline_maps: B2 axis-packet letter-count collision exclusion ---


def test_load_baseline_maps_excludes_b2_axis_packets_with_colliding_letter_count(tmp_path):
    """A B2 axis packet built at the default full-roster quorum (config/suite.yaml
    b2.quorum == cohort size) has EXACTLY the same letter count as a real B1
    baseline packet (both are len(cohort_models)+2 letters). load_baseline_maps
    must exclude axis packets (dim starting with "axis") BEFORE computing the
    max-letter-count target -- otherwise, once real B2 judging runs, those
    axis-5/8 packets get counted as B1 baseline packets, inflating the
    "Baseline packets" total (this is the bug the fix regresses)."""
    letters = list(string.ascii_uppercase[:18])  # 18-letter full roster (16 models + CAL x2)
    letters_by_judge = {jid: {letter: f"model-{i}" for i, letter in enumerate(letters)}
                         for jid in ("claude", "codex", "gemini")}

    b1_map = {"task_id": "b1.finance-01", "run_n": 1, "unit": "finance",
              "rubric_sha": "rsha", "cal_fallback": False, "base_seed": "seed1",
              "letters_by_judge": letters_by_judge}
    axis_map = {"task_id": "b2.error-recovery-01", "run_n": 1, "dim": "axis5",
                "scenario": "error-recovery-01", "present_models": [], "missing_models": [],
                "fabrication_pass": {}, "rubric_sha": "rsha", "base_seed": "seed2",
                "letters_by_judge": letters_by_judge}

    packets_dir = tmp_path / "results" / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    (packets_dir / "b1-pkt.map.json").write_text(json.dumps(b1_map), encoding="utf-8")
    (packets_dir / "b2-axis-pkt.map.json").write_text(json.dumps(axis_map), encoding="utf-8")

    baseline = p8_report.load_baseline_maps(tmp_path)

    assert set(baseline) == {"b1-pkt"}


# --- B2 judged-axis section ---


def _cfg(quorum=3, judged_axes=(5, 8)):
    class _Cfg:
        suite = {"b2": {"quorum": quorum, "judged_axes": list(judged_axes)}}
        judges = {"judges": {"claude": {}, "codex": {}, "gemini": {}}}
    return _Cfg()


def _b2_map(dim="axis5", task_id="b2.error-recovery-01", run_n=1,
            present_models=None, missing_models=None, fabrication_pass=None):
    return {
        "task_id": task_id, "run_n": run_n, "dim": dim, "scenario": "error-recovery-01",
        "present_models": present_models or [], "missing_models": missing_models or [],
        "fabrication_pass": fabrication_pass or {}, "rubric_sha": "rsha",
        "base_seed": "seed", "letters_by_judge": {},
    }


def _j(packet_id, judge_id, model_id, score, status="ok"):
    return {"packet_id": packet_id, "judge_id": judge_id, "model_id": model_id,
            "score": score, "status": status}


def test_b2_judged_section_shows_axis5_column_with_capped_score():
    """A fabrication-guard failure caps the printed axis-5 score at 2 (Task
    5's hard-cap), and the column header names the axis + '(judged)'."""
    maps = {
        "p1": _b2_map(dim="axis5", present_models=["model-a"],
                       fabrication_pass={"model-a": False}),
    }
    judgments = [
        _j("p1", "claude", "model-a", 9), _j("p1", "codex", "model-a", 9),
        _j("p1", "gemini", "model-a", 9),
        # CAL rows so axis5 calibration is accepted (not quarantined) and
        # the real model's capped score is actually printed, not hidden.
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 9), _j("p1", "codex", "CAL-weak", 2),
    ]
    section = p8_report.build_b2_judged_section(_cfg(), [], judgments, maps)

    assert "Axis 5 (judged)" in section
    assert "model-a" in section
    assert "2.0" in section          # capped, not 9.0
    assert "quarantined" not in section.lower()   # axis5 calibration accepted here


def test_b2_judged_section_marks_quarantined_axis_and_hides_partial_numbers():
    """axis8's CAL judgments are ordinal-inverted (strong <= weak on a
    judge) -> quarantined -> the axis column must read 'quarantined' for
    every model, NEVER a partial numeric score."""
    maps = {
        "p1": _b2_map(dim="axis8", task_id="b2.faith-01", present_models=["model-a"],
                       fabrication_pass={"model-a": True}),
    }
    judgments = [
        _j("p1", "claude", "model-a", 9), _j("p1", "codex", "model-a", 9),
        # CAL rows fail the ordinal invariant on codex (strong <= weak).
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 2), _j("p1", "codex", "CAL-weak", 9),
    ]
    section = p8_report.build_b2_judged_section(_cfg(), [], judgments, maps)

    assert "Axis 8 (judged)" in section
    assert "quarantined" in section
    # The row for model-a under the quarantined axis must show the literal
    # marker, not a numeric score smuggled through.
    lines = [ln for ln in section.splitlines() if ln.strip().startswith("| model-a")]
    assert lines, "expected a table row for model-a"
    assert "quarantined" in lines[0]
    assert "9.0" not in lines[0]


def test_b2_judged_section_lists_subquorum_packets_from_missing_models():
    maps = {
        "p1": _b2_map(dim="axis5", present_models=["model-a"],
                       missing_models=["model-b", "model-c"],
                       fabrication_pass={"model-a": True}),
    }
    judgments = [
        _j("p1", "claude", "model-a", 7), _j("p1", "codex", "model-a", 7),
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 9), _j("p1", "codex", "CAL-weak", 2),
    ]
    section = p8_report.build_b2_judged_section(_cfg(), [], judgments, maps)

    assert "model-b" in section
    assert "model-c" in section
    assert "sub-quorum" in section.lower() or "sub quorum" in section.lower()


def test_b2_judged_section_degrades_cleanly_when_no_axis_packets_yet():
    section = p8_report.build_b2_judged_section(_cfg(), [], [], {})
    assert "no b2 judged-axis packets" in section.lower()
    assert "Traceback" not in section


# --- end-to-end: build_report() on a synthetic two-shard repo ---


def test_build_report_end_to_end_labels_source_suite_and_judged_axes(tmp_path):
    """Full build_report() over a minimal synthetic repo (config/ copied
    from the real repo so cfg loads normally; results/ + results/packets/
    synthetic) proves the pieces compose: REPORT.md text contains an
    'Axis 5 (judged)' column, a source_suite marker, and both a v2.0.0 and
    a v2.1.0 row group are represented without crashing."""
    import shutil
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    shutil.copytree(REPO_ROOT / "grading", tmp_path / "grading")
    (tmp_path / "suite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    _write_row(tmp_path, "suite-v2.0.0", battery=2, model_id="model-a",
               task_id="b2.error-recovery-01")
    _write_row(tmp_path, "suite-v2.1.0", battery=2, model_id="model-a",
               task_id="b2.error-recovery-01", run_n=2)

    packets_dir = tmp_path / "results" / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    map_record = _b2_map(dim="axis5", present_models=["model-a"],
                          fabrication_pass={"model-a": True})
    (packets_dir / "pkt1.map.json").write_text(json.dumps(map_record), encoding="utf-8")

    judgments_path = tmp_path / "results" / "judgments.jsonl"
    judgment_rows = [
        {"schema_version": 1, "packet_id": "pkt1", "judge_id": "claude",
         "judge_model_pin": "p", "judge_cli_version": "v", "letter": "A",
         "model_id": "model-a", "score": 7, "reason": "r", "rank": 1,
         "ts": "2026-07-19T00:00:00+00:00", "status": "ok"},
        {"schema_version": 1, "packet_id": "pkt1", "judge_id": "claude",
         "judge_model_pin": "p", "judge_cli_version": "v", "letter": "B",
         "model_id": "CAL-strong", "score": 9, "reason": "r", "rank": 2,
         "ts": "2026-07-19T00:00:00+00:00", "status": "ok"},
        {"schema_version": 1, "packet_id": "pkt1", "judge_id": "claude",
         "judge_model_pin": "p", "judge_cli_version": "v", "letter": "C",
         "model_id": "CAL-weak", "score": 2, "reason": "r", "rank": 3,
         "ts": "2026-07-19T00:00:00+00:00", "status": "ok"},
    ]
    with judgments_path.open("w", encoding="utf-8", newline="\n") as f:
        for j in judgment_rows:
            f.write(json.dumps(j) + "\n")

    full_md, condensed = p8_report.build_report(tmp_path)

    assert "Axis 5 (judged)" in full_md
    assert "source_suite" in full_md
    assert "suite-v2.1.0" in full_md
    assert "Traceback" not in full_md
    assert isinstance(condensed, str) and condensed
