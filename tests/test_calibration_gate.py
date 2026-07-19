"""Tests for the non-circular B2 calibration gate (Task 6).

The frozen CAL-strong/CAL-weak anchors (Task 2, grading/calibration/b2/
<axis>.yaml) are NEVER re-tuned to please the judges. calibration_status()
only CHECKS whether the panel's already-produced scores on those anchors
satisfy two judge-independent invariants, per axis dimension, and
quarantines an axis when they don't -- no regeneration, no human gate.

Data shapes mirror aggregate.py / test_b2_aggregate.py: judgment rows are
dicts with packet_id/judge_id/model_id/score/status; a packet is a B2 axis
packet iff maps[packet_id]["dim"] starts with "axis"; CAL answers appear
under model_id == "CAL-strong" / "CAL-weak".
"""
from __future__ import annotations

from llmtest.judging.calibration_gate import calibration_status

REFSCORES = {"strong": 9, "weak": 2, "tolerance": 1}


def _map(dim="axis5", task_id="b2.t1", run_n=1):
    return {"task_id": task_id, "run_n": run_n, "dim": dim}


def _b1_map(unit="finance", task_id="b1.t1", run_n=1):
    return {"task_id": task_id, "run_n": run_n, "unit": unit}


def _j(packet_id, judge_id, model_id, score, status="ok"):
    return {"packet_id": packet_id, "judge_id": judge_id, "model_id": model_id,
            "score": score, "status": status}


# --- accepted: both invariants hold ---


def test_accepted_when_ordinal_and_drift_both_hold():
    maps = {"p1": _map("axis5")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 9),  _j("p1", "codex", "CAL-weak", 2),
        _j("p1", "gemini", "CAL-strong", 8), _j("p1", "gemini", "CAL-weak", 1),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    # median(strong)=9, median(weak)=2 -> exact match, 0 drift.
    assert status == {"axis5": "accepted"}


# --- quarantined: a judge scores CAL-strong <= CAL-weak (ordinal failure) ---


def test_quarantined_when_a_judge_scores_strong_leq_weak():
    maps = {"p1": _map("axis5")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 9),  _j("p1", "codex", "CAL-weak", 2),
        # gemini inverted: strong <= weak.
        _j("p1", "gemini", "CAL-strong", 2), _j("p1", "gemini", "CAL-weak", 8),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "quarantined"}


def test_quarantined_when_judge_scores_strong_exactly_equal_to_weak():
    """The invariant is a strict '>' -- a tie does not satisfy it."""
    maps = {"p1": _map("axis5")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 5),  _j("p1", "codex", "CAL-weak", 5),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "quarantined"}


# --- quarantined: drift beyond tolerance (even though ordinal holds) ---


def test_quarantined_when_drift_exceeds_tolerance():
    maps = {"p1": _map("axis5")}
    judgments = [
        # Every judge correctly ranks strong > weak, but the panel's
        # absolute calibration to the frozen anchors has drifted far from
        # refscores (strong=9, weak=2, tol=1).
        _j("p1", "claude", "CAL-strong", 5), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 5),  _j("p1", "codex", "CAL-weak", 2),
        _j("p1", "gemini", "CAL-strong", 5), _j("p1", "gemini", "CAL-weak", 2),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "quarantined"}


def test_drift_within_tolerance_is_accepted_at_the_boundary():
    maps = {"p1": _map("axis5")}
    judgments = [
        # median(strong)=10 -> |10-9|=1 == tolerance -> still accepted.
        _j("p1", "claude", "CAL-strong", 10), _j("p1", "claude", "CAL-weak", 3),
        _j("p1", "codex", "CAL-strong", 10),  _j("p1", "codex", "CAL-weak", 3),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "accepted"}


# --- quarantined: no CAL judgment rows at all for the axis ---


def test_quarantined_when_no_cal_rows_present():
    maps = {"p1": _map("axis5")}
    judgments = []   # nothing judged yet for this axis's CAL answers
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "quarantined"}


def test_quarantined_when_only_real_model_rows_present_no_cal():
    maps = {"p1": _map("axis5")}
    judgments = [
        _j("p1", "claude", "m1", 7),
        _j("p1", "codex", "m1", 7),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "quarantined"}


# --- multiple axes: independent status per axis (brief's own example) ---


def test_multiple_axes_get_independent_status():
    maps = {
        "p1": _map("axis5"),
        "p2": _map("axis8"),
    }
    judgments = [
        # axis5: clean accept.
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 9),  _j("p1", "codex", "CAL-weak", 2),
        # axis8: ordinal failure on codex.
        _j("p2", "claude", "CAL-strong", 9), _j("p2", "claude", "CAL-weak", 2),
        _j("p2", "codex", "CAL-strong", 2),  _j("p2", "codex", "CAL-weak", 9),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "accepted", "axis8": "quarantined"}


# --- default refscores (strong=9, weak=2, tolerance=1) applied when omitted ---


def test_default_refscores_used_when_not_passed():
    maps = {"p1": _map("axis5")}
    judgments = [
        # median(strong)=11 -> |11-9|=2 > default tolerance(1) -> quarantined.
        _j("p1", "claude", "CAL-strong", 11), _j("p1", "claude", "CAL-weak", 2),
        _j("p1", "codex", "CAL-strong", 11),  _j("p1", "codex", "CAL-weak", 2),
    ]
    status = calibration_status(judgments, maps)   # no refscores kwarg
    assert status == {"axis5": "quarantined"}


# --- per-judge aggregation across multiple packets of the same axis, via
# median (brief: "Aggregate per judge across that axis's packets with the
# median"); drift pools ALL raw CAL scores for the axis, not a median of
# per-judge medians. ---


def test_ordinal_aggregates_per_judge_across_packets_with_median():
    maps = {"p1": _map("axis5", task_id="b2.a"), "p2": _map("axis5", task_id="b2.b")}
    judgments = [
        # claude: strong scores [9, 7] -> median 8; weak scores [1, 1] -> median 1. 8>1 OK.
        _j("p1", "claude", "CAL-strong", 9), _j("p2", "claude", "CAL-strong", 7),
        _j("p1", "claude", "CAL-weak", 1),   _j("p2", "claude", "CAL-weak", 1),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "accepted"}


def test_drift_pools_all_raw_scores_not_median_of_per_judge_medians():
    """codex's per-judge strong-score median is 5 ([9, 1]), which would drift
    (|5-9|=4>1) if drift were computed from a median of per-judge medians.
    But pooling ALL raw CAL-strong scores across judges/packets for this
    axis -- [9, 9, 9, 1] -- medians to 9, exactly on the reference, so the
    axis must still be accepted. This locks in the brief's literal wording:
    'median(all CAL-strong scores for this axis)'."""
    maps = {"p1": _map("axis5", task_id="b2.a"), "p2": _map("axis5", task_id="b2.b")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 9), _j("p2", "claude", "CAL-strong", 9),
        _j("p1", "claude", "CAL-weak", 1),   _j("p2", "claude", "CAL-weak", 1),
        _j("p1", "codex", "CAL-strong", 9),  _j("p2", "codex", "CAL-strong", 1),
        _j("p1", "codex", "CAL-weak", 1),    _j("p2", "codex", "CAL-weak", 1),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    # ordinal: claude median(strong)=9 > median(weak)=1 OK;
    #          codex  median(strong)=5 > median(weak)=1 OK (ordinal only
    #          needs strong > weak per judge, not proximity to refscores).
    # drift: pooled strong=[9,9,9,1] -> median 9 (matches ref exactly);
    #        pooled weak=[1,1,1,1] -> median 1, |1-2|=1<=tol.
    assert status == {"axis5": "accepted"}


# --- error rows and non-B2 (B1) packets are excluded ---


def test_non_ok_status_rows_are_excluded():
    maps = {"p1": _map("axis5")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
        # This row, if counted, would break the ordinal invariant for codex --
        # but it's status=error, so it must be ignored entirely.
        _j("p1", "codex", "CAL-strong", 1, status="error"),
        _j("p1", "codex", "CAL-weak", 8, status="error"),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {"axis5": "accepted"}


def test_b1_unit_packets_never_appear_in_output():
    maps = {"p1": _b1_map("finance")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 9), _j("p1", "claude", "CAL-weak", 2),
    ]
    status = calibration_status(judgments, maps, refscores=REFSCORES)
    assert status == {}
