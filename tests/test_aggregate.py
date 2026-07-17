"""Tests for table-time aggregation (TESTPLAN 6.1/6.2, Task 8) -- median/spread,
drift vs refscores, kin-delta, agreement, incomplete panels, pairwise majority
vote, and byte-determinism of the rendered tables built on top of it.

Nothing here is stored: `aggregate()` is a pure function of
Store.iter_judgments() rows + the committed packet maps + the judges.yaml
kin_map + refscores.yaml -- called fresh at table-render time.
"""
from __future__ import annotations

import random

from llmtest.judging.aggregate import CAL_IDENTITIES, aggregate

REFSCORES = {"strong": 9, "weak": 2, "tolerance": 1}


def _map(task_id="b1.cybersecurity-01", run_n=1, unit="cybersecurity",
         rubric_sha="rsha1", cal_fallback=False):
    return {"task_id": task_id, "run_n": run_n, "unit": unit,
            "rubric_sha": rubric_sha, "cal_fallback": cal_fallback,
            "base_seed": "seed1", "letters_by_judge": {}}


def _j(packet_id, judge_id, letter, model_id, score, rank=1, status="ok"):
    return {"schema_version": 1, "packet_id": packet_id, "judge_id": judge_id,
            "judge_model_pin": "pin", "judge_cli_version": "v1", "letter": letter,
            "model_id": model_id if status == "ok" else None,
            "score": score if status == "ok" else None,
            "reason": "r", "rank": rank if status == "ok" else None,
            "ts": "2026-07-17T00:00:00+00:00", "status": status}


# --- median / spread math ---


def test_median_and_spread_for_full_panel():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 6, rank=2),
        _j("p1", "codex", "B", "model-a", 7, rank=2),
        _j("p1", "gemini", "C", "model-a", 8, rank=2),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    pa = next(pa for pa in agg.packet_answers if pa.model_id == "model-a")
    assert pa.median == 7
    assert pa.spread == 2
    assert pa.n_judges == 3


def test_spread_undefined_for_single_judge():
    maps = {"p1": _map()}
    judgments = [_j("p1", "claude", "A", "model-a", 6, rank=1)]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    pa = agg.packet_answers[0]
    assert pa.n_judges == 1
    assert pa.median == 6
    assert pa.spread is None


def test_median_for_two_judges_is_mean():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 4, rank=1),
        _j("p1", "codex", "B", "model-a", 8, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    pa = agg.packet_answers[0]
    assert pa.n_judges == 2
    assert pa.median == 6
    assert pa.spread == 4


# --- spread > 2 -> flag lands in FLAGS output ---


def test_spread_three_flag_recorded():
    maps = {"p1": _map(task_id="b1.cybersecurity-03", run_n=2)}
    judgments = [
        _j("p1", "claude", "A", "model-a", 3, rank=3),
        _j("p1", "codex", "B", "model-a", 6, rank=2),
        _j("p1", "gemini", "C", "model-a", 9, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert len(agg.spread_flags) == 1
    flag = agg.spread_flags[0]
    assert flag["task_id"] == "b1.cybersecurity-03"
    assert flag["run_n"] == 2
    assert flag["model_id"] == "model-a"
    assert flag["scores_by_judge"] == {"claude": 3, "codex": 6, "gemini": 9}


def test_spread_exactly_two_does_not_flag():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 5, rank=1),
        _j("p1", "codex", "B", "model-a", 7, rank=1),
        _j("p1", "gemini", "C", "model-a", 6, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.spread_flags == []


def test_single_judge_never_flags_spread():
    maps = {"p1": _map()}
    judgments = [_j("p1", "claude", "A", "model-a", 6, rank=1)]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.spread_flags == []


# --- drift detection vs refscores ---


def test_drift_flag_when_cal_strong_median_below_ref():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "D", "CAL-strong", 7, rank=1),
        _j("p1", "codex", "E", "CAL-strong", 7, rank=1),
        _j("p1", "gemini", "F", "CAL-strong", 7, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert len(agg.drift_flags) == 1
    flag = agg.drift_flags[0]
    assert flag["cal_type"] == "strong"
    assert flag["median"] == 7
    assert flag["ref"] == 9
    assert flag["delta"] == -2
    assert flag["packet_id"] == "p1"


def test_no_drift_flag_within_tolerance():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "D", "CAL-strong", 8, rank=1),
        _j("p1", "codex", "E", "CAL-strong", 8, rank=1),
        _j("p1", "gemini", "F", "CAL-strong", 8, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.drift_flags == []


def test_drift_flag_for_weak_calibration_too():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "D", "CAL-weak", 5, rank=1),
        _j("p1", "codex", "E", "CAL-weak", 5, rank=1),
        _j("p1", "gemini", "F", "CAL-weak", 5, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert len(agg.drift_flags) == 1
    assert agg.drift_flags[0]["cal_type"] == "weak"
    assert agg.drift_flags[0]["delta"] == 3


def test_drift_flags_do_not_leak_into_spread_flags_for_real_models():
    """CAL identities are excluded from model_unit_stats/model_overall (the
    scorecard is models only) but CAN still spread-flag like any letter."""
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "D", "CAL-strong", 7, rank=1),
        _j("p1", "codex", "E", "CAL-strong", 7, rank=1),
        _j("p1", "gemini", "F", "CAL-strong", 7, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.model_unit_stats == {}
    assert agg.model_overall == {}


# --- kin-delta sign correctness ---


def test_kin_delta_positive_when_judge_favors_kin_model():
    maps = {"p1": _map(task_id="b1.it_infra-01", unit="it_infra")}
    kin_map = {"gpt-oss-20b": "codex"}
    judgments = [
        # codex scores its kin model (+2 over baseline) and a non-kin model.
        _j("p1", "codex", "A", "gpt-oss-20b", 9, rank=1),
        _j("p1", "codex", "B", "other-model", 7, rank=2),
    ]
    agg = aggregate([], judgments, maps, kin_map=kin_map, refscores=REFSCORES)
    assert agg.kin_delta["codex"] == 2


def test_kin_delta_negative_when_judge_penalizes_kin_model():
    maps = {"p1": _map(task_id="b1.it_infra-01", unit="it_infra")}
    kin_map = {"gpt-oss-20b": "codex"}
    judgments = [
        _j("p1", "codex", "A", "gpt-oss-20b", 5, rank=2),
        _j("p1", "codex", "B", "other-model", 8, rank=1),
    ]
    agg = aggregate([], judgments, maps, kin_map=kin_map, refscores=REFSCORES)
    assert agg.kin_delta["codex"] == -3


def test_kin_delta_none_when_judge_has_no_kin_models_present():
    maps = {"p1": _map(task_id="b1.it_infra-01", unit="it_infra")}
    kin_map = {"gpt-oss-20b": "codex"}
    judgments = [_j("p1", "claude", "A", "other-model", 6, rank=1)]
    agg = aggregate([], judgments, maps, kin_map=kin_map,
                     refscores=REFSCORES, judge_ids=["claude", "codex"])
    assert agg.kin_delta["claude"] is None    # claude has no kin in the roster
    assert agg.kin_delta["codex"] is None     # codex's kin model never scored


def test_kin_delta_excludes_calibration_letters():
    maps = {"p1": _map(task_id="b1.it_infra-01", unit="it_infra")}
    kin_map = {"gpt-oss-20b": "codex"}
    judgments = [
        _j("p1", "codex", "A", "gpt-oss-20b", 9, rank=1),
        _j("p1", "codex", "B", "other-model", 7, rank=2),
        # CAL letters must never count as "kin" or "non-kin" data points.
        _j("p1", "codex", "C", "CAL-strong", 1, rank=3),
    ]
    agg = aggregate([], judgments, maps, kin_map=kin_map, refscores=REFSCORES)
    assert agg.kin_delta["codex"] == 2


# --- agreement / mean spread ---


def test_agreement_percent_and_mean_spread():
    maps = {"p1": _map(task_id="b1.it_infra-01"), "p2": _map(task_id="b1.it_infra-02")}
    judgments = [
        # p1/model-a: spread 1 -> agrees
        _j("p1", "claude", "A", "model-a", 6, rank=1),
        _j("p1", "codex", "B", "model-a", 7, rank=1),
        _j("p1", "gemini", "C", "model-a", 6, rank=1),
        # p2/model-a: spread 4 -> disagrees
        _j("p2", "claude", "A", "model-a", 2, rank=1),
        _j("p2", "codex", "B", "model-a", 6, rank=1),
        _j("p2", "gemini", "C", "model-a", 6, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.agreement_pct == 50.0
    assert round(agg.mean_spread, 2) == 2.5


def test_agreement_ignores_groups_with_undefined_spread():
    maps = {"p1": _map()}
    judgments = [_j("p1", "claude", "A", "model-a", 6, rank=1)]   # n=1 -> spread None
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.agreement_pct == 0.0
    assert agg.mean_spread == 0.0


# --- incomplete panels ---


def test_incomplete_panel_counted_when_fewer_than_three_judges():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 6, rank=1),
        _j("p1", "codex", "B", "model-a", 7, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.incomplete_panel_count == 1


def test_full_panel_not_counted_as_incomplete():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 6, rank=1),
        _j("p1", "codex", "B", "model-a", 7, rank=1),
        _j("p1", "gemini", "C", "model-a", 8, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.incomplete_panel_count == 0


def test_incomplete_panel_counts_packets_not_groups():
    """Two incomplete groups in the SAME packet count as one incomplete packet."""
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 6, rank=1),
        _j("p1", "claude", "B", "model-b", 7, rank=2),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.incomplete_panel_count == 1


# --- error rows: excluded from math, counted in health ---


def test_error_rows_excluded_from_math_but_counted():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 6, rank=1),
        _j("p1", "codex", "B", "model-a", 7, rank=1),
        {"schema_version": 1, "packet_id": "p1", "judge_id": "gemini",
         "judge_model_pin": "pin", "judge_cli_version": "v1", "letter": "-",
         "model_id": None, "score": None, "reason": "invalid reply", "rank": None,
         "ts": "2026-07-17T00:00:00+00:00", "status": "error"},
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    pa = agg.packet_answers[0]
    assert pa.n_judges == 2                 # gemini's error row contributes nothing
    assert agg.error_rows_count == 1


# --- cal_fallback count comes from the maps, not judgments ---


def test_cal_fallback_count_from_maps():
    maps = {
        "p1": _map(task_id="b1.it_infra-01", cal_fallback=True),
        "p2": _map(task_id="b1.it_infra-02", cal_fallback=False),
    }
    agg = aggregate([], [], maps, refscores=REFSCORES)
    assert agg.cal_fallback_count == 1


# --- model/unit means and overall ---


def test_model_unit_mean_and_overall_mean_of_unit_means():
    maps = {
        "p1": _map(task_id="b1.it_infra-01", unit="it_infra"),
        "p2": _map(task_id="b1.it_infra-02", unit="it_infra"),
        "p3": _map(task_id="b1.finance-01", unit="finance"),
    }
    judgments = [
        _j("p1", "claude", "A", "model-a", 6, rank=1),
        _j("p2", "claude", "A", "model-a", 8, rank=1),
        _j("p3", "claude", "A", "model-a", 4, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.model_unit_stats[("model-a", "it_infra")]["mean"] == 7   # (6+8)/2
    assert agg.model_unit_stats[("model-a", "finance")]["mean"] == 4
    assert agg.model_overall["model-a"] == 5.5                          # mean(7, 4)


# --- pairwise majority vote (models only) ---


def test_pairwise_majority_vote_excludes_calibration():
    maps = {"p1": _map()}
    judgments = [
        _j("p1", "claude", "A", "model-a", 8, rank=1),
        _j("p1", "claude", "B", "model-b", 5, rank=2),
        _j("p1", "codex", "A", "model-a", 8, rank=1),
        _j("p1", "codex", "B", "model-b", 5, rank=2),
        _j("p1", "gemini", "A", "model-a", 4, rank=2),   # dissenting judge
        _j("p1", "gemini", "B", "model-b", 9, rank=1),
        _j("p1", "claude", "C", "CAL-strong", 9, rank=3),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.pairwise_wins.get(("model-a", "model-b")) == 1   # 2-of-3 majority
    assert ("model-b", "model-a") not in agg.pairwise_wins
    assert not any(cal in pair for pair in agg.pairwise_wins for cal in CAL_IDENTITIES)


# --- rubric_sha selection: stale (superseded) packets excluded when known ---


def test_stale_rubric_sha_excluded_when_current_known():
    maps = {
        "p-old": _map(task_id="b1.it_infra-01", unit="it_infra", rubric_sha="old-sha"),
        "p-new": _map(task_id="b1.it_infra-01", unit="it_infra", rubric_sha="new-sha"),
    }
    judgments = [
        _j("p-old", "claude", "A", "model-a", 2, rank=1),
        _j("p-new", "claude", "A", "model-a", 9, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES,
                     current_rubric_sha={"it_infra": "new-sha"})
    assert len(agg.packet_answers) == 1
    assert agg.packet_answers[0].packet_id == "p-new"
    assert agg.packet_answers[0].median == 9


def test_rubric_sha_filter_is_noop_when_unit_unknown():
    """No current_rubric_sha entry for a unit -> nothing is filtered (today's
    reality: no grading/anchors/<unit>.md exists yet, so tables.py can't know
    a 'current' rubric_sha for any unit)."""
    maps = {
        "p-old": _map(task_id="b1.it_infra-01", unit="it_infra", rubric_sha="old-sha"),
        "p-new": _map(task_id="b1.it_infra-01", unit="it_infra", rubric_sha="new-sha"),
    }
    judgments = [
        _j("p-old", "claude", "A", "model-a", 2, rank=1),
        _j("p-new", "claude", "A", "model-a", 9, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert len(agg.packet_answers) == 2


# --- byte-determinism: shuffled input -> identical AggResult content ---


def test_aggregate_deterministic_under_shuffled_input():
    maps = {
        "p1": _map(task_id="b1.finance-01", unit="finance"),
        "p2": _map(task_id="b1.finance-02", unit="finance"),
        "p3": _map(task_id="b1.it_infra-01", unit="it_infra"),
    }
    judgments = [
        _j("p1", "claude", "A", "model-a", 3, rank=3),
        _j("p1", "codex", "B", "model-a", 6, rank=2),
        _j("p1", "gemini", "C", "model-a", 9, rank=1),
        _j("p2", "claude", "A", "model-b", 7, rank=1),
        _j("p2", "codex", "B", "model-b", 7, rank=1),
        _j("p2", "gemini", "C", "model-b", 8, rank=1),
        _j("p3", "claude", "D", "CAL-strong", 7, rank=1),
        _j("p3", "codex", "E", "CAL-strong", 7, rank=1),
        _j("p3", "gemini", "F", "CAL-strong", 7, rank=1),
    ]
    kin_map = {"model-a": "codex"}
    shuffled = judgments[:]
    random.Random(42).shuffle(shuffled)

    agg1 = aggregate([], judgments, maps, kin_map=kin_map, refscores=REFSCORES)
    agg2 = aggregate([], shuffled, dict(reversed(list(maps.items()))),
                      kin_map=kin_map, refscores=REFSCORES)

    def _snapshot(agg):
        return (
            [(pa.packet_id, pa.model_id, pa.median, pa.spread, pa.n_judges)
             for pa in agg.packet_answers],
            agg.spread_flags, agg.drift_flags, agg.kin_delta,
            round(agg.agreement_pct, 4), round(agg.mean_spread, 4),
            agg.incomplete_panel_count, agg.cal_fallback_count,
            sorted(agg.model_overall.items()),
            sorted(agg.pairwise_wins.items()),
        )

    assert _snapshot(agg1) == _snapshot(agg2)


# --- roster_filter: exclude quant-arm models from roster expansion ---


def test_quant_arm_models_excluded_from_roster():
    """When roster_filter is provided, only models in the filter get widened
    into model_roster from result rows. Models with judged scores always appear
    in model_roster regardless (scores are real data)."""
    maps = {"p1": _map(task_id="b1.cybersecurity-01", unit="cybersecurity")}
    judgments = [
        # model-a has a judged score
        _j("p1", "claude", "A", "model-a", 8, rank=1),
        # model-b is in result rows but NOT in roster_filter (e.g., role:quant-arm)
    ]
    # Result rows include both model-a and model-b
    rows = [
        {"model_id": "model-a"},
        {"model_id": "model-b"},
    ]

    # With roster_filter, only model-a should appear (it has a judged score)
    roster_filter = {"model-a"}  # model-b is excluded (quant-arm)
    agg = aggregate(rows, judgments, maps, roster_filter=roster_filter, refscores=REFSCORES)

    # model-a is in roster because it has a judged score
    assert "model-a" in agg.model_roster
    # model-b is NOT in roster (not in filter, no judged scores)
    assert "model-b" not in agg.model_roster


def test_roster_filter_retains_judged_models():
    """Models with judged scores stay in model_roster even if not in roster_filter.
    Only the row-widening expansion is restricted by the filter."""
    maps = {"p1": _map(task_id="b1.cybersecurity-01", unit="cybersecurity")}
    judgments = [
        # model-b has a judged score despite being "quant-arm"
        _j("p1", "claude", "A", "model-b", 6, rank=1),
        _j("p1", "codex", "B", "model-b", 7, rank=2),
    ]
    # Result rows try to widen the roster
    rows = [
        {"model_id": "model-a"},
        {"model_id": "model-b"},
    ]

    # roster_filter excludes both models
    roster_filter = set()
    agg = aggregate(rows, judgments, maps, roster_filter=roster_filter, refscores=REFSCORES)

    # model-a is not in roster (not in filter, no judged scores)
    assert "model-a" not in agg.model_roster
    # model-b IS in roster (has judged scores, regardless of filter)
    assert "model-b" in agg.model_roster
