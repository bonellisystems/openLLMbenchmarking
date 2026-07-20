"""Tests for B2 per-(model, axis) aggregation + fabrication hard-cap (Task 5).

B2 packet maps carry `dim` (e.g. "axis5") instead of B1's `unit`, plus a
per-answer `fabrication_pass` dict ({model_id: bool}) recorded by the
deterministic guard (Task 3). aggregate() must route dim-having packets into
`b2_axis_scores` -- median-of-judges per (packet, model), capped at 2.0 when
fabrication_pass[model_id] is False -- WITHOUT letting them leak into the B1
(model, unit) aggregation path (model_unit_stats/model_overall/scorecard).
"""
from __future__ import annotations

from llmtest.judging.aggregate import aggregate

REFSCORES = {"strong": 9, "weak": 2, "tolerance": 1}


def _b1_map(task_id="b1.cybersecurity-01", run_n=1, unit="cybersecurity",
            rubric_sha="rsha1", cal_fallback=False):
    return {"task_id": task_id, "run_n": run_n, "unit": unit,
            "rubric_sha": rubric_sha, "cal_fallback": cal_fallback,
            "base_seed": "seed1", "letters_by_judge": {}}


def _b2_map(task_id="b2.error-recovery-01", run_n=1, dim="axis5",
            scenario="error-recovery-01", present_models=None,
            missing_models=None, fabrication_pass=None, rubric_sha="rsha2"):
    return {
        "task_id": task_id, "run_n": run_n, "dim": dim, "scenario": scenario,
        "present_models": present_models or [], "missing_models": missing_models or [],
        "fabrication_pass": fabrication_pass or {}, "rubric_sha": rubric_sha,
        "cal_fallback": False, "base_seed": "seed2", "letters_by_judge": {},
    }


def _j(packet_id, judge_id, letter, model_id, score, rank=1, status="ok"):
    return {"schema_version": 1, "packet_id": packet_id, "judge_id": judge_id,
            "judge_model_pin": "pin", "judge_cli_version": "v1", "letter": letter,
            "model_id": model_id if status == "ok" else None,
            "score": score if status == "ok" else None,
            "reason": "r", "rank": rank if status == "ok" else None,
            "ts": "2026-07-19T00:00:00+00:00", "status": status}


# --- (a) three judges' axis-5 scores median correctly per model ---


def test_b2_axis_score_medians_per_model():
    maps = {"p1": _b2_map(fabrication_pass={"m1": True, "m2": True})}
    judgments = [
        _j("p1", "claude", "A", "m1", 6, rank=1),
        _j("p1", "codex", "B", "m1", 7, rank=1),
        _j("p1", "gemini", "C", "m1", 8, rank=1),
        _j("p1", "claude", "D", "m2", 3, rank=2),
        _j("p1", "codex", "E", "m2", 4, rank=2),
        _j("p1", "gemini", "F", "m2", 5, rank=2),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.b2_axis_scores[("m1", "axis5")] == 7   # median(6,7,8)
    assert agg.b2_axis_scores[("m2", "axis5")] == 4   # median(3,4,5)


def test_b2_axis_scores_average_across_multiple_packets_same_axis():
    """A model appearing in two B2 packets for the same axis -> mean of the
    two per-packet medians (mirrors B1's mean-of-medians per (model, unit))."""
    maps = {
        "p1": _b2_map(task_id="b2.scenario-a", run_n=1,
                       fabrication_pass={"m1": True}),
        "p2": _b2_map(task_id="b2.scenario-b", run_n=1,
                       fabrication_pass={"m1": True}),
    }
    judgments = [
        _j("p1", "claude", "A", "m1", 4, rank=1),
        _j("p1", "codex", "B", "m1", 4, rank=1),
        _j("p1", "gemini", "C", "m1", 4, rank=1),
        _j("p2", "claude", "A", "m1", 8, rank=1),
        _j("p2", "codex", "B", "m1", 8, rank=1),
        _j("p2", "gemini", "C", "m1", 8, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.b2_axis_scores[("m1", "axis5")] == 6   # mean(4, 8)


# --- (b) fabrication hard-cap at 2 ---


def test_fabrication_fail_caps_score_at_two_even_if_median_is_nine():
    maps = {"p1": _b2_map(fabrication_pass={"m1": False, "m2": True})}
    judgments = [
        _j("p1", "claude", "A", "m1", 9, rank=1),
        _j("p1", "codex", "B", "m1", 9, rank=1),
        _j("p1", "gemini", "C", "m1", 9, rank=1),
        _j("p1", "claude", "D", "m2", 9, rank=2),
        _j("p1", "codex", "E", "m2", 9, rank=2),
        _j("p1", "gemini", "F", "m2", 9, rank=2),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.b2_axis_scores[("m1", "axis5")] == 2      # capped: fabrication_pass False
    assert agg.b2_axis_scores[("m2", "axis5")] == 9      # not capped: fabrication_pass True


def test_fabrication_pass_missing_entry_does_not_cap():
    """fabrication_pass.get(model_id) is None (guard not applicable / model
    absent from the dict) -> NOT capped. Only an explicit False caps."""
    maps = {"p1": _b2_map(fabrication_pass={})}   # no entry for m1 at all
    judgments = [
        _j("p1", "claude", "A", "m1", 9, rank=1),
        _j("p1", "codex", "B", "m1", 9, rank=1),
        _j("p1", "gemini", "C", "m1", 9, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    assert agg.b2_axis_scores[("m1", "axis5")] == 9


# --- B1 path untouched / uncontaminated by B2 packets ---


def test_b2_packets_excluded_from_b1_model_unit_aggregation():
    """A mixed B1+B2 judgment set: B2 packets must not appear in
    model_unit_stats/model_overall/model roster's unit-keyed math, and must
    not show up as bogus unit="" rows."""
    maps = {
        "p1": _b1_map(task_id="b1.cybersecurity-01", unit="cybersecurity"),
        "p2": _b2_map(task_id="b2.error-recovery-01", fabrication_pass={"m1": True}),
    }
    judgments = [
        _j("p1", "claude", "A", "m1", 6, rank=1),
        _j("p1", "codex", "B", "m1", 7, rank=1),
        _j("p1", "gemini", "C", "m1", 8, rank=1),
        _j("p2", "claude", "A", "m1", 9, rank=1),
        _j("p2", "codex", "B", "m1", 9, rank=1),
        _j("p2", "gemini", "C", "m1", 9, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    # B1 aggregation sees ONLY the B1 packet's median (7), unaffected by the
    # B2 packet's 9s.
    assert agg.model_unit_stats[("m1", "cybersecurity")]["mean"] == 7
    assert agg.model_overall["m1"] == 7
    assert all(unit != "" for (_model, unit) in agg.model_unit_stats)
    # And the B2 packet's answer landed in b2_axis_scores, not packet_answers'
    # unit-keyed math.
    assert agg.b2_axis_scores[("m1", "axis5")] == 9


def test_b1_only_aggregation_byte_identical_with_or_without_b2_present():
    """Presence of B2 code paths must not perturb a B1-only aggregation:
    running aggregate() on a B1-only judgment set produces the same
    B1-relevant fields as it did before Task 5 (no dim-having packets at all,
    so this is really just a regression guard on the B1 path)."""
    maps = {
        "p1": _b1_map(task_id="b1.finance-01", unit="finance"),
        "p2": _b1_map(task_id="b1.it_infra-01", unit="it_infra"),
    }
    judgments = [
        _j("p1", "claude", "A", "model-a", 3, rank=3),
        _j("p1", "codex", "B", "model-a", 6, rank=2),
        _j("p1", "gemini", "C", "model-a", 9, rank=1),
        _j("p2", "claude", "A", "model-b", 7, rank=1),
        _j("p2", "codex", "B", "model-b", 7, rank=1),
        _j("p2", "gemini", "C", "model-b", 8, rank=1),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)

    # Concrete numeric assertions, computed by hand from the fixture above,
    # so this test catches a systematic B1-path regression (e.g. a wrong
    # mean/median/agreement formula) and not just non-determinism between
    # two identical calls.
    #   p1/model-a scores {3, 6, 9} -> median 6, spread 6 (n=3)
    #   p2/model-b scores {7, 7, 8} -> median 7, spread 1 (n=3)
    assert agg.model_unit_stats[("model-a", "finance")] == {"mean": 6, "sd": 0.0, "n": 1}
    assert agg.model_unit_stats[("model-b", "it_infra")] == {"mean": 7, "sd": 0.0, "n": 1}
    assert agg.model_overall["model-a"] == 6
    assert agg.model_overall["model-b"] == 7
    assert agg.model_roster == ["model-a", "model-b"]
    # agreement_pct/mean_spread over the 2 defined-spread groups: only p2
    # (spread=1) is within the spread<=1 agreement bar; p1 (spread=6) isn't.
    assert round(agg.agreement_pct, 4) == 50.0
    assert round(agg.mean_spread, 4) == 3.5
    assert agg.incomplete_panel_count == 0
    assert agg.cal_fallback_count == 0
    assert agg.error_rows_count == 0
    # Each packet has only one model ranked -> no pairs, no pairwise data.
    assert agg.pairwise_wins == {}

    def _snapshot(a):
        return (
            [(pa.packet_id, pa.model_id, pa.median, pa.spread, pa.n_judges)
             for pa in a.packet_answers],
            a.spread_flags, a.drift_flags,
            sorted(a.model_unit_stats.items()), sorted(a.model_overall.items()),
            a.model_roster, a.kin_delta,
            round(a.agreement_pct, 4), round(a.mean_spread, 4),
            a.incomplete_panel_count, a.cal_fallback_count, a.error_rows_count,
            sorted(a.pairwise_wins.items()),
        )

    agg2 = aggregate([], judgments, maps, refscores=REFSCORES)
    assert _snapshot(agg) == _snapshot(agg2)
    assert agg.b2_axis_scores == {}


def test_b2_axis_packet_never_contaminates_pairwise_wins():
    """CRITICAL regression: the pairwise_wins loop re-iterates `judgments`
    and reads j["rank"] for every packet. B2 axis packets DO carry rank, and
    at full-roster quorum routinely have 2+ real models ranked in one
    packet -- if that loop isn't guarded by dim the same way the B1/B2 split
    above is, those axis rankings blend into the B1 unit-based head-to-head
    matrix under the same (winner, loser) keys.

    Fixture: one B1 packet with two models (model-a beats model-b on every
    judge's ranking) plus one B2 axis packet with two DIFFERENT models (m1,
    m2) also fully ranked by every judge -- exactly the shape the old test
    missed, since its B2 packet had only a single model so the inner
    pairwise loop never executed for it.
    """
    maps = {
        "p1": _b1_map(task_id="b1.finance-01", unit="finance"),
        "p2": _b2_map(task_id="b2.error-recovery-01", dim="axis5",
                       present_models=["m1", "m2"],
                       fabrication_pass={"m1": True, "m2": True}),
    }
    judgments = [
        # B1 packet: model-a ranked 1st, model-b ranked 2nd by all 3 judges
        # -> unambiguous pairwise winner (model-a, model-b).
        _j("p1", "claude", "A", "model-a", 8, rank=1),
        _j("p1", "claude", "D", "model-b", 4, rank=2),
        _j("p1", "codex", "B", "model-a", 8, rank=1),
        _j("p1", "codex", "E", "model-b", 4, rank=2),
        _j("p1", "gemini", "C", "model-a", 8, rank=1),
        _j("p1", "gemini", "F", "model-b", 4, rank=2),
        # B2 axis packet: m1 ranked 1st, m2 ranked 2nd by all 3 judges --
        # would (pre-fix) produce a bogus pairwise_wins[("m1", "m2")] entry.
        _j("p2", "claude", "G", "m1", 9, rank=1),
        _j("p2", "claude", "H", "m2", 3, rank=2),
        _j("p2", "codex", "I", "m1", 9, rank=1),
        _j("p2", "codex", "J", "m2", 3, rank=2),
        _j("p2", "gemini", "K", "m1", 9, rank=1),
        _j("p2", "gemini", "L", "m2", 3, rank=2),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)

    # pairwise_wins must contain ONLY the B1-derivable entry -- nothing that
    # could only have come from the B2 packet's two models.
    assert agg.pairwise_wins == {("model-a", "model-b"): 1}
    assert ("m1", "m2") not in agg.pairwise_wins
    assert ("m2", "m1") not in agg.pairwise_wins
    # No key in pairwise_wins should mention a B2-only model at all.
    models_in_pairwise = {m for pair in agg.pairwise_wins for m in pair}
    assert models_in_pairwise == {"model-a", "model-b"}
    # And the B2 packet's own data still correctly landed in b2_axis_scores.
    assert agg.b2_axis_scores[("m1", "axis5")] == 9
    assert agg.b2_axis_scores[("m2", "axis5")] == 3


def test_b2_cal_identities_excluded_from_axis_scores():
    """CAL-strong and CAL-weak letters (calibration identities) must never
    appear as model_ids in b2_axis_scores, matching the B1 path exclusion."""
    maps = {
        "p1": _b2_map(
            fabrication_pass={"m1": True, "CAL-strong": True, "CAL-weak": True}
        ),
    }
    judgments = [
        _j("p1", "claude", "A", "m1", 5, rank=1),
        _j("p1", "codex", "B", "m1", 6, rank=1),
        _j("p1", "gemini", "C", "m1", 7, rank=1),
        _j("p1", "claude", "D", "CAL-strong", 9, rank=2),
        _j("p1", "codex", "E", "CAL-strong", 9, rank=2),
        _j("p1", "gemini", "F", "CAL-strong", 9, rank=2),
        _j("p1", "claude", "G", "CAL-weak", 2, rank=3),
        _j("p1", "codex", "H", "CAL-weak", 2, rank=3),
        _j("p1", "gemini", "I", "CAL-weak", 2, rank=3),
    ]
    agg = aggregate([], judgments, maps, refscores=REFSCORES)
    # m1 should appear in b2_axis_scores
    assert ("m1", "axis5") in agg.b2_axis_scores
    # CAL-strong and CAL-weak must NEVER appear as keys in b2_axis_scores
    for key in agg.b2_axis_scores:
        model_id = key[0]
        assert model_id not in {"CAL-strong", "CAL-weak"}, \
            f"CAL identity {model_id} should not appear in b2_axis_scores"
