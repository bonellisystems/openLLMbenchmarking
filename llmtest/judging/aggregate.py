"""Table-time aggregation (TESTPLAN 6.1/6.2, Task 8) -- median/spread/kin/drift/agreement.

Nothing here is stored: `aggregate()` is a PURE function of Store.iter_judgments()
rows, the committed packet maps (results/packets/<packet_id>.map.json), the
judges.yaml kin_map, and grading/calibration/refscores.yaml. Call it fresh at
table-render time (tables.py) -- a judging re-run can never desync a stored
aggregate because none is ever written.

Grouping key: a judgment row already carries the RESOLVED `model_id` (real
model_id, or the literal "CAL-strong"/"CAL-weak" for calibration letters --
see judging/runner.py). Letters themselves are per-(packet, judge) blinding
permutations and are NOT a stable cross-judge identity, so every aggregate
here groups by (packet_id, model_id) -- "one packet's answer", scored by
however many judges produced an ok row for it. This is what the plan's
shorthand "per (packet, letter)" means once letters are resolved back to
identities.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CAL_IDENTITIES = {"CAL-strong", "CAL-weak"}
FULL_PANEL_SIZE = 3   # claude/codex/gemini (TESTPLAN 6.1) -- fewer = incomplete panel


def load_refscores(path: Path | str) -> dict:
    """grading/calibration/refscores.yaml -> {strong, weak, tolerance} with
    Task-8-mandated defaults ({strong: 9, weak: 2, tolerance: 1}) filled in
    for any key the file omits."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {
        "strong": data.get("strong", 9),
        "weak": data.get("weak", 2),
        "tolerance": data.get("tolerance", 1),
    }


def load_maps(maps_dir: Path | str) -> dict[str, dict]:
    """{packet_id: map_dict} for every committed results/packets/*.map.json.
    Returns {} when the directory doesn't exist yet (no packets built)."""
    maps_dir = Path(maps_dir)
    out: dict[str, dict] = {}
    if not maps_dir.exists():
        return out
    for p in sorted(maps_dir.glob("*.map.json")):
        packet_id = p.name[: -len(".map.json")]
        out[packet_id] = json.loads(p.read_text(encoding="utf-8"))
    return out


@dataclass
class PacketAnswer:
    """One packet's answer (one model identity's scores within one packet),
    pooled across every judge that produced an ok row for it."""
    packet_id: str
    task_id: str
    run_n: int
    unit: str
    model_id: str                 # real model_id, or "CAL-strong"/"CAL-weak"
    scores: dict = field(default_factory=dict)   # judge_id -> int score (ok only)

    @property
    def n_judges(self) -> int:
        return len(self.scores)

    @property
    def median(self):
        return statistics.median(self.scores.values())

    @property
    def spread(self):
        """max - min, only defined for n_judges >= 2 (per plan: 'spread only
        defined for n>=2')."""
        if self.n_judges < 2:
            return None
        vals = self.scores.values()
        return max(vals) - min(vals)


@dataclass
class AggResult:
    packet_answers: list = field(default_factory=list)      # list[PacketAnswer], stable-sorted
    spread_flags: list = field(default_factory=list)        # list[dict], stable-sorted
    drift_flags: list = field(default_factory=list)         # list[dict], stable-sorted
    model_unit_stats: dict = field(default_factory=dict)    # (model_id, unit) -> {mean, sd, n}
    model_overall: dict = field(default_factory=dict)       # model_id -> mean of unit means
    model_roster: list = field(default_factory=list)        # sorted model_ids "present"
    kin_delta: dict = field(default_factory=dict)           # judge_id -> float | None
    agreement_pct: float = 0.0
    mean_spread: float = 0.0
    incomplete_panel_count: int = 0
    cal_fallback_count: int = 0
    error_rows_count: int = 0
    pairwise_wins: dict = field(default_factory=dict)       # (winner, loser) -> packet win count


def aggregate(
    rows: list,
    judgments: list,
    maps: dict,
    *,
    kin_map: dict | None = None,
    refscores: dict | None = None,
    judge_ids: list | None = None,
    current_rubric_sha: dict | None = None,
) -> AggResult:
    """Compute every table-time aggregate from raw judgments.

    Args:
        rows: Store.iter_rows() result rows -- used only to widen the model
            roster (a model that ran but has no judgments yet still gets a
            scorecard column of all "-").
        judgments: Store.iter_judgments() rows.
        maps: {packet_id: map_dict} as returned by `load_maps` (or hand-built
            in tests) -- supplies task_id/run_n/unit/rubric_sha/cal_fallback,
            the metadata judgment rows themselves don't carry.
        kin_map: config/judges.yaml's `kin_map:` dict ({model_id: judge_id}).
        refscores: grading/calibration/refscores.yaml, as loaded by
            `load_refscores` (or the Task-8 default when omitted).
        judge_ids: the full configured judge panel (e.g. sorted(judges.yaml
            `judges:`)); when given, `kin_delta` reports one entry per judge
            even for a judge with zero ok judgments so far (value None).
            Defaults to the judge_ids actually seen in `judgments`.
        current_rubric_sha: optional {unit: rubric_sha} of the CURRENTLY
            checked-out anchor file per unit (TESTPLAN 6.2: "aggregation at
            table time selects judgments matching the checked-out
            rubric_sha"). A packet whose map's rubric_sha doesn't match its
            unit's entry is a superseded (re-minted) packet and is excluded
            entirely. A unit absent from this dict is not filtered (today's
            reality before grading/anchors/ exists for any unit).
    """
    kin_map = kin_map or {}
    refscores = refscores or {"strong": 9, "weak": 2, "tolerance": 1}
    current_rubric_sha = current_rubric_sha or {}

    error_rows_count = 0
    groups: dict[tuple, dict] = {}   # (packet_id, model_id) -> {judge_id: score}
    for j in judgments:
        if j.get("status") != "ok":
            error_rows_count += 1
            continue
        packet_id = j["packet_id"]
        m = maps.get(packet_id)
        if m is None:
            continue   # no committed map for this packet -- can't attribute unit/task, skip
        unit = m.get("unit")
        want_sha = current_rubric_sha.get(unit)
        if want_sha is not None and m.get("rubric_sha") != want_sha:
            continue   # superseded packet (rubric changed since this was judged)
        key = (packet_id, j["model_id"])
        groups.setdefault(key, {})[j["judge_id"]] = j["score"]

    packet_answers = [
        PacketAnswer(
            packet_id=packet_id, model_id=model_id,
            task_id=maps[packet_id].get("task_id", ""),
            run_n=maps[packet_id].get("run_n", 0),
            unit=maps[packet_id].get("unit", ""),
            scores=dict(scores),
        )
        for (packet_id, model_id), scores in groups.items()
    ]
    packet_answers.sort(key=lambda pa: (pa.task_id, pa.run_n, pa.packet_id, pa.model_id))

    # --- spread flags: spread > 2, ANY letter (calibration letters included --
    # the interface doesn't restrict this one to real models). ---
    spread_flags = []
    for pa in packet_answers:
        if pa.spread is not None and pa.spread > 2:
            spread_flags.append({
                "task_id": pa.task_id, "run_n": pa.run_n, "model_id": pa.model_id,
                "packet_id": pa.packet_id,
                "scores_by_judge": dict(sorted(pa.scores.items())),
            })
    spread_flags.sort(key=lambda f: (f["task_id"], f["run_n"], f["model_id"], f["packet_id"]))

    # --- calibration drift: CAL-strong/CAL-weak median vs refscores ---
    tolerance = refscores["tolerance"]
    drift_flags = []
    for pa in packet_answers:
        if pa.model_id not in CAL_IDENTITIES:
            continue
        cal_type = "strong" if pa.model_id == "CAL-strong" else "weak"
        ref = refscores[cal_type]
        delta = pa.median - ref
        if abs(delta) > tolerance:
            drift_flags.append({
                "packet_id": pa.packet_id, "task_id": pa.task_id, "run_n": pa.run_n,
                "cal_type": cal_type, "median": pa.median, "ref": ref, "delta": delta,
            })
    drift_flags.sort(key=lambda f: (f["task_id"], f["run_n"], f["packet_id"], f["cal_type"]))

    # --- agreement % (spread <= 1) + mean spread, over every group where
    # spread is defined (n_judges >= 2) ---
    defined = [pa for pa in packet_answers if pa.spread is not None]
    if defined:
        agreement_pct = 100.0 * sum(1 for pa in defined if pa.spread <= 1) / len(defined)
        mean_spread = sum(pa.spread for pa in defined) / len(defined)
    else:
        agreement_pct = 0.0
        mean_spread = 0.0

    # --- incomplete panels: distinct PACKETS containing >=1 group with
    # n_judges < FULL_PANEL_SIZE ---
    incomplete_panel_count = len({pa.packet_id for pa in packet_answers
                                   if pa.n_judges < FULL_PANEL_SIZE})

    # --- cal_fallback count: from the maps (packet builder used the global
    # strong.md/weak.md fallback), not from judgments ---
    cal_fallback_count = sum(1 for m in maps.values() if m.get("cal_fallback"))

    # --- kin-delta per judge: mean(score on kin models) - mean(score on
    # non-kin), real-model letters only ---
    seen_judge_ids = {j["judge_id"] for j in judgments}
    all_judge_ids = sorted(seen_judge_ids | set(judge_ids or []))
    kin_delta: dict[str, float | None] = {}
    for judge_id in all_judge_ids:
        kin_scores, nonkin_scores = [], []
        for pa in packet_answers:
            if pa.model_id in CAL_IDENTITIES:
                continue
            score = pa.scores.get(judge_id)
            if score is None:
                continue
            if kin_map.get(pa.model_id) == judge_id:
                kin_scores.append(score)
            else:
                nonkin_scores.append(score)
        if kin_scores and nonkin_scores:
            kin_delta[judge_id] = (sum(kin_scores) / len(kin_scores)
                                    - sum(nonkin_scores) / len(nonkin_scores))
        else:
            kin_delta[judge_id] = None

    # --- per (model, unit) mean of medians (+sd); overall = mean of unit means ---
    model_unit_medians: dict[tuple, list] = {}
    for pa in packet_answers:
        if pa.model_id in CAL_IDENTITIES:
            continue
        model_unit_medians.setdefault((pa.model_id, pa.unit), []).append(pa.median)

    model_unit_stats = {}
    for key, medians in model_unit_medians.items():
        mean = sum(medians) / len(medians)
        sd = statistics.pstdev(medians) if len(medians) > 1 else 0.0
        model_unit_stats[key] = {"mean": mean, "sd": sd, "n": len(medians)}

    model_unit_means: dict[str, list] = {}
    for (model_id, _unit), stats in model_unit_stats.items():
        model_unit_means.setdefault(model_id, []).append(stats["mean"])
    model_overall = {model_id: sum(means) / len(means)
                      for model_id, means in model_unit_means.items()}

    # --- model roster: models with aggregate data, widened by any model_id
    # present in `rows` (so a model that ran but isn't judged yet still gets
    # a scorecard column) ---
    model_roster = sorted(set(model_overall) | {r["model_id"] for r in rows if r.get("model_id")})

    # --- pairwise majority-vote win matrix from rankings (models only) ---
    # Per packet, per model pair: each judge votes for whichever of the two
    # it ranked better; the pair's winner is whichever model got the
    # majority of judge votes for that packet. Ties record no winner.
    by_packet_judge: dict[str, dict[str, dict]] = {}
    for j in judgments:
        if j.get("status") != "ok" or j["model_id"] in CAL_IDENTITIES:
            continue
        by_packet_judge.setdefault(j["packet_id"], {}).setdefault(
            j["judge_id"], {})[j["model_id"]] = j["rank"]

    pairwise_wins: dict[tuple, int] = {}
    for packet_id in sorted(by_packet_judge):
        judge_rank_maps = by_packet_judge[packet_id]
        models_in_packet = sorted({m for rm in judge_rank_maps.values() for m in rm})
        for i, model_a in enumerate(models_in_packet):
            for model_b in models_in_packet[i + 1:]:
                votes_a = votes_b = 0
                for rank_map in judge_rank_maps.values():
                    if model_a in rank_map and model_b in rank_map:
                        if rank_map[model_a] < rank_map[model_b]:
                            votes_a += 1
                        elif rank_map[model_b] < rank_map[model_a]:
                            votes_b += 1
                if votes_a > votes_b:
                    pairwise_wins[(model_a, model_b)] = pairwise_wins.get((model_a, model_b), 0) + 1
                elif votes_b > votes_a:
                    pairwise_wins[(model_b, model_a)] = pairwise_wins.get((model_b, model_a), 0) + 1

    return AggResult(
        packet_answers=packet_answers,
        spread_flags=spread_flags,
        drift_flags=drift_flags,
        model_unit_stats=model_unit_stats,
        model_overall=model_overall,
        model_roster=model_roster,
        kin_delta=kin_delta,
        agreement_pct=agreement_pct,
        mean_spread=mean_spread,
        incomplete_panel_count=incomplete_panel_count,
        cal_fallback_count=cal_fallback_count,
        error_rows_count=error_rows_count,
        pairwise_wins=pairwise_wins,
    )
