import json
import random
import shutil
from pathlib import Path

from llmtest.judging.aggregate import aggregate
from llmtest.store import Store
from llmtest.tables import (
    _current_rubric_sha,
    render_flags,
    render_scorecard,
    render_serving_table,
    run_tables,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

def _r(model, cond, tps):
    return {"model_id": model, "hf_repo": f"org/{model}", "condition": cond,
            "timing_authoritative": True, "status": "ok",
            "response_meta": {"decode_tps": tps, "pp_tps": 0, "ttft_ms": 0},
            "tags": []}

def test_serving_table_deterministic_and_authority_filtered():
    rows = [_r("b", "cond=PEAK", 100.15), _r("a", "cond=PEAK", 50.0),
            dict(_r("c", "cond=PEAK", 999.0), timing_authoritative=False)]
    out1 = render_serving_table(rows)
    out2 = render_serving_table(list(reversed(rows)))
    assert out1 == out2
    assert "org/a" in out1 and "org/b" in out1
    assert "999" not in out1
    assert "100.2" in out1     # fixed rounding


# --- scorecard.md / FLAGS.md rendering (Task 8) ---

UNITS = ["coding", "finance", "it_infra"]


def _map(task_id, run_n=1, unit="finance", rubric_sha="sha1", cal_fallback=False):
    return {"task_id": task_id, "run_n": run_n, "unit": unit, "rubric_sha": rubric_sha,
            "cal_fallback": cal_fallback, "base_seed": "s", "letters_by_judge": {}}


def _j(packet_id, judge_id, model_id, score, rank, status="ok"):
    return {"schema_version": 1, "packet_id": packet_id, "judge_id": judge_id,
            "judge_model_pin": "pin", "judge_cli_version": "v1", "letter": "A",
            "model_id": model_id, "score": score, "reason": "r", "rank": rank,
            "ts": "2026-07-17T00:00:00+00:00", "status": status}


def _sample_maps_and_judgments():
    maps = {
        "p1": _map("b1.finance-01", unit="finance"),
        "p2": _map("b1.it_infra-01", unit="it_infra"),
    }
    judgments = [
        _j("p1", "claude", "model-a", 8, 1),
        _j("p1", "codex", "model-a", 6, 1),
        _j("p1", "gemini", "model-a", 9, 1),   # spread 3 -> flags
        _j("p2", "claude", "model-b", 7, 1),
        _j("p2", "codex", "model-b", 7, 1),
        _j("p2", "gemini", "model-b", 7, 1),
    ]
    return maps, judgments


def _sample_agg():
    maps, judgments = _sample_maps_and_judgments()
    refscores = {"strong": 9, "weak": 2, "tolerance": 1}
    return aggregate([], judgments, maps, kin_map={}, refscores=refscores,
                      judge_ids=["claude", "codex", "gemini"])


def test_scorecard_grid_units_alpha_models_by_overall_desc():
    agg = _sample_agg()
    out = render_scorecard(agg, UNITS)
    lines = out.splitlines()
    header = next(l for l in lines if l.startswith("| Unit |"))
    assert header.index("model-a") < header.index("model-b")   # 8.0 overall > 7.0
    assert "| coding | - | - |" in out                          # unit with no data at all
    assert "8.0" in out and "7.0" in out


def test_scorecard_missing_cell_is_dash():
    agg = _sample_agg()
    out = render_scorecard(agg, UNITS)
    it_infra_row = next(l for l in out.splitlines() if l.startswith("| it_infra"))
    assert it_infra_row.split("|")[2].strip() == "-"            # model-a never scored in it_infra


def test_scorecard_health_block_fields():
    agg = _sample_agg()
    out = render_scorecard(agg, UNITS)
    assert "Agreement" in out
    assert "Mean spread" in out
    assert "Kin-delta" in out and "n/a" in out       # no kin_map given -> every judge n/a
    assert "Drift flags: 0" in out
    assert "Spread flags: 1" in out
    assert "Incomplete panels: 0" in out


def test_scorecard_no_data_placeholder():
    agg = aggregate([], [], {}, refscores={"strong": 9, "weak": 2, "tolerance": 1})
    out = render_scorecard(agg, UNITS)
    assert "no judgment data yet" in out
    assert "Spread flags: 0" in out


def test_scorecard_deterministic_shuffled_judgments():
    maps, judgments = _sample_maps_and_judgments()
    shuffled = judgments[:]
    random.Random(7).shuffle(shuffled)
    refscores = {"strong": 9, "weak": 2, "tolerance": 1}
    agg1 = aggregate([], judgments, maps, refscores=refscores, judge_ids=["claude", "codex", "gemini"])
    agg2 = aggregate([], shuffled, dict(reversed(list(maps.items()))), refscores=refscores,
                      judge_ids=["gemini", "codex", "claude"])
    out1 = render_scorecard(agg1, UNITS)
    out2 = render_scorecard(agg2, UNITS)
    assert out1 == out2


def test_flags_spread_and_none_placeholder_when_empty():
    agg = _sample_agg()
    out = render_flags(agg)
    assert "b1.finance-01" in out and "model-a" in out
    assert "claude=8, codex=6, gemini=9" in out
    assert "## Drift flags" in out
    drift_section = out.split("## Drift flags")[1]
    assert "(none)" in drift_section


def test_flags_drift_row_rendered():
    maps = {"p1": _map("b1.finance-01", unit="finance")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 7, 1),
        _j("p1", "codex", "CAL-strong", 7, 1),
        _j("p1", "gemini", "CAL-strong", 7, 1),
    ]
    refscores = {"strong": 9, "weak": 2, "tolerance": 1}
    agg = aggregate([], judgments, maps, refscores=refscores)
    out = render_flags(agg)
    assert "strong" in out
    assert "-2.0" in out or "-2" in out


def test_flags_deterministic_shuffled_input():
    maps, judgments = _sample_maps_and_judgments()
    shuffled = judgments[:]
    random.Random(3).shuffle(shuffled)
    refscores = {"strong": 9, "weak": 2, "tolerance": 1}
    agg1 = aggregate([], judgments, maps, refscores=refscores)
    agg2 = aggregate([], shuffled, dict(reversed(list(maps.items()))), refscores=refscores)
    assert render_flags(agg1) == render_flags(agg2)


def test_flags_spread_row_includes_packet_column():
    """Spread flag rows must include a Packet column (first 12 hex chars of packet_id)."""
    agg = _sample_agg()
    out = render_flags(agg)
    # Extract spread flags section
    spread_section = out.split("## Drift flags")[0]
    assert "| Packet |" in spread_section or "Packet" in spread_section
    # Check that packet_id (or its first 12 chars) appears in the spread row
    assert "p1" in spread_section


def test_flags_drift_ref_formatted_with_decimal():
    """Drift flag ref column should be formatted with 1 decimal place (e.g. '9.0', not '9')."""
    maps = {"p1": _map("b1.finance-01", unit="finance")}
    judgments = [
        _j("p1", "claude", "CAL-strong", 7, 1),
        _j("p1", "codex", "CAL-strong", 7, 1),
        _j("p1", "gemini", "CAL-strong", 7, 1),
    ]
    refscores = {"strong": 9, "weak": 2, "tolerance": 1}
    agg = aggregate([], judgments, maps, refscores=refscores)
    out = render_flags(agg)
    # The ref value should be formatted with 1 decimal: "9.0" not "9"
    assert "9.0" in out


# --- run_tables integration: real config + a tmp results tree, byte-clean twice ---


def _scaffold_root(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    shutil.copytree(REPO_ROOT / "grading", tmp_path / "grading")
    (tmp_path / "results" / "packets").mkdir(parents=True)
    return tmp_path


def test_run_tables_writes_three_files_byte_clean_on_rerun(tmp_path):
    root = _scaffold_root(tmp_path)
    store = Store(root / "results")

    # _scaffold_root copies the repo's real grading/anchors -- since Task 9
    # populated real anchor files, run_tables() now filters judgments to the
    # CURRENTLY checked-out rubric_sha per unit (TESTPLAN 6.2). Build maps
    # with the real sha so this fixture's judgments survive that filter
    # regardless of anchor content, instead of hardcoding a stale "sha1".
    rubric = _current_rubric_sha(root)
    _, judgments = _sample_maps_and_judgments()
    maps = {
        "p1": _map("b1.finance-01", unit="finance",
                    rubric_sha=rubric.get("finance", "sha1")),
        "p2": _map("b1.it_infra-01", unit="it_infra",
                    rubric_sha=rubric.get("it_infra", "sha1")),
    }
    for packet_id, m in maps.items():
        (root / "results" / "packets" / f"{packet_id}.map.json").write_text(
            json.dumps(m, sort_keys=True), encoding="utf-8")
    for j in judgments:
        store.append_judgment(j)

    assert run_tables(root) == 0
    scorecard1 = (root / "results" / "tables" / "scorecard.md").read_bytes()
    flags1 = (root / "results" / "FLAGS.md").read_bytes()
    serving1 = (root / "results" / "tables" / "serving.md").read_bytes()

    assert run_tables(root) == 0
    scorecard2 = (root / "results" / "tables" / "scorecard.md").read_bytes()
    flags2 = (root / "results" / "FLAGS.md").read_bytes()
    serving2 = (root / "results" / "tables" / "serving.md").read_bytes()

    assert scorecard1 == scorecard2
    assert flags1 == flags2
    assert serving1 == serving2
    assert b"\r\n" not in scorecard1 and b"\r\n" not in flags1
    assert b"model-a" in scorecard1


def test_run_tables_handles_empty_results(tmp_path):
    root = _scaffold_root(tmp_path)
    assert run_tables(root) == 0
    assert (root / "results" / "tables" / "scorecard.md").exists()
    assert (root / "results" / "FLAGS.md").exists()
