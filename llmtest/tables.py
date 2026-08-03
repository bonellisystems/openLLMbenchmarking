"""Byte-deterministic tables (TESTPLAN 7.5): pure functions of rows, stable sorts, fixed float fmt, LF newlines."""
import hashlib
from pathlib import Path

from llmtest.judging.aggregate import AggResult, aggregate, load_maps, load_refscores
from llmtest.judging.runner import resolve_cohort_models
from llmtest.store import Store


def _fmt1(x: float) -> str:
    return f"{x:.1f}"


def _fmt2(x: float) -> str:
    return f"{x:.2f}"


def render_serving_table(rows: list[dict]) -> str:
    keep = [r for r in rows
            if r.get("timing_authoritative") and r.get("status") == "ok"
            and "non-reportable" not in r.get("tags", [])]
    keep.sort(key=lambda r: (r["model_id"], r["condition"]))
    lines = ["# Serving (Battery 5) — timing_authoritative rows only", "",
             "| Model | Condition | decode t/s | PP t/s | TTFT ms |",
             "|---|---|---|---|---|"]
    for r in keep:
        m = r["response_meta"]
        lines.append(f"| {r['hf_repo']} | {r['condition']} | {m.get('decode_tps', 0):.1f} "
                     f"| {m.get('pp_tps', 0):.1f} | {m.get('ttft_ms', 0):.0f} |")
    return "\n".join(lines) + "\n"


def render_scorecard(agg: AggResult, units: list[str]) -> str:
    """results/tables/scorecard.md: units (rows, sorted alpha) x models
    (columns, sorted by overall desc then name); cell = unit mean, 1-decimal;
    header row includes an Overall row; footer = suite-health block."""
    units_sorted = sorted(units)

    def _sort_key(model_id: str):
        overall = agg.model_overall.get(model_id)
        return (overall is None, -(overall if overall is not None else 0.0), model_id)

    models_sorted = sorted(agg.model_roster, key=_sort_key)

    lines = ["# Scorecard (Battery 1) -- table-time aggregation, computed fresh "
             "every run, never stored", ""]
    if not models_sorted:
        lines.append("(no judgment data yet)")
    else:
        header = "| Unit | " + " | ".join(models_sorted) + " |"
        sep = "|---|" + "---|" * len(models_sorted)
        lines += [header, sep]
        overall_cells = [_fmt1(agg.model_overall[m]) if m in agg.model_overall else "-"
                          for m in models_sorted]
        lines.append("| Overall | " + " | ".join(overall_cells) + " |")
        for unit in units_sorted:
            cells = []
            for m in models_sorted:
                stats = agg.model_unit_stats.get((m, unit))
                cells.append(_fmt1(stats["mean"]) if stats else "-")
            lines.append(f"| {unit} | " + " | ".join(cells) + " |")

    lines += ["", "## Suite Health", ""]
    lines.append(f"- Agreement (spread <=1): {agg.agreement_pct:.1f}%")
    lines.append(f"- Mean spread: {agg.mean_spread:.2f}")
    kin_parts = [
        f"{judge_id}={_fmt2(agg.kin_delta[judge_id]) if agg.kin_delta[judge_id] is not None else 'n/a'}"
        for judge_id in sorted(agg.kin_delta)
    ]
    lines.append(f"- Kin-delta: {', '.join(kin_parts) if kin_parts else 'n/a'}")
    lines.append(f"- Drift flags: {len(agg.drift_flags)}")
    lines.append(f"- Spread flags: {len(agg.spread_flags)}")
    lines.append(f"- Incomplete panels: {agg.incomplete_panel_count}")
    lines.append(f"- Cal-fallback packets: {agg.cal_fallback_count}")
    lines.append(f"- Errored judge-packets: {agg.error_rows_count}")
    return "\n".join(lines) + "\n"


def render_flags(agg: AggResult) -> str:
    """results/FLAGS.md: one row per spread>2 flag + per drift flag, stable sort."""
    lines = ["# FLAGS -- spread>2 and calibration-drift triage "
             "(table-time, regenerated every run)", ""]

    lines += ["## Spread flags (max-min > 2)", ""]
    if agg.spread_flags:
        lines += ["| Packet | Task | Run | Model | Scores by judge |", "|---|---|---|---|---|"]
        for f in agg.spread_flags:
            sbj = ", ".join(f"{judge_id}={score}"
                             for judge_id, score in sorted(f["scores_by_judge"].items()))
            packet_short = f['packet_id'][:12]
            lines.append(f"| {packet_short} | {f['task_id']} | {f['run_n']} | {f['model_id']} | {sbj} |")
    else:
        lines.append("(none)")

    lines += ["", "## Drift flags (|median - ref| > tolerance)", ""]
    if agg.drift_flags:
        lines += ["| Packet | Task | Run | CAL | Median | Ref | Delta |",
                   "|---|---|---|---|---|---|---|"]
        for f in agg.drift_flags:
            lines.append(f"| {f['packet_id']} | {f['task_id']} | {f['run_n']} | {f['cal_type']} "
                          f"| {_fmt1(f['median'])} | {_fmt1(f['ref'])} | {f['delta']:+.1f} |")
    else:
        lines.append("(none)")

    return "\n".join(lines) + "\n"


def _rubric_inputs(root: Path):
    """(judge_prompt_bytes, [anchor paths]) or None when the grading tree is absent."""
    judge_prompt_path = root / "grading" / "judge_prompt.md"
    anchors_dir = root / "grading" / "anchors"
    if not judge_prompt_path.exists() or not anchors_dir.exists():
        return None
    return judge_prompt_path.read_bytes(), sorted(anchors_dir.glob("*.md"))


def _current_rubric_sha(root: Path) -> dict:
    """{unit: rubric_sha} for every unit with a grading/anchors/<unit>.md file CURRENTLY
    checked out (TESTPLAN 6.2). Mirrors packets.py's own formula. One STRING per unit --
    this is the value stamped into a packet when it is built, so it must stay a scalar.
    Aggregation should use `_acceptable_rubric_shas`, which tolerates line endings.
    """
    got = _rubric_inputs(root)
    if got is None:
        return {}
    prompt, anchors = got
    return {p.stem: hashlib.sha256(p.read_bytes() + prompt).hexdigest() for p in anchors}


def _acceptable_rubric_shas(root: Path) -> dict:
    """{unit: {sha, ...}} -- every spelling of the CURRENT rubric that aggregation should
    treat as a match.

    This hash is taken over FILE BYTES, and the anchors are ordinary text: git hands them
    to Windows as CRLF and to Linux as LF, so the same rubric hashes differently per
    platform. The packets were built on Windows, so on a Linux checkout every packet
    compared as "superseded by a rubric change", every judgment was dropped, and the
    flagship B1 scorecard silently regenerated EMPTY - all dashes, agreement 0.0%. It
    broke CI's byte-clean gate, but worse, anyone cloning on Linux would have produced a
    blank scorecard with no indication anything was wrong.

    Both spellings are accepted, each derived from normalised text so either is computable
    on any platform. That matches the existing packets without rewriting their recorded
    provenance.
    """
    got = _rubric_inputs(root)
    if got is None:
        return {}
    prompt, anchors = got
    lf_p = prompt.replace(b"\r\n", b"\n")
    crlf_p = lf_p.replace(b"\n", b"\r\n")
    out = {}
    for p in anchors:
        lf_a = p.read_bytes().replace(b"\r\n", b"\n")
        crlf_a = lf_a.replace(b"\n", b"\r\n")
        out[p.stem] = {hashlib.sha256(lf_a + lf_p).hexdigest(),
                       hashlib.sha256(crlf_a + crlf_p).hexdigest()}
    return out


def run_tables(root: str | Path = ".") -> int:
    root = Path(root).resolve()
    store = Store(root / "results")
    # CURRENT rows only (latest suite version per cell; hardware-superseded cells
    # withdrawn until replaced) — otherwise serving.md would keep quoting timings from
    # retired measurements. Store.iter_rows also reads the shakedown shard; rowselect
    # excludes it by version-string.
    from llmtest.rowselect import effective_suite_rows, load_superseded
    rows = effective_suite_rows(list(store.iter_rows()), load_superseded(root))
    out = root / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "serving.md").write_text(render_serving_table(rows),
                                    encoding="utf-8", newline="\n")

    from llmtest.registry import load_config
    cfg = load_config(root)

    judgments = list(store.iter_judgments())
    maps = load_maps(root / "results" / "packets")
    kin_map = cfg.judges.get("kin_map", {})
    judge_ids = sorted(cfg.judges.get("judges", {}))
    refscores_path = root / "grading" / "calibration" / "refscores.yaml"
    refscores = load_refscores(refscores_path) if refscores_path.exists() else None

    # Compute roster_filter using same rule as runner.resolve_cohort_models():
    # exclude models with role=quant-arm from scorecard roster expansion
    roster_filter = set(resolve_cohort_models(cfg))

    agg = aggregate(rows, judgments, maps, kin_map=kin_map, refscores=refscores,
                     judge_ids=judge_ids, current_rubric_sha=_acceptable_rubric_shas(root),
                     roster_filter=roster_filter)

    units = cfg.suite.get("b1", {}).get("units_tier1", [])
    (out / "scorecard.md").write_text(render_scorecard(agg, units),
                                       encoding="utf-8", newline="\n")
    (root / "results" / "FLAGS.md").write_text(render_flags(agg),
                                                encoding="utf-8", newline="\n")

    print(f"tables: wrote serving.md from {len(rows)} rows; "
          f"scorecard.md/FLAGS.md from {len(judgments)} judgments across {len(maps)} packets")
    return 0
