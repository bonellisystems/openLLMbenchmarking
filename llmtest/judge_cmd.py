"""llmtest judge — CLI entrypoint for the judge runner (TESTPLAN 6.1/7.5)."""
from __future__ import annotations

from pathlib import Path

from llmtest.registry import load_config
from llmtest.store import Store
from llmtest.judging.runner import resolve_cohort_models, run_pending


def run_judge(args, root: str | Path = ".") -> int:
    if not getattr(args, "pending", False):
        print("judge: pass --pending (no other modes implemented yet)")
        return 2

    root = Path(root).resolve()
    cfg = load_config(root)
    store = Store(root / "results")
    judges_cfg = cfg.judges["judges"]

    judge_filter = getattr(args, "judge", None)
    if judge_filter is not None and judge_filter not in judges_cfg:
        print(f"judge: unknown judge id {judge_filter!r}; configured: {sorted(judges_cfg)}")
        return 2

    rows = list(store.iter_rows())
    cohort_models = resolve_cohort_models(cfg)
    b2_quorum = cfg.suite.get("b2", {}).get("quorum")

    result = run_pending(
        rows=rows, root=root, store=store,
        rubric_dir=root / "grading" / "anchors",
        calibration_dir=root / "grading" / "calibration",
        out_artifacts=root / "artifacts" / "packets",
        out_maps=root / "results" / "packets",
        judge_prompt_path=root / "grading" / "judge_prompt.md",
        judges_cfg=judges_cfg, cohort_models=cohort_models,
        b2_quorum=b2_quorum,
        judge_filter=judge_filter,
        packets_only=getattr(args, "packets_only", False),
        fake=getattr(args, "fake", False),
        retry_errors=getattr(args, "retry_errors", False),
    )

    print(f"judge: {len(result.packets)} packets built, {len(result.skipped)} cohorts skipped, "
          f"{result.judgments_written} judgments written, {result.errors_written} error rows")
    for s in result.skipped:
        print(f"  SKIP {s['task_id']} run={s['run_n']}: {s['reason']}")
    return 0
