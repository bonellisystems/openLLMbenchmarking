"""llmtest status — done counts from resume keys; --judging is the judging-pipeline matrix (TESTPLAN 6.1/7.5)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from llmtest.store import Store


def run_status(root: str | Path = ".", *, judging: bool = False) -> int:
    root = Path(root).resolve()
    store = Store(root / "results")
    if judging:
        return _run_status_judging(root, store)
    counts = Counter()
    for r in store.iter_rows():
        counts[(r["battery"], r["model_id"], r["status"])] += 1
    if not counts:
        print("status: no rows yet")
        return 0
    for (battery, model, status), n in sorted(counts.items()):
        print(f"B{battery} {model:24s} {status:8s} {n}")
    return 0


def _run_status_judging(root: Path, store: Store) -> int:
    from llmtest.registry import load_config
    from llmtest.judging.runner import (JUDGED_BATTERIES, resolve_cohort_models,
                                         run_pending, summarize_judging)

    cfg = load_config(root)
    rows = list(store.iter_rows())
    cohort_models = resolve_cohort_models(cfg)
    judges_cfg = cfg.judges["judges"]
    judge_ids = sorted(judges_cfg)

    # packets_only=True: same cohort-completeness computation the runner
    # does, without invoking any judge -- a status view must never burn quota.
    result = run_pending(
        rows=rows, root=root, store=store,
        rubric_dir=root / "grading" / "anchors",
        calibration_dir=root / "grading" / "calibration",
        out_artifacts=root / "artifacts" / "packets",
        out_maps=root / "results" / "packets",
        judge_prompt_path=root / "grading" / "judge_prompt.md",
        judges_cfg=judges_cfg, cohort_models=cohort_models,
        packets_only=True,
    )

    for battery in sorted(JUDGED_BATTERIES):
        print(f"B{battery} judging: {len(result.packets)} cohorts complete (packets built), "
              f"{len(result.skipped)} incomplete/skipped")
    for s in result.skipped:
        print(f"  MISSING {s['task_id']} run={s['run_n']}: {s['reason']}")

    counts = summarize_judging(store, result.packets, judge_ids)
    print(f"  packet x judge: {counts['done']} done, {counts['pending']} pending, "
          f"{counts['error']} error")
    return 0
