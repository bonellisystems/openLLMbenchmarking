"""P3 Task 10 vertical-slice driver -- proves the B1 + judging pipeline end-to-end on a
single task/model before the Task 11 authoring fan-out, and seeds the Task 12 quota
dry-run driver.

B1.preflight() requires ALL 15 Tier-1 unit dirs to exist and parse >=1 task each; only
`cybersecurity` is authored as of Task 10, so `llmtest run --battery 1 ...` would abort
at the preflight gate. This script bypasses preflight entirely and drives
B1Business.plan()/.execute() directly (the same call shape run_cmd.py uses internally),
filtered to one task/model/run_n -- a "real" B1 row via the real ServerManager/GPU, with
none of the other 14 units required to exist yet.

Cohort completeness normally requires every non-quant-arm roster model (11 today) to have
an ok row before a judging packet is built. For a one-model slice, `--cohort` overrides
`cohort_models` passed to build_cohort_packets() directly (bypassing
config/suite.yaml's b1.cohort_models, which stays unset until the real Task 12 dry-run)
so a single-model cohort is "complete" and a packet gets built.

Usage:
    python scripts/slice_b1.py run   --model gpt-oss-20b --task cybersecurity-01 --run-n 1
    python scripts/slice_b1.py judge --cohort gpt-oss-20b --fake
    python -m llmtest tables   # separate, real CLI -- reads what this script wrote
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.batteries.b1_business import B1Business          # noqa: E402
from llmtest.registry import load_config                      # noqa: E402
from llmtest.run_cmd import RunContext                         # noqa: E402
from llmtest.store import Store                                # noqa: E402


def do_run(args) -> int:
    cfg = load_config(ROOT)
    store = Store(ROOT / "results")
    battery = B1Business()

    # plan() with model_filter narrows to one model; force=False (fresh run_n=1..n_runs).
    items = battery.plan(cfg, store, model_filter=args.model, force=False)
    task_id = f"b1.{args.task}"
    items = [i for i in items if i.task_id == task_id and i.run_n == args.run_n]
    if not items:
        print(f"slice_b1 run: no planned item for model={args.model} task={task_id} "
              f"run_n={args.run_n} (check the model exists in registry.yaml and the "
              f"unit dir/task file exist)")
        return 1

    done = store.existing_row_ids()
    pending = [i for i in items if args.force or i.row_id not in done]
    if not pending:
        print(f"slice_b1 run: row already present for {items[0].row_id[:12]} "
              f"(pass --force to re-measure)")
        return 0

    ctx = RunContext(cfg=cfg, store=store, root=ROOT, keep_server=False, debug=False)
    try:
        for item in pending:
            for row in battery.execute(item, ctx):
                appended = store.append(row)
                print(f"slice_b1 run: row_id={row['row_id']} status={row['status']} "
                      f"needs_judging={row['needs_judging']} appended={appended}")
    finally:
        # VRAM drain: always teardown, mirroring run_cmd.run_run()'s finally block.
        if ctx.server is not None:
            ctx.server.teardown()
            print("slice_b1 run: server torn down (VRAM released)")
    return 0


def do_judge(args) -> int:
    from llmtest.judging.runner import run_pending

    cfg = load_config(ROOT)
    store = Store(ROOT / "results")
    rows = list(store.iter_rows())
    cohort_models = [m.strip() for m in args.cohort.split(",") if m.strip()]

    result = run_pending(
        rows=rows, root=ROOT, store=store,
        rubric_dir=ROOT / "grading" / "anchors",
        calibration_dir=ROOT / "grading" / "calibration",
        out_artifacts=ROOT / "artifacts" / "packets",
        out_maps=ROOT / "results" / "packets",
        judge_prompt_path=ROOT / "grading" / "judge_prompt.md",
        judges_cfg=cfg.judges["judges"], cohort_models=cohort_models,
        fake=args.fake,
    )
    print(f"slice_b1 judge: {len(result.packets)} packets built, "
          f"{len(result.skipped)} cohorts skipped, "
          f"{result.judgments_written} judgments written, "
          f"{result.errors_written} error rows")
    for p in result.packets:
        print(f"  packet_id={p.packet_id} task_id={p.task_id} run_n={p.run_n} "
              f"unit={p.unit} cal_fallback={p.cal_fallback}")
    for s in result.skipped:
        print(f"  SKIP {s['task_id']} run={s['run_n']}: {s['reason']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)

    r = sub.add_parser("run", help="plan()+execute() one B1 task/model/run_n directly "
                                    "(bypasses preflight)")
    r.add_argument("--model", default="gpt-oss-20b")
    r.add_argument("--task", default="cybersecurity-01", help="task id without the 'b1.' prefix")
    r.add_argument("--run-n", type=int, default=1)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=do_run)

    j = sub.add_parser("judge", help="build packets + judge pending, with an explicit "
                                      "cohort override")
    j.add_argument("--cohort", required=True,
                    help="comma-separated model_ids treated as the FULL cohort "
                         "(overrides suite.yaml's b1.cohort_models)")
    j.add_argument("--fake", action="store_true", help="use FakeJudgeAdapter, no live CLIs")
    j.set_defaults(func=do_judge)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
