"""P3 Task 13 lean shakedown: generate gpt-oss-20b on ONE task per unit across
all 15 Tier-1 units, confirming every unit produces non-empty output at the new
ctx=32k / raised-budget config. Writes to the real shakedown store; RESUMABLE via
row_id dedupe (re-invoke after a timeout to continue). No judging.

    python scripts/smoke_b1.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.batteries.b1_business import B1Business          # noqa: E402
from llmtest.registry import load_config                      # noqa: E402
from llmtest.run_cmd import RunContext                         # noqa: E402
from llmtest.store import Store                                # noqa: E402


def _unit_of(task_id: str) -> str:
    # "b1.cybersecurity-01" -> "cybersecurity"
    return task_id.split(".", 1)[1].rsplit("-", 1)[0]


def main() -> int:
    cfg = load_config(ROOT)
    store = Store(ROOT / "results")
    battery = B1Business()

    items = battery.plan(cfg, store, model_filter="gpt-oss-20b", force=False)
    # one task per unit (lowest task_id), run_n == 1
    first_per_unit: dict[str, object] = {}
    for it in items:
        if it.run_n != 1:
            continue
        u = _unit_of(it.task_id)
        if u not in first_per_unit or it.task_id < first_per_unit[u].task_id:
            first_per_unit[u] = it
    picks = [first_per_unit[u] for u in sorted(first_per_unit)]
    print(f"smoke_b1: {len(picks)} units -> {[p.task_id for p in picks]}")

    done = store.existing_row_ids()
    pending = [p for p in picks if p.row_id not in done]
    print(f"smoke_b1: {len(picks) - len(pending)} already done, {len(pending)} pending")
    if not pending:
        print("smoke_b1: 0 pending -- all units already smoked")
        return 0

    ctx = RunContext(cfg=cfg, store=store, root=ROOT, keep_server=False, debug=False)
    empties = []
    try:
        for i, item in enumerate(pending, 1):
            t0 = time.time()
            try:
                for row in battery.execute(item, ctx):
                    store.append(row)
                    chars = (row.get("metrics") or {}).get("chars", 0)
                    flag = "" if chars > 0 else "  <<< EMPTY"
                    if chars == 0:
                        empties.append(item.task_id)
                    print(f"  [{i}/{len(pending)}] {item.task_id:28s} "
                          f"status={row['status']} chars={chars}{flag} "
                          f"({time.time()-t0:.1f}s)")
            except Exception as e:
                empties.append(item.task_id)
                print(f"  [{i}/{len(pending)}] EXEC-ERROR {item.task_id}: {e}")
    finally:
        if ctx.server is not None:
            ctx.server.teardown()
            print("smoke_b1: server torn down (VRAM released)")

    if empties:
        print(f"smoke_b1: FAIL -- {len(empties)} empty/errored units: {empties}")
        return 1
    print("smoke_b1: PASS -- all units produced non-empty output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
