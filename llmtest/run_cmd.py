"""llmtest run — plan/diff/execute loop with free resume (TESTPLAN 7.2/7.5)."""
import json
from pathlib import Path

from llmtest.registry import load_config
from llmtest.store import Store


def _results_dir(root: Path) -> Path:
    return root / "results"


def _get_battery(battery_id: int):
    from llmtest import batteries
    return batteries.get(battery_id)


def run_run(args) -> int:
    root = Path(".").resolve()
    cfg = load_config(root)
    store = Store(_results_dir(root))
    battery = _get_battery(args.battery)
    items = battery.plan(cfg, store, model_filter=args.model)
    if args.task_id:
        items = [i for i in items if i.task_id == args.task_id]
    if args.condition:
        items = [i for i in items if i.condition == args.condition]
    done = store.existing_row_ids()
    pending = [i for i in items if args.force or i.row_id not in done]
    print(f"run: {len(items)} planned, {len(pending)} pending")
    ctx = RunContext(cfg=cfg, store=store, root=root,
                     keep_server=args.keep_server, debug=args.debug)
    failures = 0
    try:
        for item in pending:
            try:
                for row in battery.execute(item, ctx):
                    appended = store.append(row)
                    if not appended and args.force:
                        failures += 1
                        print(f"EXEC-ERROR {item.task_id} {item.condition}: "
                              "--force re-ran the item but the row key already exists — "
                              "new measurement DISCARDED (run_n bump/supersede design "
                              "pending, see docs/backlog-p3.md)")
                    if args.debug:
                        dbg = root / "artifacts" / "debug"
                        dbg.mkdir(parents=True, exist_ok=True)
                        (dbg / f"{row['row_id']}.json").write_text(
                            json.dumps(row, indent=2), encoding="utf-8")
            except Exception as e:                    # row-level containment
                failures += 1
                print(f"EXEC-ERROR {item.task_id} {item.condition}: {e}")
    finally:
        if not args.keep_server and ctx.server is not None:
            ctx.server.teardown()
    print(f"run: done, {failures} failures")
    return 1 if failures else 0


class RunContext:
    """Handed to Battery.execute(). Lazily builds ServerManager on first use."""
    def __init__(self, *, cfg, store, root, keep_server, debug):
        self.cfg = cfg
        self.store = store
        self.root = root
        self.keep_server = keep_server
        self.debug = debug
        self._server = None

    @property
    def server(self):
        return self._server

    def server_manager(self):
        if self._server is None:
            from llmtest.server import ServerManager
            self._server = ServerManager(self.cfg, self.store)
        return self._server
