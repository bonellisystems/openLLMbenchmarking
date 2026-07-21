"""task-b8local -- the coordinator's entry point for a REAL local B8 run:
plan()+execute() B8Harness WorkItems against a model served MANUALLY
(outside `llmtest.server.ServerManager` -- the standard Windows
ServerManager launch path needs adaptation this box doesn't have yet), by
injecting the already-live endpoint straight into `ctx.b8_endpoint` (the
seam `llmtest/batteries/b8_harness.py::execute()` added for exactly this:
`getattr(ctx, "b8_endpoint", None) or ctx.server_manager().request_endpoint(
...)`). This script never calls `ctx.server_manager()` at all -- it doesn't
need to, and `RunContext.server_manager()` only builds a `ServerManager` on
first call, so simply never calling it means one is never constructed.

Prerequisite: `llama-server` already running and healthy at the URL you
pass via `--endpoint-url`, e.g.:

    llama-server -m <gguf> -ngl 99 -c 40960 --jinja -fa on \\
        --spec-type ngram-mod --spec-ngram-mod-n-match 32 --cache-ram 0 \\
        --host 127.0.0.1 --port 8080

Usage:
    python scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080
    python scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080/v1 \\
        --model gpt-oss-20b --task py-bugfix-01 --limit 3
    python scripts/run_b8_local.py --help          # dry-run-safe: parses
                                                     # and exits, launches
                                                     # nothing (see main())

Every `battery.execute()` call in the loop below drives the REAL
`OpenCodeAdapter` (a live `opencode` subprocess) and the REAL `run_oracle`
(a live Docker container, `python:3.11-slim` for the Python task manifests
via suite.yaml's `b8.sandbox.oracle_image`) -- neither `ctx.b8_adapters` nor
`ctx.b8_run_oracle` is set here, so `b8_harness.py`'s own default-resolution
(`_resolve_adapter`/`_resolve_run_oracle`) picks the real implementations,
exactly as it does under `llmtest run --battery 8` (llmtest/run_cmd.py).
Resume/dedup is entirely `B8Harness.plan()`'s job (see b8_harness.py's
module docstring RESUME note) -- this script does NOT additionally filter
by `store.existing_row_ids()` the way `run_cmd.py` does for other
batteries; that check is a structural no-op for B8 (a WorkItem's row_id
never matches its own eventual row's row_id -- attempt_id/exec_sha are
stamped only inside execute()), so omitting it here changes nothing.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.batteries.b8_harness import B8Harness   # noqa: E402
from llmtest.registry import load_config              # noqa: E402
from llmtest.run_cmd import RunContext                 # noqa: E402
from llmtest.store import Store                         # noqa: E402


@dataclass
class ManualEndpoint:
    """The object `ctx.b8_endpoint` must be, per the seam's contract
    (`llmtest/batteries/b8_harness.py::execute()` +
    `llmtest/harness/opencode.py::OpenCodeAdapter.setup()`):

      - `base_url` -- read by `OpenCodeAdapter.setup()` via
        `getattr(endpoint, "base_url", None) or str(endpoint)`, then used as
        `f"{base_url}/v1"` for the `local` provider's `baseURL`
        (`_write_opencode_config`). MUST be the bare origin (e.g.
        "http://127.0.0.1:8080"), NOT already `/v1`-suffixed -- see
        `_normalize_base_url` below for why (double `/v1` breaks every
        call). Matches `llmtest.server.EndpointHandle.base_url`'s own shape
        exactly (`f"http://127.0.0.1:{port}"`, no `/v1`).
      - `session_id` -- read by `execute()` via
        `getattr(endpoint, "session_id", "unknown")`, stored on the row for
        provenance only (schema.ResultRow.session_id). Any stable string is
        fine; this is not looked up against `results/sessions.jsonl` the
        way a real `ServerManager`-launched session_id would be, since no
        `SessionRow` exists for a manually-served endpoint.
      - `normalized_config` -- read by `execute()` via
        `getattr(endpoint, "normalized_config", {}) or {}`, folded into the
        row's `det_checks`/provenance-adjacent `server_profile.flags`
        (`_execution_provenance_sha`'s inputs). Provenance-only, like
        `session_id` -- does not gate or configure anything downstream.

    `pid`/`_mgr` (present on the real `EndpointHandle`) are deliberately
    absent: nothing in the B8 execute() path reads them (only
    `base_url`/`session_id`/`normalized_config` are ever touched), and this
    script never owns the server process's lifecycle (the coordinator
    launched `llama-server` manually and tears it down manually too) -- so
    there is nothing for a `pid`/`_mgr`-shaped teardown hook to do here.
    """
    base_url: str
    session_id: str
    normalized_config: dict = field(default_factory=dict)


def _normalize_base_url(url: str) -> str:
    """Strip a trailing '/v1' (with or without a trailing slash) so the
    injected `base_url` is always the BARE origin. `OpenCodeAdapter.
    _write_opencode_config` appends "/v1" itself (`f"{self.endpoint}/v1"`)
    -- passing an already-'/v1'-suffixed URL straight through would double
    it into ".../v1/v1" and silently break every OpenCode call (no test
    catches this: the injection tests use a mock adapter that never reads
    `base_url` at all). Accepts either shape a coordinator might paste --
    llama-server's raw "http://127.0.0.1:8080" or the OpenAI-style
    ".../v1" -- and returns the same, correct bare origin either way."""
    u = url.rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")]
    return u


def _unit_task_id(task_id: str) -> str:
    """'b8.py-bugfix-01' -> 'py-bugfix-01' (bare manifest id, what --task
    matches against, alongside the full 'b8.'-prefixed form)."""
    return task_id.split(".", 1)[1] if "." in task_id else task_id


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint-url", required=True,
                   help="Base URL of the already-running, manually-served "
                        "endpoint (e.g. http://127.0.0.1:8080 or "
                        "http://127.0.0.1:8080/v1 -- a trailing /v1 is "
                        "stripped automatically).")
    p.add_argument("--model", default=None,
                   help="Restrict to one model id (must be in suite.yaml's "
                        "b8.models). Default: every model in b8.models.")
    p.add_argument("--task", default=None,
                   help="Restrict to one B8 task id, bare ('py-bugfix-01') "
                        "or full ('b8.py-bugfix-01'). Default: every "
                        "planned task.")
    p.add_argument("--limit", type=int, default=None,
                   help="Execute at most N planned (and not-yet-recorded) "
                        "WorkItems this invocation. Default: all of them.")
    p.add_argument("--force", action="store_true",
                   help="Passed through to B8Harness.plan() -- emit exactly "
                        "one more replicate beyond whatever's already "
                        "recorded per (model, harness, task) cell, instead "
                        "of filling up to b8.replicates.")
    p.add_argument("--session-id", default=None,
                   help="Override the session_id stamped on emitted rows "
                        "(provenance only). Default: a fresh 'manual-<uuid8>'.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(ROOT)
    b8cfg = cfg.suite["b8"]
    base_url = _normalize_base_url(args.endpoint_url)
    print(f"run_b8_local: endpoint base_url resolved to {base_url!r} "
          f"(from --endpoint-url {args.endpoint_url!r})")

    session_id = args.session_id or f"manual-{uuid.uuid4().hex[:8]}"
    endpoint = ManualEndpoint(
        base_url=base_url, session_id=session_id,
        normalized_config={"runtime": "manual", "ctx": b8cfg.get("ctx"),
                           "kv_dtype": b8cfg.get("kv"), "endpoint_url": base_url})

    store = Store(ROOT / "results")
    ctx = RunContext(cfg=cfg, store=store, root=ROOT, keep_server=True, debug=False)
    ctx.b8_endpoint = endpoint   # the seam -- ctx.server_manager() is never called

    battery = B8Harness()
    items = battery.plan(cfg, store, model_filter=args.model, force=args.force)
    if args.task:
        items = [i for i in items
                 if i.task_id == args.task or _unit_task_id(i.task_id) == args.task]
    if not items:
        print("run_b8_local: 0 planned WorkItems for the given --model/--task "
              "filters (already fully recorded? check --force, or widen the filter)")
        return 0
    if args.limit is not None:
        items = items[: args.limit]

    print(f"run_b8_local: {len(items)} WorkItem(s) to execute "
          f"(model_filter={args.model!r}, task_filter={args.task!r}, "
          f"limit={args.limit}, force={args.force})")

    failures = 0
    for i, item in enumerate(items, 1):
        t0 = time.time()
        try:
            rows = battery.execute(item, ctx)
        except Exception as e:                        # row-level containment
            failures += 1
            print(f"  [{i}/{len(items)}] EXEC-ERROR model={item.model_id} "
                  f"task={item.task_id} run_n={item.run_n}: {e!r}")
            continue
        for row in rows:
            appended = store.append(row)
            m = row["metrics"]
            dt = time.time() - t0
            print(f"  [{i}/{len(items)}] model={item.model_id} task={item.task_id} "
                  f"run_n={item.run_n} -> completion={m['completion']} "
                  f"terminal_status={m['terminal_status']} steps={m['steps']} "
                  f"tokens_prompt={m['tokens_prompt']} tokens_completion={m['tokens_completion']} "
                  f"appended={appended} row_id={row['row_id'][:12]} ({dt:.1f}s)")

    print(f"run_b8_local: done, {failures} EXEC-ERROR(s) out of {len(items)} planned")
    return 1 if failures else 0


if __name__ == "__main__":
    # Guard: nothing above this point launches a process, hits the network,
    # or touches Docker -- `python scripts/run_b8_local.py --help` (or a
    # plain `import` of this module, e.g. from a test) is dry-run-safe.
    # Only reaching main() with real args (real subprocess/Docker calls
    # live inside battery.execute() -> OpenCodeAdapter/run_oracle) does
    # anything live.
    raise SystemExit(main())
