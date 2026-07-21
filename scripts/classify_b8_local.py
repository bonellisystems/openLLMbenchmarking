"""B8 first-failure classify pass (task-b8classify) -- the piece that
actually RUNS `llmtest.harness.failure_class.classify_first_failure` over
real B8 rows, which nothing did before this script existed (the classifier
itself, and the trace-persistence it needs, both predate this file --
`llmtest.harness.failure_class` and `llmtest.batteries.b8_harness.execute()`
persisting `Trace` to `artifacts/b8_traces/<row_id>.json` -- but no CALLER
ever invoked the classifier against real rows, so `scripts/p8_report.py`'s
first-failure-class distribution always showed "(unclassified)").

WHAT THIS DOES, end to end, per suite_version:
  1. Load every battery=8 row with `metrics.completion is False` from the
     real results store (`results/rows-<suite_version>.jsonl`).
  2. Skip rows already present in the sibling classification store (unless
     `--redo`) -- resume-safe, mirrors every other battery's dedup idiom.
  3. For each pending row: reload its persisted `Trace` via
     `response_meta.trace_ref` (written by `b8_harness.execute()`), reload
     its `B8Task` manifest via `llmtest.harness.tasks.load_b8_tasks`
     (matched on `task_id`, stripping the `"b8."` prefix), then run
     `classify_first_failure(trace, task, completed=False,
     classifiers=<panel>, oracle_detail=row["det_checks"]["oracle"]
     ["detail"])` -- Wave 1b: the row's own oracle rejection reason is
     threaded through as TRUSTED evidence so the panel can tell (b) from
     (c) instead of guessing from the bare completion boolean alone.
  4. Append `{row_id, task_id, model_id, suite_version, label, source, ts}`
     to `results/b8_classifications-<suite_version>.jsonl` -- a SIBLING,
     append-only store, never a mutation of `rows-<suite_version>.jsonl`
     itself. This is a deliberate, not incidental, choice: a row's row_id
     is derived from a fixed set of fields that do NOT include a
     classification verdict, and `Store.append()` (llmtest/store.py) is a
     structural no-op when handed an already-seen row_id -- it returns
     `False` without writing anything -- so there is no way to "update" a
     stored row in place through the existing Store API. `scripts/
     p8_report.py`'s `build_b8_section` reads this sibling store
     (`_load_b8_classifications`) and fills in a row's
     `metrics['first_failure_class']` from it wherever the row itself
     doesn't already carry that key, so re-running this script followed by
     `scripts/p8_report.py` shows the real distribution.

PANEL: real mode builds one classifier per `config/judges.yaml` judge entry
via `llmtest.judging.adapters.make_adapter` (the SAME frozen claude/codex/
gemini pins the numeric B1 judge panel uses) -- classify_first_failure's
`classifiers` argument accepts these directly (`BaseAdapter.invoke(...)` IS
the `.invoke()` shape the classifier interface expects). `--fake` swaps in
a tiny offline stub (no subprocess at all) for a dry run of the whole
pipeline (trace load -> task reload -> classify_first_failure -> sibling-
store append -> p8_report pickup) with zero CLI cost/quota.

`packet_dir` is pinned to `<root>/artifacts/packets` in real mode -- the
gemini/agy judge entry in config/judges.yaml hardcodes
`--add-dir ...\\artifacts\\packets` (agy's headless file-read grant is
scoped to exactly that directory), so a classifier packet written anywhere
else would be unreadable by a real agy call. See
`llmtest.harness.failure_class.panel_classify`'s docstring.

Usage:
    python scripts/classify_b8_local.py --fake                    # dry run, no live CLI
    python scripts/classify_b8_local.py                           # real panel via judges.yaml
    python scripts/classify_b8_local.py --suite suite-v2.1.0 --limit 20
    python scripts/classify_b8_local.py --redo                    # re-classify already-recorded rows
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llmtest.harness.failure_class import classify_first_failure   # noqa: E402
from llmtest.harness.tasks import load_b8_tasks                    # noqa: E402
from llmtest.harness.trace import Trace                            # noqa: E402
from llmtest.judging.adapters import make_adapter                  # noqa: E402
from llmtest.registry import load_config                           # noqa: E402
from llmtest.store import Store                                    # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FakeClassifier:
    """--fake stub panel seat: deterministic, offline, no subprocess at all
    -- mirrors `llmtest.judging.adapters.FakeJudgeAdapter`'s role in
    `scripts/p8_judge.py`. A single fixed-label seat is enough to exercise
    the FULL pipeline (trace load -> task reload -> classify_first_failure
    -> panel majority-of-1 -> sibling-store append -> p8_report pickup)
    without a live classifier CLI."""

    def __init__(self, label: str = "c"):
        self.label = label

    def classify(self, blinded_text: str) -> str:
        return self.label


def task_suffix(task_id: str) -> str:
    """'b8.py-bugfix-01' -> 'py-bugfix-01' (B8Task.id has no 'b8.' prefix;
    a stored row's task_id always does -- see b8_harness.py's module
    docstring, `_base_condition`/`item.task_id = f"b8.{task.id}"`)."""
    return task_id.split(".", 1)[1] if "." in task_id else task_id


def load_task_map(root: Path) -> dict[str, object]:
    """{B8Task.id: B8Task} for every manifest under suite/b8_harness/."""
    return {task.id: task for task in load_b8_tasks(root)}


def load_trace_for_row(root: Path, row: dict) -> Trace | None:
    """Reload the full `Trace` a row's `response_meta.trace_ref` points at
    (written by `b8_harness.execute()`). Mirrors `llmtest.judging.packets.
    _read_artifact_text`'s "relative to artifacts/, with a root-relative
    fallback" convention. Returns `None` (never raises) for a row with no
    `trace_ref` at all, or one whose file is missing -- the caller SKIPS
    such rows rather than crashing the whole pass; that's the caller's
    call, not this loader's."""
    trace_ref = (row.get("response_meta") or {}).get("trace_ref")
    if not trace_ref:
        return None
    for candidate in (root / "artifacts" / trace_ref, root / trace_ref):
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return Trace.from_dict(data)
    return None


def classifications_path(root: Path, suite_version: str) -> Path:
    return root / "results" / f"b8_classifications-{suite_version}.jsonl"


def already_classified_row_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row_id = json.loads(line).get("row_id")
            if row_id:
                out.add(row_id)
    return out


def append_classification(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def build_panel(cfg, *, fake: bool) -> list:
    """The real classifier panel (one adapter per `config/judges.yaml`
    judge, alphabetical for a deterministic invocation order) -- SAME
    frozen pins the numeric B1 judge panel uses, reused as-is via
    `make_adapter` (no separate classifier-specific config). `--fake`
    swaps in a single offline `FakeClassifier` seat instead."""
    if fake:
        return [FakeClassifier("c")]
    judges_cfg = cfg.judges["judges"]
    return [make_adapter(judge_id, judges_cfg[judge_id]) for judge_id in sorted(judges_cfg)]


def classify_pending_rows(root: Path, cfg, suite_version: str, *, fake: bool,
                           limit: int | None = None, redo: bool = False,
                           log=print) -> dict:
    """The actual pass: load failed rows -> skip already-classified (unless
    `redo`) -> reload trace+task per row -> classify -> append to the
    sibling store. Returns a small summary dict (counts) so both `main()`
    and tests can assert on the outcome without scraping stdout."""
    store = Store(root / "results")
    failed_rows = [
        r for r in store.iter_rows()
        if r.get("battery") == 8 and r.get("suite_version") == suite_version
        and r.get("metrics", {}).get("completion") is False
    ]

    out_path = classifications_path(root, suite_version)
    already = set() if redo else already_classified_row_ids(out_path)
    pending = [r for r in failed_rows if r["row_id"] not in already]
    if limit is not None:
        pending = pending[:limit]

    log(f"[classify_b8_local] {len(failed_rows)} failed B8 row(s) for suite "
        f"{suite_version!r}, {len(pending)} pending (fake={fake}, redo={redo})")

    summary = {"failed_total": len(failed_rows), "pending": len(pending),
               "classified": 0, "skipped_no_trace": 0, "skipped_no_task": 0}
    if not pending:
        return summary

    tasks_by_id = load_task_map(root)
    classifiers = build_panel(cfg, fake=fake)
    packet_dir = None if fake else (root / "artifacts" / "packets")

    for row in pending:
        trace = load_trace_for_row(root, row)
        if trace is None:
            print(f"SKIP row_id={row['row_id']} task_id={row.get('task_id')}: "
                  f"no persisted Trace (missing/unresolvable response_meta.trace_ref)",
                  file=sys.stderr)
            summary["skipped_no_trace"] += 1
            continue

        task = tasks_by_id.get(task_suffix(row.get("task_id", "")))
        if task is None:
            print(f"SKIP row_id={row['row_id']} task_id={row.get('task_id')}: "
                  f"no matching B8Task manifest (load_b8_tasks)", file=sys.stderr)
            summary["skipped_no_task"] += 1
            continue

        # Wave 1b: thread the row's own oracle rejection reason through as
        # TRUSTED evidence (`det_checks.oracle.detail` -- the same string
        # `b8_harness.execute()` computed from `run_oracle` and stored on
        # this row) -- see `classify_first_failure`'s docstring and
        # `llmtest.harness.failure_class`'s "ORACLE DETAIL" module-doc
        # section. Guarded against a missing/malformed `det_checks`/
        # `oracle` key (older rows, or a hand-seeded test fixture) rather
        # than assuming the nested shape is always present.
        oracle_detail = ((row.get("det_checks") or {}).get("oracle") or {}).get("detail")

        label, source = classify_first_failure(
            trace, task, completed=False, classifiers=classifiers, packet_dir=packet_dir,
            oracle_detail=oracle_detail)

        record = {
            "row_id": row["row_id"], "task_id": row.get("task_id"),
            "model_id": row.get("model_id"), "suite_version": suite_version,
            "label": label, "source": source, "ts": _now(),
        }
        append_classification(out_path, record)
        summary["classified"] += 1
        log(f"row_id={row['row_id']} task_id={row.get('task_id')} model={row.get('model_id')} "
            f"-> {label} ({source})")

    log(f"[classify_b8_local] wrote {summary['classified']} classification(s) -> {out_path} "
        f"(skipped: {summary['skipped_no_trace']} no-trace, "
        f"{summary['skipped_no_task']} no-task)")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="repo root (default: parent of scripts/)")
    ap.add_argument("--suite", default=None,
                     help="suite_version to classify (default: config/suite.yaml's)")
    ap.add_argument("--fake", action="store_true",
                     help="stub classifier panel, no live CLI, no subprocess at all")
    ap.add_argument("--limit", type=int, default=None,
                     help="debug: cap the number of rows classified this run")
    ap.add_argument("--redo", action="store_true",
                     help="re-classify rows already present in the sibling classification store")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else _REPO_ROOT
    cfg = load_config(root)
    suite_version = args.suite or cfg.suite["suite_version"]

    classify_pending_rows(root, cfg, suite_version, fake=args.fake,
                          limit=args.limit, redo=args.redo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
