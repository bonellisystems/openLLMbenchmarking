"""Battery 2 -- tool calling (TESTPLAN 5.2). Row generation for tool-use scenarios.

8 axes, scored separately, never blended (TESTPLAN 5.2 canonical numbering):
  1 schema adherence · 2 correct tool selection · 3 parallel calls ·
  4 chained/dependent calls · 5 error recovery (JUDGED) · 6 abstention ·
  7 long-context calls · 8 faithfulness to tool results (JUDGED, shared w/ B3).

Axes 1-4, 6-7 are scored fully deterministically in det_checks (see
b2_fixtures.score_axes). Axes 5 and 8 get a best-effort deterministic
"fabrication trap" floor plus needs_judging=True -- their real score is a
judge's job. NOTE (scope of this build): B2 rows are NOT yet wired into the
judging pipeline -- llmtest/judging/runner.py's JUDGED_BATTERIES is still
{1} by design (its own comment: "kept as a set... so a future battery can
opt into judging without touching the filter's shape"), and packets.py's
build_cohort_packets() hardcodes B1's unit/anchor-file resolution
(grading/anchors/<unit>.md keyed off "b1.<unit>-NN" task_ids), which has no
B2 equivalent yet (B2 tasks have no "unit", they have `scenario`/`axes`).
Wiring B2 into that pipeline (new anchor files under grading/anchors/,
scenario-aware task_id parsing, JUDGED_BATTERIES = {1, 2}) is future work;
this battery only sets the row-level needs_judging flag correctly per
TESTPLAN 7.4's contract ("det checks inline; needs_judging flags").
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register
from llmtest.batteries.b2_fixtures import load_tasks, score_axes, validate_expect_block, validate_tool_schemas


@register
class B2ToolCalling(Battery):
    id = 2

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        """Generate WorkItems for B2 tool-calling tasks.

        Covers all registry models WITHOUT role=quant-arm (same roster filter
        as B1Business.plan()), all fixtures under suite/b2_toolcalling/, and
        run_n in 1..cfg.suite["b2"]["n_runs"]. One constant condition for the
        whole battery (mirrors B1) -- per-task context differences (e.g. the
        axis-7 long-context task) live in fixture content, not the condition
        string, since cfg.suite["b2"]["ctx"] is sized to fit every B2 task.
        """
        order = cfg.suite["condition_order"]
        sv = cfg.suite["suite_version"]
        n_runs = cfg.suite["b2"]["n_runs"]

        tasks = load_tasks(cfg.root)

        condition = schema.canonical_condition(
            {"runtime": "fork", "spec": "ngram32", "kv": "q8",
             "ctx": "40k", "cond": "B2"},
            order
        )

        items = []
        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            if m.get("role") == "quant-arm":
                continue
            if str(m.get("local_path", "")).startswith("TO-"):
                continue

            for task in tasks:
                task_id = f"b2.{task.id}"
                fixture_sha = task.fixture_sha

                if force:
                    existing = [r["run_n"] for r in store.iter_rows()
                               if r["task_id"] == task_id
                               and r["model_id"] == model_id
                               and r["condition"] == condition]
                    run_ns = [(max(existing) + 1) if existing else 1]
                else:
                    run_ns = range(1, n_runs + 1)

                for run_n in run_ns:
                    rid = schema.compute_row_id(
                        suite_version=sv, model_id=model_id,
                        quant_sha256=m["provenance"]["sha256"], battery=2,
                        task_id=task_id, fixture_sha=fixture_sha,
                        condition=condition, run_n=run_n)

                    items.append(WorkItem(
                        row_id=rid, model_id=model_id, battery=2,
                        task_id=task_id, condition=condition,
                        run_n=run_n,
                        payload={
                            "model": m,
                            "task_id": task.id,
                            "fixture_sha": fixture_sha,
                            "suite_version": sv,
                            "tools": task.tools,
                            "messages": task.messages,
                            "expect": task.expect,
                            "axes": task.axes,
                        }))

        return items

    def preflight(self, ctx) -> list[dict]:
        """TESTPLAN 5.2: "preflight(): all tool schemas parse."

        One selftest row per loaded task (status=ok/error), plus a single
        guard row if the fixture directory is missing or empty entirely.
        All rows tagged ["selftest"], cond=SELFTEST, run_n=1. A malformed
        fixture FILE raises ValueError loud from load_tasks() (mirrors B1),
        so it is never turned into a graceful error row -- it aborts the run
        before any row is written, same as B1Business.preflight().
        """
        rows = []
        order = ctx.cfg.suite["condition_order"]
        sv = ctx.cfg.suite["suite_version"]
        condition = schema.canonical_condition({"cond": "SELFTEST"}, order)
        model_info = ctx.cfg.registry["models"].get("gpt-oss-20b", {})

        tasks = load_tasks(ctx.root)

        if not tasks:
            fixture_sha = hashlib.sha256(b"b2_toolcalling-empty").hexdigest()
            rid = schema.compute_row_id(
                suite_version=sv, model_id="selftest", quant_sha256="0" * 64,
                battery=2, task_id="b2.selftest.no_fixtures",
                fixture_sha=fixture_sha, condition=condition, run_n=1)
            row = schema.ResultRow.new(
                suite_version=sv, model_id="selftest",
                hf_repo=model_info.get("hf_repo", "N/A"),
                quant_file=model_info.get("quant_file", "N/A"),
                quant_sha256="0" * 64, tier="selftest", battery=2,
                task_id="b2.selftest.no_fixtures", fixture_sha=fixture_sha,
                condition=condition, run_n=1,
                session_id="selftest", status="error",
                error_detail="no fixtures found under suite/b2_toolcalling/",
                tags=["selftest"])
            return [row.to_dict()]

        for task in tasks:
            errs = validate_tool_schemas(task.tools)
            errs += validate_expect_block(task)

            rid = schema.compute_row_id(
                suite_version=sv, model_id="selftest", quant_sha256="0" * 64,
                battery=2, task_id=f"b2.selftest.{task.id}",
                fixture_sha=task.fixture_sha, condition=condition, run_n=1)

            if not errs:
                row = schema.ResultRow.new(
                    suite_version=sv, model_id="selftest",
                    hf_repo=model_info.get("hf_repo", "N/A"),
                    quant_file=model_info.get("quant_file", "N/A"),
                    quant_sha256="0" * 64, tier="selftest", battery=2,
                    task_id=f"b2.selftest.{task.id}", fixture_sha=task.fixture_sha,
                    condition=condition, run_n=1,
                    session_id="selftest", status="ok",
                    tags=["selftest"])
            else:
                row = schema.ResultRow.new(
                    suite_version=sv, model_id="selftest",
                    hf_repo=model_info.get("hf_repo", "N/A"),
                    quant_file=model_info.get("quant_file", "N/A"),
                    quant_sha256="0" * 64, tier="selftest", battery=2,
                    task_id=f"b2.selftest.{task.id}", fixture_sha=task.fixture_sha,
                    condition=condition, run_n=1,
                    session_id="selftest", status="error",
                    error_detail="; ".join(errs),
                    tags=["selftest"])
            rows.append(row.to_dict())

        return rows

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        """Execute a B2 tool-calling task.

        - Request endpoint with ctx from suite config (b2.ctx), kv="q8_0"
        - Chat call with temperature omitted (runtime default), tools param set
        - Score all applicable axes deterministically (score_axes)
        - Save the full raw response JSON as the artifact under artifacts/b2/<row_id>.json
        - needs_judging=True iff the task's axes intersect {5, 8}
        """
        cfg = ctx.cfg
        model = item.payload["model"]
        fixture_sha = item.payload["fixture_sha"]
        suite_version = item.payload["suite_version"]
        tools = item.payload["tools"]
        messages = item.payload["messages"]
        expect = item.payload["expect"]
        axes = item.payload["axes"]

        endpoint = ctx.server_manager().request_endpoint(
            item.model_id, ctx=cfg.suite["b2"]["ctx"], kv="q8_0",
            timing_authoritative=False)

        max_tokens = cfg.suite["b2"].get("max_tokens", 2000)
        response = endpoint.chat(messages, max_tokens=max_tokens, temperature=None, tools=tools)

        # score_axes only needs the axis-relevant fields of a Task -- build a
        # lightweight shim rather than re-loading fixtures from disk here
        # (plan() already rode tools/expect/axes through the payload).
        from types import SimpleNamespace
        task_shim = SimpleNamespace(tools=tools, expect=expect, axes=axes)
        det_checks, needs_judging, metrics = score_axes(response, task_shim)

        artifacts_root = (ctx.root / "artifacts" / "b2") if hasattr(ctx, 'root') else (Path("artifacts") / "b2")
        artifacts_root.mkdir(parents=True, exist_ok=True)

        artifact_path = artifacts_root / f"{item.row_id}.json"
        artifact_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        row = schema.ResultRow.new(
            suite_version=suite_version, model_id=item.model_id,
            hf_repo=model.get("hf_repo", ""),
            quant_file=model.get("quant_file", ""),
            quant_sha256=model["provenance"]["sha256"],
            tier="T1", battery=2,
            task_id=item.task_id, fixture_sha=fixture_sha,
            condition=item.condition, run_n=item.run_n,
            session_id=endpoint.session_id,
            sampling={"temp": "runtime-default", "max_tokens": max_tokens},
            det_checks=det_checks,
            needs_judging=needs_judging,
            metrics=metrics,
            timing_authoritative=False,
            artifacts={"response": {"sha256": artifact_sha,
                                    "relpath": f"b2/{item.row_id}.json"}},
            status="ok",
            tags=[]
        )

        if response.get("timings"):
            row.response_meta.update({
                "predicted_n": response["timings"].get("predicted_n"),
                "predicted_per_second": response["timings"].get("predicted_per_second")
            })

        return [row.to_dict()]
