"""Battery 1 — business task execution (TESTPLAN 1.X). Row generation for MSP classification tasks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register
from llmtest.batteries.b1_fixtures import load_unit_tasks, check_signals


@register
class B1Business(Battery):
    id = 1

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        """Generate WorkItems for B1 business tasks.

        Covers all registry models WITHOUT role=quant-arm, all available unit tasks,
        and run_n in 1..cfg.suite["b1"]["n_runs"].
        """
        order = cfg.suite["condition_order"]
        sv = cfg.suite["suite_version"]
        n_runs = cfg.suite["b1"]["n_runs"]

        items = []

        # Load all available tasks from the real tree
        all_tasks = {}  # unit -> [Task]
        for unit in cfg.suite["b1"]["units_tier1"]:
            tasks = load_unit_tasks(cfg.root, unit)
            if tasks:
                all_tasks[unit] = tasks

        # Iterate over registry models
        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            # Skip models with role=quant-arm
            if m.get("role") == "quant-arm":
                continue
            # Skip models without a real local_path
            if str(m.get("local_path", "")).startswith("TO-"):
                continue

            # For each available task
            for unit, tasks in all_tasks.items():
                for task in tasks:
                    # fixture_sha is the hash of the unit directory name (deterministic)
                    fixture_sha = hashlib.sha256(unit.encode("utf-8")).hexdigest()

                    # For each run
                    for run_n in range(1, n_runs + 1):
                        if force:
                            # Find the max run_n for this task+model+condition
                            existing = [r["run_n"] for r in store.iter_rows()
                                       if r["task_id"] == f"b1.{task.id}"
                                       and r["model_id"] == model_id]
                            run_n = (max(existing) + 1) if existing else 1

                        # Build canonical condition
                        condition = schema.canonical_condition(
                            {"runtime": "fork", "spec": "ngram32", "kv": "q8",
                             "ctx": "16k", "cond": "B1"},
                            order
                        )

                        # Compute row_id
                        rid = schema.compute_row_id(
                            suite_version=sv, model_id=model_id,
                            quant_sha256=m["provenance"]["sha256"], battery=1,
                            task_id=f"b1.{task.id}", fixture_sha=fixture_sha,
                            condition=condition, run_n=run_n)

                        items.append(WorkItem(
                            row_id=rid, model_id=model_id, battery=1,
                            task_id=f"b1.{task.id}", condition=condition,
                            run_n=run_n,
                            payload={
                                "model": m,
                                "task_id": task.id,
                                "fixture_sha": fixture_sha,
                                "suite_version": sv
                            }))

        return items

    def preflight(self, ctx) -> list[dict]:
        """Validate that all Tier-1 unit dirs exist and have ≥1 task each.

        Returns selftest rows: status="ok" per unit, "error" naming the unit.
        All rows have tags=["selftest"], cond=SELFTEST, run_n=1.
        """
        rows = []
        order = ctx.cfg.suite["condition_order"]
        sv = ctx.cfg.suite["suite_version"]

        for unit in ctx.cfg.suite["b1"]["units_tier1"]:
            # fixture_sha is the hash of the unit directory name
            fixture_sha = hashlib.sha256(unit.encode("utf-8")).hexdigest()

            # Try to load tasks from this unit
            tasks = load_unit_tasks(ctx.root, unit)

            # Compute row_id for the selftest
            condition = schema.canonical_condition({"cond": "SELFTEST"}, order)
            rid = schema.compute_row_id(
                suite_version=sv, model_id="selftest", quant_sha256="0" * 64,
                battery=1, task_id=f"b1.selftest.{unit}",
                fixture_sha=fixture_sha, condition=condition, run_n=1)

            # Get model info for the row (use a placeholder for selftest)
            model_info = ctx.cfg.registry["models"].get("gpt-oss-20b", {})

            if tasks:
                # Unit exists and has tasks
                row = schema.ResultRow.new(
                    suite_version=sv, model_id="selftest",
                    hf_repo=model_info.get("hf_repo", "N/A"),
                    quant_file=model_info.get("quant_file", "N/A"),
                    quant_sha256="0" * 64, tier="selftest", battery=1,
                    task_id=f"b1.selftest.{unit}", fixture_sha=fixture_sha,
                    condition=condition, run_n=1,
                    session_id="selftest", status="ok",
                    tags=["selftest"])
            else:
                # Unit dir missing or no tasks
                row = schema.ResultRow.new(
                    suite_version=sv, model_id="selftest",
                    hf_repo=model_info.get("hf_repo", "N/A"),
                    quant_file=model_info.get("quant_file", "N/A"),
                    quant_sha256="0" * 64, tier="selftest", battery=1,
                    task_id=f"b1.selftest.{unit}", fixture_sha=fixture_sha,
                    condition=condition, run_n=1,
                    session_id="selftest", status="error",
                    error_detail=f"unit dir missing or no tasks: {unit}",
                    tags=["selftest"])

            rows.append(row.to_dict())

        return rows

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        """Execute a B1 business task.

        - Request endpoint with ctx=16384, kv="q8_0"
        - Chat call with temperature omitted (runtime default)
        - Check signals against the response
        - Save artifact under artifacts/b1/<row_id>.txt
        - Return a row with needs_judging=True
        """
        cfg = ctx.cfg
        model = item.payload["model"]
        task_id = item.payload["task_id"]
        fixture_sha = item.payload["fixture_sha"]
        suite_version = item.payload["suite_version"]

        # Load the task to get prompt and signals
        unit = task_id.split("-")[0]  # e.g., "cybersecurity-01" -> "cybersecurity"
        tasks = load_unit_tasks(cfg.root, unit)
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            raise RuntimeError(f"task {task_id} not found in unit {unit}")

        # Request endpoint
        endpoint = ctx.server_manager().request_endpoint(
            item.model_id, ctx=16384, kv="q8_0", timing_authoritative=False)

        # Make chat call (temperature omitted from request body)
        messages = [{"role": "user", "content": task.prompt}]
        max_tokens = cfg.suite["b1"]["max_tokens_by_class"].get(task.cls, 1600)

        response = endpoint.chat(messages, max_tokens=max_tokens, temperature=None)

        # Extract response text
        text = response["choices"][0]["message"]["content"]

        # Check signals
        det_checks = check_signals(text, task.signals)

        # Save artifact
        artifacts_root = (ctx.root / "artifacts" / "b1") if hasattr(ctx, 'root') else (Path("artifacts") / "b1")
        artifacts_root.mkdir(parents=True, exist_ok=True)

        artifact_path = artifacts_root / f"{item.row_id}.txt"
        artifact_path.write_text(text, encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        # Build the row
        row = schema.ResultRow.new(
            suite_version=suite_version, model_id=item.model_id,
            hf_repo=model.get("hf_repo", ""),
            quant_file=model.get("quant_file", ""),
            quant_sha256=model["provenance"]["sha256"],
            tier="T1", battery=1,
            task_id=item.task_id, fixture_sha=fixture_sha,
            condition=item.condition, run_n=item.run_n,
            session_id=endpoint.session_id,
            sampling={"max_tokens": max_tokens},
            det_checks=det_checks,
            needs_judging=True,
            metrics={"chars": len(text)},
            timing_authoritative=False,
            artifacts={"b1": {"sha256": artifact_sha,
                             "relpath": f"b1/{item.row_id}.txt"}},
            status="ok",
            tags=[]
        )

        # Add response_meta from timings if present
        if response.get("timings"):
            row.response_meta.update({
                "predicted_n": response["timings"].get("predicted_n"),
                "predicted_per_second": response["timings"].get("predicted_per_second")
            })

        return [row.to_dict()]
