"""Battery 3 — hallucination curve (TESTPLAN 5.3). Fabrication-under-pressure
probes (unanswerable / false-premise / fabricated-artifact traps + a
closed-domain control + a multi-turn consistency probe), scored
deterministically. needs_judging=False on every row — TESTPLAN 5.3/mission
scope for this battery is fully proxy-checkable; nothing here is genuinely
subjective."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register
from llmtest.batteries.b3_fixtures import load_tasks, score_hallucination


@register
class B3Hallucination(Battery):
    id = 3

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        """Generate WorkItems for B3 hallucination tasks.

        Covers all registry models WITHOUT role=quant-arm (and with a real
        local_path), all fixture tasks in suite/b3_hallucination/, and
        run_n in 1..cfg.suite["b3"]["n_runs"].
        """
        order = cfg.suite["condition_order"]
        sv = cfg.suite["suite_version"]
        n_runs = cfg.suite["b3"]["n_runs"]

        tasks = load_tasks(cfg.root)

        # Condition is constant for B3 tasks — compute once, outside every
        # loop. Deliberately shares B1's runtime/kv/ctx so the ServerManager
        # can reuse an already-running endpoint (TESTPLAN 7.3 "config-match
        # reuse") when B1 and B3 run back to back.
        condition = schema.canonical_condition(
            {"runtime": "fork", "spec": "ngram32", "kv": "q8",
             "ctx": "32k", "cond": "B3"},
            order
        )

        items = []

        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            # Skip models with role=quant-arm
            if m.get("role") == "quant-arm":
                continue
            # Skip models without a real local_path
            if str(m.get("local_path", "")).startswith("TO-"):
                continue

            for task in tasks:
                task_id = f"b3.{task.id}"
                # fixture_sha is the per-task content hash from the loader
                # (sha256 of the task YAML bytes) — NOT a battery-wide constant.
                fixture_sha = task.fixture_sha

                if force:
                    # Exactly ONE new item at max(existing run_n for this
                    # (model, task, condition)) + 1 — mirrors B1's
                    # condition-scoped force-bump.
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
                        quant_sha256=m["provenance"]["sha256"], battery=3,
                        task_id=task_id, fixture_sha=fixture_sha,
                        condition=condition, run_n=run_n)

                    items.append(WorkItem(
                        row_id=rid, model_id=model_id, battery=3,
                        task_id=task_id, condition=condition,
                        run_n=run_n,
                        payload={
                            "model": m,
                            "task_id": task.id,
                            "fixture_sha": fixture_sha,
                            "suite_version": sv,
                            # Ride everything score_hallucination() needs
                            # along so execute() doesn't re-load the fixture
                            # tree.
                            "turns": task.turns,
                            "expect": task.expect,
                            "cls": task.cls,
                            "category": task.category,
                            "difficulty": task.difficulty,
                            "hedge_signals": task.hedge_signals,
                            "trap_signals": task.trap_signals,
                            "answer_signals": task.answer_signals,
                        }))

        return items

    def preflight(self, ctx) -> list[dict]:
        """Validate that every configured category has >=1 authored task.

        Returns selftest rows: status="ok" per category, "error" naming the
        category. All rows have tags=["selftest"], cond=SELFTEST, run_n=1.
        """
        rows = []
        order = ctx.cfg.suite["condition_order"]
        sv = ctx.cfg.suite["suite_version"]

        tasks = load_tasks(ctx.root)
        counts: dict[str, int] = {}
        for t in tasks:
            counts[t.category] = counts.get(t.category, 0) + 1

        model_info = ctx.cfg.registry["models"].get("gpt-oss-20b", {})
        condition = schema.canonical_condition({"cond": "SELFTEST"}, order)

        for category in ctx.cfg.suite["b3"]["categories"]:
            # Selftest rows aren't tied to any one task file, so there's no
            # single content hash to use here — same sanctioned exception
            # as B1's preflight (deterministic hash of the category name,
            # used ONLY for these selftest rows).
            fixture_sha = hashlib.sha256(category.encode("utf-8")).hexdigest()

            rid = schema.compute_row_id(
                suite_version=sv, model_id="selftest", quant_sha256="0" * 64,
                battery=3, task_id=f"b3.selftest.{category}",
                fixture_sha=fixture_sha, condition=condition, run_n=1)

            if counts.get(category, 0) > 0:
                row = schema.ResultRow.new(
                    suite_version=sv, model_id="selftest",
                    hf_repo=model_info.get("hf_repo", "N/A"),
                    quant_file=model_info.get("quant_file", "N/A"),
                    quant_sha256="0" * 64, tier="selftest", battery=3,
                    task_id=f"b3.selftest.{category}", fixture_sha=fixture_sha,
                    condition=condition, run_n=1,
                    session_id="selftest", status="ok",
                    tags=["selftest"])
            else:
                row = schema.ResultRow.new(
                    suite_version=sv, model_id="selftest",
                    hf_repo=model_info.get("hf_repo", "N/A"),
                    quant_file=model_info.get("quant_file", "N/A"),
                    quant_sha256="0" * 64, tier="selftest", battery=3,
                    task_id=f"b3.selftest.{category}", fixture_sha=fixture_sha,
                    condition=condition, run_n=1,
                    session_id="selftest", status="error",
                    error_detail=f"no authored task tagged category: {category}",
                    tags=["selftest"])

            rows.append(row.to_dict())

        return rows

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        """Execute a B3 hallucination task.

        - Request endpoint with ctx from suite config (b3.ctx), kv="q8_0"
        - Run the (possibly multi-turn) conversation, temperature omitted
          (runtime default)
        - Score the FINAL turn's response deterministically via
          score_hallucination()
        - Save the full transcript under artifacts/b3/<row_id>.txt
        - Return a row with needs_judging=False (fully deterministic battery)
        """
        cfg = ctx.cfg
        model = item.payload["model"]
        task_id = item.payload["task_id"]
        fixture_sha = item.payload["fixture_sha"]
        suite_version = item.payload["suite_version"]
        turns = item.payload["turns"]
        cls = item.payload["cls"]

        endpoint = ctx.server_manager().request_endpoint(
            item.model_id, ctx=cfg.suite["b3"]["ctx"], kv="q8_0",
            timing_authoritative=False)

        max_tokens = cfg.suite["b3"]["max_tokens_by_class"].get(cls, 1500)

        messages: list[dict] = []
        transcript_parts: list[str] = []
        last_text = ""
        last_response: dict = {}
        for turn_prompt in turns:
            messages.append({"role": "user", "content": turn_prompt})
            response = endpoint.chat(messages, max_tokens=max_tokens, temperature=None)
            last_text = response["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": last_text})
            transcript_parts.append(f"USER: {turn_prompt}\n\nASSISTANT: {last_text}")
            last_response = response

        # score_hallucination() only needs expect/hedge_signals/trap_signals/
        # answer_signals — those rode in the payload from plan(), so no
        # second fixture-tree load here.
        task_stub = SimpleNamespace(
            expect=item.payload["expect"],
            hedge_signals=item.payload["hedge_signals"],
            trap_signals=item.payload["trap_signals"],
            answer_signals=item.payload["answer_signals"])
        det_checks = score_hallucination(last_text, task_stub)

        transcript = "\n\n---\n\n".join(transcript_parts)
        artifacts_root = (ctx.root / "artifacts" / "b3") if hasattr(ctx, 'root') else (Path("artifacts") / "b3")
        artifacts_root.mkdir(parents=True, exist_ok=True)

        artifact_path = artifacts_root / f"{item.row_id}.txt"
        artifact_path.write_text(transcript, encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        row = schema.ResultRow.new(
            suite_version=suite_version, model_id=item.model_id,
            hf_repo=model.get("hf_repo", ""),
            quant_file=model.get("quant_file", ""),
            quant_sha256=model["provenance"]["sha256"],
            tier="T1", battery=3,
            task_id=item.task_id, fixture_sha=fixture_sha,
            condition=item.condition, run_n=item.run_n,
            session_id=endpoint.session_id,
            sampling={"temp": "runtime-default", "max_tokens": max_tokens},
            det_checks=det_checks,
            needs_judging=False,
            metrics={
                "category": item.payload["category"],
                "difficulty": item.payload["difficulty"],
                "expect": item.payload["expect"],
                "turns": len(turns),
                # Plain booleans (not the {"pass": ...}-wrapped det_checks
                # entries) so table-time curve aggregation can group/mean
                # directly on metrics.fabricated by metrics.difficulty.
                "hedged": det_checks["hedged"]["pass"],
                "fabricated": det_checks["fabricated"]["pass"],
                "correct": det_checks["correct"]["pass"],
                "chars": len(last_text),
            },
            timing_authoritative=False,
            artifacts={"response": {"sha256": artifact_sha,
                                    "relpath": f"b3/{item.row_id}.txt"}},
            status="ok",
            tags=[]
        )

        # Add response_meta from the LAST turn's timings, if present.
        if last_response.get("timings"):
            row.response_meta.update({
                "predicted_n": last_response["timings"].get("predicted_n"),
                "predicted_per_second": last_response["timings"].get("predicted_per_second")
            })

        return [row.to_dict()]
