"""Battery 6 -- agentic coding (TESTPLAN 5.6). First-pass build.

Scope note: TESTPLAN 5.6 specifies a full one-shot game roster (Snake ... flight
sim) scored by a headless-Playwright deterministic gate (load/motion/input probes)
plus an N=6 hint-escalation self-correction loop. This first-pass build takes the
deliberately narrower shape requested for the initial implementation: two task
families -- (a) FROM-SCRATCH small program/function/CLI generation and (b)
PLANTED-BUG one-shot self-correction (find + fix, no iterative loop yet) -- scored
purely by STATIC, DETERMINISTIC signal checks. See b6-report.md for the full list
of deltas vs the TESTPLAN spec and why (Playwright gate / loop protocol / game
roster are follow-up work, not in this pass).

SAFETY: model-generated code is NEVER executed inside this framework. Scoring
extracts a fenced code block from the response and runs (1) static signal checks
(required constructs present / fix root-cause token present / buggy line no longer
present) and (2) for Python tasks only, a compile()-only syntax check -- compile()
parses/byte-compiles but never runs the code. See b6_fixtures.compile_check's
docstring for the exact safety argument.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register
from llmtest.batteries.b6_fixtures import (
    load_tasks, extract_code_block, check_code_signals, compile_check,
)


def _condition(order: list[str], ctx_label: str) -> str:
    return schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": ctx_label, "cond": "B6"},
        order)


@register
class B6AgenticCoding(Battery):
    id = 6

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        """Generate WorkItems for B6 agentic-coding tasks.

        Covers all registry models WITHOUT role=quant-arm, all fixture tasks in
        suite/b6_agenticcoding/, and run_n in 1..cfg.suite["b6"]["n_runs"].
        Mirrors B1Business.plan()'s shape (fixture_sha per task rides in the
        payload; force bumps run_n scoped to (model, task, condition)).
        """
        order = cfg.suite["condition_order"]
        sv = cfg.suite["suite_version"]
        n_runs = cfg.suite["b6"]["n_runs"]

        tasks = load_tasks(cfg.root)
        condition = _condition(order, cfg.suite["b6"]["ctx_label"])

        items = []
        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            if m.get("role") == "quant-arm":
                continue
            if str(m.get("local_path", "")).startswith("TO-"):
                continue

            for task in tasks:
                task_id = f"b6.{task.id}"
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
                        quant_sha256=m["provenance"]["sha256"], battery=6,
                        task_id=task_id, fixture_sha=fixture_sha,
                        condition=condition, run_n=run_n)

                    items.append(WorkItem(
                        row_id=rid, model_id=model_id, battery=6,
                        task_id=task_id, condition=condition, run_n=run_n,
                        payload={
                            "model": m,
                            "task_id": task.id,
                            "fixture_sha": fixture_sha,
                            "suite_version": sv,
                            "track": task.track,
                            "language": task.language,
                            "prompt": task.prompt,
                            "required_signals": task.required_signals,
                            "fix_signals": task.fix_signals,
                            "regression_signals": task.regression_signals,
                        }))

        return items

    def preflight(self, ctx) -> list[dict]:
        """Lightweight fixture-existence selftest: the suite/b6_agenticcoding
        dir exists and loads >=1 task.

        Not the TESTPLAN §5.6 "known-good gate" (that requires a Playwright
        harness that doesn't exist yet in this first pass) -- this only
        guards against the fixture directory being absent/empty/malformed
        before a run burns GPU time.
        """
        rows = []
        order = ctx.cfg.suite["condition_order"]
        sv = ctx.cfg.suite["suite_version"]
        fixture_sha = hashlib.sha256(b"b6_agenticcoding").hexdigest()
        condition = schema.canonical_condition({"cond": "SELFTEST"}, order)

        rid = schema.compute_row_id(
            suite_version=sv, model_id="selftest", quant_sha256="0" * 64,
            battery=6, task_id="b6.selftest.fixtures",
            fixture_sha=fixture_sha, condition=condition, run_n=1)

        model_info = ctx.cfg.registry["models"].get("gpt-oss-20b", {})

        try:
            tasks = load_tasks(ctx.root)
        except ValueError as e:
            row = schema.ResultRow.new(
                suite_version=sv, model_id="selftest",
                hf_repo=model_info.get("hf_repo", "N/A"),
                quant_file=model_info.get("quant_file", "N/A"),
                quant_sha256="0" * 64, tier="selftest", battery=6,
                task_id="b6.selftest.fixtures", fixture_sha=fixture_sha,
                condition=condition, run_n=1,
                session_id="selftest", status="error",
                error_detail=f"fixture load failed: {e}", tags=["selftest"])
            rows.append(row.to_dict())
            return rows

        if tasks:
            row = schema.ResultRow.new(
                suite_version=sv, model_id="selftest",
                hf_repo=model_info.get("hf_repo", "N/A"),
                quant_file=model_info.get("quant_file", "N/A"),
                quant_sha256="0" * 64, tier="selftest", battery=6,
                task_id="b6.selftest.fixtures", fixture_sha=fixture_sha,
                condition=condition, run_n=1,
                session_id="selftest", status="ok", tags=["selftest"])
        else:
            row = schema.ResultRow.new(
                suite_version=sv, model_id="selftest",
                hf_repo=model_info.get("hf_repo", "N/A"),
                quant_file=model_info.get("quant_file", "N/A"),
                quant_sha256="0" * 64, tier="selftest", battery=6,
                task_id="b6.selftest.fixtures", fixture_sha=fixture_sha,
                condition=condition, run_n=1,
                session_id="selftest", status="error",
                error_detail="suite/b6_agenticcoding has no task-*.yaml fixtures",
                tags=["selftest"])

        rows.append(row.to_dict())
        return rows

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        """Execute a B6 agentic-coding task.

        - Request endpoint with ctx/max_tokens from suite config (b6.ctx,
          b6.max_tokens_by_track), spec=ngram32 (edit/codegen is the
          n-gram-accelerated case per CLAUDE.md).
        - Chat call with temperature omitted (runtime default), same as B1.
        - Extract a fenced code block from the response (never executed).
        - Static signal checks: required constructs, and for bugfix tasks,
          fix-evidence + no-op/regression detection.
        - Python tasks only: compile()-only syntax check.
        - needs_judging=True: correctness/completeness axes need a judge
          (deterministic checks can't verify runtime behavior without
          executing untrusted code).
        """
        cfg = ctx.cfg
        model = item.payload["model"]
        task_id = item.payload["task_id"]
        fixture_sha = item.payload["fixture_sha"]
        suite_version = item.payload["suite_version"]
        track = item.payload["track"]
        language = item.payload["language"]
        prompt = item.payload["prompt"]
        required_signals = item.payload["required_signals"]
        fix_signals = item.payload["fix_signals"]
        regression_signals = item.payload["regression_signals"]

        endpoint = ctx.server_manager().request_endpoint(
            item.model_id, ctx=cfg.suite["b6"]["ctx"], kv="q8_0",
            flags_overlay={"spec": "ngram32"},
            timing_authoritative=False)

        messages = [{"role": "user", "content": prompt}]
        max_tokens = cfg.suite["b6"]["max_tokens_by_track"].get(track, 4000)

        response = endpoint.chat(messages, max_tokens=max_tokens, temperature=None)
        text = response["choices"][0]["message"]["content"]

        code = extract_code_block(text, language)
        code_for_checks = code if code is not None else ""

        det_checks = {"code_extracted": {"pass": code is not None}}
        det_checks.update(check_code_signals(code_for_checks, required_signals, "required"))
        if track == "bugfix":
            det_checks.update(check_code_signals(code_for_checks, fix_signals, "fix"))
            det_checks.update(check_code_signals(code_for_checks, regression_signals, "regression"))
        if language == "python":
            det_checks["compile_ok"] = compile_check(code_for_checks)

        artifacts_root = (ctx.root / "artifacts" / "b6") if hasattr(ctx, "root") else (Path("artifacts") / "b6")
        artifacts_root.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts_root / f"{item.row_id}.txt"
        artifact_path.write_text(text, encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        row = schema.ResultRow.new(
            suite_version=suite_version, model_id=item.model_id,
            hf_repo=model.get("hf_repo", ""),
            quant_file=model.get("quant_file", ""),
            quant_sha256=model["provenance"]["sha256"],
            tier="T1", battery=6,
            task_id=item.task_id, fixture_sha=fixture_sha,
            condition=item.condition, run_n=item.run_n,
            session_id=endpoint.session_id,
            sampling={"temp": "runtime-default", "max_tokens": max_tokens},
            det_checks=det_checks,
            needs_judging=True,
            metrics={"chars": len(text), "code_chars": len(code_for_checks),
                     "track": track, "language": language},
            timing_authoritative=False,
            artifacts={"response": {"sha256": artifact_sha,
                                    "relpath": f"b6/{item.row_id}.txt"}},
            status="ok",
            tags=[])

        if response.get("timings"):
            row.response_meta.update({
                "predicted_n": response["timings"].get("predicted_n"),
                "predicted_per_second": response["timings"].get("predicted_per_second")
            })

        return [row.to_dict()]
