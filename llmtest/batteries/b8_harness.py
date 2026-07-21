"""Battery 8 -- real agent-harness execution (Part 2 Phase 2 integration).

Drives real models through a real external agent harness (OpenCode first;
`llmtest.harness.opencode.OpenCodeAdapter`) against B8's versioned task
manifests (`llmtest.harness.tasks`), and scores completion with the
anti-gaming completion oracle (`run_oracle`) rather than the static
content-signal checks B1/B6/B7 use -- a B8 row measures whether the agent
actually got the task done in a real multi-step tool-use loop, not just
whether one chat response contains the right tokens.

ROW IDENTITY -- fully additive, per the resolved design (task-7-brief.md):
`schema.compute_row_id`'s signature is UNCHANGED. B8 gets two new pieces of
identity that ride entirely inside the existing `condition` string (via
`schema.canonical_condition`, which only emits keys present in `parts` --
so B1-B7's condition strings, whose parts dicts never contain these keys,
come out byte-identical before and after this file exists):

  - `attempt_id` -- a unique id per PHYSICAL execution (the actual
    subprocess run of the harness). Injectable via `ctx.b8_attempt_id`
    (tests drive this directly); defaults to a fresh `uuid4().hex` per
    `execute()` call otherwise.
  - `execution_provenance_sha` -- sha256 over a canonical JSON blob of
    `{harness_version, litellm_version, server_profile, rendered_prompt}`
    (`_execution_provenance_sha`), computed at `execute()` time. Changing
    any of those four inputs changes the row_id; none of them depend on
    wall-clock time or randomness, so the SAME attempt_id replayed under
    the SAME harness/profile/prompt reproduces the SAME row_id (no
    accidental fan-out), while a genuinely different physical attempt
    (fresh attempt_id) never collides with a prior one.

`run_n` keeps carrying the LOGICAL replicate number (`replicate_n`, 1..N)
exactly as every other battery's `run_n` does -- attempt_id/exec_sha are
the physical-execution axis, replicate_n is the logical-repeat axis; they
are orthogonal and both live in the row, one via `run_n`, the other via
`condition`.

`plan()` computes each `WorkItem`'s condition/row_id from the BASE identity
only (`cond=B8`, `harness`, `task`) -- attempt_id/exec_sha don't exist yet
at planning time (they're stamped once the physical run actually happens).
`execute()` recomputes the FULL condition (base + attempt_id + exec_sha)
and lets `schema.ResultRow.new` compute the row's own, more specific
row_id from that -- so a WorkItem's `row_id` is a planning-time id (used by
`plan()` itself, see the resume note below), not the final row's row_id.
This is intentional, not a bug: `validate_row` recomputes row_id from the
ROW's own fields regardless of what any WorkItem said, so there is never a
mismatch at the schema level.

RESUME: because the final row_id is never known until execute() (it depends
on attempt_id/exec_sha, stamped only inside execute()), `run_cmd.py`'s own
`item.row_id not in store.existing_row_ids()` gate can never match a B8
WorkItem against an already-completed B8 row (their row_ids are computed
from different-shaped conditions by construction) -- so unlike B1/B6/B7,
that check is a structural no-op for this battery, not a resume mechanism.
`plan()` is therefore the ONLY place resume/dedup can happen for B8: it
reads `store.iter_rows()` and, for each (model, harness, task) cell, skips
any `replicate_n` that already has a recorded row for that cell (matched by
parsing `harness`/`task` back out of the stored row's own condition string
-- attempt_id/exec_sha are deliberately ignored for this match, since they
vary per physical attempt and are not part of the logical cell). Default
(force=False): emit WorkItems only for replicate_ns 1..N not yet recorded.
`force=True`: mirrors B6/B7's bump idiom, emitting exactly one WorkItem for
one replicate_n beyond whatever's already recorded (an explicit "run one
more attempt" request, not a full re-cross).

CONTAINMENT (Wave 2, B8 validity program): `cfg.suite["b8"]["sandbox"]`
is the seam. `enabled: true` (suite.yaml's default as of Wave 2) threads
`sandbox.image` (a Node-capable image, `b8-sandbox:1` -- has `opencode` on
PATH, unlike the unrelated `nvidia/cuda:...-base` `Sandbox` pin, which is
for `run_oracle`'s anti-gaming validation container, a completely
different concern) plus `sandbox.cpus`/`memory`/`pids_limit` into the
`OpenCodeAdapter` the default factory (`_DEFAULT_HARNESS_FACTORIES`)
constructs -- from there, `OpenCodeAdapter.run()` itself drives `opencode`
inside a disposable, hardened container instead of a host subprocess (see
`llmtest.harness.opencode`'s own module docstring, CONTAINMENT section,
for the container design + two live findings). `enabled: false` is the
ORIGINAL host-execution path, kept as a fallback specifically so
`ctx.b8_adapters`-injected unit tests (every `execute()` test in
`test_b8.py`/`test_b8_local.py`) stay green with no Docker involved at
all -- that seam bypasses `_resolve_adapter`'s factory entirely regardless
of `sandbox.enabled`, so the flag only matters for the REAL, uninjected
factory path a live run actually takes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register
from llmtest.batteries.b8_fixtures import load_tasks
from llmtest.harness.opencode import OpenCodeAdapter
from llmtest.harness.tasks import materialize_repo, run_oracle

# harness name -> factory(model_id, budgets, sandbox_cfg) -> HarnessAdapter.
# Only "opencode" is wired to a real adapter today (Task 4); a test
# overrides this entirely via ctx.b8_adapters (see _resolve_adapter).
# `max_steps` is threaded from `budgets["steps"]` (Wave 1a) --
# OpenCodeAdapter uses it for a NATIVE, best-effort mid-run step cap (see
# its own docstring); the post-hoc `budget_exceeded` check below in
# `execute()` is what actually enforces both the step AND completion-token
# budgets regardless of whether the native cap fired.
#
# `sandbox_cfg` (Wave 2 CONTAINMENT, `b8.sandbox` from suite.yaml) threads
# straight into `OpenCodeAdapter`'s own container-mode constructor args --
# `sandbox_image` is `None` (host-subprocess path, byte-for-byte the
# pre-Wave-2 behavior) unless `sandbox_cfg["enabled"]` is true, in which
# case it's `sandbox_cfg["image"]`. `cpus`/`memory`/`pids_limit` are only
# ever READ by the adapter when `sandbox_image` is set, so they're passed
# through with sane defaults regardless.
_DEFAULT_HARNESS_FACTORIES = {
    "opencode": lambda model_id, budgets, sandbox_cfg: OpenCodeAdapter(
        model=model_id, wall_clock_s=budgets.get("wall_clock_s"),
        max_steps=budgets.get("steps"),
        sandbox_image=((sandbox_cfg or {}).get("image")
                       if (sandbox_cfg or {}).get("enabled") else None),
        cpus=(sandbox_cfg or {}).get("cpus", 2.0),
        mem_limit=(sandbox_cfg or {}).get("memory", "4g"),
        pids_limit=str((sandbox_cfg or {}).get("pids_limit", "512"))),
}


def _condition_parts(condition: str) -> dict:
    return dict(p.split("=", 1) for p in condition.split(";") if p)


def _base_condition(order: list[str], harness_name: str, task_id: str) -> str:
    """The identity known at plan() time -- battery marker + harness + task.
    No attempt_id/execution_provenance_sha yet (see module docstring)."""
    return schema.canonical_condition(
        {"cond": "B8", "harness": harness_name, "task": task_id}, order)


def _full_condition(order: list[str], harness_name: str, task_id: str,
                    attempt_id: str, execution_provenance_sha: str) -> str:
    """The identity stamped at execute() time -- base + the physical-
    execution axis. This is what actually rides in the emitted ROW's
    condition (and therefore its row_id)."""
    return schema.canonical_condition(
        {"cond": "B8", "harness": harness_name, "task": task_id,
         "attempt_id": attempt_id,
         "execution_provenance_sha": execution_provenance_sha}, order)


def _execution_provenance_sha(*, harness_version: str, litellm_version: str,
                              server_profile: dict, rendered_prompt: str) -> str:
    """sha256 over a canonical (sorted-key, separator-tight) JSON encoding
    of the four inputs the brief names: harness version, the LiteLLM proxy
    version (empty string for a direct adapter like OpenCode -- there is no
    LiteLLM hop), the server profile (launch flags + a template hash), and
    the rendered prompt actually handed to the harness. Deliberately
    contains NO timestamp/uuid/random input -- that's what makes "same
    attempt_id -> same row_id" hold on a byte-identical replay (module
    docstring)."""
    payload = {
        "harness_version": harness_version,
        "litellm_version": litellm_version,
        "server_profile": server_profile,
        "rendered_prompt": rendered_prompt,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _resolve_adapter(ctx, harness_name: str, model_id: str, budgets: dict,
                     sandbox_cfg: dict | None):
    """The real adapter for `harness_name`, unless `ctx.b8_adapters` (a
    {harness_name: HarnessAdapter instance} dict) is present -- the
    no-Docker/no-live-harness test seam: a test sets this to
    {"opencode": MockHarnessAdapter(...)} and execute() never touches a
    real subprocess (or, for `sandbox_cfg["enabled"]`, real Docker) at
    all -- `sandbox_cfg` is simply unused on the injected-adapter path,
    which is exactly what keeps every existing `ctx.b8_adapters` test
    green regardless of `b8.sandbox.enabled`'s value."""
    injected = getattr(ctx, "b8_adapters", None)
    if injected is not None:
        if harness_name not in injected:
            raise KeyError(f"no adapter injected for harness {harness_name!r}")
        return injected[harness_name]
    factory = _DEFAULT_HARNESS_FACTORIES.get(harness_name)
    if factory is None:
        raise KeyError(f"unknown harness {harness_name!r} (no default adapter factory)")
    return factory(model_id, budgets, sandbox_cfg)


def _resolve_run_oracle(ctx):
    """The real `run_oracle` (needs Docker -- see `llmtest.harness.tasks`),
    unless `ctx.b8_run_oracle` (a `(task, workspace, root=...) -> (bool,
    str)` callable) is present -- the test seam that lets `execute()` be
    exercised with a controllable completion outcome and no Docker.

    Always returns something callable as `(task, workspace, root=...,
    oracle_image=...)` -- `execute()` passes `oracle_image` unconditionally
    (the b8.sandbox.oracle_image seam, task-b8local). An injected
    `ctx.b8_run_oracle` keeps its ORIGINAL, oracle_image-agnostic signature
    (every existing test injects `lambda task, workspace, root=".": ...`)
    by being wrapped in a thin adapter that accepts and discards
    `oracle_image` before delegating -- so no existing seam consumer needs
    to change. Only the real `run_oracle` (`llmtest.harness.tasks`) actually
    receives `oracle_image`."""
    injected = getattr(ctx, "b8_run_oracle", None)
    if injected is not None:
        return lambda task, workspace, root=".", oracle_image=None: injected(
            task, workspace, root=root)
    return run_oracle


def _resolve_attempt_id(ctx) -> str:
    """A fresh id per physical execution. Injectable via `ctx.b8_attempt_id`
    (a fixed string -- what the row-identity tests drive to prove two
    different attempt_ids never collide, and the same attempt_id
    round-trips to the same row_id); otherwise a fresh `uuid4().hex` (this
    is battery/orchestration code, not a workflow script, so uuid4 is fine
    here per the resolved design)."""
    injected = getattr(ctx, "b8_attempt_id", None)
    if injected:
        return injected
    return uuid.uuid4().hex


@register
class B8Harness(Battery):
    id = 8

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        """One WorkItem per (model, harness, task, replicate_n), crossed
        from `cfg.suite["b8"]["models"]` x `["harnesses"]` x
        `load_tasks(cfg.root)` x `range(1, replicates+1)` -- `load_tasks`
        further narrowed by the optional `b8.tasks` allowlist (task-
        b8expand, see below) before the cross.

        `models`/`harnesses` are read from the b8: config block itself
        (NOT the full non-quant-arm registry roster B1/B6/B7 use) -- B8 is
        deliberately a small subset (each replicate is a real, possibly
        slow, external-process harness run against a live endpoint).

        `b8.tasks` (additive, task-b8expand): when `cfg.suite["b8"]` has a
        non-empty `tasks` list, only `B8Task.id` values in that list are
        crossed -- e.g. suite.yaml sets this to the 6 real Python task ids
        so a live run doesn't cross the 5 bash placeholder manifests
        (task-01..05.yaml) that were never meant to be exercised for real.
        Absent or empty: every loaded task is crossed, unchanged from
        before this key existed.

        Resume/dedup happens HERE (see module docstring's RESUME section,
        not via `run_cmd.py`'s row_id-membership check, which is a
        structural no-op for this battery): for each (model, harness, task)
        cell, `replicate_n` values already represented by a stored row for
        that cell (matched on `harness`/`task` parsed back out of the
        row's own condition, ignoring attempt_id/exec_sha) are skipped.
        `force=True` mirrors B6/B7's bump idiom: exactly one more
        replicate_n beyond whatever's already recorded.
        """
        order = cfg.suite["condition_order"]
        sv = cfg.suite["suite_version"]
        b8cfg = cfg.suite["b8"]
        replicates = b8cfg["replicates"]

        tasks = load_tasks(cfg.root)
        # `b8.tasks` allowlist (additive, task-b8expand): a non-empty list
        # restricts plan() to exactly those B8Task.id values (e.g. the 6
        # real Python tasks in suite.yaml), so a live run can target only
        # tasks that are actually agent-solvable/executable, leaving the
        # bash placeholder manifests (task-01..05.yaml) loaded but unused.
        # Absent or empty (falsy) -- every existing caller/test predating
        # this key -- falls straight through to every loaded task,
        # byte-for-byte the pre-task-b8expand behavior.
        task_allowlist = b8cfg.get("tasks")
        if task_allowlist:
            allowed_ids = set(task_allowlist)
            tasks = [task for task in tasks if task.id in allowed_ids]
        rows = list(store.iter_rows())

        items = []
        for model_id in b8cfg["models"]:
            if model_filter and model_id != model_filter:
                continue
            m = cfg.registry["models"][model_id]

            for harness_name in b8cfg["harnesses"]:
                for task in tasks:
                    task_id = f"b8.{task.id}"
                    fixture_sha = task.fixture_sha
                    condition = _base_condition(order, harness_name, task.id)

                    done_runs = {
                        r["run_n"] for r in rows
                        if r["task_id"] == task_id and r["model_id"] == model_id
                        and _condition_parts(r["condition"]).get("harness") == harness_name
                        and _condition_parts(r["condition"]).get("task") == task.id
                    }
                    if force:
                        run_ns = [(max(done_runs) + 1) if done_runs else 1]
                    else:
                        run_ns = [n for n in range(1, replicates + 1) if n not in done_runs]

                    for run_n in run_ns:
                        rid = schema.compute_row_id(
                            suite_version=sv, model_id=model_id,
                            quant_sha256=m["provenance"]["sha256"], battery=8,
                            task_id=task_id, fixture_sha=fixture_sha,
                            condition=condition, run_n=run_n)

                        items.append(WorkItem(
                            row_id=rid, model_id=model_id, battery=8,
                            task_id=task_id, condition=condition, run_n=run_n,
                            payload={
                                "model": m,
                                "task": task,
                                "harness": harness_name,
                                "suite_version": sv,
                                "fixture_sha": fixture_sha,
                            }))
        return items

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        """Materialize the task workspace -> run the harness adapter (real
        or injected) -> score completion via the anti-gaming oracle (real
        or injected) -> stamp attempt_id/execution_provenance_sha into the
        FULL condition -> emit one schema-valid row.

        `needs_judging=False`: completion is decided by the deterministic
        anti-gaming oracle (`run_oracle`), not a judge -- mirrors every
        other deterministically-scored battery's convention (first-failure
        CLASSIFICATION, which does need a blinded panel for the ambiguous
        cases, is Task 8's scope, not this one's).
        """
        cfg = ctx.cfg
        order = cfg.suite["condition_order"]
        b8cfg = cfg.suite["b8"]
        budgets = b8cfg["budgets"]
        sandbox_cfg = b8cfg.get("sandbox")

        model = item.payload["model"]
        task = item.payload["task"]
        harness_name = item.payload["harness"]
        suite_version = item.payload["suite_version"]
        fixture_sha = item.payload["fixture_sha"]

        root = ctx.root if hasattr(ctx, "root") else Path(".")
        workspace = (root / "artifacts" / "b8_workspaces"
                    / f"{item.task_id}-run{item.run_n}-{uuid.uuid4().hex[:8]}")
        materialize_repo(task, workspace)

        # Endpoint-injection seam (task-b8local): `ctx.b8_endpoint`, when
        # present, is used AS-IS and `ctx.server_manager()` is never
        # touched -- this is what lets a coordinator serve the model
        # MANUALLY (outside ServerManager, e.g. because the standard
        # Windows ServerManager path needs adaptation for a given box) and
        # simply hand execute() the already-live endpoint. Mirrors the
        # existing `ctx.b8_adapters` / `ctx.b8_run_oracle` / `ctx.
        # b8_attempt_id` seams: `getattr` with a `None` default, so a
        # `ctx` that has no `b8_endpoint` attribute at all (every existing
        # caller, including run_cmd.py's real `RunContext`) falls straight
        # through to the original `server_manager().request_endpoint(...)`
        # path, byte-for-byte unchanged.
        endpoint = getattr(ctx, "b8_endpoint", None)
        if endpoint is None:
            endpoint = ctx.server_manager().request_endpoint(
                item.model_id, ctx=b8cfg.get("ctx", 32768), kv=b8cfg.get("kv", "q8_0"),
                flags_overlay={"spec": "ngram32"}, timing_authoritative=False)

        adapter = _resolve_adapter(ctx, harness_name, item.model_id, budgets, sandbox_cfg)
        adapter.setup(task, endpoint, workspace)
        try:
            trace = adapter.run()
        finally:
            adapter.teardown()

        # Configurable oracle image (task-b8local): b8.sandbox.oracle_image
        # in suite.yaml (e.g. "python:3.11-slim" for the Python-shaped
        # manifests, which need python3 to import/execute the agent's
        # solution -- the pinned nvidia/cuda:...-base image has none). Absent
        # from suite.yaml (or ctx.b8_run_oracle injected) -> None, which
        # _resolve_run_oracle/run_oracle both treat as "use the pin", so this
        # is inert for every caller that predates this key.
        oracle_image = (b8cfg.get("sandbox") or {}).get("oracle_image")
        run_oracle_fn = _resolve_run_oracle(ctx)
        completed, oracle_detail = run_oracle_fn(task, workspace, root=root,
                                                 oracle_image=oracle_image)

        # -- Budget enforcement + the scoring contract (Wave 1a, B8 -------
        # measurement-validity: codex/kimi strategy review) ---------------
        # `wall_clock_s` was already enforced (the adapter's own subprocess
        # timeout -> terminal_status=="killed" on expiry); `tokens`/`steps`
        # were only ever RECORDED before this point -- a model that solved
        # a task in 5 steps and one that stumbled for 200 steps and got
        # lucky were scored identically, which is not a valid basis for a
        # ranking. Both are now enforced post-hoc, here: a run that burned
        # more COMPLETION tokens or STEPS than its budget is a budget
        # failure, credited as a completion NEVER -- even when the oracle
        # would otherwise have passed. See suite.yaml's b8.budgets doc
        # comment for the completion-vs-total-token distinction and the
        # OpenCode native-step-cap note (a soft, best-effort nudge, not a
        # hard stop -- this post-hoc check is the actual backstop for both
        # dimensions).
        budget_tokens = budgets.get("tokens")
        budget_steps = budgets.get("steps")
        budget_exceeded = bool(
            (budget_tokens is not None and trace.tokens_completion > budget_tokens)
            or (budget_steps is not None and trace.steps > budget_steps))

        # The row's `metrics.terminal_status` mirrors the RAW trace status
        # unchanged whenever that status is already non-"completed" --
        # "killed"/"infra-error" are already the more specific, accurate
        # outcome description (no branching there -- see
        # test_execute_handles_non_completed_terminal_status). "budget-
        # exceeded" is substituted ONLY when the harness itself thought the
        # run finished cleanly but this post-hoc check disagrees -- the one
        # case the harness had no way to have flagged on its own.
        effective_terminal_status = trace.terminal_status
        if trace.terminal_status == "completed" and budget_exceeded:
            effective_terminal_status = "budget-exceeded"

        # A row's `completion` is credited ONLY when all three hold: the
        # harness itself reported a clean finish, the run stayed within
        # BOTH budgets, and the anti-gaming oracle passed. This is what
        # makes every credited completion COMPARABLE across models,
        # regardless of how many tokens/turns a given model burned getting
        # there -- the entire point of this wave.
        completed_final = (trace.terminal_status == "completed"
                          and not budget_exceeded and completed)

        # The Trace persisted to artifacts/b8_traces/<row_id>.json (reloaded
        # by scripts/classify_b8_local.py) must show the SAME effective
        # terminal_status the row's own metrics do -- otherwise
        # llmtest.harness.failure_class.classify_first_failure's
        # deterministic budget/step-exhaustion check (label "e") could
        # never see this outcome for a run the harness itself called
        # "completed". `trace` itself (used below for steps/tokens/
        # subagent_spawned) is left untouched; only this copy's
        # terminal_status differs when it differs at all.
        persisted_trace = trace
        if effective_terminal_status != trace.terminal_status:
            persisted_trace = dataclasses.replace(
                trace, terminal_status=effective_terminal_status)

        attempt_id = _resolve_attempt_id(ctx)
        harness_version = adapter.version()
        litellm_version = getattr(adapter, "litellm_version", "") or ""
        server_profile = {
            "flags": getattr(endpoint, "normalized_config", {}) or {},
            # task-derived, not endpoint-derived: the closest existing,
            # already-deterministic hash of "which task-repo template
            # underlies this run" (llmtest.harness.tasks.B8Task.
            # setup_repo_sha, computed once at manifest-load time) --
            # distinct from `rendered_prompt` below, which hashes the
            # literal prompt text actually handed to the harness.
            "template_sha": task.setup_repo_sha,
        }
        rendered_prompt = getattr(adapter, "rendered_prompt", None) or task.prompt

        exec_sha = _execution_provenance_sha(
            harness_version=harness_version, litellm_version=litellm_version,
            server_profile=server_profile, rendered_prompt=rendered_prompt)

        condition = _full_condition(order, harness_name, task.id, attempt_id, exec_sha)

        row = schema.ResultRow.new(
            suite_version=suite_version, model_id=item.model_id,
            hf_repo=model.get("hf_repo", ""), quant_file=model.get("quant_file", ""),
            quant_sha256=model["provenance"]["sha256"], tier="T1", battery=8,
            task_id=item.task_id, fixture_sha=fixture_sha,
            condition=condition, run_n=item.run_n,
            session_id=getattr(endpoint, "session_id", "unknown"),
            sampling={"harness": harness_name, "wall_clock_s": budgets.get("wall_clock_s")},
            needs_judging=False,
            # Surfaces WHY run_oracle rejected a run (e.g. "out-of-bounds
            # edit: sneaky.sh", "protected file tampered: NOTES.md", an
            # oracle timeout) on every completion=False row -- Task 8
            # (first-failure classification) reads this, not just the bare
            # completed bool in metrics below.
            det_checks={"oracle": {"pass": completed, "detail": oracle_detail}},
            metrics={
                "completion": completed_final,
                "steps": trace.steps,
                "tokens_prompt": trace.tokens_prompt,
                "tokens_completion": trace.tokens_completion,
                "terminal_status": effective_terminal_status,
                "subagent_spawned": trace.subagent_spawned,
                # Wave 1a auditability: the caveat behind a budget-forced
                # completion=False must be reconstructable from the row
                # alone, not just inferable from terminal_status=="budget-
                # exceeded" -- `budget_exceeded` is recorded regardless of
                # whether it was the DECIDING factor (e.g. still True here
                # even on an already-killed/infra-error row, for audit),
                # alongside the configured budgets it was checked against
                # (the observed values are `steps`/`tokens_completion`
                # above -- compare directly against these two).
                "budget_exceeded": budget_exceeded,
                "budget_tokens_completion": budget_tokens,
                "budget_steps": budget_steps,
            },
            timing_authoritative=False,
            status="ok", tags=[])

        # SCORE-INDEPENDENCE (Wave 1a item 4): `metrics.completion` above is
        # derived ONLY from run_oracle's verdict + trace.terminal_status +
        # the budget check -- this module never imports
        # llmtest.harness.failure_class and classify_first_failure is never
        # called here, so a row's completion can never be influenced by a
        # classifier verdict/label (that pass runs separately and later,
        # via scripts/classify_b8_local.py, into a SIBLING store -- see its
        # own module docstring). This assertion is a live regression guard,
        # not a no-op: it fails the moment a future edit starts stuffing a
        # classifier label into this same metrics dict instead.
        assert "first_failure_class" not in row.metrics

        # Trace persistence (task-b8classify): the full `Trace` (every
        # event, not just the summary counts already in `metrics` above) is
        # written to artifacts/b8_traces/<row_id>.json -- `row.row_id` is
        # already final at this point (schema.ResultRow.new computed it
        # from the row's own fields, the same inputs `_full_condition`
        # above just fed into `condition`), so the trace file and the row
        # that references it can never disagree on which row_id they
        # belong to. `response_meta.trace_ref` is set AFTER `ResultRow.new`
        # returns -- purely additive (response_meta plays no part in
        # row_id's preimage; schema.validate_row only requires it be a
        # dict) -- so this cannot change row_id or break the schema/
        # validator. `scripts/classify_b8_local.py` (the classify pass)
        # reads this back via `Trace.from_dict(json.loads(...))` to
        # reconstruct the exact Trace `classify_first_failure` needs; a
        # stored row's own `metrics` never carried enough (no per-event
        # detail, only aggregate counts).
        trace_relpath = f"b8_traces/{row.row_id}.json"
        trace_path = root / "artifacts" / trace_relpath
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(persisted_trace.to_dict(), sort_keys=True), encoding="utf-8")
        row.response_meta["trace_ref"] = trace_relpath

        return [row.to_dict()]
