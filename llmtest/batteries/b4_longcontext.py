"""Battery 4 -- long context (TESTPLAN 5.4). NIAH single/multi-needle + multi-hop
retrieval over synthetic documents assembled at execute() time (never stored as a
literal 256k-token fixture); KV-quant (f16/q8/q4) x ctx-tier (16k/64k/128k/256k)
sweep. Scored deterministically -- needle recall is checkable, needs_judging=False
on every row."""
from __future__ import annotations

import hashlib
from pathlib import Path

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register
from llmtest.batteries.b4_fixtures import (build_document, check_needle_signals,
                                           load_longcontext_tasks)

# condition "kv" short label (matches suite.yaml condition_vocab.kv) -> the literal
# llama.cpp KV-cache dtype string ServerManager.request_endpoint()/
# compose_fork_flags() expect for -ctk/-ctv.
KV_DTYPE = {"f16": "f16", "q8": "q8_0", "q4": "q4_0"}


def ctx_label(ctx_tokens: int) -> str:
    """262144 -> '256k'. Every configured B4 ctx value is an exact KiB multiple."""
    return f"{ctx_tokens // 1024}k"


def tiers_for_model(ctx_tiers: list[int], claimed_ctx: int) -> list[int]:
    """Ctx tiers this model actually claims support for (TESTPLAN 5.4: 'ctx tiers
    ... where the model claims support' -- tiers above claimed_ctx are dropped for
    that model). If EVERY configured tier exceeds claimed_ctx, the model's own max
    rides as a single substituted tier instead of the model getting zero B4 rows
    (TESTPLAN: capped models are 'tested at their max ... tagged
    fits-short-context, not skipped'). No roster model currently triggers the
    substitution branch (every claimed_ctx >= the smallest configured tier); it
    exists so a future short-context intake addition is never silently dropped."""
    kept = [t for t in ctx_tiers if t <= claimed_ctx]
    return kept if kept else [claimed_ctx]


def arm_fits_estimate(model: dict, tiers_cfg: dict, kv_short: str, ctx_tokens: int,
                      tier: str = "T1") -> bool:
    """Estimated physical VRAM fit for ONE (kv, ctx) arm at its ACTUAL requested
    context length. registry.fits() always checks the fixed 128k kv_floor_ctx
    regardless of the caller's ctx -- it answers "does this artifact belong on
    this tier at all" (a tier-PLACEMENT question, TESTPLAN 3.1). This answers "does
    this specific B4 sweep point boot" (a planning question), by reusing the exact
    same kv_bytes_per_token table + hybrid-linear-attention 0.25x discount so the
    two never silently diverge on the underlying model math.

    `tier` is which physical card the sweep is being PLANNED for, and it defaults to
    T1 (the 24GB laptop) so existing behaviour and tests are unchanged. It must be set
    to the card actually being used, because it decides which arms exist at all: on T1
    every tier above 16k prunes away, and a 57.6GB model like laguna-s-2.1 gets ZERO
    arms and silently contributes no B4 rows. The frozen roster's B4 rows were produced
    on an RTX PRO 6000 (results/sessions.jsonl records
    hardware_sku=rtx-pro-6000, measured_usable_vram_gb=94.0), i.e. T3 - which is why
    they contain 64k/128k/256k arms that T1 planning cannot produce. Set it via
    suite.yaml b4.plan_tier when running on a bigger card.
    """
    t = tiers_cfg["tiers"][tier]
    usable = float(t["usable_gb"])
    weights = float(model["weights_gb"])
    overhead = float(tiers_cfg["runtime_overhead_gb"])
    kv_dtype = KV_DTYPE[kv_short]
    kv_per_tok = (model.get("kv_bytes_per_token") or {}).get(kv_dtype) \
        or tiers_cfg["kv_bytes_per_token"][kv_dtype]
    if model.get("arch", {}).get("hybrid_linear_attn"):
        kv_per_tok = kv_per_tok * 0.25
    kv_gb = kv_per_tok * ctx_tokens / (1024 ** 3)
    return weights + kv_gb + overhead <= usable


def model_arms(model_id: str, m: dict, b4cfg: dict, tiers_cfg: dict) -> dict[tuple[int, str], list[str]]:
    """(ctx_tokens, kv_short) -> tags, for ONE model. Two sweep dimensions union:

    - STANDARD capability sweep (TESTPLAN 5.4 "Capability" bullet): kv=standard_kv
      across claimed_ctx-filtered tiers, PRUNED to arms that pass
      arm_fits_estimate -- keeps the full-roster grid from planning launches that
      would never boot.
    - KV-QUANT QUALITY sweep (TESTPLAN 5.4 "KV-quant quality cost" bullet):
      kv_sweep_models get the full kv_sweep across claimed_ctx-filtered tiers, and
      the kv_spot_check model additionally gets its named (model, ctx_tiers,
      kv_sweep) point -- NEITHER is fit-pruned (TESTPLAN explicitly names these
      arms; empirically discovering "does q4 KV survive at 256k" is the point of
      the sweep). An arm arm_fits_estimate predicts won't fit is still planned but
      tagged fits-short-context as an advisory instead of being dropped.

    An arm covered by BOTH dimensions (e.g. the primary model's kv=standard_kv
    point) is treated as unpruned -- the quality-sweep membership wins.
    """
    claimed_ctx = m["claimed_ctx"]
    ctx_tiers = b4cfg["ctx_tiers"]
    standard_kv = b4cfg["standard_kv"]
    kv_sweep_models = set(b4cfg.get("kv_sweep_models") or [])
    spot = b4cfg.get("kv_spot_check") or {}
    configured_tiers = set(ctx_tiers) | set(spot.get("ctx_tiers", []))
    eff_tiers = tiers_for_model(ctx_tiers, claimed_ctx)

    unpruned: dict[tuple[int, str], bool] = {}
    for ctx_tokens in eff_tiers:
        unpruned.setdefault((ctx_tokens, standard_kv), False)

    if model_id in kv_sweep_models:
        for kv in b4cfg["kv_sweep"]:
            for ctx_tokens in eff_tiers:
                unpruned[(ctx_tokens, kv)] = True

    if model_id == spot.get("model"):
        for kv in spot.get("kv_sweep", []):
            for ctx_tokens in tiers_for_model(spot.get("ctx_tiers", []), claimed_ctx):
                unpruned[(ctx_tokens, kv)] = True

    result: dict[tuple[int, str], list[str]] = {}
    for (ctx_tokens, kv_short), ride_through in unpruned.items():
        fits_ok = arm_fits_estimate(m, tiers_cfg, kv_short, ctx_tokens,
                                    b4cfg.get("plan_tier", "T1"))
        if not ride_through and not fits_ok:
            continue                       # standard-sweep arm: physically pruned
        tags = set()
        if not fits_ok:
            tags.add("fits-short-context")
        if ctx_tokens not in configured_tiers:
            tags.add("fits-short-context")  # substituted model-max row
        result[(ctx_tokens, kv_short)] = sorted(tags)
    return result


def _fixture_sha_dir() -> str:
    # Sanctioned exception (mirrors b1_business.preflight Finding 1): B4 selftest
    # rows aren't tied to one task file, so there's no single content hash to use.
    return hashlib.sha256(b"b4_longcontext").hexdigest()


@register
class B4LongContext(Battery):
    id = 4

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        order = cfg.suite["condition_order"]
        sv = cfg.suite["suite_version"]
        b4cfg = cfg.suite["b4"]
        n_runs = b4cfg["n_runs"]

        tasks = load_longcontext_tasks(cfg.root)
        items = []

        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            if m.get("role") == "quant-arm":
                continue
            if str(m.get("local_path", "")).startswith("TO-"):
                continue

            arms = model_arms(model_id, m, b4cfg, cfg.tiers)
            if not arms:
                continue

            for task in tasks:
                task_id = f"b4.{task.id}"
                fixture_sha = task.fixture_sha

                for (ctx_tokens, kv_short), tags in sorted(arms.items()):
                    condition = schema.canonical_condition(
                        {"runtime": "fork", "spec": "ngram32", "kv": kv_short,
                         "ctx": ctx_label(ctx_tokens), "cond": "B4"}, order)

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
                            quant_sha256=m["provenance"]["sha256"], battery=4,
                            task_id=task_id, fixture_sha=fixture_sha,
                            condition=condition, run_n=run_n)

                        items.append(WorkItem(
                            row_id=rid, model_id=model_id, battery=4,
                            task_id=task_id, condition=condition, run_n=run_n,
                            payload={
                                "model": m,
                                "fixture_sha": fixture_sha,
                                "suite_version": sv,
                                "kind": task.kind,
                                "filler_template": task.filler_template,
                                "needles": task.needles,
                                "question": task.question,
                                "signals": task.signals,
                                "ctx_tokens": ctx_tokens,
                                "kv_short": kv_short,
                                "tags": tags,
                            }))

        return items

    def preflight(self, ctx) -> list[dict]:
        """selftest rows (TESTPLAN 7.4): (1) fixtures load, (2) the document
        BUILDER reaches ~declared length at every configured ctx tier ('corpora
        exist at declared lengths', translated to a build-at-execute-time design
        -- there is no literal 256k file whose existence to check, so this checks
        the mechanism that stands in for one)."""
        rows = []
        order = ctx.cfg.suite["condition_order"]
        sv = ctx.cfg.suite["suite_version"]
        b4cfg = ctx.cfg.suite["b4"]
        model_info = ctx.cfg.registry["models"].get("gpt-oss-20b", {})

        tasks = load_longcontext_tasks(ctx.root)
        fixture_sha = _fixture_sha_dir()
        condition = schema.canonical_condition({"cond": "SELFTEST"}, order)

        if not tasks:
            rows.append(schema.ResultRow.new(
                suite_version=sv, model_id="selftest",
                hf_repo=model_info.get("hf_repo", "N/A"),
                quant_file=model_info.get("quant_file", "N/A"),
                quant_sha256="0" * 64, tier="selftest", battery=4,
                task_id="b4.selftest.fixtures", fixture_sha=fixture_sha,
                condition=condition, run_n=1, session_id="selftest",
                status="error", error_detail="suite/b4_longcontext has no tasks",
                tags=["selftest"]).to_dict())
            return rows

        rows.append(schema.ResultRow.new(
            suite_version=sv, model_id="selftest",
            hf_repo=model_info.get("hf_repo", "N/A"),
            quant_file=model_info.get("quant_file", "N/A"),
            quant_sha256="0" * 64, tier="selftest", battery=4,
            task_id="b4.selftest.fixtures", fixture_sha=fixture_sha,
            condition=condition, run_n=1, session_id="selftest",
            status="ok", metrics={"n_tasks": len(tasks)},
            tags=["selftest"]).to_dict())

        sample = tasks[0]
        reserve = b4cfg.get("reserve_tokens", 1024)
        for ctx_tokens in b4cfg["ctx_tiers"]:
            target_tokens = max(1, ctx_tokens - reserve)
            doc = build_document(sample.filler_template, target_tokens,
                                 sample.needles, sample.question)
            approx_tokens = len(doc) // 4
            ok = approx_tokens >= target_tokens * 0.9
            label = ctx_label(ctx_tokens)
            tier_condition = schema.canonical_condition(
                {"cond": "SELFTEST", "ctx": label}, order)
            tier_fixture_sha = hashlib.sha256(f"b4-corpus-{ctx_tokens}".encode()).hexdigest()
            rows.append(schema.ResultRow.new(
                suite_version=sv, model_id="selftest",
                hf_repo=model_info.get("hf_repo", "N/A"),
                quant_file=model_info.get("quant_file", "N/A"),
                quant_sha256="0" * 64, tier="selftest", battery=4,
                task_id=f"b4.selftest.corpus.{label}", fixture_sha=tier_fixture_sha,
                condition=tier_condition, run_n=1, session_id="selftest",
                status="ok" if ok else "error",
                error_detail=None if ok else
                    f"built doc ~{approx_tokens} tokens, wanted >= {int(target_tokens * 0.9)}",
                metrics={"approx_tokens": approx_tokens, "target_tokens": target_tokens},
                tags=["selftest"]).to_dict())

        return rows

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        cfg = ctx.cfg
        b4cfg = cfg.suite["b4"]
        model = item.payload["model"]
        ctx_tokens = item.payload["ctx_tokens"]
        kv_short = item.payload["kv_short"]
        reserve = b4cfg.get("reserve_tokens", 1024)
        target_tokens = max(1, ctx_tokens - reserve)

        document = build_document(item.payload["filler_template"], target_tokens,
                                  item.payload["needles"], item.payload["question"])

        endpoint = ctx.server_manager().request_endpoint(
            item.model_id, ctx=ctx_tokens, kv=KV_DTYPE[kv_short],
            timing_authoritative=False)

        max_tokens = b4cfg["max_tokens"]
        response = endpoint.chat([{"role": "user", "content": document}],
                                 max_tokens=max_tokens, temperature=None)
        text = response["choices"][0]["message"]["content"] or ""  # null on reasoning models via proxies

        det_checks = check_needle_signals(text, item.payload["signals"])
        n_signals = len(det_checks)
        n_pass = sum(1 for v in det_checks.values() if v.get("pass"))

        artifacts_root = (ctx.root / "artifacts" / "b4") if hasattr(ctx, "root") \
            else (Path("artifacts") / "b4")
        artifacts_root.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts_root / f"{item.row_id}.txt"
        artifact_path.write_text(text, encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        row = schema.ResultRow.new(
            suite_version=item.payload["suite_version"], model_id=item.model_id,
            hf_repo=model.get("hf_repo", ""), quant_file=model.get("quant_file", ""),
            quant_sha256=model["provenance"]["sha256"], tier="T1", battery=4,
            task_id=item.task_id, fixture_sha=item.payload["fixture_sha"],
            condition=item.condition, run_n=item.run_n,
            session_id=endpoint.session_id,
            sampling={"temp": "runtime-default", "max_tokens": max_tokens},
            det_checks=det_checks,
            needs_judging=False,
            metrics={"needle_recall": (n_pass / n_signals) if n_signals else 0.0,
                    "prompt_chars": len(document)},
            timing_authoritative=False,
            artifacts={"response": {"sha256": artifact_sha,
                                    "relpath": f"b4/{item.row_id}.txt"}},
            status="ok",
            tags=list(item.payload.get("tags", [])))

        if response.get("timings"):
            row.response_meta.update({
                "predicted_n": response["timings"].get("predicted_n"),
                "predicted_per_second": response["timings"].get("predicted_per_second"),
                "prompt_n": response["timings"].get("prompt_n"),
                "prompt_per_second": response["timings"].get("prompt_per_second"),
            })

        return [row.to_dict()]
