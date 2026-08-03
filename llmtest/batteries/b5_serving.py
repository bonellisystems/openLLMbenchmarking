"""Battery 5 — throughput & serving (TESTPLAN 5.5). Owns server lifecycle; mints timing_authoritative rows."""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import yaml

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register

_FIXTURE = Path("suite/b5_serving/prompts.yaml")


def _fixture_sha(root: Path) -> str:
    return hashlib.sha256((root / _FIXTURE).read_bytes()).hexdigest()


def peak_metrics(timings: dict, *, ttft_ms: float) -> dict:
    return {"decode_tps": float(timings.get("predicted_per_second", 0.0)),
            "pp_tps": float(timings.get("prompt_per_second", 0.0)),
            "ttft_ms": float(ttft_ms),
            "tokens_out": int(timings.get("predicted_n", 0)),
            "tokens_in": int(timings.get("prompt_n", 0)),
            "n_drafted": timings.get("draft_n"),
            "n_accepted": timings.get("draft_n_accepted"),
            "accept_rate": (timings.get("draft_n_accepted", 0) / timings["draft_n"])
            if timings.get("draft_n") else None}


def concurrency_metrics(per_stream: list[dict], *, elapsed_s: float) -> dict:
    total = sum(int(t.get("predicted_n", 0)) for t in per_stream)
    speeds = [float(t.get("predicted_per_second", 0.0)) for t in per_stream]
    return {"aggregate_tps": total / elapsed_s if elapsed_s else 0.0,
            "per_stream_tps_mean": sum(speeds) / len(speeds) if speeds else 0.0,
            "streams_ok": len(per_stream)}


def build_sustained_prompt(paragraph: str, target_tokens: int, question: str) -> str:
    approx_chars = target_tokens * 4
    body = (paragraph + "\n") * (approx_chars // max(len(paragraph), 1) + 1)
    return body[:approx_chars] + "\n\n" + question


def _conditions(order, extra_runtimes=False):
    def c(**kw):
        return schema.canonical_condition(kw, order)
    conds = [c(runtime="fork", spec="ngram32", kv="q8", cond="PEAK"),
             c(runtime="fork", spec="ngram32", kv="q8", cond="SUSTAINED32K"),
             c(runtime="fork", spec="off", kv="q8", cond="PEAK"),
             c(runtime="fork", spec="off", kv="q8", cond="SUSTAINED32K")]
    conds += [c(runtime="fork", spec="ngram32", kv="q8", cond="PEAK", conc=n)
              for n in (2, 4, 8, 16)]
    if extra_runtimes:
        # ollama: no spec key (ngram is fork-only); no vllm arms yet.
        conds += [c(runtime="ollama", kv="q8", cond="PEAK"),
                  c(runtime="ollama", kv="q8", cond="SUSTAINED32K")]
    return conds


@register
class B5Serving(Battery):
    id = 5

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        order = cfg.suite["condition_order"]
        fx = _fixture_sha(cfg.root)
        sv = cfg.suite["suite_version"]
        extra_runtimes = cfg.suite.get("b5_extra_runtimes", False)
        items = []
        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            if str(m.get("local_path", "")).startswith("TO-"):
                continue                    # artifact not on disk yet
            if m.get("role") == "api-reference":
                # B5 is timing-authoritative on LOCAL serving. Pointed at an API
                # (via a proxy) there are no llama timings, so peak_metrics would
                # return zeros and the row would still be written with
                # timing_authoritative=True - a fabricated benchmark number.
                # Every other battery has a role skip; B5 had none.
                continue
            for cond in _conditions(order, extra_runtimes=extra_runtimes):
                if force:
                    existing = [r["run_n"] for r in store.iter_rows()
                               if r["task_id"] == "b5.serving" and r["model_id"] == model_id
                               and r["condition"] == cond]
                    run_n = (max(existing) + 1) if existing else 1
                else:
                    run_n = 1
                rid = schema.compute_row_id(
                    suite_version=sv, model_id=model_id,
                    quant_sha256=m["provenance"]["sha256"], battery=5,
                    task_id="b5.serving", fixture_sha=fx,
                    condition=cond, run_n=run_n)
                items.append(WorkItem(row_id=rid, model_id=model_id, battery=5,
                                      task_id="b5.serving", condition=cond,
                                      run_n=run_n,
                                      payload={"model": m, "fixture_sha": fx,
                                               "suite_version": sv}))
        return items

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        cfg = ctx.cfg
        fixture = yaml.safe_load((cfg.root / _FIXTURE).read_text(encoding="utf-8"))
        parts = dict(pair.split("=") for pair in item.condition.split(";"))
        conc = int(parts.get("conc", 1))
        overlay = {"spec": "off"} if parts.get("spec") == "off" else None
        ctx_len = 36864 if parts["cond"] == "SUSTAINED32K" else 8192
        mgr = ctx.server_manager()
        handle = mgr.request_endpoint(item.model_id, runtime=parts.get("runtime", "fork"),
                                      flags_overlay=overlay, parallel=conc,
                                      ctx=ctx_len, kv="q8_0",
                                      timing_authoritative=True)
        actual_max_tokens = None
        if conc > 1:
            results, errors, lock = [], [], threading.Lock()
            actual_max_tokens = fixture["conc_max_tokens"]
            def worker():
                try:
                    d = handle.chat([{"role": "user", "content": fixture["conc_prompt"]}],
                                    max_tokens=fixture["conc_max_tokens"])
                except Exception as e:
                    with lock:
                        errors.append(e)
                    return
                with lock:
                    results.append(d.get("timings", {}))
            t0 = time.time()
            threads = [threading.Thread(target=worker) for _ in range(conc)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            if errors:
                raise RuntimeError(
                    f"{len(errors)}/{conc} concurrent streams failed: {errors[0]}")
            metrics = concurrency_metrics(results, elapsed_s=time.time() - t0)
            resp_meta = {"decode_tps": metrics["per_stream_tps_mean"],
                         "pp_tps": None, "ttft_ms": None,
                         "tokens_out": sum(int(r.get("predicted_n", 0)) for r in results)}
        else:
            if parts["cond"] == "SUSTAINED32K":
                prompt = build_sustained_prompt(fixture["sustained_filler_paragraph"],
                                                fixture["sustained_ctx_tokens"],
                                                fixture["sustained_question"])
                actual_max_tokens = fixture["sustained_max_tokens"]
            else:
                prompt = fixture["peak_prompt"]
                actual_max_tokens = fixture["peak_max_tokens"]
            t0 = time.time()
            d = handle.chat([{"role": "user", "content": prompt}],
                            max_tokens=actual_max_tokens)
            ttft = d.get("timings", {}).get("prompt_ms", (time.time() - t0) * 1000)
            metrics = resp_meta = peak_metrics(d.get("timings", {}), ttft_ms=ttft)
        m = item.payload["model"]
        row = schema.ResultRow.new(
            suite_version=item.payload["suite_version"], model_id=item.model_id,
            hf_repo=m["hf_repo"], quant_file=m["quant_file"],
            quant_sha256=m["provenance"]["sha256"], tier="T1", battery=5,
            task_id=item.task_id, fixture_sha=item.payload["fixture_sha"],
            condition=item.condition, run_n=item.run_n,
            session_id=handle.session_id,
            sampling={"temp": 0.0, "max_tokens": actual_max_tokens, "top_p": None, "seed": None},
            response_meta={k: v for k, v in resp_meta.items() if v is not None},
            metrics={k: v for k, v in metrics.items() if v is not None},
            timing_authoritative=True)
        return [row.to_dict()]
