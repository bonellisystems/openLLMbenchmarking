from llmtest.batteries.b5_serving import peak_metrics, concurrency_metrics, build_sustained_prompt


def test_peak_metrics_maps_fork_timings():
    t = {"predicted_per_second": 150.5, "prompt_per_second": 2000.0,
         "predicted_n": 800, "prompt_n": 30, "predicted_ms": 5314.0}
    m = peak_metrics(t, ttft_ms=210.0)
    assert m["decode_tps"] == 150.5 and m["pp_tps"] == 2000.0
    assert m["ttft_ms"] == 210.0 and m["tokens_out"] == 800


def test_concurrency_metrics_aggregates():
    per = [{"predicted_n": 400, "predicted_per_second": 90.0},
           {"predicted_n": 400, "predicted_per_second": 88.0}]
    m = concurrency_metrics(per, elapsed_s=5.0)
    assert m["aggregate_tps"] == 160.0            # 800 tokens / 5 s
    assert m["per_stream_tps_mean"] == 89.0
    assert m["streams_ok"] == 2


def test_sustained_prompt_hits_target_length():
    p = build_sustained_prompt("word " * 40, 32000, "Q?")
    assert len(p) // 4 >= 30000                   # ~4 chars/token heuristic


def test_conditions_extra_runtimes_gate():
    from llmtest.batteries.b5_serving import _conditions
    order = ["runtime", "spec", "kv", "ctx", "cond", "conc"]
    base = _conditions(order)
    assert not any("ollama" in c for c in base)
    extra = _conditions(order, extra_runtimes=True)
    assert "runtime=ollama;kv=q8;cond=PEAK" in extra
    assert "runtime=ollama;kv=q8;cond=SUSTAINED32K" in extra
    assert len(extra) == len(base) + 2


def test_concurrent_worker_failure_raises(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace
    import pytest as _pytest
    from llmtest.batteries.b5_serving import B5Serving
    calls = {"n": 0}
    class FlakyHandle:
        session_id = "s-x"
        def chat(self, messages, max_tokens):
            with threading.Lock():
                calls["n"] += 1
                if calls["n"] % 2 == 0:
                    raise RuntimeError("boom")
            return {"timings": {"predicted_n": 10, "predicted_per_second": 50.0}}
    class FakeMgr:
        def request_endpoint(self, *a, **k):
            return FlakyHandle()
    ctx = SimpleNamespace(cfg=None, server_manager=lambda: FakeMgr())
    # build a minimal ctx.cfg with root pointing at repo so fixture loads
    from pathlib import Path
    from llmtest.registry import load_config
    ctx.cfg = load_config(Path(".").resolve())
    item = SimpleNamespace(model_id="gpt-oss-20b", task_id="b5.serving",
                           condition="runtime=fork;spec=ngram32;kv=q8;cond=PEAK;conc=4",
                           run_n=1,
                           payload={"model": ctx.cfg.registry["models"]["gpt-oss-20b"],
                                    "fixture_sha": "f"*64,
                                    "suite_version": "suite-v2.0.0-shakedown"})
    with _pytest.raises(RuntimeError, match="streams failed"):
        B5Serving().execute(item, ctx)
