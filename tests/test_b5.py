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
