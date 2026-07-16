from llmtest.tables import render_serving_table

def _r(model, cond, tps):
    return {"model_id": model, "hf_repo": f"org/{model}", "condition": cond,
            "timing_authoritative": True, "status": "ok",
            "response_meta": {"decode_tps": tps, "pp_tps": 0, "ttft_ms": 0},
            "tags": []}

def test_serving_table_deterministic_and_authority_filtered():
    rows = [_r("b", "cond=PEAK", 100.15), _r("a", "cond=PEAK", 50.0),
            dict(_r("c", "cond=PEAK", 999.0), timing_authoritative=False)]
    out1 = render_serving_table(rows)
    out2 = render_serving_table(list(reversed(rows)))
    assert out1 == out2
    assert "org/a" in out1 and "org/b" in out1
    assert "999" not in out1
    assert "100.2" in out1     # fixed rounding
