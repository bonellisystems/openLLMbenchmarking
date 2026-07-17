from llmtest.canary import evaluate_canary

BAND = {"min_speedup": 3.5, "max_speedup": 7.0}

def test_canary_pass_inside_band():
    ok, msg = evaluate_canary(100.0, 500.0, BAND)
    assert ok and "5.00x" in msg

def test_canary_fail_below_band():
    ok, msg = evaluate_canary(100.0, 200.0, BAND)
    assert not ok and "2.00x" in msg
