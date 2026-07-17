from pathlib import Path
import pytest
from llmtest.registry import load_config
from llmtest.server import compose_fork_flags, normalize_config

ROOT = Path(__file__).resolve().parents[1]

def test_compose_fork_flags_standard_plus_overlay():
    cfg = load_config(ROOT)
    flags = compose_fork_flags(cfg, ctx=8192, parallel=1, kv="q8_0", overlay=None)
    assert "--spec-type ngram-mod" in flags and "-c 8192" in flags
    assert "-ctk q8_0" in flags and "-ctv q8_0" in flags
    off = compose_fork_flags(cfg, ctx=8192, parallel=4, kv="f16", overlay={"spec": "off"})
    assert "--spec-type none" in off and "-np 4" in off and "-ctk" not in off

def test_normalize_config_cross_runtime_shape():
    n = normalize_config(runtime="fork", ctx=8192, kv="q8_0",
                         spec="ngram32", parallel=2, flash_attn=True)
    assert n == {"ctx": 8192, "kv_dtype": "q8_0", "flash_attn": True,
                 "spec_type": "ngram32", "spec_params": {"n_match": 32}, "parallel": 2}

def test_never_below_nmatch_16_guard():
    cfg = load_config(ROOT)
    with pytest.raises(ValueError):
        compose_fork_flags(cfg, ctx=8192, parallel=1, kv="q8_0",
                           overlay={"spec": "ngram8"})
