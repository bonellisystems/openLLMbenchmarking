from pathlib import Path
from llmtest.registry import load_config, fits

ROOT = Path(__file__).resolve().parents[1]

def test_config_loads_all_six_models():
    cfg = load_config(ROOT)
    assert set(cfg.registry["models"]) >= {
        "gpt-oss-20b", "qwen3.6-35b-a3b", "gemma-4-26b-a4b",
        "ornith-1.0-35b", "qwen3.6-27b-dense", "qwen3-coder-30b"}
    assert cfg.suite["condition_order"][0] == "runtime"

def test_fits_small_model_t1():
    cfg = load_config(ROOT)
    r = fits(cfg.registry["models"]["gpt-oss-20b"], cfg.tiers, "q8_0", tier="T1")
    assert r.fits is True

def test_fits_flags_short_context_not_reject():
    cfg = load_config(ROOT)
    fat = dict(cfg.registry["models"]["qwen3.6-27b-dense"], weights_gb=21.5)
    r = fits(fat, cfg.tiers, "f16", tier="T1")
    assert r.fits is True and r.fits_short_context is True
