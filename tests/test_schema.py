import pytest
from llmtest import schema

ORDER = ["runtime", "spec", "kv", "ctx", "cond", "conc"]

def test_condition_is_order_canonical():
    a = schema.canonical_condition({"conc": 8, "runtime": "fork", "cond": "PEAK"}, ORDER)
    b = schema.canonical_condition({"runtime": "fork", "cond": "PEAK", "conc": 8}, ORDER)
    assert a == b == "runtime=fork;cond=PEAK;conc=8"

def test_condition_rejects_unknown_key():
    with pytest.raises(ValueError):
        schema.canonical_condition({"bogus": 1}, ORDER)

def test_row_id_stable_and_sensitive():
    base = dict(suite_version="suite-v2.0.0", model_id="gpt-oss-20b",
                quant_sha256="a" * 64, battery=5, task_id="b5.decode",
                fixture_sha="f" * 64, condition="runtime=fork;cond=PEAK", run_n=1)
    r1 = schema.compute_row_id(**base)
    assert r1 == schema.compute_row_id(**base)          # stable
    assert r1 != schema.compute_row_id(**{**base, "run_n": 2})  # sensitive

def test_validate_row_catches_missing_and_bad_fields():
    row = schema.ResultRow.new(
        suite_version="suite-v2.0.0", model_id="gpt-oss-20b",
        hf_repo="unsloth/gpt-oss-20b-GGUF", quant_file="gpt-oss-20b-F16.gguf",
        quant_sha256="a" * 64, tier="T1", battery=5, task_id="b5.decode",
        fixture_sha="f" * 64, condition="runtime=fork;cond=PEAK", run_n=1,
        session_id="s1",
    ).to_dict()
    assert schema.validate_row(row) == []
    bad = dict(row); bad.pop("row_id"); assert "row_id" in " ".join(schema.validate_row(bad))
    bad2 = dict(row); bad2["timing_authoritative"] = "yes"
    assert any("timing_authoritative" in e for e in schema.validate_row(bad2))
