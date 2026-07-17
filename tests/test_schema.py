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


def _judgment(**overrides):
    d = dict(schema_version=schema.SCHEMA_VERSION, packet_id="p1", judge_id="claude",
              judge_model_pin="pin-x", judge_cli_version="v1", letter="A",
              model_id="model-a", score=7, reason="solid", rank=1, ts="2026-07-17T00:00:00+00:00",
              status="ok")
    d.update(overrides)
    return d


def test_validate_judgment_happy_path_ok_and_error():
    assert schema.validate_judgment(_judgment()) == []
    err = _judgment(letter="-", model_id=None, score=None, reason="timeout", rank=None,
                     status="error")
    assert schema.validate_judgment(err) == []


def test_validate_judgment_missing_field():
    bad = _judgment(); bad.pop("score")
    assert any("score" in e for e in schema.validate_judgment(bad))


def test_validate_judgment_score_required_for_ok_status():
    bad = _judgment(score=None)
    assert any("None" in e or "int" in e for e in schema.validate_judgment(bad))


def test_validate_judgment_score_must_be_none_on_error():
    bad = _judgment(status="error", letter="-", score=5)
    assert any("None" in e for e in schema.validate_judgment(bad))


def test_validate_judgment_score_out_of_range():
    bad = _judgment(score=11)
    assert any("range" in e for e in schema.validate_judgment(bad))


def test_validate_judgment_bool_score_rejected():
    bad = _judgment(score=True)
    assert schema.validate_judgment(bad) != []


def test_validate_judgment_bad_status_rejected():
    bad = _judgment(status="nope")
    assert any("status" in e for e in schema.validate_judgment(bad))


def test_validate_judgment_letter_status_invariant():
    # letter == "-" must imply status == "error"
    bad_ok_dash = _judgment(letter="-", status="ok", score=7)
    assert any("-" in e for e in schema.validate_judgment(bad_ok_dash))

    # status == "error" must imply letter == "-"
    bad_error_letter = _judgment(letter="A", status="error", score=None)
    assert any("-" in e for e in schema.validate_judgment(bad_error_letter))
