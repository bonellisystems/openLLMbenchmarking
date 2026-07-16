import pytest
from llmtest import schema
from llmtest.store import Store, SchemaError

def _row(run_n=1):
    return schema.ResultRow.new(
        suite_version="suite-v2.0.0", model_id="m", hf_repo="org/r",
        quant_file="q.gguf", quant_sha256="a"*64, tier="T1", battery=5,
        task_id="b5.x", fixture_sha="f"*64, condition="cond=PEAK",
        run_n=run_n, session_id="s1").to_dict()

def test_append_dedupes_by_row_id(tmp_path):
    s = Store(tmp_path)
    assert s.append(_row()) is True
    assert s.append(_row()) is False
    assert len(list(s.iter_rows())) == 1
    assert (tmp_path / "rows-suite-v2.0.0.jsonl").exists()

def test_append_rejects_invalid(tmp_path):
    bad = _row(); bad["status"] = "nope"
    with pytest.raises(SchemaError):
        Store(tmp_path).append(bad)

def test_resume_index_reads_all_shards(tmp_path):
    s = Store(tmp_path)
    r1, r2 = _row(1), _row(2)
    s.append(r1); s.append(r2)
    assert Store(tmp_path).existing_row_ids() == {r1["row_id"], r2["row_id"]}

def test_jsonl_lines_end_lf_only(tmp_path):
    s = Store(tmp_path)
    s.append(_row()); s.append_session({"session_id": "s1"})
    for f in list(tmp_path.glob("rows-*.jsonl")) + [tmp_path / "sessions.jsonl"]:
        raw = f.read_bytes()
        assert b"\r" not in raw, f"CRLF found in {f.name}"
