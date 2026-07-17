from pathlib import Path
from types import SimpleNamespace
from llmtest import run_cmd
from llmtest.store import Store
from llmtest import schema

class FakeBattery:
    id = 5
    def plan(self, cfg, store, model_filter=None):
        row = schema.ResultRow.new(
            suite_version="suite-v2.0.0", model_id="m", hf_repo="o/r",
            quant_file="q", quant_sha256="a"*64, tier="T1", battery=5,
            task_id="b5.fake", fixture_sha="f"*64, condition="cond=PEAK",
            run_n=1, session_id="pending")
        return [SimpleNamespace(row_id=row.row_id, model_id="m",
                                task_id="b5.fake", condition="cond=PEAK",
                                run_n=1, payload={"row": row})]
    def execute(self, item, ctx):
        r = item.payload["row"]; r.session_id = "s-fake"; return [r.to_dict()]

def test_run_skips_done_items(tmp_path, monkeypatch):
    monkeypatch.setattr(run_cmd, "_get_battery", lambda i: FakeBattery())
    monkeypatch.setattr(run_cmd, "_results_dir", lambda root: tmp_path)
    args = SimpleNamespace(suite="smoke", model=None, battery=5, task_id=None,
                           condition=None, force=False, keep_server=False, debug=False)
    assert run_cmd.run_run(args) == 0
    assert len(list(Store(tmp_path).iter_rows())) == 1
    assert run_cmd.run_run(args) == 0                 # resume: nothing re-executed
    assert len(list(Store(tmp_path).iter_rows())) == 1


def test_force_rerun_fails_loudly_when_row_discarded(tmp_path, monkeypatch):
    monkeypatch.setattr(run_cmd, "_get_battery", lambda i: FakeBattery())
    monkeypatch.setattr(run_cmd, "_results_dir", lambda root: tmp_path)
    args = SimpleNamespace(suite="smoke", model=None, battery=5, task_id=None,
                           condition=None, force=False, keep_server=False, debug=False)
    assert run_cmd.run_run(args) == 0                  # first run writes the row
    forced = SimpleNamespace(**{**vars(args), "force": True})
    assert run_cmd.run_run(forced) == 1                # forced re-run must FAIL loudly
    from llmtest.store import Store
    assert len(list(Store(tmp_path).iter_rows())) == 1  # old row retained
