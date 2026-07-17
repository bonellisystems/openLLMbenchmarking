from types import SimpleNamespace
from llmtest import run_cmd, schema
from llmtest.store import Store

def _args(**kw):
    base = dict(suite="smoke", model=None, battery=5, task_id=None,
                condition=None, force=False, keep_server=False, debug=False)
    base.update(kw)
    return SimpleNamespace(**base)

def _row(run_n=1, task="b1.x"):
    return schema.ResultRow.new(
        suite_version="suite-v2.0.0", model_id="m", hf_repo="o/r",
        quant_file="q", quant_sha256="a"*64, tier="T1", battery=5,
        task_id=task, fixture_sha="f"*64, condition="cond=PEAK",
        run_n=run_n, session_id="s")

class PreflightFailBattery:
    id = 5
    def preflight(self, ctx):
        r = _row(task="b5.selftest")
        d = r.to_dict(); d["status"] = "error"; d["tags"] = ["selftest"]
        d["row_id"] = schema.compute_row_id(
            suite_version=d["suite_version"], model_id=d["model_id"],
            quant_sha256=d["quant_sha256"], battery=5, task_id="b5.selftest",
            fixture_sha=d["fixture_sha"], condition=d["condition"], run_n=1)
        return [d]
    def plan(self, cfg, store, model_filter=None, force=False):
        raise AssertionError("plan() must not run when preflight fails")
    def execute(self, item, ctx):
        return []

class ForceBattery:
    id = 5
    def preflight(self, ctx):
        return []
    def plan(self, cfg, store, model_filter=None, force=False):
        existing = [r["run_n"] for r in store.iter_rows() if r["task_id"] == "b1.x"]
        n = (max(existing) + 1) if (force and existing) else 1
        row = _row(run_n=n)
        return [SimpleNamespace(row_id=row.row_id, model_id="m", task_id="b1.x",
                                condition="cond=PEAK", run_n=n,
                                payload={"row": row})]
    def execute(self, item, ctx):
        return [item.payload["row"].to_dict()]

def test_preflight_failure_aborts_battery(tmp_path, monkeypatch):
    monkeypatch.setattr(run_cmd, "_get_battery", lambda i: PreflightFailBattery())
    monkeypatch.setattr(run_cmd, "_results_dir", lambda root: tmp_path)
    assert run_cmd.run_run(_args()) == 1
    rows = list(Store(tmp_path).iter_rows())
    assert len(rows) == 1 and "selftest" in rows[0]["tags"]

def test_force_bumps_run_n_appends_new_row(tmp_path, monkeypatch):
    monkeypatch.setattr(run_cmd, "_get_battery", lambda i: ForceBattery())
    monkeypatch.setattr(run_cmd, "_results_dir", lambda root: tmp_path)
    assert run_cmd.run_run(_args()) == 0
    assert run_cmd.run_run(_args(force=True)) == 0
    runs = sorted(r["run_n"] for r in Store(tmp_path).iter_rows())
    assert runs == [1, 2]
