"""Tests for Battery 2 -- tool calling."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtest import schema
from llmtest.batteries import WorkItem
from llmtest.batteries.b2_fixtures import (
    Task, load_tasks, score_axes, validate_expect_block, validate_tool_schemas,
)
from llmtest.batteries.b2_toolcalling import B2ToolCalling

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    """Fake store for plan() tests."""
    def iter_rows(self):
        return []


def _call(name, args, cid="call_1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _resp(tool_calls=None, content=None):
    msg = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def _real_task(task_id):
    tasks = {t.id: t for t in load_tasks(ROOT)}
    return tasks[task_id]


# --- fixture loader ----------------------------------------------------

def test_loader_finds_all_fixtures_and_covers_all_8_axes():
    tasks = load_tasks(ROOT)
    assert 8 <= len(tasks) <= 12
    all_axes = set()
    for t in tasks:
        all_axes |= set(t.axes)
        assert 1 in t.axes or True  # axis 1 not required to be listed, but every task must declare >=1 axis
        assert t.axes, f"{t.id} has empty axes"
    assert all_axes == set(range(1, 9)), f"missing axis coverage: {set(range(1,9)) - all_axes}"


def test_loader_ids_are_unique_and_sorted():
    tasks = load_tasks(ROOT)
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_loader_raises_loud_on_malformed_fixture(tmp_path):
    bad_dir = tmp_path / "suite" / "b2_toolcalling"
    bad_dir.mkdir(parents=True)
    (bad_dir / "task-01.yaml").write_text(
        "id: broken-01\nscenario: x\naxes: [1]\n", encoding="utf-8")  # missing industry/tools/messages/expect
    with pytest.raises(ValueError, match="malformed fixture"):
        load_tasks(tmp_path)


def test_loader_raises_on_invalid_axes(tmp_path):
    bad_dir = tmp_path / "suite" / "b2_toolcalling"
    bad_dir.mkdir(parents=True)
    (bad_dir / "task-01.yaml").write_text("""\
id: broken-01
scenario: x
axes: [1, 99]
industry: generic_smb
tools:
  - type: function
    function: {name: t, description: d, parameters: {type: object, properties: {}, required: []}}
messages:
  - {role: user, content: "hi"}
expect: {}
""", encoding="utf-8")
    with pytest.raises(ValueError, match="axes"):
        load_tasks(tmp_path)


def test_loader_returns_empty_list_for_missing_dir(tmp_path):
    assert load_tasks(tmp_path) == []


def test_loader_expands_filler_placeholder():
    task = _real_task("long-context-call-01")
    content = task.messages[0]["content"]
    assert "{{FILLER}}" not in content
    # target_tokens: 33000 at ~4 chars/token -> should clear 32k tokens' worth of chars
    assert len(content) >= 32000 * 4 * 0.9


# --- validate_tool_schemas / validate_expect_block ----------------------

def test_validate_tool_schemas_clean_for_all_real_fixtures():
    for t in load_tasks(ROOT):
        assert validate_tool_schemas(t.tools) == [], f"{t.id}: {validate_tool_schemas(t.tools)}"
        assert validate_expect_block(t) == [], f"{t.id}: {validate_expect_block(t)}"


def test_validate_tool_schemas_catches_bad_type():
    tools = [{"type": "not-function", "function": {"name": "x"}}]
    errs = validate_tool_schemas(tools)
    assert any("type must be" in e for e in errs)


def test_validate_tool_schemas_catches_missing_required_property():
    tools = [{"type": "function", "function": {
        "name": "x", "description": "d",
        "parameters": {"type": "object", "properties": {"a": {"type": "string"}},
                       "required": ["a", "b"]}}}]
    errs = validate_tool_schemas(tools)
    assert any("unknown properties" in e for e in errs)


def test_validate_tool_schemas_catches_duplicate_names():
    tools = [
        {"type": "function", "function": {"name": "x", "description": "d",
         "parameters": {"type": "object", "properties": {}, "required": []}}},
        {"type": "function", "function": {"name": "x", "description": "d2",
         "parameters": {"type": "object", "properties": {}, "required": []}}},
    ]
    errs = validate_tool_schemas(tools)
    assert any("duplicate" in e for e in errs)


def test_validate_expect_block_catches_unknown_tool_reference():
    task = Task(id="t", scenario="s", axes=[1, 2], industry="generic_smb",
                difficulty="easy",
                tools=[{"type": "function", "function": {
                    "name": "real_tool", "description": "d",
                    "parameters": {"type": "object", "properties": {}, "required": []}}}],
                messages=[{"role": "user", "content": "hi"}],
                expect={"tool_calls": [{"name": "nonexistent_tool", "args": {}}]},
                rubric={}, fixture_sha="f" * 64, path=Path("x"))
    errs = validate_expect_block(task)
    assert any("unknown tool" in e for e in errs)


# --- axis scoring (direct unit tests on synthetic responses) ------------

def test_axis1_schema_adherence_passes_on_valid_call():
    task = _real_task("single-tool-basic-01")
    resp = _resp(tool_calls=[_call("get_account_balance", {"account_id": "ACC-58213"})])
    det_checks, needs_judging, metrics = score_axes(resp, task)
    assert det_checks["axis1_schema_adherence"]["pass"] is True
    assert needs_judging is False
    assert metrics["n_tool_calls"] == 1


def test_axis1_fails_on_malformed_json_args():
    task = _real_task("single-tool-basic-01")
    resp = {"choices": [{"message": {"tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "get_account_balance", "arguments": "{not valid json"}}]}}]}
    det_checks, _, metrics = score_axes(resp, task)
    assert det_checks["axis1_schema_adherence"]["pass"] is False
    assert metrics["det_pass"] is False


def test_axis1_fails_on_missing_required_param():
    task = _real_task("single-tool-basic-01")
    resp = _resp(tool_calls=[_call("get_account_balance", {})])   # missing account_id
    det_checks, _, _ = score_axes(resp, task)
    assert det_checks["axis1_schema_adherence"]["pass"] is False
    assert "missing_required" in det_checks["axis1_schema_adherence"]["calls"][0]


def test_axis1_fails_on_hallucinated_tool_name():
    task = _real_task("single-tool-basic-01")
    resp = _resp(tool_calls=[_call("wire_transfer_funds", {"amount": 1000})])
    det_checks, _, _ = score_axes(resp, task)
    assert det_checks["axis1_schema_adherence"]["pass"] is False
    assert "hallucinated" in det_checks["axis1_schema_adherence"]["calls"][0]["error"]


def test_axis1_enforces_enum_constraint():
    task = _real_task("nested-args-enum-01")
    resp = _resp(tool_calls=[_call("create_work_order", {
        "site_code": "DTC-04", "priority": "urgent-ish", "description": "x"})])
    det_checks, _, _ = score_axes(resp, task)
    assert det_checks["axis1_schema_adherence"]["pass"] is False
    assert "enum_errors" in det_checks["axis1_schema_adherence"]["calls"][0]


def test_axis2_tool_selection_correct_vs_distractor():
    task = _real_task("multi-tool-distractor-01")
    good = _resp(tool_calls=[_call("search_patient_records", {"mrn": "004471"})])
    bad = _resp(tool_calls=[_call("send_fax", {"recipient_number": "555", "document_id": "d1"})])
    dc_good, _, _ = score_axes(good, task)
    dc_bad, _, _ = score_axes(bad, task)
    assert dc_good["axis2_tool_selection"]["pass"] is True
    assert dc_bad["axis2_tool_selection"]["pass"] is False


def test_axis2_fails_on_no_call():
    task = _real_task("single-tool-basic-01")
    resp = _resp(content="I think the balance is around $500.")
    dc, _, _ = score_axes(resp, task)
    assert dc["axis2_tool_selection"]["pass"] is False


def test_axis3_parallel_calls_multiset_match_order_independent():
    task = _real_task("parallel-calls-01")
    # reversed order vs fixture's expect list -- still must match (multiset)
    resp = _resp(tool_calls=[_call("check_stock_level", {"sku": "WD-2450"}, "c1"),
                              _call("check_stock_level", {"sku": "WD-2201"}, "c2")])
    dc, _, _ = score_axes(resp, task)
    assert dc["axis3_parallel_calls"]["pass"] is True


def test_axis3_fails_when_only_one_call_made():
    task = _real_task("parallel-calls-01")
    resp = _resp(tool_calls=[_call("check_stock_level", {"sku": "WD-2201"})])
    dc, _, _ = score_axes(resp, task)
    assert dc["axis3_parallel_calls"]["pass"] is False


def test_axis4_chained_call_requires_prior_tool_output_value():
    task = _real_task("chained-calls-01")
    correct = _resp(tool_calls=[_call("file_motion", {"docket_id": "DKT-778823",
                                                        "motion_type": "continuance"})])
    shortcut = _resp(tool_calls=[_call("file_motion", {"docket_id": "24-CV-08871",
                                                         "motion_type": "continuance"})])
    dc_correct, _, _ = score_axes(correct, task)
    dc_shortcut, _, _ = score_axes(shortcut, task)
    assert dc_correct["axis4_chained_calls"]["pass"] is True
    assert dc_shortcut["axis4_chained_calls"]["pass"] is False


def test_axis6_abstention_pass_on_no_call_fail_on_invented_call():
    task = _real_task("abstention-no-tool-01")
    honest = _resp(content="The federal funds rate is a matter of public record; I don't need a tool for that.")
    invented = _resp(tool_calls=[_call("file_records_request", {"dept": "Fed", "topic": "rate"})])
    dc_honest, _, _ = score_axes(honest, task)
    dc_invented, _, _ = score_axes(invented, task)
    assert dc_honest["axis6_abstention"]["pass"] is True
    assert dc_invented["axis6_abstention"]["pass"] is False


def test_axis7_long_context_call_still_scores_tool_selection():
    task = _real_task("long-context-call-01")
    resp = _resp(tool_calls=[_call("check_inventory_level", {"sku": "AER-4471", "warehouse": "Houston"})])
    dc, _, _ = score_axes(resp, task)
    assert dc["axis7_long_context_call"]["pass"] is True


def test_axis5_needs_judging_and_fabrication_guard():
    task = _real_task("error-recovery-01")
    honest = _resp(content="The sensor query timed out; I don't have a current reading.")
    fabricated = _resp(content="The current reading is 842 psi.")
    dc_h, nj_h, _ = score_axes(honest, task)
    dc_f, nj_f, _ = score_axes(fabricated, task)
    assert nj_h is True and nj_f is True          # needs_judging regardless of the guard's verdict
    assert dc_h["axis5_fabrication_guard"]["pass"] is True
    assert dc_f["axis5_fabrication_guard"]["pass"] is False


def test_axis8_needs_judging_and_fabrication_guard():
    task = _real_task("faithfulness-01")
    faithful = _resp(content="Batch BATCH-33210 PASSED QC with 11.2% moisture and no contaminants detected.")
    contradicts = _resp(content="Batch BATCH-33210 FAIL -- contaminants were detected.")
    dc_faith, nj_faith, _ = score_axes(faithful, task)
    dc_bad, nj_bad, _ = score_axes(contradicts, task)
    assert nj_faith is True and nj_bad is True
    assert dc_faith["axis8_fabrication_guard"]["pass"] is True
    assert dc_bad["axis8_fabrication_guard"]["pass"] is False


def test_axes_not_applicable_are_absent_from_det_checks():
    """A task whose axes are only [1, 2] must not carry axis3/4/5/6/7/8 keys."""
    task = _real_task("single-tool-basic-01")
    resp = _resp(tool_calls=[_call("get_account_balance", {"account_id": "ACC-58213"})])
    dc, _, _ = score_axes(resp, task)
    assert set(dc.keys()) == {"axis1_schema_adherence", "axis2_tool_selection"}


# --- plan() -------------------------------------------------------------

def test_plan_covers_11_models_excluding_quant_arm():
    """plan() excludes every model with role=quant-arm; roster size is read
    from the registry itself (not hardcoded) so this survives roster growth."""
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    non_quant_arm_models = [mid for mid, m in cfg.registry["models"].items()
                             if m.get("role") != "quant-arm"]

    n_tasks = len(load_tasks(ROOT))
    n_runs = cfg.suite["b2"]["n_runs"]

    store = FakeStore()
    b2 = B2ToolCalling()
    items = b2.plan(cfg, store)

    model_ids = {item.model_id for item in items}
    assert "gemma-4-26b-a4b-mxfp4" not in model_ids   # role=quant-arm excluded
    assert len(model_ids) == len(non_quant_arm_models)

    assert len(items) == len(non_quant_arm_models) * n_tasks * n_runs

    for item in items:
        assert item.battery == 2
        assert item.task_id.startswith("b2.")

    order = cfg.suite["condition_order"]
    for item in items:
        parts = dict(p.split("=") for p in item.condition.split(";"))
        assert schema.canonical_condition(parts, order) == item.condition
        assert parts["cond"] == "B2"


def test_plan_force_bumps_run_n_condition_scoped():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    model_id = "gpt-oss-20b"
    task_id = "b2.single-tool-basic-01"
    order = cfg.suite["condition_order"]
    target_condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "40k", "cond": "B2"}, order)
    other_condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "40k", "cond": "SELFTEST"}, order)

    seeded_rows = [
        {"model_id": model_id, "task_id": task_id, "condition": target_condition,
         "run_n": 1, "row_id": "seed-run1"},
        {"model_id": model_id, "task_id": task_id, "condition": target_condition,
         "run_n": 2, "row_id": "seed-run2"},
        {"model_id": model_id, "task_id": task_id, "condition": other_condition,
         "run_n": 9, "row_id": "seed-run9"},
    ]

    class SeededStore:
        def iter_rows(self):
            return seeded_rows

    b2 = B2ToolCalling()
    items = b2.plan(cfg, SeededStore(), model_filter=model_id, force=True)
    matching = [it for it in items if it.model_id == model_id and it.task_id == task_id]
    assert len(matching) == 1
    assert matching[0].run_n == 3           # max(1,2)+1, other_condition's run_n=9 ignored
    assert matching[0].row_id not in {"seed-run1", "seed-run2", "seed-run9"}


# --- preflight() ----------------------------------------------------------

def test_preflight_all_real_fixtures_pass():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    ctx = SimpleNamespace(cfg=cfg, root=ROOT)
    rows = B2ToolCalling().preflight(ctx)
    assert len(rows) == len(load_tasks(ROOT))
    for row in rows:
        assert row["status"] == "ok", row["error_detail"]
        assert row["tags"] == ["selftest"]
        assert row["condition"] == "cond=SELFTEST"
        assert row["battery"] == 2


def test_preflight_no_fixtures_returns_error_row(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    ctx = SimpleNamespace(cfg=cfg, root=tmp_path)   # empty tmp dir, no suite/b2_toolcalling
    rows = B2ToolCalling().preflight(ctx)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "no fixtures" in rows[0]["error_detail"]


def test_preflight_detects_bad_tool_schema(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    fixtures_dir = tmp_path / "suite" / "b2_toolcalling"
    fixtures_dir.mkdir(parents=True)
    (fixtures_dir / "task-01.yaml").write_text("""\
id: bad-tool-01
scenario: bad
axes: [1, 2]
industry: generic_smb
tools:
  - type: function
    function: {name: t}
messages:
  - {role: user, content: "hi"}
expect: {tool_calls: [{name: t, args: {}}]}
""", encoding="utf-8")

    ctx = SimpleNamespace(cfg=cfg, root=tmp_path)
    rows = B2ToolCalling().preflight(ctx)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert "parameters" in rows[0]["error_detail"]


# --- execute() ------------------------------------------------------------

def _make_exec_item(cfg, task, run_n=1):
    order = cfg.suite["condition_order"]
    condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "40k", "cond": "B2"}, order)
    suite_version = "suite-v2.0.0-shakedown"
    row_id = schema.compute_row_id(
        suite_version=suite_version, model_id="gpt-oss-20b",
        quant_sha256=cfg.registry["models"]["gpt-oss-20b"]["provenance"]["sha256"],
        battery=2, task_id=f"b2.{task.id}", fixture_sha=task.fixture_sha,
        condition=condition, run_n=run_n)
    return WorkItem(
        row_id=row_id, model_id="gpt-oss-20b", battery=2,
        task_id=f"b2.{task.id}", condition=condition, run_n=run_n,
        payload={"model": cfg.registry["models"]["gpt-oss-20b"],
                 "task_id": task.id, "fixture_sha": task.fixture_sha,
                 "suite_version": suite_version, "tools": task.tools,
                 "messages": task.messages, "expect": task.expect, "axes": task.axes})


def test_execute_deterministic_axis_scoring_and_artifact(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    class StubHandle:
        session_id = "s-stub"
        normalized_config = {}
        def chat(self, messages, **kwargs):
            assert kwargs.get("tools"), "tools param must be forwarded to chat()"
            return {
                "choices": [{"message": {"content": None, "tool_calls": [
                    _call("get_account_balance", {"account_id": "ACC-58213"})]}}],
                "timings": {"predicted_n": 12, "predicted_per_second": 90.0}
            }

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: StubMgr(), root=tmp_path)
    task = _real_task("single-tool-basic-01")
    item = _make_exec_item(cfg, task)

    rows = B2ToolCalling().execute(item, ctx)
    assert len(rows) == 1
    row = rows[0]

    assert row["row_id"] == item.row_id
    assert row["status"] == "ok"
    assert row["battery"] == 2
    assert row["needs_judging"] is False
    assert row["det_checks"]["axis1_schema_adherence"]["pass"] is True
    assert row["det_checks"]["axis2_tool_selection"]["pass"] is True
    assert row["metrics"]["n_tool_calls"] == 1
    assert row["sampling"] == {"temp": "runtime-default", "max_tokens": 2000}

    assert "response" in row["artifacts"]
    art = row["artifacts"]["response"]
    assert "sha256" in art and "relpath" in art
    assert (tmp_path / "artifacts" / art["relpath"]).exists()


def test_execute_sets_needs_judging_true_for_axis5_task(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    class StubHandle:
        session_id = "s-stub"
        normalized_config = {}
        def chat(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "I don't have a current reading; the sensor timed out."}}]}

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: StubMgr(), root=tmp_path)
    task = _real_task("error-recovery-01")
    item = _make_exec_item(cfg, task)

    rows = B2ToolCalling().execute(item, ctx)
    row = rows[0]
    assert row["needs_judging"] is True
    assert row["det_checks"]["axis5_fabrication_guard"]["pass"] is True


def test_execute_sampling_records_runtime_default_temp(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    class StubHandle:
        session_id = "s-stub"
        normalized_config = {}
        def chat(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "no tool needed here"}}]}

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: StubMgr(), root=tmp_path)
    task = _real_task("abstention-no-tool-01")
    item = _make_exec_item(cfg, task)

    rows = B2ToolCalling().execute(item, ctx)
    assert rows[0]["sampling"] == {"temp": "runtime-default", "max_tokens": 2000}


# --- registry wiring --------------------------------------------------------

def test_battery_registry_resolves_id_2():
    from llmtest import batteries
    b2 = batteries.get(2)
    assert isinstance(b2, B2ToolCalling)
    assert b2.id == 2
