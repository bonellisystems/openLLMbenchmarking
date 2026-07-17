"""Tests for Battery 7 — harness/config sensitivity matrix."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtest import schema
from llmtest import batteries
from llmtest.batteries import WorkItem
from llmtest.batteries.b7_harnessmatrix import (
    B7HarnessMatrix, _matrix_cells, _baseline_condition, _check_json_format,
    _check_tool_call, _signal_agreement, _word_jaccard,
)
from llmtest.batteries.b7_fixtures import load_probe_tasks, lint_probe_tasks
from llmtest.registry import load_config

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    """Fake store for plan() tests — no seeded rows."""
    def iter_rows(self):
        return []


# ---------------------------------------------------------------------------
# Battery registry
# ---------------------------------------------------------------------------

def test_battery_7_registers():
    b7 = batteries.get(7)
    assert isinstance(b7, B7HarnessMatrix)
    assert b7.id == 7


# ---------------------------------------------------------------------------
# Fixture loader + lint
# ---------------------------------------------------------------------------

def test_loader_reads_8_probes_and_hashes():
    tasks = load_probe_tasks(ROOT)
    assert len(tasks) == 8
    ids = [t.id for t in tasks]
    assert ids == sorted(ids)
    assert ids[0] == "probe-01" and ids[-1] == "probe-08"
    for t in tasks:
        assert len(t.fixture_sha) == 64


def test_loader_reads_tool_call_fields():
    tasks = load_probe_tasks(ROOT)
    tool_task = next(t for t in tasks if t.id == "probe-02")
    assert tool_task.expects_tool_call is True
    assert tool_task.expected_tool_name == "get_ticket_status"
    assert tool_task.tool_schema["function"]["name"] == "get_ticket_status"

    non_tool_task = next(t for t in tasks if t.id == "probe-01")
    assert non_tool_task.expects_tool_call is False
    assert non_tool_task.tool_schema is None


def test_loader_reads_response_format():
    tasks = load_probe_tasks(ROOT)
    json_task = next(t for t in tasks if t.id == "probe-04")
    assert json_task.response_format == "json"
    text_task = next(t for t in tasks if t.id == "probe-01")
    assert text_task.response_format == "text"


def test_loader_raises_on_malformed(tmp_path):
    probes_dir = tmp_path / "probes"
    probes_dir.mkdir()
    (probes_dir / "probe-01.yaml").write_text("id: [broken", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fixture"):
        load_probe_tasks(tmp_path, "probes")


def test_loader_raises_on_missing_tool_schema(tmp_path):
    probes_dir = tmp_path / "probes"
    probes_dir.mkdir()
    (probes_dir / "probe-01.yaml").write_text(
        "id: probe-01\nprompt: hi\nsignals: []\nexpects_tool_call: true\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fixture|tool_schema"):
        load_probe_tasks(tmp_path, "probes")


def test_loader_raises_on_bad_response_format(tmp_path):
    probes_dir = tmp_path / "probes"
    probes_dir.mkdir()
    (probes_dir / "probe-01.yaml").write_text(
        "id: probe-01\nprompt: hi\nsignals: []\nresponse_format: xml\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fixture|response_format"):
        load_probe_tasks(tmp_path, "probes")


def test_lint_clean_on_real_fixtures():
    tasks = load_probe_tasks(ROOT)
    assert lint_probe_tasks(tasks) == []


def test_lint_catches_bad_id_and_missing_signals():
    from llmtest.batteries.b7_fixtures import Task
    bad = [
        Task(id="probeX", prompt="p", signals=[], expects_tool_call=False,
             tool_schema=None, expected_tool_name=None, response_format="text",
             fixture_sha="0" * 64, path=Path("fake.yaml")),
    ]
    errs = lint_probe_tasks(bad)
    assert any("id" in e for e in errs)
    assert any("no signals" in e for e in errs)


def test_lint_catches_bad_regex_signal():
    from llmtest.batteries.b7_fixtures import Task
    bad = [
        Task(id="probe-99", prompt="p", signals=[{"type": "regex", "value": "(unclosed"}],
             expects_tool_call=False, tool_schema=None, expected_tool_name=None,
             response_format="text", fixture_sha="0" * 64, path=Path("fake.yaml")),
    ]
    errs = lint_probe_tasks(bad)
    assert any("compile" in e for e in errs)


# ---------------------------------------------------------------------------
# Matrix cells
# ---------------------------------------------------------------------------

def test_matrix_cells_baseline_plus_ofat_variants():
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    cells = _matrix_cells(cfg, order)
    names = [n for n, _ in cells]
    # baseline + one variant per non-baseline value of each of the 4
    # dimensions (each dimension currently has exactly 2 values -> 4 variants)
    assert names[0] == "baseline"
    assert len(cells) == 5
    assert set(names) == {"baseline", "sysp-minimal", "temp-tdef",
                          "toolfmt-prompted", "spec-off"}
    # every condition string is distinct
    conditions = [c for _, c in cells]
    assert len(set(conditions)) == len(conditions)
    # every condition round-trips through canonical_condition
    for cond in conditions:
        parts = dict(p.split("=") for p in cond.split(";"))
        assert schema.canonical_condition(parts, order) == cond
    # baseline condition matches _baseline_condition()
    assert cells[0][1] == _baseline_condition(cfg, order)


def test_matrix_cells_encode_b7_marker():
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    for _, cond in _matrix_cells(cfg, order):
        assert "cond=B7" in cond


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------

def test_plan_item_count_roster_x_probes_x_cells_x_runs():
    cfg = load_config(ROOT)
    models = cfg.registry["models"]
    non_quant_arm = [mid for mid, m in models.items() if m.get("role") != "quant-arm"]
    assert len(non_quant_arm) == 11  # 12 registry models, 1 quant-arm excluded

    store = FakeStore()
    b7 = B7HarnessMatrix()
    items = b7.plan(cfg, store)

    n_probes = 8
    n_cells = 5
    n_runs = cfg.suite["b7"]["n_runs"]
    assert n_runs == 2
    assert len(items) == len(non_quant_arm) * n_probes * n_cells * n_runs  # 880

    model_ids = {item.model_id for item in items}
    assert "gemma-4-26b-a4b-mxfp4" not in model_ids
    assert len(model_ids) == 11

    for item in items:
        assert item.battery == 7
        assert item.task_id.startswith("b7.")


def test_plan_condition_encodes_cell_distinctly():
    """For a fixed (model, probe, run_n), the 5 matrix cells must produce 5
    distinct row_ids/conditions — the whole point of the battery."""
    cfg = load_config(ROOT)
    store = FakeStore()
    b7 = B7HarnessMatrix()
    items = b7.plan(cfg, store, model_filter="gpt-oss-20b")

    same_probe_run1 = [it for it in items
                       if it.task_id == "b7.probe-01" and it.run_n == 1]
    assert len(same_probe_run1) == 5
    conditions = {it.condition for it in same_probe_run1}
    row_ids = {it.row_id for it in same_probe_run1}
    assert len(conditions) == 5
    assert len(row_ids) == 5


def test_plan_force_bumps_run_n_condition_scoped():
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    model_id = "gpt-oss-20b"
    task_id = "b7.probe-01"
    baseline_cond = _baseline_condition(cfg, order)
    other_cond = "cond=B7;runtime=fork;kv=q8;ctx=8k;sysp=minimal;temp=t0;toolfmt=native;spec=ngram32"

    seeded_rows = [
        {"model_id": model_id, "task_id": task_id, "condition": baseline_cond,
         "run_n": 1, "row_id": "seed-run1"},
        {"model_id": model_id, "task_id": task_id, "condition": baseline_cond,
         "run_n": 2, "row_id": "seed-run2"},
        {"model_id": model_id, "task_id": task_id, "condition": other_cond,
         "run_n": 9, "row_id": "seed-run9"},
    ]

    class SeededStore:
        def iter_rows(self):
            return seeded_rows

    b7 = B7HarnessMatrix()
    items = b7.plan(cfg, SeededStore(), model_filter=model_id, force=True)
    matching = [it for it in items
               if it.model_id == model_id and it.task_id == task_id
               and it.condition == baseline_cond]
    assert len(matching) == 1
    assert matching[0].run_n == 3
    assert matching[0].row_id not in {"seed-run1", "seed-run2", "seed-run9"}


# ---------------------------------------------------------------------------
# execute() — det_checks helpers
# ---------------------------------------------------------------------------

def test_check_json_format_plain_and_fenced():
    assert _check_json_format('{"a": 1}') is True
    assert _check_json_format('```json\n{"a": 1}\n```') is True
    assert _check_json_format("not json at all") is False


def test_check_tool_call_native():
    response = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "get_ticket_status"}}]}}]}
    out = _check_tool_call(response, "", "native", "get_ticket_status")
    assert out["pass"] is True

    response_wrong = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "other_fn"}}]}}]}
    out2 = _check_tool_call(response_wrong, "", "native", "get_ticket_status")
    assert out2["pass"] is False


def test_check_tool_call_prompted():
    text = "TOOL_CALL: get_ticket_status(ticket_id=TCK-9027)\nLooking it up..."
    out = _check_tool_call({}, text, "prompted", "get_ticket_status")
    assert out["pass"] is True
    out2 = _check_tool_call({}, "no marker here", "prompted", "get_ticket_status")
    assert out2["pass"] is False


def test_signal_agreement_full_and_partial():
    a = {"contains-0": {"pass": True}, "regex-1": {"pass": False}}
    b = {"contains-0": {"pass": True}, "regex-1": {"pass": False}}
    out = _signal_agreement(a, b, 0.8)
    assert out["agreement_rate"] == 1.0 and out["pass"] is True

    c = {"contains-0": {"pass": False}, "regex-1": {"pass": False}}
    out2 = _signal_agreement(a, c, 0.8)
    assert out2["agreement_rate"] == 0.5 and out2["pass"] is False


def test_word_jaccard_identical_and_disjoint():
    assert _word_jaccard("hello world", "hello world") == 1.0
    assert _word_jaccard("hello world", "goodbye moon") == 0.0


# ---------------------------------------------------------------------------
# execute() — full row construction with a stub handle
# ---------------------------------------------------------------------------

def _probe01_item(cfg, condition, run_n=1):
    tasks = load_probe_tasks(cfg.root)
    probe = next(t for t in tasks if t.id == "probe-01")
    sv = cfg.suite["suite_version"]
    row_id = schema.compute_row_id(
        suite_version=sv, model_id="gpt-oss-20b",
        quant_sha256=cfg.registry["models"]["gpt-oss-20b"]["provenance"]["sha256"],
        battery=7, task_id="b7.probe-01", fixture_sha=probe.fixture_sha,
        condition=condition, run_n=run_n)
    return WorkItem(
        row_id=row_id, model_id="gpt-oss-20b", battery=7, task_id="b7.probe-01",
        condition=condition, run_n=run_n,
        payload={"model": cfg.registry["models"]["gpt-oss-20b"],
                 "fixture_sha": probe.fixture_sha, "suite_version": sv,
                 "prompt": probe.prompt, "signals": probe.signals,
                 "expects_tool_call": probe.expects_tool_call,
                 "tool_schema": probe.tool_schema,
                 "expected_tool_name": probe.expected_tool_name,
                 "response_format": probe.response_format})


class StubHandle:
    session_id = "s-stub"
    normalized_config = {}

    def __init__(self, text):
        self.text = text

    def chat(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self.text}}],
               "timings": {"predicted_n": 20, "predicted_per_second": 90.0}}


class StubMgr:
    def __init__(self, text):
        self.text = text

    def request_endpoint(self, *a, **k):
        return StubHandle(self.text)


def test_execute_baseline_row_needs_judging_false_and_artifact_saved(tmp_path):
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    baseline_cond = _baseline_condition(cfg, order)
    item = _probe01_item(cfg, baseline_cond)

    ctx = SimpleNamespace(cfg=cfg, store=FakeStore(),
                          server_manager=lambda: StubMgr(
                              "Fix the 0x8004010F error via the OST/offline data file "
                              "repair; expect a reply within 4 business hours."),
                          root=tmp_path)
    rows = B7HarnessMatrix().execute(item, ctx)
    assert len(rows) == 1
    row = rows[0]
    assert row["needs_judging"] is False
    assert row["status"] == "ok"
    assert row["battery"] == 7
    assert row["det_checks"]["contains-0"]["pass"] is True
    assert "response" in row["artifacts"]
    artifact_path = tmp_path / "artifacts" / row["artifacts"]["response"]["relpath"]
    assert artifact_path.exists()
    # baseline row has no vs-baseline comparison keys (nothing to compare to)
    assert "signal_agreement_vs_baseline" not in row["det_checks"]


def test_execute_variant_computes_signal_agreement_vs_baseline(tmp_path):
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    baseline_cond = _baseline_condition(cfg, order)

    baseline_text = ("Fix the 0x8004010F error via the OST/offline data file "
                     "repair; expect a reply within 4 business hours.")
    baseline_item = _probe01_item(cfg, baseline_cond)

    class RecordingStore(FakeStore):
        def __init__(self):
            self.rows = []
        def iter_rows(self):
            return self.rows

    store = RecordingStore()
    ctx = SimpleNamespace(cfg=cfg, store=store,
                          server_manager=lambda: StubMgr(baseline_text), root=tmp_path)
    baseline_row = B7HarnessMatrix().execute(baseline_item, ctx)[0]
    store.rows.append(baseline_row)

    # variant cell: sysp=minimal, same content signals hold (agreement should be 1.0)
    variant_cond = next(c for n, c in _matrix_cells(cfg, order) if n == "sysp-minimal")
    variant_item = _probe01_item(cfg, variant_cond)
    ctx.server_manager = lambda: StubMgr(baseline_text)  # same content, different cell
    variant_row = B7HarnessMatrix().execute(variant_item, ctx)[0]

    agreement = variant_row["det_checks"]["signal_agreement_vs_baseline"]
    assert agreement["pass"] is True
    assert agreement["agreement_rate"] == 1.0
    assert variant_row["metrics"]["word_jaccard_vs_baseline"] == 1.0


def test_execute_variant_missing_baseline_row_degrades_gracefully(tmp_path):
    """If the baseline row hasn't been computed yet (e.g. a filtered/partial
    run), the variant row must still succeed — just without the vs-baseline
    checks, not crash."""
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    variant_cond = next(c for n, c in _matrix_cells(cfg, order) if n == "temp-tdef")
    item = _probe01_item(cfg, variant_cond)
    ctx = SimpleNamespace(cfg=cfg, store=FakeStore(),
                          server_manager=lambda: StubMgr("some answer text"),
                          root=tmp_path)
    row = B7HarnessMatrix().execute(item, ctx)[0]
    assert row["status"] == "ok"
    assert "signal_agreement_vs_baseline" not in row["det_checks"]


def test_execute_ngram_off_byte_identical_vs_baseline(tmp_path):
    """Direct test of the project's lossless-ngram-at-temp0 claim: the
    spec-off cell (temp still t0) should be flagged byte_identical_vs_baseline
    True when text matches, False when it doesn't."""
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    baseline_cond = _baseline_condition(cfg, order)
    baseline_text = "identical response text"

    class RecordingStore(FakeStore):
        def __init__(self):
            self.rows = []
        def iter_rows(self):
            return self.rows

    store = RecordingStore()
    ctx = SimpleNamespace(cfg=cfg, store=store,
                          server_manager=lambda: StubMgr(baseline_text), root=tmp_path)
    baseline_row = B7HarnessMatrix().execute(_probe01_item(cfg, baseline_cond), ctx)[0]
    store.rows.append(baseline_row)

    spec_off_cond = next(c for n, c in _matrix_cells(cfg, order) if n == "spec-off")

    # Case 1: identical text -> pass True
    ctx.server_manager = lambda: StubMgr(baseline_text)
    row_same = B7HarnessMatrix().execute(_probe01_item(cfg, spec_off_cond), ctx)[0]
    assert row_same["det_checks"]["byte_identical_vs_baseline"]["pass"] is True

    # Case 2: different text -> pass False
    ctx.server_manager = lambda: StubMgr("different response text")
    row_diff = B7HarnessMatrix().execute(_probe01_item(cfg, spec_off_cond), ctx)[0]
    assert row_diff["det_checks"]["byte_identical_vs_baseline"]["pass"] is False


def test_execute_tool_probe_native_and_prompted(tmp_path):
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    tasks = load_probe_tasks(cfg.root)
    probe = next(t for t in tasks if t.id == "probe-02")
    sv = cfg.suite["suite_version"]

    def make_item(condition):
        row_id = schema.compute_row_id(
            suite_version=sv, model_id="gpt-oss-20b",
            quant_sha256=cfg.registry["models"]["gpt-oss-20b"]["provenance"]["sha256"],
            battery=7, task_id="b7.probe-02", fixture_sha=probe.fixture_sha,
            condition=condition, run_n=1)
        return WorkItem(row_id=row_id, model_id="gpt-oss-20b", battery=7,
                        task_id="b7.probe-02", condition=condition, run_n=1,
                        payload={"model": cfg.registry["models"]["gpt-oss-20b"],
                                 "fixture_sha": probe.fixture_sha, "suite_version": sv,
                                 "prompt": probe.prompt, "signals": probe.signals,
                                 "expects_tool_call": probe.expects_tool_call,
                                 "tool_schema": probe.tool_schema,
                                 "expected_tool_name": probe.expected_tool_name,
                                 "response_format": probe.response_format})

    baseline_cond = _baseline_condition(cfg, order)

    class NativeToolHandle(StubHandle):
        def chat(self, messages, **kwargs):
            assert kwargs.get("tools") is not None
            return {"choices": [{"message": {"content": "Ticket TCK-9027 is open.",
                                              "tool_calls": [{"function":
                                                             {"name": "get_ticket_status"}}]}}],
                   "timings": {}}

    ctx = SimpleNamespace(cfg=cfg, store=FakeStore(),
                          server_manager=lambda: SimpleNamespace(
                              request_endpoint=lambda *a, **k: NativeToolHandle("x")),
                          root=tmp_path)
    row = B7HarnessMatrix().execute(make_item(baseline_cond), ctx)[0]
    assert row["det_checks"]["tool_call_compliance"]["pass"] is True

    prompted_cond = next(c for n, c in _matrix_cells(cfg, order) if n == "toolfmt-prompted")

    class PromptedToolHandle(StubHandle):
        def chat(self, messages, **kwargs):
            assert kwargs.get("tools") is None
            return {"choices": [{"message": {
                "content": "TOOL_CALL: get_ticket_status(ticket_id=TCK-9027)\nChecking..."}}],
                   "timings": {}}

    ctx.server_manager = lambda: SimpleNamespace(
        request_endpoint=lambda *a, **k: PromptedToolHandle("x"))
    row2 = B7HarnessMatrix().execute(make_item(prompted_cond), ctx)[0]
    assert row2["det_checks"]["tool_call_compliance"]["pass"] is True


def test_execute_json_probe_format_check(tmp_path):
    cfg = load_config(ROOT)
    order = cfg.suite["condition_order"]
    tasks = load_probe_tasks(cfg.root)
    probe = next(t for t in tasks if t.id == "probe-04")
    sv = cfg.suite["suite_version"]
    baseline_cond = _baseline_condition(cfg, order)
    row_id = schema.compute_row_id(
        suite_version=sv, model_id="gpt-oss-20b",
        quant_sha256=cfg.registry["models"]["gpt-oss-20b"]["provenance"]["sha256"],
        battery=7, task_id="b7.probe-04", fixture_sha=probe.fixture_sha,
        condition=baseline_cond, run_n=1)
    item = WorkItem(row_id=row_id, model_id="gpt-oss-20b", battery=7,
                    task_id="b7.probe-04", condition=baseline_cond, run_n=1,
                    payload={"model": cfg.registry["models"]["gpt-oss-20b"],
                             "fixture_sha": probe.fixture_sha, "suite_version": sv,
                             "prompt": probe.prompt, "signals": probe.signals,
                             "expects_tool_call": probe.expects_tool_call,
                             "tool_schema": probe.tool_schema,
                             "expected_tool_name": probe.expected_tool_name,
                             "response_format": probe.response_format})
    good_json = '{"open_tickets": 14, "sla_breaches": 2, "on_call_engineer": "J. Alvarez"}'
    ctx = SimpleNamespace(cfg=cfg, store=FakeStore(),
                          server_manager=lambda: StubMgr(good_json), root=tmp_path)
    row = B7HarnessMatrix().execute(item, ctx)[0]
    assert row["det_checks"]["format_json"]["pass"] is True

    ctx.server_manager = lambda: StubMgr("not json")
    row2 = B7HarnessMatrix().execute(item, ctx)[0]
    assert row2["det_checks"]["format_json"]["pass"] is False


# ---------------------------------------------------------------------------
# suite.yaml wiring
# ---------------------------------------------------------------------------

def test_suite_yaml_has_b7_block_and_condition_vocab():
    cfg = load_config(ROOT)
    assert "b7" in cfg.suite
    assert cfg.suite["b7"]["n_runs"] >= 2
    assert "B7" in cfg.suite["condition_vocab"]["cond"]
    for k in ("sysp", "temp", "toolfmt"):
        assert k in cfg.suite["condition_order"]
