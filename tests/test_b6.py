"""Tests for Battery 6 -- agentic coding (from-scratch codegen + planted-bug self-correction)."""
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtest.batteries.b6_agenticcoding import B6AgenticCoding
from llmtest.batteries import WorkItem
from llmtest.batteries import b6_fixtures as f
from llmtest import schema

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    """Fake store for plan() tests."""
    def iter_rows(self):
        return []


def _load_task(task_id: str) -> f.Task:
    tasks = f.load_tasks(ROOT)
    return next(t for t in tasks if t.id == task_id)


def _make_exec_item(cfg, task: f.Task, run_n=1, model_id="gpt-oss-20b"):
    """Build a WorkItem matching what plan() produces: fixture_sha + prompt/
    signals ride in payload so execute() doesn't re-load the fixture YAML."""
    order = cfg.suite["condition_order"]
    condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "32k", "cond": "B6"}, order)
    suite_version = cfg.suite["suite_version"]
    row_id = schema.compute_row_id(
        suite_version=suite_version, model_id=model_id,
        quant_sha256=cfg.registry["models"][model_id]["provenance"]["sha256"],
        battery=6, task_id=f"b6.{task.id}", fixture_sha=task.fixture_sha,
        condition=condition, run_n=run_n)
    return WorkItem(
        row_id=row_id, model_id=model_id, battery=6, task_id=f"b6.{task.id}",
        condition=condition, run_n=run_n,
        payload={"model": cfg.registry["models"][model_id],
                 "task_id": task.id, "fixture_sha": task.fixture_sha,
                 "suite_version": suite_version, "track": task.track,
                 "language": task.language, "prompt": task.prompt,
                 "required_signals": task.required_signals,
                 "fix_signals": task.fix_signals,
                 "regression_signals": task.regression_signals})


def _stub_ctx(tmp_path, response_text, timings=None):
    class StubHandle:
        session_id = "s-stub"
        normalized_config = {}
        def chat(self, messages, **kwargs):
            return {"choices": [{"message": {"content": response_text}}],
                    "timings": timings or {}}

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    return SimpleNamespace(cfg=cfg, server_manager=lambda: StubMgr(), root=tmp_path)


# --- fixture loader ---------------------------------------------------------

def test_loader_reads_10_tasks_sorted_by_id():
    tasks = f.load_tasks(ROOT)
    assert len(tasks) == 10
    ids = [t.id for t in tasks]
    assert ids == sorted(ids)
    assert {t.track for t in tasks} == {"scratch", "bugfix"}
    assert {t.language for t in tasks} == {"python", "bash", "sql", "js"}


def test_loader_scratch_task_fields():
    task = _load_task("scratch-01")
    assert task.track == "scratch"
    assert task.language == "python"
    assert task.difficulty == "easy"
    assert task.required_signals
    assert task.buggy_code is None
    assert len(task.fixture_sha) == 64


def test_loader_bugfix_task_fields():
    task = _load_task("bugfix-01")
    assert task.track == "bugfix"
    assert task.buggy_code is not None
    assert "def summarize(nums)" in task.buggy_code
    assert task.fix_signals
    assert task.regression_signals


def test_loader_raises_on_missing_required_key(tmp_path):
    unit_dir = tmp_path / "suite" / "b6_agenticcoding"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text(
        "id: scratch-01\ntrack: scratch\nlanguage: python\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fixture"):
        f.load_tasks(tmp_path)


def test_loader_raises_on_invalid_track(tmp_path):
    unit_dir = tmp_path / "suite" / "b6_agenticcoding"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text("""\
id: weird-01
track: not_a_real_track
language: python
difficulty: easy
prompt: hi
""", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid track"):
        f.load_tasks(tmp_path)


def test_loader_raises_bugfix_missing_buggy_code(tmp_path):
    unit_dir = tmp_path / "suite" / "b6_agenticcoding"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text("""\
id: bugfix-99
track: bugfix
language: python
difficulty: easy
prompt: fix it
""", encoding="utf-8")
    with pytest.raises(ValueError, match="buggy_code"):
        f.load_tasks(tmp_path)


def test_loader_raises_bugfix_missing_regression_signals(tmp_path):
    unit_dir = tmp_path / "suite" / "b6_agenticcoding"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text("""\
id: bugfix-99
track: bugfix
language: python
difficulty: easy
buggy_code: "x = 1"
prompt: fix it
""", encoding="utf-8")
    with pytest.raises(ValueError, match="regression_signals"):
        f.load_tasks(tmp_path)


def test_loader_missing_dir_returns_empty():
    assert f.load_tasks(Path("/definitely/not/a/real/path")) == []


# --- code extraction ---------------------------------------------------------

def test_extract_code_block_matches_language_tag():
    text = "blah\n```python\ndef f():\n    return 1\n```\nmore text"
    code = f.extract_code_block(text, "python")
    assert code == "def f():\n    return 1"


def test_extract_code_block_falls_back_to_untagged_fence():
    text = "```\nSELECT 1;\n```"
    code = f.extract_code_block(text, "sql")
    assert code == "SELECT 1;"


def test_extract_code_block_falls_back_to_first_fence_on_language_mismatch():
    text = "```javascript\nconsole.log(1)\n```"
    code = f.extract_code_block(text, "python")
    assert code == "console.log(1)"


def test_extract_code_block_none_when_no_fence():
    assert f.extract_code_block("just prose, no code fence here", "python") is None


# --- signal checking ---------------------------------------------------------

def test_check_code_signals_contains_regex_absent():
    code = "def add_item(item, cart=None):\n    if cart is None:\n        cart = []\n"
    sigs = [{"type": "contains", "value": "def add_item("},
            {"type": "regex", "value": r"cart\s*=\s*None"},
            {"type": "absent", "value": "cart=[]"}]
    out = f.check_code_signals(code, sigs, "fix")
    assert out["fix.contains-0"]["pass"] is True
    assert out["fix.regex-1"]["pass"] is True
    assert out["fix.absent-2"]["pass"] is True


def test_check_code_signals_absent_fails_when_pattern_still_present():
    code = "def add_item(item, cart=[]):\n    cart.append(item)\n"
    out = f.check_code_signals(code, [{"type": "absent", "value": "cart=[]"}], "regression")
    assert out["regression.absent-0"]["pass"] is False


def test_check_code_signals_bad_regex_no_crash():
    out = f.check_code_signals("code", [{"type": "regex", "value": "(unclosed"}], "required")
    assert out["required.regex-0"]["pass"] is False
    assert "error" in out["required.regex-0"]


def test_check_code_signals_unknown_type():
    out = f.check_code_signals("code", [{"type": "bogus", "value": "x"}], "required")
    assert out["required.bogus-0"]["pass"] is False


def test_check_code_signals_prefix_namespaces_keys():
    """required/fix/regression each restart index at 0 -- prefix must keep keys distinct."""
    sigs = [{"type": "contains", "value": "x"}]
    req = f.check_code_signals("x", sigs, "required")
    fix = f.check_code_signals("x", sigs, "fix")
    assert set(req) == {"required.contains-0"}
    assert set(fix) == {"fix.contains-0"}


# --- compile() safety --------------------------------------------------------

def test_compile_check_valid_python():
    result = f.compile_check("def f(x):\n    return x + 1\n")
    assert result["pass"] is True


def test_compile_check_syntax_error():
    result = f.compile_check("def f(x)\n    return x + 1\n")
    assert result["pass"] is False
    assert "SyntaxError" in result["error"]


def test_compile_check_empty_code():
    result = f.compile_check("")
    assert result["pass"] is False


def test_compile_check_never_executes(monkeypatch):
    """compile() must never execute the code. Patch builtins.exec/eval to blow up
    if called at all -- then feed compile_check code that WOULD raise/exit/mutate
    state if it were ever actually run, and confirm it's scored as valid syntax
    without exec/eval ever firing."""
    def _boom(*a, **k):
        raise AssertionError("exec/eval must never be called scoring model code")
    monkeypatch.setattr(builtins, "exec", _boom)
    monkeypatch.setattr(builtins, "eval", _boom)

    dangerous = "import sys\nsys.exit(1)\n"
    result = f.compile_check(dangerous)
    assert result["pass"] is True

    also_dangerous = "raise SystemExit('nope')\n"
    result2 = f.compile_check(also_dangerous)
    assert result2["pass"] is True


# --- Battery.plan() -----------------------------------------------------------

def test_plan_covers_11_models_x_10_tasks_x_3_runs():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    models = cfg.registry["models"]
    quant_arm = [mid for mid, m in models.items() if m.get("role") == "quant-arm"]
    assert len(quant_arm) == 1

    tasks = f.load_tasks(ROOT)
    assert len(tasks) == 10

    store = FakeStore()
    b6 = B6AgenticCoding()
    items = b6.plan(cfg, store)

    # 11 non-quant-arm models x 10 fixture tasks x 3 runs = 330
    assert len(items) == 330

    model_ids = {item.model_id for item in items}
    assert quant_arm[0] not in model_ids
    assert len(model_ids) == 11

    for item in items:
        assert item.battery == 6
        assert item.task_id.startswith("b6.")

    order = cfg.suite["condition_order"]
    for item in items:
        parts = dict(p.split("=") for p in item.condition.split(";"))
        assert item.condition == schema.canonical_condition(parts, order)
        assert parts["cond"] == "B6"


def test_plan_model_filter():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    items = B6AgenticCoding().plan(cfg, FakeStore(), model_filter="gpt-oss-20b")
    assert items
    assert {i.model_id for i in items} == {"gpt-oss-20b"}
    assert len(items) == 10 * 3  # 10 tasks x 3 runs


def test_plan_force_bumps_run_n_condition_scoped():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)

    model_id = "gpt-oss-20b"
    task = _load_task("scratch-01")
    task_id = f"b6.{task.id}"
    order = cfg.suite["condition_order"]
    target_condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "32k", "cond": "B6"}, order)
    other_condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "32k", "cond": "PEAK"}, order)

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

    items = B6AgenticCoding().plan(cfg, SeededStore(), model_filter=model_id, force=True)
    matching = [it for it in items if it.task_id == task_id]
    assert len(matching) == 1
    assert matching[0].run_n == 3
    assert matching[0].row_id not in {"seed-run1", "seed-run2", "seed-run9"}


# --- Battery.execute() ---------------------------------------------------------

def test_execute_scratch_correct_code_passes_all_checks(tmp_path):
    task = _load_task("scratch-01")
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = (
        "Here you go:\n\n```python\n"
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n            return False\n"
        "    return True\n```\n"
    )
    ctx = _stub_ctx(tmp_path, response, timings={"predicted_n": 40, "predicted_per_second": 90.0})
    rows = B6AgenticCoding().execute(item, ctx)
    assert len(rows) == 1
    row = rows[0]

    assert row["row_id"] == item.row_id
    assert row["needs_judging"] is True
    assert row["status"] == "ok"
    assert row["battery"] == 6
    assert row["det_checks"]["code_extracted"]["pass"] is True
    assert row["det_checks"]["compile_ok"]["pass"] is True
    assert all(v["pass"] for k, v in row["det_checks"].items() if k.startswith("required."))
    assert "response" in row["artifacts"]
    assert row["metrics"]["track"] == "scratch"
    assert row["metrics"]["language"] == "python"


def test_execute_scratch_wrong_symbol_fails_required_signal(tmp_path):
    """Valid Python, but doesn't define the required function name."""
    task = _load_task("scratch-01")
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = "```python\ndef check_prime(n):\n    return True\n```"
    ctx = _stub_ctx(tmp_path, response)
    row = B6AgenticCoding().execute(item, ctx)[0]

    assert row["det_checks"]["code_extracted"]["pass"] is True
    assert row["det_checks"]["compile_ok"]["pass"] is True
    assert row["det_checks"]["required.regex-0"]["pass"] is False  # def is_prime( absent


def test_execute_scratch_broken_syntax_fails_compile(tmp_path):
    task = _load_task("scratch-01")
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = "```python\ndef is_prime(n)\n    return True\n```"  # missing colon
    ctx = _stub_ctx(tmp_path, response)
    row = B6AgenticCoding().execute(item, ctx)[0]

    assert row["det_checks"]["code_extracted"]["pass"] is True
    assert row["det_checks"]["compile_ok"]["pass"] is False
    assert "SyntaxError" in row["det_checks"]["compile_ok"]["error"]


def test_execute_scratch_no_code_fence_fails_extraction(tmp_path):
    task = _load_task("scratch-01")
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = "Sure, here's a description of a prime-checking algorithm in prose."
    ctx = _stub_ctx(tmp_path, response)
    row = B6AgenticCoding().execute(item, ctx)[0]

    assert row["det_checks"]["code_extracted"]["pass"] is False
    # compile_check on empty code must not pass either
    assert row["det_checks"]["compile_ok"]["pass"] is False


def test_execute_bugfix_correct_fix_passes_fix_and_regression_signals(tmp_path):
    task = _load_task("bugfix-02")  # running_totals off-by-one
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = (
        "```python\n"
        "def running_totals(nums):\n"
        "    totals = []\n"
        "    running = 0\n"
        "    for i in range(len(nums)):\n"
        "        running += nums[i]\n"
        "        totals.append(running)\n"
        "    return totals\n```"
    )
    ctx = _stub_ctx(tmp_path, response)
    row = B6AgenticCoding().execute(item, ctx)[0]

    assert row["det_checks"]["required.contains-0"]["pass"] is True
    assert row["det_checks"]["fix.regex-0"]["pass"] is True
    assert row["det_checks"]["regression.absent-0"]["pass"] is True
    assert row["det_checks"]["compile_ok"]["pass"] is True


def test_execute_bugfix_noop_fails_regression_signal(tmp_path):
    """Model echoes the buggy code back unchanged -- the no-op/DNF case."""
    task = _load_task("bugfix-02")
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = (
        "```python\n"
        "def running_totals(nums):\n"
        "    totals = []\n"
        "    running = 0\n"
        "    for i in range(len(nums) - 1):\n"
        "        running += nums[i]\n"
        "        totals.append(running)\n"
        "    return totals\n```"
    )
    ctx = _stub_ctx(tmp_path, response)
    row = B6AgenticCoding().execute(item, ctx)[0]

    assert row["det_checks"]["regression.absent-0"]["pass"] is False  # bug still there
    assert row["det_checks"]["fix.regex-0"]["pass"] is False          # fix pattern absent
    assert row["det_checks"]["compile_ok"]["pass"] is True            # still valid syntax though


def test_execute_bugfix_non_python_language_has_no_compile_check(tmp_path):
    task = _load_task("bugfix-05")  # sql
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = (
        "```sql\nSELECT c.id, c.name\nFROM customers c\n"
        "LEFT JOIN orders o ON c.id = o.customer_id\n"
        "WHERE o.customer_id IS NULL;\n```"
    )
    ctx = _stub_ctx(tmp_path, response)
    row = B6AgenticCoding().execute(item, ctx)[0]

    assert "compile_ok" not in row["det_checks"]
    assert row["det_checks"]["fix.regex-0"]["pass"] is True
    assert row["det_checks"]["regression.absent-0"]["pass"] is True


def test_execute_sampling_records_runtime_default_temp(tmp_path):
    task = _load_task("scratch-03")
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    ctx = _stub_ctx(tmp_path, "```bash\ntar -czf /tmp/x_$(date +%s).tar.gz $1\n```")
    row = B6AgenticCoding().execute(item, ctx)[0]
    assert row["sampling"] == {"temp": "runtime-default", "max_tokens": 6000}  # scratch track


def test_execute_never_calls_exec_or_eval(tmp_path, monkeypatch):
    """End-to-end safety: the full execute() path (extraction + signal checks +
    compile-only syntax check) must never invoke exec()/eval() on model output,
    even for a task whose code would raise/exit if actually run."""
    def _boom(*a, **k):
        raise AssertionError("exec/eval must never be called scoring model code")
    monkeypatch.setattr(builtins, "exec", _boom)
    monkeypatch.setattr(builtins, "eval", _boom)

    task = _load_task("scratch-01")
    item = _make_exec_item(_load_cfg(), task, run_n=1)
    response = "```python\nimport sys\nsys.exit(1)\ndef is_prime(n):\n    return True\n```"
    ctx = _stub_ctx(tmp_path, response)
    rows = B6AgenticCoding().execute(item, ctx)  # must not raise
    assert rows[0]["status"] == "ok"
    assert rows[0]["det_checks"]["compile_ok"]["pass"] is True


# --- Battery.preflight() --------------------------------------------------------

def test_preflight_ok_when_fixtures_present():
    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    ctx = SimpleNamespace(cfg=cfg, root=ROOT)
    rows = B6AgenticCoding().preflight(ctx)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["tags"] == ["selftest"]
    assert rows[0]["battery"] == 6


def test_preflight_error_when_fixtures_missing(tmp_path):
    from llmtest.registry import load_config
    cfg = load_config(ROOT)
    ctx = SimpleNamespace(cfg=cfg, root=tmp_path)  # no suite/b6_agenticcoding here
    rows = B6AgenticCoding().preflight(ctx)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"


def _load_cfg():
    from llmtest.registry import load_config
    return load_config(ROOT)
