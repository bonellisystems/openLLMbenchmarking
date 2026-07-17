"""Tests for Battery 4 -- long context (TESTPLAN 5.4)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtest import schema
from llmtest.batteries import WorkItem
from llmtest.batteries.b4_fixtures import (build_document, check_needle_signals,
                                           load_longcontext_tasks)
from llmtest.batteries.b4_longcontext import (B4LongContext, arm_fits_estimate,
                                              ctx_label, model_arms, tiers_for_model)

ROOT = Path(__file__).resolve().parents[1]


class FakeStore:
    def iter_rows(self):
        return []


def _cfg():
    from llmtest.registry import load_config
    return load_config(ROOT)


# --- fixture loader + lint ---------------------------------------------------

def test_loader_reads_all_eight_tasks_and_hashes():
    tasks = load_longcontext_tasks(ROOT)
    assert len(tasks) == 8
    ids = {t.id for t in tasks}
    assert ids == {"single-needle-01", "single-needle-02", "multi-needle-01",
                   "multi-needle-02", "multi-hop-01", "multi-hop-02",
                   "distractor-01", "distractor-02"}
    for t in tasks:
        assert len(t.fixture_sha) == 64
        assert t.kind in {"single_needle", "multi_needle", "multi_hop", "distractor"}
        assert t.needles
        for n in t.needles:
            assert 0 <= n["depth_pct"] <= 100


def test_loader_raises_on_malformed(tmp_path):
    unit_dir = tmp_path / "suite" / "b4_longcontext"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text("id: [broken", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fixture"):
        load_longcontext_tasks(tmp_path)


def test_loader_raises_on_missing_needles(tmp_path):
    unit_dir = tmp_path / "suite" / "b4_longcontext"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text("""\
id: x-01
kind: single_needle
filler_template: "filler\\n"
needles: []
question: Q?
signals: []
""", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        load_longcontext_tasks(tmp_path)


def test_loader_raises_on_bad_depth_pct(tmp_path):
    unit_dir = tmp_path / "suite" / "b4_longcontext"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text("""\
id: x-01
kind: single_needle
filler_template: "filler\\n"
needles:
  - {depth_pct: 150, text: "nope"}
question: Q?
signals: []
""", encoding="utf-8")
    with pytest.raises(ValueError, match="depth_pct"):
        load_longcontext_tasks(tmp_path)


def test_validate_cmd_lints_b4_fixtures(tmp_path, capsys):
    """Mirrors tests/test_fixture_lint.py's B1 pattern for B4: missing keys, bad
    kind, bad depth_pct, and bad signal values must all surface as VALIDATE-ERROR
    lines, not silently pass."""
    import shutil
    import yaml
    from llmtest.validate_cmd import run_validate

    shutil.copytree(ROOT / "config", tmp_path / "config")
    suite_yaml = tmp_path / "config" / "suite.yaml"
    suite_data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    suite_data["b4"] = {"ctx_tiers": [16384]}
    suite_yaml.write_text(yaml.dump(suite_data), encoding="utf-8")
    (tmp_path / "TESTPLAN.md").write_text("# Test Plan\n", encoding="utf-8")

    fx_dir = tmp_path / "suite" / "b4_longcontext"
    fx_dir.mkdir(parents=True)

    (fx_dir / "task-01.yaml").write_text("""\
id: bad-kind-01
kind: not_a_real_kind
filler_template: "filler\\n"
needles:
  - {depth_pct: 50, text: "the code is X"}
question: Q?
signals:
  - {type: contains, value: "X"}
""", encoding="utf-8")

    (fx_dir / "task-02.yaml").write_text("""\
id: bad-depth-02
kind: single_needle
filler_template: "filler\\n"
needles:
  - {depth_pct: 999, text: "the code is Y"}
question: Q?
signals:
  - {type: numeric, value: "not a number"}
""", encoding="utf-8")

    exit_code = run_validate(tmp_path)
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "task-01.yaml" in out and "not_a_real_kind" in out
    assert "task-02.yaml" in out and "depth_pct" in out
    assert "signal 0" in out


# --- document builder ---------------------------------------------------------

def _single_needle_task():
    return next(t for t in load_longcontext_tasks(ROOT) if t.id == "single-needle-01")


def test_build_document_reaches_target_length():
    task = _single_needle_task()
    target_tokens = 15360   # 16k tier minus a 1024-token reserve
    doc = build_document(task.filler_template, target_tokens, task.needles, task.question)
    assert len(doc) // 4 >= target_tokens * 0.9


def test_build_document_plants_needle_near_declared_depth():
    task = _single_needle_task()
    doc = build_document(task.filler_template, 4000, task.needles, task.question)
    needle_text = task.needles[0]["text"]
    assert needle_text in doc
    offset = doc.index(needle_text)
    body_len = len(doc)
    # needle depth_pct is 50 -- must land roughly in the middle third, not glued
    # to either end (loose bound: this is a heuristic placement, not exact).
    assert 0.25 * body_len < offset < 0.75 * body_len


def test_build_document_needle_order_independent_of_list_order():
    """Descending-offset insertion means the SAME needles produce the SAME
    document regardless of the order they appear in the fixture's needles list."""
    needles_asc = [{"depth_pct": 10, "text": "AAA-1111"}, {"depth_pct": 80, "text": "BBB-2222"}]
    needles_desc = list(reversed(needles_asc))
    doc_a = build_document("filler text here\n", 2000, needles_asc, "Q?")
    doc_b = build_document("filler text here\n", 2000, needles_desc, "Q?")
    assert doc_a == doc_b


# --- signal checker -------------------------------------------------------------

def test_check_needle_signals_contains_and_not_contains():
    sig = [{"type": "contains", "value": "8823"}, {"type": "not_contains", "value": "4471"}]
    out = check_needle_signals("The code for Site B is 8823.", sig)
    assert out["contains-0"]["pass"] is True
    assert out["not_contains-1"]["pass"] is True

    out2 = check_needle_signals("The code for Site B is 8823, not Site A's 4471.", sig)
    assert out2["contains-0"]["pass"] is True
    assert out2["not_contains-1"]["pass"] is False


# --- ctx_label / tiers_for_model / arm_fits_estimate ----------------------------

def test_ctx_label():
    assert ctx_label(16384) == "16k"
    assert ctx_label(65536) == "64k"
    assert ctx_label(131072) == "128k"
    assert ctx_label(262144) == "256k"


def test_tiers_for_model_drops_tiers_above_claimed_ctx():
    tiers = [16384, 65536, 131072, 262144]
    assert tiers_for_model(tiers, 131072) == [16384, 65536, 131072]
    assert tiers_for_model(tiers, 262144) == tiers


def test_tiers_for_model_substitutes_max_when_all_tiers_exceed_claim():
    """A hypothetical model claiming only 8k context (below every configured
    tier) must still get ONE row at its own max, not zero rows (TESTPLAN 5.4:
    'tested at their max ... tagged fits-short-context, not skipped')."""
    tiers = [16384, 65536, 131072, 262144]
    assert tiers_for_model(tiers, 8192) == [8192]


def test_arm_fits_estimate_matches_registry_fits_direction():
    """Sanity cross-check against the real registry: qwen3.6-35b-a3b (hybrid
    linear attn, 0.25x KV discount) fits q8 at 128k but not f16 at 128k -- the
    exact split TESTPLAN 5.4 documents ('128k/256k points are q8/q4-only arms')."""
    cfg = _cfg()
    m = cfg.registry["models"]["qwen3.6-35b-a3b"]
    assert arm_fits_estimate(m, cfg.tiers, "q8", 131072) is True
    assert arm_fits_estimate(m, cfg.tiers, "f16", 131072) is False
    assert arm_fits_estimate(m, cfg.tiers, "q4", 262144) is True
    assert arm_fits_estimate(m, cfg.tiers, "q8", 262144) is False


# --- model_arms: the sweep-composition logic ------------------------------------

def test_model_arms_designated_sweep_model_gets_full_grid_with_advisory_tags():
    cfg = _cfg()
    m = cfg.registry["models"]["qwen3.6-35b-a3b"]
    arms = model_arms("qwen3.6-35b-a3b", m, cfg.suite["b4"], cfg.tiers)
    assert len(arms) == 12                      # 3 kv x 4 ctx tiers, none pruned
    assert arms[(131072, "f16")] == ["fits-short-context"]
    assert arms[(262144, "f16")] == ["fits-short-context"]
    assert arms[(262144, "q8")] == ["fits-short-context"]
    assert arms[(131072, "q8")] == []            # fits fine -- no advisory tag
    assert arms[(262144, "q4")] == []


def test_model_arms_standard_model_is_fit_pruned_not_tagged():
    """A non-designated model's standard-kv sweep only plans arms that pass the
    fit estimate -- infeasible points are DROPPED, not tagged (that pruning is
    what keeps the full-roster grid from planning launches that never boot)."""
    cfg = _cfg()
    m = cfg.registry["models"]["gpt-oss-20b"]
    arms = model_arms("gpt-oss-20b", m, cfg.suite["b4"], cfg.tiers)
    assert set(arms.keys()) == {(16384, "q8"), (65536, "q8")}
    assert all(tags == [] for tags in arms.values())


def test_model_arms_spot_check_model_adds_named_point_unpruned():
    cfg = _cfg()
    m = cfg.registry["models"]["qwen3.6-27b-dense"]
    arms = model_arms("qwen3.6-27b-dense", m, cfg.suite["b4"], cfg.tiers)
    assert set(arms.keys()) == {(16384, "q8"), (32768, "f16"), (32768, "q4")}
    # f16 spot-check point rides through even though the estimate predicts it's
    # tight on T1 -- advisory tag, not pruned (TESTPLAN names this exact point).
    assert arms[(32768, "f16")] == ["fits-short-context"]
    assert arms[(32768, "q4")] == []


def test_model_arms_every_roster_model_gets_at_least_one_arm():
    """No non-quant-arm roster model is silently dropped to zero B4 coverage."""
    cfg = _cfg()
    for model_id, m in cfg.registry["models"].items():
        if m.get("role") == "quant-arm":
            continue
        arms = model_arms(model_id, m, cfg.suite["b4"], cfg.tiers)
        assert arms, f"{model_id} got zero B4 arms"


# --- plan() --------------------------------------------------------------------

def test_plan_full_grid_excludes_quant_arm_and_matches_summed_arms():
    cfg = _cfg()
    b4 = B4LongContext()
    items = b4.plan(cfg, FakeStore())

    model_ids = {i.model_id for i in items}
    assert "gemma-4-26b-a4b-mxfp4" not in model_ids     # quant-arm excluded
    assert len(model_ids) == 11

    total_arms = sum(len(model_arms(mid, m, cfg.suite["b4"], cfg.tiers))
                     for mid, m in cfg.registry["models"].items()
                     if m.get("role") != "quant-arm")
    n_tasks = len(load_longcontext_tasks(cfg.root))
    n_runs = cfg.suite["b4"]["n_runs"]
    assert len(items) == total_arms * n_tasks * n_runs

    for item in items:
        assert item.battery == 4
        assert item.task_id.startswith("b4.")

    order = cfg.suite["condition_order"]
    for item in items:
        parts = dict(p.split("=") for p in item.condition.split(";"))
        assert schema.canonical_condition(parts, order) == item.condition
        assert parts["cond"] == "B4"


def test_plan_condition_encodes_kv_and_ctx_distinctly():
    """Two arms differing ONLY by kv, or ONLY by ctx, must produce different
    condition strings and different row_ids -- the whole point of the sweep
    being resumable/idempotent per TESTPLAN 7.2."""
    cfg = _cfg()
    b4 = B4LongContext()
    items = b4.plan(cfg, FakeStore(), model_filter="qwen3.6-35b-a3b")
    single_task_items = [i for i in items if i.task_id == "b4.single-needle-01"]
    conditions = {i.condition for i in single_task_items}
    assert len(conditions) == 12          # 12 distinct (kv, ctx) arms for this model
    row_ids = {i.row_id for i in single_task_items}
    assert len(row_ids) == 12

    q8_128k = next(i for i in single_task_items if "kv=q8;ctx=128k" in i.condition)
    q4_128k = next(i for i in single_task_items if "kv=q4;ctx=128k" in i.condition)
    q8_256k = next(i for i in single_task_items if "kv=q8;ctx=256k" in i.condition)
    assert q8_128k.condition != q4_128k.condition     # kv varies, ctx fixed
    assert q8_128k.condition != q8_256k.condition     # ctx varies, kv fixed
    assert q8_128k.row_id != q4_128k.row_id
    assert q8_128k.row_id != q8_256k.row_id


def test_plan_force_bumps_run_n_condition_scoped():
    cfg = _cfg()
    model_id = "gpt-oss-20b"
    task_id = "b4.single-needle-01"
    order = cfg.suite["condition_order"]
    target_condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "16k", "cond": "B4"}, order)
    other_condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": "q8", "ctx": "64k", "cond": "B4"}, order)

    seeded_rows = [
        {"model_id": model_id, "task_id": task_id, "condition": target_condition,
         "run_n": 1, "row_id": "seed-run1"},
        {"model_id": model_id, "task_id": task_id, "condition": other_condition,
         "run_n": 9, "row_id": "seed-run9"},
    ]

    class SeededStore:
        def iter_rows(self):
            return seeded_rows

    b4 = B4LongContext()
    items = b4.plan(cfg, SeededStore(), model_filter=model_id, force=True)
    matching = [i for i in items if i.condition == target_condition
               and i.task_id == task_id]
    assert len(matching) == 1
    assert matching[0].run_n == 2                 # max(1) + 1, not influenced by run_n=9 elsewhere


# --- preflight() -----------------------------------------------------------------

def test_preflight_reports_fixture_count_and_corpus_builder_selftest():
    cfg = _cfg()
    ctx = SimpleNamespace(cfg=cfg, root=ROOT)
    rows = B4LongContext().preflight(ctx)

    fixtures_row = next(r for r in rows if r["task_id"] == "b4.selftest.fixtures")
    assert fixtures_row["status"] == "ok"
    assert fixtures_row["metrics"]["n_tasks"] == 8

    corpus_rows = [r for r in rows if r["task_id"].startswith("b4.selftest.corpus.")]
    assert len(corpus_rows) == len(cfg.suite["b4"]["ctx_tiers"])
    for r in corpus_rows:
        assert r["status"] == "ok", r.get("error_detail")
        assert r["tags"] == ["selftest"]


def test_preflight_missing_fixtures_returns_error_row(tmp_path):
    cfg = _cfg()
    ctx = SimpleNamespace(cfg=cfg, root=tmp_path)   # no suite/b4_longcontext here
    rows = B4LongContext().preflight(ctx)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["task_id"] == "b4.selftest.fixtures"


# --- execute() ---------------------------------------------------------------

def _make_item(cfg, ctx_tokens=16384, kv_short="q8", tags=None):
    task = _single_needle_task()
    order = cfg.suite["condition_order"]
    condition = schema.canonical_condition(
        {"runtime": "fork", "spec": "ngram32", "kv": kv_short,
         "ctx": ctx_label(ctx_tokens), "cond": "B4"}, order)
    suite_version = cfg.suite["suite_version"]
    row_id = schema.compute_row_id(
        suite_version=suite_version, model_id="gpt-oss-20b",
        quant_sha256=cfg.registry["models"]["gpt-oss-20b"]["provenance"]["sha256"],
        battery=4, task_id=f"b4.{task.id}", fixture_sha=task.fixture_sha,
        condition=condition, run_n=1)
    return WorkItem(
        row_id=row_id, model_id="gpt-oss-20b", battery=4, task_id=f"b4.{task.id}",
        condition=condition, run_n=1,
        payload={"model": cfg.registry["models"]["gpt-oss-20b"],
                "fixture_sha": task.fixture_sha, "suite_version": suite_version,
                "kind": task.kind, "filler_template": task.filler_template,
                "needles": task.needles, "question": task.question,
                "signals": task.signals, "ctx_tokens": ctx_tokens,
                "kv_short": kv_short, "tags": tags or []})


def test_execute_correct_needle_scores_pass(tmp_path):
    cfg = _cfg()
    captured = {}

    class StubHandle:
        session_id = "s-stub"
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return {"choices": [{"message": {"content": "The code is MW-7742-DELTA."}}],
                    "timings": {"predicted_n": 12, "predicted_per_second": 80.0,
                               "prompt_n": 4000, "prompt_per_second": 3000.0}}

    class StubMgr:
        def request_endpoint(self, model_id, **kwargs):
            captured["endpoint_kwargs"] = kwargs
            return StubHandle()

    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: StubMgr(), root=tmp_path)
    item = _make_item(cfg, ctx_tokens=16384, kv_short="q4")

    rows = B4LongContext().execute(item, ctx)
    assert len(rows) == 1
    row = rows[0]

    assert row["row_id"] == item.row_id
    assert row["needs_judging"] is False
    assert row["status"] == "ok"
    assert row["battery"] == 4
    assert row["det_checks"]["contains-0"]["pass"] is True
    assert row["metrics"]["needle_recall"] == 1.0
    assert "response" in row["artifacts"]

    # kv short label "q4" must map to the literal llama.cpp dtype "q4_0", and ctx
    # must be passed through as the raw token count, not the "16k" condition label.
    assert captured["endpoint_kwargs"]["ctx"] == 16384
    assert captured["endpoint_kwargs"]["kv"] == "q4_0"


def test_execute_wrong_needle_scores_fail_but_status_ok(tmp_path):
    cfg = _cfg()

    class StubHandle:
        session_id = "s-stub"
        def chat(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "I could not find that information."}}],
                    "timings": {}}

    class StubMgr:
        def request_endpoint(self, *a, **k):
            return StubHandle()

    ctx = SimpleNamespace(cfg=cfg, server_manager=lambda: StubMgr(), root=tmp_path)
    item = _make_item(cfg, tags=["fits-short-context"])

    rows = B4LongContext().execute(item, ctx)
    row = rows[0]

    assert row["status"] == "ok"                  # execution succeeded
    assert row["det_checks"]["contains-0"]["pass"] is False   # scoring failed
    assert row["metrics"]["needle_recall"] == 0.0
    assert row["tags"] == ["fits-short-context"]   # advisory tag carried through
