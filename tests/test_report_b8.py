"""Tests for the Wilson score interval (`llmtest.harness.stats.wilson`) and
the P8 report's B8 section (Task 9 -- subagent canary + B8 report section +
Wilson intervals, the final Phase-3 task).

No real B8 rows exist yet on disk (real harness runs deferred to a
Blackwell box -- see task-9-brief.md), so `build_b8_section` is exercised
here entirely against SYNTHETIC battery=8 rows written in the same on-disk
shape `llmtest/batteries/b8_harness.py::execute()` emits (condition string
built the same way, via `cond=B8;harness=...;task=...;attempt_id=...;
execution_provenance_sha=...`, `metrics` carrying `completion`/`steps`/
`tokens_prompt`/`tokens_completion`/`terminal_status`/`subagent_spawned`).

Mirrors `tests/test_report_b2.py`'s import-by-path trick (scripts/ isn't a
package) and its `_write_row`-style row-writer helper pattern.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "p8_report", REPO_ROOT / "scripts" / "p8_report.py")
p8_report = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("p8_report", p8_report)
_SPEC.loader.exec_module(p8_report)

from llmtest.harness.stats import wilson  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Wilson score interval -- known values + edge cases
# ---------------------------------------------------------------------------


def test_wilson_known_value_5_of_10_at_default_z():
    """5/10 at z=1.96 is a textbook Wilson-interval value: (0.237, 0.763)
    to 3 decimal places -- verifies the exact closed-form, not just
    "some interval was returned"."""
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_3_of_5():
    """The value this file's synthetic B8 test group below actually uses
    (3 completions / 5 replicates) -- pinned here independently of the
    report test so a report-side regression and a stats-side regression
    are never conflated."""
    lo, hi = wilson(3, 5)
    assert lo == pytest.approx(0.2307, abs=1e-3)
    assert hi == pytest.approx(0.8824, abs=1e-3)


def test_wilson_n_zero_returns_documented_sentinel():
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_zero_of_n():
    """0/n: lower bound is (clamped to) exactly 0.0, upper bound is a
    finite interval strictly less than 1.0 (there IS information in "0
    successes out of n trials" -- it doesn't collapse to "anything from 0
    to 100%")."""
    lo, hi = wilson(0, 5)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < hi < 1.0


def test_wilson_n_of_n():
    """n/n (all successes): lower bound is a finite interval strictly
    greater than 0.0, upper bound is (clamped to) exactly 1.0 -- this is a
    genuine property of the (uncorrected) Wilson interval at the k==n
    boundary, not a smoothing artifact; see the docstring in
    llmtest/harness/stats.py for the algebra."""
    lo, hi = wilson(5, 5)
    assert 0.0 < lo < 1.0
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_1_of_1():
    """k=n=1 -- the smallest possible "all successes" case. Bounds must
    stay within [0,1] (never overshoot like the naive normal-approximation
    Wald interval would), and the lower bound is well below 1.0 (a single
    success carries little information, so the interval must stay wide)."""
    lo, hi = wilson(1, 1)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert lo < 0.3       # wide interval -- one success alone proves little
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_wilson_rejects_k_out_of_range():
    with pytest.raises(ValueError):
        wilson(6, 5)
    with pytest.raises(ValueError):
        wilson(-1, 5)


@pytest.mark.parametrize("k,n", [(0, 1), (1, 2), (2, 3), (7, 20), (50, 50)])
def test_wilson_bounds_always_within_unit_interval(k, n):
    lo, hi = wilson(k, n)
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# 2. build_b8_section -- synthetic B8 rows
# ---------------------------------------------------------------------------


def _write_b8_row(root: Path, suite_version: str, *, model_id="model-a",
                   harness="opencode", task="repo-fix", run_n=1, completion=True,
                   steps=4, tokens_prompt=100, tokens_completion=50,
                   terminal_status="completed", subagent_spawned="no",
                   first_failure_class=None) -> None:
    """Writes one synthetic B8 row in the same on-disk shape
    `llmtest/batteries/b8_harness.py::execute()` emits -- condition built
    the same way (`cond=B8;harness=...;task=...;attempt_id=...;
    execution_provenance_sha=...`) so `build_b8_section` exercises the SAME
    condition-parsing path a real B8 row would hit, not a hand-rolled
    shortcut."""
    path = root / "results" / f"rows-{suite_version}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    condition = (f"cond=B8;harness={harness};task={task};"
                 f"attempt_id=attempt-{model_id}-{harness}-{run_n};"
                 f"execution_provenance_sha=deadbeef{run_n}")
    metrics = {
        "completion": completion,
        "steps": steps,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "terminal_status": terminal_status,
        "subagent_spawned": subagent_spawned,
    }
    if first_failure_class is not None:
        metrics["first_failure_class"] = first_failure_class
    row = {
        "schema_version": 1,
        "row_id": f"{suite_version}-{model_id}-{harness}-{task}-{run_n}",
        "parent_id": None, "suite_version": suite_version, "fixture_sha": "sha",
        "code_sha": "unknown", "battery": 8, "task_id": f"b8.{task}",
        "condition": condition, "run_n": run_n, "model_id": model_id,
        "hf_repo": "o/r", "quant_file": "q.gguf", "quant_sha256": "qsha",
        "tier": "T1", "session_id": "s", "sampling": {"harness": harness},
        "ts": "2026-07-19T00:00:00+00:00", "request": {}, "response_meta": {},
        "det_checks": {"oracle": {"pass": completion, "detail": "stub"}},
        "needs_judging": False, "metrics": metrics, "timing_authoritative": False,
        "artifacts": {}, "status": "ok", "error_detail": None, "tags": [],
    }
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def _cfg():
    class _Cfg:
        suite = {}
    return _Cfg()


def test_build_b8_section_degrades_when_no_b8_rows(tmp_path):
    section = p8_report.build_b8_section(tmp_path, _cfg(), [])
    assert "(no B8 data yet)" in section
    assert "Traceback" not in section


def test_build_b8_section_shows_completion_kn_and_wilson_interval(tmp_path):
    """5 replicates for (model-a, opencode), 3 completed -- the section
    must show the raw '3/5' AND a Wilson interval for it, not a blended
    single-number probability."""
    caveats: list[str] = []
    for run_n, ok in zip(range(1, 6), [True, True, True, False, False]):
        _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode",
                       run_n=run_n, completion=ok,
                       first_failure_class=(None if ok else "a"))
    rows = p8_report.load_rows(tmp_path, "suite-v2.1.0", caveats)

    section = p8_report.build_b8_section(tmp_path, _cfg(), rows)

    assert "3/5" in section
    lo, hi = wilson(3, 5)
    assert f"{lo * 100:.1f}%" in section
    assert f"{hi * 100:.1f}%" in section
    assert "Wilson" in section


def test_build_b8_section_shows_first_failure_class_breakdown(tmp_path):
    caveats: list[str] = []
    for run_n, ok, cls in zip(range(1, 6), [True, True, True, False, False],
                               [None, None, None, "a", None]):
        _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode",
                       run_n=run_n, completion=ok, first_failure_class=cls)
    rows = p8_report.load_rows(tmp_path, "suite-v2.1.0", caveats)

    section = p8_report.build_b8_section(tmp_path, _cfg(), rows)

    # One failed row carries first_failure_class="a", the other failed row
    # carries none at all -- must show up as unclassified, not silently
    # dropped or crashing.
    assert "(unclassified)" in section
    assert "first-failure-class" in section.lower() or "first failure class" in section.lower()


def test_build_b8_section_subagent_canary_line_for_yes_no_group(tmp_path):
    caveats: list[str] = []
    for run_n, spawned in zip(range(1, 6), ["yes", "yes", "no", "no", "no"]):
        _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode",
                       run_n=run_n, subagent_spawned=spawned)
    rows = p8_report.load_rows(tmp_path, "suite-v2.1.0", caveats)

    section = p8_report.build_b8_section(tmp_path, _cfg(), rows)

    assert "2/5" in section  # 2 yes out of 5


def test_build_b8_section_honors_not_applicable_canary_never_false_zero_percent(tmp_path):
    """A second synthetic harness with no delegation primitive: every row
    carries subagent_spawned='not_applicable'. The canary line for THIS
    group must read 'not_applicable', never '0%' (a harness incapable of
    delegating is not the same signal as one that tried and failed)."""
    caveats: list[str] = []
    for run_n in range(1, 6):
        _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-b", harness="hermes-native",
                       run_n=run_n, subagent_spawned="not_applicable")
    # also seed an unrelated group so the section has >1 (model,harness) row
    for run_n in range(1, 3):
        _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode",
                       run_n=run_n, subagent_spawned="no")
    rows = p8_report.load_rows(tmp_path, "suite-v2.1.0", caveats)

    section = p8_report.build_b8_section(tmp_path, _cfg(), rows)
    lines = section.splitlines()

    # Find the table row for model-b / hermes-native and check ITS canary
    # CELL specifically (the last '|'-delimited content cell), not just
    # that "not_applicable" appears somewhere on the line -- the row also
    # renders a Wilson-CI cell like "[X%, 100.0%]" whose upper bound can
    # itself contain the substring "0%" (a whole-line "0%" check would be
    # a false negative against a CORRECT implementation).
    hermes_lines = [ln for ln in lines if "model-b" in ln and "hermes-native" in ln]
    assert hermes_lines, "expected a row for model-b / hermes-native"
    cells = [c.strip() for c in hermes_lines[0].strip().strip("|").split("|")]
    canary_cell = cells[-1]
    assert canary_cell == "not_applicable"
    assert "0%" not in canary_cell


def test_build_b8_section_labels_source_suite(tmp_path):
    caveats: list[str] = []
    _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode", run_n=1)
    rows = p8_report.load_rows(tmp_path, "suite-v2.1.0", caveats)

    section = p8_report.build_b8_section(tmp_path, _cfg(), rows)

    assert "source_suite" in section
    assert "suite-v2.1.0" in section


def test_build_b8_section_never_blends_two_source_suites(tmp_path):
    """A (hypothetical) v2.0.0 B8 row and a v2.1.0 B8 row must render as
    two separately-labeled sub-blocks, mirroring every other battery
    section's _split_by_source_suite discipline -- never one combined
    k/N across both suite versions."""
    caveats: list[str] = []
    _write_b8_row(tmp_path, "suite-v2.0.0", model_id="model-a", harness="opencode", run_n=1)
    _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode", run_n=1)
    rows = p8_report.load_rows(tmp_path, "suite-v2.0.0", caveats)

    section = p8_report.build_b8_section(tmp_path, _cfg(), rows)

    assert "suite-v2.0.0" in section
    assert "suite-v2.1.0" in section
    # Each shard's group should show 1/1, never a blended 2/2.
    assert "1/1" in section
    assert "2/2" not in section


# ---------------------------------------------------------------------------
# 3. Zero B8 rows -- idempotent-safe (mirrors every other battery section)
# ---------------------------------------------------------------------------


def test_build_b8_section_zero_rows_from_other_battery_only(tmp_path):
    """Rows exist, but none are battery=8 -- must still degrade cleanly,
    exactly like every other battery's "(no B<n> rows yet)" convention."""
    caveats: list[str] = []
    path = tmp_path / "results" / "rows-suite-v2.1.0.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    b1_row = {
        "schema_version": 1, "row_id": "r1", "parent_id": None,
        "suite_version": "suite-v2.1.0", "fixture_sha": "sha", "code_sha": "unknown",
        "battery": 1, "task_id": "b1.finance-01", "condition": "cond=B1", "run_n": 1,
        "model_id": "model-a", "hf_repo": "o/r", "quant_file": "q.gguf",
        "quant_sha256": "qsha", "tier": "T1", "session_id": "s", "sampling": {},
        "ts": "2026-07-19T00:00:00+00:00", "request": {}, "response_meta": {},
        "det_checks": {}, "needs_judging": False, "metrics": {},
        "timing_authoritative": False, "artifacts": {}, "status": "ok",
        "error_detail": None, "tags": [],
    }
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(b1_row) + "\n")
    rows = p8_report.load_rows(tmp_path, "suite-v2.1.0", caveats)

    section = p8_report.build_b8_section(tmp_path, _cfg(), rows)
    assert "(no B8 data yet)" in section


# ---------------------------------------------------------------------------
# 4. Full report integrates B8 without breaking the existing sections
# ---------------------------------------------------------------------------


def test_build_report_end_to_end_integrates_b8_section(tmp_path):
    """Full `build_report()` over a minimal synthetic repo: real config/ +
    grading/ copied in (so cfg loads normally, same trick as
    test_report_b2.py's end-to-end test), plus a B1 row (so the B1 section
    has something to render) and two B8 rows for the SAME model/harness/
    task/run_n as an existing config-declared B8 task -- proving the B8
    section renders alongside every other section without crashing and
    without breaking the rest of the report."""
    import shutil
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    shutil.copytree(REPO_ROOT / "grading", tmp_path / "grading")
    (tmp_path / "suite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

    b1_row = {
        "schema_version": 1, "row_id": "r1", "parent_id": None,
        "suite_version": "suite-v2.0.0", "fixture_sha": "sha", "code_sha": "unknown",
        "battery": 1, "task_id": "b1.finance-01", "condition": "cond=B1", "run_n": 1,
        "model_id": "model-a", "hf_repo": "o/r", "quant_file": "q.gguf",
        "quant_sha256": "qsha", "tier": "T1", "session_id": "s", "sampling": {},
        "ts": "2026-07-19T00:00:00+00:00", "request": {}, "response_meta": {},
        "det_checks": {}, "needs_judging": False, "metrics": {},
        "timing_authoritative": False, "artifacts": {}, "status": "ok",
        "error_detail": None, "tags": [],
    }
    b1_path = tmp_path / "results" / f"rows-suite-v2.0.0.jsonl"
    b1_path.parent.mkdir(parents=True, exist_ok=True)
    with b1_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(b1_row) + "\n")

    _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode", run_n=1)
    _write_b8_row(tmp_path, "suite-v2.1.0", model_id="model-a", harness="opencode",
                   run_n=2, completion=False, first_failure_class="d")

    full_md, condensed = p8_report.build_report(tmp_path)

    assert "B8" in full_md
    assert "1/2" in full_md  # 1 completion out of 2 replicates
    assert "Wilson" in full_md
    assert "Traceback" not in full_md
    assert isinstance(condensed, str) and condensed
