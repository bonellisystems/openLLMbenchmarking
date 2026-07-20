"""TDD tests for the B8 first-failure classification pipeline (Task 8).

No live classifier CLI is invoked anywhere here -- classifiers are either
tiny `.classify(text) -> label` fakes, or (implicitly, via `RaisingFake`) a
guard proving the panel was never consulted for a deterministic verdict.
"""
from __future__ import annotations

import pytest

from llmtest.harness.failure_class import (classify_first_failure, panel_classify,
                                            parse_categorical_reply)
from llmtest.harness.tasks import B8Task
from llmtest.harness.trace import Trace, TraceEvent
from llmtest.judging.adapters import JudgeReply


# -- fixtures ------------------------------------------------------------


def _task(prompt: str = "Fix add() so it returns the sum, not the difference.") -> B8Task:
    return B8Task(
        id="edit-01", shape="edit", setup_repo_sha="sha-x",
        allowed_tools=["read_file", "write_file"],
        budgets={"wall_clock_s": 60, "tokens": 2000, "steps": 6},
        oracle=["bash", "-c", "true"], protected_shas={},
        task_version="1.0.0", fixture_sha="sha-x",
        setup_repo={"add.py": "def add(a, b):\n    return a - b\n"},
        oracle_files={"oracle_test.sh": "#!/bin/bash\necho PASS\n"},
        protected_paths=[], allowed_diff_paths=["add.py"],
        prompt=prompt, path=None,
    )


class RaisingClassifier:
    """A classifier that fails the test if the panel ever calls it --
    used to assert deterministic verdicts short-circuit before the panel."""

    def classify(self, blinded_text: str) -> str:
        raise AssertionError("panel must not be consulted when a deterministic verdict applies")


class FakeClassifier:
    def __init__(self, label):
        self.label = label

    def classify(self, blinded_text: str) -> str:
        return self.label


def _completed_trace() -> Trace:
    """A trace whose HARNESS completed cleanly (tool calls parsed, tool
    ran, terminal_status == 'completed') -- the (c) task-logic shape: only
    the oracle's completed=False flag (supplied separately by the caller)
    reveals this run as a FAILURE."""
    events = [
        TraceEvent(kind="turn", payload={}),
        TraceEvent(kind="tool_call", payload={
            "tool": "write_file", "callID": "1",
            "input": {"path": "add.py", "content": "def add(a, b):\n    return a + 1\n"},
            "parsed": True,
        }),
        TraceEvent(kind="tool_result", payload={"status": "completed", "output": "ok"}),
        TraceEvent(kind="terminal", payload={"finish": "stop"}),
    ]
    return Trace.from_events(events, terminal_status="completed",
                              tokens_prompt=20, tokens_completion=10,
                              subagent_spawned="no")


# -- 1. unparsed tool call -> deterministic (a), panel not consulted -----


def test_unparsed_tool_call_is_deterministic_a():
    events = [
        TraceEvent(kind="turn", payload={}),
        TraceEvent(kind="tool_call", payload={
            "tool": "write_file", "callID": "1", "input": "not-valid-json{{{",
            "parsed": False,
        }),
        TraceEvent(kind="tool_result", payload={"status": "error", "output": None}),
        TraceEvent(kind="terminal", payload={}),
    ]
    trace = Trace.from_events(events, terminal_status="failed-task",
                               tokens_prompt=10, tokens_completion=0,
                               subagent_spawned="no")

    label, source = classify_first_failure(
        trace, _task(), classifiers=[RaisingClassifier()])

    assert (label, source) == ("a", "deterministic")


# -- 2. harness-error terminal -> deterministic (d), panel not consulted -


def test_infra_error_terminal_is_deterministic_d():
    events = [TraceEvent(kind="terminal", payload={"launch_error": "binary not found"})]
    trace = Trace.from_events(events, terminal_status="infra-error",
                               tokens_prompt=0, tokens_completion=0,
                               subagent_spawned="no")

    label, source = classify_first_failure(
        trace, _task(), classifiers=[RaisingClassifier()])

    assert (label, source) == ("d", "deterministic")


# -- 3. completed-but-wrong-logic -> panel, majority wins -----------------


def test_completed_but_wrong_logic_routes_to_panel_majority():
    trace = _completed_trace()
    classifiers = [FakeClassifier("c"), FakeClassifier("c"), FakeClassifier("b")]

    label, source = classify_first_failure(
        trace, _task(), completed=False, classifiers=classifiers)

    assert (label, source) == ("c", "panel")


# -- 4. panel tie -> unknown -----------------------------------------------


def test_panel_tie_is_unknown():
    trace = _completed_trace()

    three_way = [FakeClassifier("b"), FakeClassifier("c"), FakeClassifier("d")]
    label, source = classify_first_failure(
        trace, _task(), completed=False, classifiers=three_way)
    assert (label, source) == ("unknown", "panel")

    even_split = [FakeClassifier("c"), FakeClassifier("b")]
    label, source = classify_first_failure(
        trace, _task(), completed=False, classifiers=even_split)
    assert (label, source) == ("unknown", "panel")


# -- 5. passing run -> not_applicable, panel not consulted ----------------


def test_passing_run_is_not_applicable():
    trace = _completed_trace()

    label, source = classify_first_failure(
        trace, _task(), completed=True, classifiers=[RaisingClassifier()])

    assert (label, source) == ("not_applicable", "deterministic")


# -- 6. abstention: invalid label doesn't count, but is reported ----------


def test_panel_abstention_is_reported():
    trace = _completed_trace()
    classifiers = [FakeClassifier("a"), FakeClassifier("a"), FakeClassifier("not-a-real-label")]

    result = panel_classify(trace, _task(), classifiers, completed=False)
    assert result.label == "a"
    assert result.abstentions == 1
    assert result.votes == {"a": 2}

    label, source = classify_first_failure(
        trace, _task(), completed=False, classifiers=classifiers)
    assert (label, source) == ("a", "panel")


# -- adapter-shaped classifier (BaseAdapter.invoke -> JudgeReply), not just
# the simpler .classify() shape -- covers parse_categorical_reply and the
# .invoke() branch of _invoke_classifier, both named as a supported
# classifier shape in the brief. Still no subprocess: FakeAdapterClassifier
# stands in for a real judge-CLI adapter's .invoke() return value.


class FakeAdapterClassifier:
    def __init__(self, raw: str):
        self._raw = raw

    def invoke(self, packet_text, expected_letters, timeout=300, packet_path=None) -> JudgeReply:
        return JudgeReply(raw=self._raw, parsed=None, error=None)


def test_parse_categorical_reply_extracts_label_from_prose():
    label, error = parse_categorical_reply(
        'Sure, here is my answer:\n{"label": "C", "reason": "wrong sum logic"}\nDone.')
    assert (label, error) == ("c", None)

    label, error = parse_categorical_reply("not json at all")
    assert label is None
    assert error is not None


def test_adapter_shaped_classifier_drives_panel_via_invoke():
    trace = _completed_trace()
    classifiers = [
        FakeAdapterClassifier('{"label": "c"}'),
        FakeAdapterClassifier('{"label": "c"}'),
        FakeAdapterClassifier('garbled non-JSON reply'),  # abstains
    ]

    result = panel_classify(trace, _task(), classifiers, completed=False)
    assert result.label == "c"
    assert result.abstentions == 1
    assert result.votes == {"c": 2}

    label, source = classify_first_failure(
        trace, _task(), completed=False, classifiers=classifiers)
    assert (label, source) == ("c", "panel")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
