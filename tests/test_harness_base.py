"""Contract test for the harness-independent core (Task 1): Trace/TraceEvent
schema, HarnessAdapter ABC, and MockHarnessAdapter (TESTPLAN Part 2 Phase 1)."""
from __future__ import annotations

import pytest

from llmtest.harness.base import MockHarnessAdapter
from llmtest.harness.trace import Trace, TraceEvent


def _scripted_events() -> list[TraceEvent]:
    return [
        TraceEvent(kind="turn", payload={"role": "user", "text": "do the task"}),
        TraceEvent(kind="tool_call", payload={"name": "read_file", "args": {}}),
        TraceEvent(kind="tool_result", payload={"ok": True}),
        TraceEvent(kind="turn", payload={"role": "assistant", "text": "done"}),
        TraceEvent(kind="terminal", payload={"reason": "task complete"}),
    ]


def test_mock_adapter_setup_run_teardown_sequence_and_scripted_trace():
    adapter = MockHarnessAdapter(scripted_events=_scripted_events(),
                                  terminal_status="completed",
                                  tokens_prompt=100, tokens_completion=50,
                                  subagent_spawned="no")

    adapter.setup(task={"id": "t1"}, endpoint="http://localhost:8080", workspace="/tmp/ws")
    trace = adapter.run()
    adapter.teardown()

    assert adapter.calls == ["setup", "run", "teardown"]

    assert isinstance(trace, Trace)
    assert trace.terminal_status == "completed"
    assert trace.steps == 2  # two "turn" events in the scripted list
    assert trace.tokens_prompt == 100
    assert trace.tokens_completion == 50
    assert trace.subagent_spawned == "no"
    assert adapter.version() == "mock-1.0"


def test_trace_from_events_derives_steps_from_turn_events():
    events = _scripted_events()
    trace = Trace.from_events(events, terminal_status="completed",
                               tokens_prompt=10, tokens_completion=5,
                               subagent_spawned="not_applicable")
    assert trace.steps == sum(1 for e in events if e.kind == "turn")


def test_invalid_terminal_status_raises_value_error():
    with pytest.raises(ValueError):
        Trace.from_events(_scripted_events(), terminal_status="bogus",
                           tokens_prompt=0, tokens_completion=0,
                           subagent_spawned="no")


def test_invalid_subagent_spawned_raises_value_error():
    with pytest.raises(ValueError):
        Trace.from_events(_scripted_events(), terminal_status="completed",
                           tokens_prompt=0, tokens_completion=0,
                           subagent_spawned="maybe")


def test_invalid_event_kind_raises_value_error():
    with pytest.raises(ValueError):
        TraceEvent(kind="not_a_kind", payload={})
