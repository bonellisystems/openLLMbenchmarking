"""HarnessAdapter ABC + MockHarnessAdapter (Task 1) -- the interface a real
agent-harness driver (Claude Code, etc., in later tasks) implements to
produce a normalized `Trace`. No real harness, Docker, or network here --
`MockHarnessAdapter` is a deterministic scripted stand-in for the contract
test.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from llmtest.harness.trace import Trace, TraceEvent


class HarnessAdapter(ABC):
    """Drives one model through one agent harness for one task.

    Lifecycle is strictly `setup()` -> `run()` -> `teardown()`; `version()`
    may be called at any point and reports the harness's own version string
    (not this adapter's).
    """

    @abstractmethod
    def setup(self, task, endpoint, workspace) -> None: ...

    @abstractmethod
    def run(self) -> Trace: ...

    @abstractmethod
    def teardown(self) -> None: ...

    @abstractmethod
    def version(self) -> str: ...


class MockHarnessAdapter(HarnessAdapter):
    """Deterministic scripted `HarnessAdapter` -- no real harness process,
    no I/O. Records the `setup`/`run`/`teardown` call sequence in `self.calls`
    so the contract test can assert ordering; `run()` just packages the
    scripted event list (via `Trace.from_events`) and `version()` returns a
    fixed string.
    """

    def __init__(self, scripted_events: list[TraceEvent], terminal_status: str,
                 tokens_prompt: int, tokens_completion: int, subagent_spawned: str):
        self.scripted_events = scripted_events
        self.terminal_status = terminal_status
        self.tokens_prompt = tokens_prompt
        self.tokens_completion = tokens_completion
        self.subagent_spawned = subagent_spawned
        self.calls: list[str] = []

    def setup(self, task, endpoint, workspace) -> None:
        self.calls.append("setup")

    def run(self) -> Trace:
        self.calls.append("run")
        return Trace.from_events(self.scripted_events,
                                  terminal_status=self.terminal_status,
                                  tokens_prompt=self.tokens_prompt,
                                  tokens_completion=self.tokens_completion,
                                  subagent_spawned=self.subagent_spawned)

    def teardown(self) -> None:
        self.calls.append("teardown")

    def version(self) -> str:
        return "mock-1.0"
