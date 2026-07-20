"""Normalized Trace schema (Task 1) -- the harness-independent shape every
`HarnessAdapter.run()` returns, regardless of which real agent harness
produced it.

`steps` is DERIVED, not supplied directly: it is the count of `"turn"`
events in `events`. Build a `Trace` via `Trace.from_events(...)` rather than
the dataclass constructor directly so that derivation happens in one place;
`__post_init__` enforces the invariant either way, so direct construction
with a mismatched `steps` raises `ValueError` rather than silently
constructing an inconsistent `Trace`. Token counts are stored fields
populated by the adapter (e.g. from scripted
data in `MockHarnessAdapter`, or a real harness's own accounting in later
tasks) -- this module has no opinion on where they come from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

VALID_TERMINAL_STATUSES = {"completed", "failed-task", "budget-exceeded",
                            "infra-error", "killed"}
VALID_SUBAGENT_SPAWNED = {"yes", "no", "not_applicable"}
VALID_EVENT_KINDS = {"turn", "tool_call", "tool_result", "subagent_spawn", "terminal"}


@dataclass
class TraceEvent:
    """One normalized event in a harness run's timeline."""

    kind: str
    payload: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_EVENT_KINDS:
            raise ValueError(
                f"invalid TraceEvent.kind: {self.kind!r} (must be one of "
                f"{sorted(VALID_EVENT_KINDS)})")


@dataclass
class Trace:
    """Normalized harness-run trace -- the common output shape all
    `HarnessAdapter.run()` implementations produce."""

    events: list[TraceEvent]
    terminal_status: str
    steps: int
    tokens_prompt: int
    tokens_completion: int
    subagent_spawned: str

    def __post_init__(self) -> None:
        if self.terminal_status not in VALID_TERMINAL_STATUSES:
            raise ValueError(
                f"invalid Trace.terminal_status: {self.terminal_status!r} "
                f"(must be one of {sorted(VALID_TERMINAL_STATUSES)})")
        if self.subagent_spawned not in VALID_SUBAGENT_SPAWNED:
            raise ValueError(
                f"invalid Trace.subagent_spawned: {self.subagent_spawned!r} "
                f"(must be one of {sorted(VALID_SUBAGENT_SPAWNED)})")
        expected = sum(1 for e in self.events if e.kind == "turn")
        if self.steps != expected:
            raise ValueError(
                f"steps ({self.steps}) must equal the number of 'turn' events "
                f"({expected})")

    @classmethod
    def from_events(cls, events: list[TraceEvent], terminal_status: str,
                     tokens_prompt: int, tokens_completion: int,
                     subagent_spawned: str) -> Trace:
        """Construct a `Trace`, deriving `steps` as the count of `"turn"`
        events in `events`. This is the intended construction path -- the
        dataclass constructor itself requires `steps` explicitly, but
        `__post_init__` enforces that it matches the turn-event count
        regardless of construction path, so going through here (rather than
        computing `steps` by hand) is the easiest way to satisfy the
        invariant."""
        steps = sum(1 for e in events if e.kind == "turn")
        return cls(events=events, terminal_status=terminal_status, steps=steps,
                    tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
                    subagent_spawned=subagent_spawned)
