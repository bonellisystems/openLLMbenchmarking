"""Battery plugin ABC (TESTPLAN 7.4) — minimal by design; the ROW SCHEMA is the real interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class WorkItem:
    row_id: str
    model_id: str
    battery: int
    task_id: str
    condition: str
    run_n: int
    payload: dict = field(default_factory=dict)


class Battery(ABC):
    id: int

    @abstractmethod
    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]: ...

    def preflight(self, ctx) -> list[dict]:
        return []

    @abstractmethod
    def execute(self, item: WorkItem, ctx) -> list[dict]: ...

    def build_judge_packets(self, rows) -> list:
        return []


_REGISTRY: dict[int, type] = {}


def register(cls):
    _REGISTRY[cls.id] = cls
    return cls


def get(battery_id: int) -> Battery:
    if battery_id not in _REGISTRY:
        # import battery modules lazily so registration side-effects run
        if battery_id == 5:
            from llmtest.batteries import b5_serving  # noqa: F401
    if battery_id not in _REGISTRY:
        raise KeyError(f"unknown battery {battery_id}")
    return _REGISTRY[battery_id]()
