"""Battery-aware judging dimension: B1 scores per business unit, B2 per axis.
Replaces the B1-hardcoded _unit_from_task_id seam so the packet builder,
aggregator, and report share one resolver."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

JUDGED_B2_AXES = (5, 8)

@dataclass(frozen=True)
class Dim:
    kind: str   # "unit" | "axis"
    value: str

def _unit_from_b1(task_id: str) -> str:
    after = task_id.split(".", 1)[1] if "." in task_id else task_id
    unit, sep, _num = after.rpartition("-")
    if not sep:
        raise ValueError(f"B1 task_id not in b1.<unit>-NN form: {task_id!r}")
    return unit

def resolve_dims(battery: int, task_id: str, axes: list[int] | None) -> list[Dim]:
    if battery == 1:
        if not task_id.startswith("b1."):
            raise ValueError(f"B1 resolver got non-b1 task_id: {task_id!r}")
        return [Dim("unit", _unit_from_b1(task_id))]
    if battery == 2:
        if axes is None:
            raise ValueError(f"B2 resolver requires axes for {task_id!r}")
        return [Dim("axis", f"axis{a}") for a in sorted(set(axes) & set(JUDGED_B2_AXES))]
    raise ValueError(f"no judging dimension defined for battery {battery}")

def rubric_ref(dim: Dim) -> str:
    return f"anchors/{dim.value}.md" if dim.kind == "unit" else f"fixture:rubric.{dim.value}"

def cal_ref(dim: Dim, root: Path | None = None) -> Path:
    base = Path(root) if root else Path(".")
    if dim.kind == "axis":
        return base / "grading" / "calibration" / "b2" / f"{dim.value}.yaml"
    raise ValueError("B1 unit CAL resolution stays in packets.py; cal_ref is B2-only")
