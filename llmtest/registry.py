"""Config loaders + fits() — ONE code path for tier placement AND VRAM preflight (TESTPLAN 3.1/7.3)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    root: Path
    tiers: dict
    registry: dict
    suite: dict
    budgets: dict
    judges: dict
    runtime_pins: dict


def _load(root: Path, name: str) -> dict:
    with (root / "config" / f"{name}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(root: Path | str) -> Config:
    root = Path(root)
    cfg = Config(root=root, tiers=_load(root, "tiers"),
                 registry=_load(root, "registry"), suite=_load(root, "suite"),
                 budgets=_load(root, "budgets"), judges=_load(root, "judges"),
                 runtime_pins=_load(root, "runtime_pins"))
    sv = os.environ.get("LLMTEST_SUITE_VERSION")
    if sv:
        cfg.suite["suite_version"] = sv
    return cfg


@dataclass
class FitResult:
    fits: bool
    fits_short_context: bool
    detail: str


def fits(model: dict, tiers_cfg: dict, kv_dtype: str, *, tier: str) -> FitResult:
    t = tiers_cfg["tiers"][tier]
    usable = float(t["usable_gb"])
    weights = float(model["weights_gb"])
    overhead = float(tiers_cfg["runtime_overhead_gb"])
    kv_per_tok = (model.get("kv_bytes_per_token") or {}).get(kv_dtype) \
        or tiers_cfg["kv_bytes_per_token"][kv_dtype]
    if model.get("arch", {}).get("hybrid_linear_attn"):
        kv_per_tok = kv_per_tok * 0.25   # ~75%-linear attention discount
    kv_gb = kv_per_tok * int(tiers_cfg["kv_floor_ctx"]) / (1024 ** 3)
    need_full = weights + kv_gb + overhead
    if need_full <= usable:
        return FitResult(True, False, f"fits {need_full:.1f}<={usable:.1f} GB @128k {kv_dtype}")
    if weights + overhead <= usable:
        return FitResult(True, True, f"weights fit; 128k KV misses: need {need_full:.1f} GB")
    return FitResult(False, False, f"weights alone exceed tier: {weights + overhead:.1f}>{usable:.1f}")
