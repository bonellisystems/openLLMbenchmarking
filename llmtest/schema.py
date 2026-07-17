"""Row schema — THE interface (TESTPLAN §7.2). schema_version 1."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

SCHEMA_VERSION = 1
STATUSES = {"ok", "error", "dnf", "excluded"}


def canonical_condition(parts: dict, order: list[str]) -> str:
    unknown = set(parts) - set(order)
    if unknown:
        raise ValueError(f"unknown condition keys: {sorted(unknown)}")
    return ";".join(f"{k}={parts[k]}" for k in order if k in parts)


def compute_row_id(*, suite_version, model_id, quant_sha256, battery,
                   task_id, fixture_sha, condition, run_n) -> str:
    joined = "|".join([suite_version, model_id, quant_sha256, str(battery),
                       task_id, fixture_sha, condition, str(run_n)])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ResultRow:
    schema_version: int
    row_id: str
    parent_id: str | None
    suite_version: str
    fixture_sha: str
    code_sha: str            # provenance only — never in the key
    battery: int
    task_id: str
    condition: str
    run_n: int
    model_id: str
    hf_repo: str
    quant_file: str
    quant_sha256: str
    tier: str
    session_id: str
    sampling: dict
    ts: str
    request: dict            # {fixture_id, prompt_sha256}
    response_meta: dict      # tokens_in/out, ttft_ms, decode_tps, pp_tps, finish_reason,
                             # truncated, n_drafted, n_accepted, accept_rate
    det_checks: dict
    needs_judging: bool
    metrics: dict
    timing_authoritative: bool
    artifacts: dict          # name -> {sha256, relpath}
    status: str
    error_detail: str | None
    tags: list

    @classmethod
    def new(cls, *, suite_version, model_id, hf_repo, quant_file, quant_sha256,
            tier, battery, task_id, fixture_sha, condition, run_n, session_id,
            parent_id=None, code_sha="unknown", sampling=None, request=None,
            response_meta=None, det_checks=None, needs_judging=False,
            metrics=None, timing_authoritative=False, artifacts=None,
            status="ok", error_detail=None, tags=None) -> "ResultRow":
        return cls(
            schema_version=SCHEMA_VERSION,
            row_id=compute_row_id(
                suite_version=suite_version, model_id=model_id,
                quant_sha256=quant_sha256, battery=battery, task_id=task_id,
                fixture_sha=fixture_sha, condition=condition, run_n=run_n),
            parent_id=parent_id, suite_version=suite_version,
            fixture_sha=fixture_sha, code_sha=code_sha, battery=battery,
            task_id=task_id, condition=condition, run_n=run_n,
            model_id=model_id, hf_repo=hf_repo, quant_file=quant_file,
            quant_sha256=quant_sha256, tier=tier, session_id=session_id,
            sampling=sampling or {}, ts=_now(), request=request or {},
            response_meta=response_meta or {}, det_checks=det_checks or {},
            needs_judging=needs_judging, metrics=metrics or {},
            timing_authoritative=timing_authoritative,
            artifacts=artifacts or {}, status=status,
            error_detail=error_detail, tags=list(tags or []),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionRow:
    schema_version: int
    session_id: str
    ts: str
    runtime: str             # llamacpp-fork | ollama | vllm
    runtime_build: str
    normalized_config: dict  # {ctx, kv_dtype, flash_attn, spec_type, spec_params, parallel}
    raw_invocation: str
    hardware_sku: str
    measured_usable_vram_gb: float
    tp_degree: int
    topology: str | None
    driver_env: dict
    power_mode: str
    ac_state: str
    timing_authoritative: bool

    def to_dict(self) -> dict:
        return asdict(self)


_REQUIRED = [f for f in ResultRow.__dataclass_fields__]  # all fields required present
_BOOLS = ["needs_judging", "timing_authoritative"]
_DICTS = ["sampling", "request", "response_meta", "det_checks", "metrics", "artifacts"]


def validate_row(d: dict) -> list[str]:
    errs = []
    for f in _REQUIRED:
        if f not in d:
            errs.append(f"missing field: {f}")
    if errs:
        return errs
    if d["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    for f in _BOOLS:
        if not isinstance(d[f], bool):
            errs.append(f"{f} must be bool")
    for f in _DICTS:
        if not isinstance(d[f], dict):
            errs.append(f"{f} must be dict")
    if not isinstance(d["tags"], list):
        errs.append("tags must be list")
    if d["status"] not in STATUSES:
        errs.append(f"status must be one of {sorted(STATUSES)}")
    expect = compute_row_id(
        suite_version=d["suite_version"], model_id=d["model_id"],
        quant_sha256=d["quant_sha256"], battery=d["battery"],
        task_id=d["task_id"], fixture_sha=d["fixture_sha"],
        condition=d["condition"], run_n=d["run_n"])
    if d["row_id"] != expect:
        errs.append("row_id does not match idempotency key hash")
    try:
        json.dumps(d)
    except (TypeError, ValueError):
        errs.append("row is not JSON-serializable")
    return errs


# --- Judgment rows (TESTPLAN 6.1/7.5, Task 7) ---
# One row per (packet_id, judge_id, letter); idempotency key is that triple
# (Store.append_judgment). A failed judge call after retry writes ONE row
# per (packet_id, judge_id) with letter="-", score=None, status="error".

JUDGMENT_STATUSES = {"ok", "error"}
_JUDGMENT_REQUIRED = [
    "schema_version", "packet_id", "judge_id", "judge_model_pin", "judge_cli_version",
    "letter", "model_id", "score", "reason", "rank", "ts", "status",
]


def validate_judgment(d: dict) -> list[str]:
    errs = []
    for f in _JUDGMENT_REQUIRED:
        if f not in d:
            errs.append(f"missing field: {f}")
    if errs:
        return errs
    if d["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    if d["status"] not in JUDGMENT_STATUSES:
        errs.append(f"status must be one of {sorted(JUDGMENT_STATUSES)}")
    score = d["score"]
    if d["status"] == "error":
        if score is not None:
            errs.append("score must be None when status == 'error'")
    else:
        if isinstance(score, bool) or not isinstance(score, int):
            errs.append(f"score must be an int 0-10 when status != 'error': {score!r}")
        elif not (0 <= score <= 10):
            errs.append(f"score out of range 0-10: {score}")
    if d["letter"] == "-" and d["status"] != "error":
        errs.append("letter '-' requires status == 'error'")
    if d["status"] == "error" and d["letter"] != "-":
        errs.append("status == 'error' requires letter == '-'")
    for f in ("packet_id", "judge_id", "letter"):
        if not isinstance(d[f], str) or not d[f]:
            errs.append(f"{f} must be a non-empty string")
    try:
        json.dumps(d)
    except (TypeError, ValueError):
        errs.append("judgment row is not JSON-serializable")
    return errs
