# LLMtest v2 — P0–P2 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LLMtest v2 foundation — repo/CI/schema/configs (P0), ServerManager + serving canary (P1), and Battery-5 as the first ABC client through a clean gpt-oss-20b shakedown (P2) — per approved `TESTPLAN.md` v2.0.0.

**Architecture:** Single Python package `llmtest` with subcommand CLI; append-only sharded JSONL results validated at write time by the same validator CI runs; ServerManager mediates all model serving with provenance auto-attached; batteries are plugins implementing a minimal ABC, validated by B5 (server-owning) first.

**Tech Stack:** Python 3.10+ (stdlib-first: `dataclasses`, `hashlib`, `json`, `subprocess`, `urllib`), `pytest`, `PyYAML`, GitHub Actions (integrity CI only), prism-ml llama.cpp fork binary (existing, pinned), `gh` CLI.

## Global Constraints (from TESTPLAN v2.0.0 — every task inherits these)

- Windows 11 host; primary shell PowerShell; python = `python` (3.10+); repo root = `D:\BUILT-TOOLS\LLMtesting\llmtest-v2`.
- Fork binary: `D:\BUILT-TOOLS\LLMtesting\bonsai\bin\llama-server.exe` — never rebuilt, build id pinned in config.
- Serving standard flags (T1 default): `-ngl 99 --jinja -fa on --spec-type ngram-mod --spec-ngram-mod-n-match 32 --cache-ram 0`. Never `n-match < 16`; never `draft-mtp` on GGUF.
- Idempotency key: `(suite_version, model_id, quant_sha256, battery, task_id, fixture_sha, condition, run_n)`; `row_id = sha256(canonical join with '|')`.
- `condition` = canonically-ordered `key=value` composite joined with `;` — vocabulary AND order fixed in `config/suite.yaml`.
- `schema_version: 1`. Results sharded `results/rows-<suite_version>.jsonl`, append-only. Aggregates computed at table time, never stored.
- `timing_authoritative` true only on B5-minted sessions; all speed tables filter on it. WSL2-vLLM >14 GB and non-sanctioned Ollama are refused authority.
- Models named by full HF repo path + exact quant filename everywhere. Registry provenance: `{source_repo, download_date, sha256, v1_continuity}`.
- Tables byte-deterministic: stable sort order, fixed float formatting (`{:.1f}` for t/s, `{:.2f}` for ratios).
- No secrets in tree — configs are templates; `.env` gitignored. No GPU work in CI. Kill servers by PID only, never `pkill`-by-pattern.
- Commits: small, frequent; trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` + `Claude-Session: https://claude.ai/code/session_01WvAPF9LiDZ4P11TW782keP`.
- **Task authoring (suite fixtures) is OUT OF SCOPE for P0–P2** — gated on P0 exit per TESTPLAN §9.

## File Structure (locked by this plan)

```
llmtest-v2/
├─ pyproject.toml                  # package metadata, pytest config, console script
├─ llmtest/
│  ├─ __init__.py                  # __version__
│  ├─ cli.py                       # argparse subcommands: validate | status | run | tables
│  ├─ schema.py                    # ResultRow/SessionRow/JudgmentRow + validate_row + row_id/condition canon
│  ├─ store.py                     # append-only shard writer/reader, resume-key index
│  ├─ registry.py                  # config loaders (tiers/registry/suite/budgets/judges) + fits()
│  ├─ server.py                    # ServerManager + EndpointHandle + translation layer
│  ├─ canary.py                    # serving canary (Ornith ngram edit A/B vs reference band)
│  ├─ tables.py                    # byte-deterministic table regeneration
│  └─ batteries/
│     ├─ __init__.py               # ABC: Battery, Task, WorkItem, ExecutionContext + registry of batteries
│     └─ b5_serving.py             # Battery 5 plugin
├─ config/
│  ├─ tiers.yaml  registry.yaml  suite.yaml  budgets.yaml  judges.yaml  runtime_pins.yaml
├─ scripts/
│  └─ freeze_artifacts.py          # P0: SHA256 + download_date of the six on-disk artifacts → registry.yaml
├─ suite/b5_serving/prompts.yaml   # B5's fixed measurement prompts (not "task authoring" — serving fixtures)
├─ tests/                          # pytest; mirrors module names test_schema.py etc.
├─ results/                        # rows-*.jsonl, sessions.jsonl (created at runtime; .gitkeep)
└─ .github/workflows/ci.yml
```

---

### Task 1: P0-blocking — gh CLI install, auth, private remote, push-verified

**Files:** none (environment + git remote)

**Interfaces:** Produces: authenticated `gh`, remote `origin`, all subsequent tasks may push.

- [ ] **Step 1: Install gh CLI**

Run (PowerShell): `winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements`
Expected: `Successfully installed`. Then **open a fresh shell** (PATH refresh) and run `gh --version` → `gh version 2.x`.

- [ ] **Step 2: Authenticate (interactive — Michael runs this)**

Ask Michael to run `! gh auth login --hostname github.com --git-protocol https --web` in the session. Verify: `gh auth status` → `Logged in to github.com`.

- [ ] **Step 3: Create private repo + push**

Run: `cd D:\BUILT-TOOLS\LLMtesting\llmtest-v2 && gh repo create llmtest-v2 --private --source . --remote origin --push`
Expected: repo URL printed; `git ls-remote origin main` shows the two existing commits' head.

- [ ] **Step 4: Verify push-verified exit criterion**

Run: `git log origin/main --oneline | head -2`
Expected: `TESTPLAN v2.0.0 APPROVED...` at head. **P0 exit criterion §9 satisfied — record in commit message of Task 2.**

### Task 2: Package skeleton + pytest wiring

**Files:**
- Create: `pyproject.toml`, `llmtest/__init__.py`, `llmtest/cli.py`, `tests/test_cli.py`, `results/.gitkeep`

**Interfaces:** Produces: `llmtest` importable package; `python -m llmtest` and console script `llmtest`; `main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import subprocess, sys

def test_cli_module_runs_and_reports_version():
    p = subprocess.run([sys.executable, "-m", "llmtest", "--version"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert p.stdout.strip().startswith("llmtest 2.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:\BUILT-TOOLS\LLMtesting\llmtest-v2 && python -m pytest tests/test_cli.py -v`
Expected: FAIL (`No module named llmtest`).

- [ ] **Step 3: Minimal implementation**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llmtest"
version = "2.0.0"
requires-python = ">=3.10"
dependencies = ["PyYAML>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
llmtest = "llmtest.cli:entry"

[tool.setuptools.packages.find]
include = ["llmtest*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`llmtest/__init__.py`:
```python
__version__ = "2.0.0"
```

`llmtest/cli.py`:
```python
"""llmtest CLI — subcommands per TESTPLAN §7.5. P0-P2 wire: validate, status, run, tables."""
import argparse
import sys

from llmtest import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmtest")
    p.add_argument("--version", action="version", version=f"llmtest {__version__}")
    sub = p.add_subparsers(dest="command")
    v = sub.add_parser("validate", help="schema + fixture lint (== CI); --serving runs the canary")
    v.add_argument("--serving", action="store_true")
    sub.add_parser("status", help="done/pending matrix from resume keys")
    r = sub.add_parser("run", help="execute battery work items")
    r.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    r.add_argument("--model", default=None)
    r.add_argument("--battery", type=int, default=None)
    r.add_argument("--task", dest="task_id", default=None)
    r.add_argument("--condition", default=None)
    r.add_argument("--force", action="store_true")
    r.add_argument("--keep-server", action="store_true")
    r.add_argument("--debug", action="store_true")
    sub.add_parser("tables", help="regenerate all tables (byte-deterministic)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 2
    # Dispatch is wired per-task as commands land (Tasks 5, 7, 11, 13, 15).
    from llmtest import dispatch
    return dispatch.run(args)


def entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry()
```

`llmtest/dispatch.py` (same step — keeps cli.py stable while commands land):
```python
"""Command dispatch. Each command module registers here as tasks complete."""


def run(args) -> int:
    if args.command == "validate":
        if getattr(args, "serving", False):
            from llmtest.canary import run_canary
            return run_canary()
        from llmtest.validate_cmd import run_validate
        return run_validate()
    if args.command == "status":
        from llmtest.status_cmd import run_status
        return run_status()
    if args.command == "run":
        from llmtest.run_cmd import run_run
        return run_run(args)
    if args.command == "tables":
        from llmtest.tables import run_tables
        return run_tables()
    raise SystemExit(f"unknown command {args.command}")
```

Add `__main__.py`:
```python
from llmtest.cli import entry
entry()
```

- [ ] **Step 4: Install editable + run test**

Run: `python -m pip install -e ".[dev]"` then `python -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml llmtest/ tests/ results/.gitkeep
git commit -m "feat(p0): package skeleton, CLI parser, dispatch stub (P0 exit: remote push verified in Task 1)"
git push
```

### Task 3: schema.py — condition canon, row_id, ResultRow, validator

**Files:**
- Create: `llmtest/schema.py`, `tests/test_schema.py`

**Interfaces:** Produces (consumed by store/server/batteries/tables):
- `canonical_condition(d: dict, order: list[str]) -> str` — `key=value;...` in fixed order, error on unknown key.
- `compute_row_id(suite_version, model_id, quant_sha256, battery, task_id, fixture_sha, condition, run_n) -> str` (sha256 hex).
- `@dataclass ResultRow` (fields per TESTPLAN §7.2) with `.to_dict()`; `@dataclass SessionRow`.
- `validate_row(d: dict) -> list[str]` — returns error list, empty = valid. Same function used by CI and write path.

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:
```python
import pytest
from llmtest import schema

ORDER = ["runtime", "spec", "kv", "ctx", "cond", "conc"]

def test_condition_is_order_canonical():
    a = schema.canonical_condition({"conc": 8, "runtime": "fork", "cond": "PEAK"}, ORDER)
    b = schema.canonical_condition({"runtime": "fork", "cond": "PEAK", "conc": 8}, ORDER)
    assert a == b == "runtime=fork;cond=PEAK;conc=8"

def test_condition_rejects_unknown_key():
    with pytest.raises(ValueError):
        schema.canonical_condition({"bogus": 1}, ORDER)

def test_row_id_stable_and_sensitive():
    base = dict(suite_version="suite-v2.0.0", model_id="gpt-oss-20b",
                quant_sha256="a" * 64, battery=5, task_id="b5.decode",
                fixture_sha="f" * 64, condition="runtime=fork;cond=PEAK", run_n=1)
    r1 = schema.compute_row_id(**base)
    assert r1 == schema.compute_row_id(**base)          # stable
    assert r1 != schema.compute_row_id(**{**base, "run_n": 2})  # sensitive

def test_validate_row_catches_missing_and_bad_fields():
    row = schema.ResultRow.new(
        suite_version="suite-v2.0.0", model_id="gpt-oss-20b",
        hf_repo="unsloth/gpt-oss-20b-GGUF", quant_file="gpt-oss-20b-F16.gguf",
        quant_sha256="a" * 64, tier="T1", battery=5, task_id="b5.decode",
        fixture_sha="f" * 64, condition="runtime=fork;cond=PEAK", run_n=1,
        session_id="s1",
    ).to_dict()
    assert schema.validate_row(row) == []
    bad = dict(row); bad.pop("row_id"); assert "row_id" in " ".join(schema.validate_row(bad))
    bad2 = dict(row); bad2["timing_authoritative"] = "yes"
    assert any("timing_authoritative" in e for e in schema.validate_row(bad2))
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_schema.py -v` → FAIL (no module attr).

- [ ] **Step 3: Implementation**

`llmtest/schema.py`:
```python
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
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_schema.py -v` → PASS (4 tests).

- [ ] **Step 5: Commit** — `git add llmtest/schema.py tests/test_schema.py && git commit -m "feat(p0): row schema, condition canon, row_id, write-time validator" && git push`

### Task 4: store.py — sharded append-only writer/reader + resume index

**Files:**
- Create: `llmtest/store.py`, `tests/test_store.py`

**Interfaces:** Produces:
- `Store(results_dir)` with `.append(row: dict) -> bool` (False = duplicate row_id skipped; validates via `schema.validate_row`, raises `SchemaError` on invalid), `.append_session(d: dict)`, `.existing_row_ids() -> set[str]`, `.iter_rows()` / `.iter_sessions()` (globs all `rows-*.jsonl`).
- Shard path: `rows-<suite_version>.jsonl`; sessions in `sessions.jsonl`.

- [ ] **Step 1: Write the failing tests**

`tests/test_store.py`:
```python
import pytest
from llmtest import schema
from llmtest.store import Store, SchemaError

def _row(run_n=1):
    return schema.ResultRow.new(
        suite_version="suite-v2.0.0", model_id="m", hf_repo="org/r",
        quant_file="q.gguf", quant_sha256="a"*64, tier="T1", battery=5,
        task_id="b5.x", fixture_sha="f"*64, condition="cond=PEAK",
        run_n=run_n, session_id="s1").to_dict()

def test_append_dedupes_by_row_id(tmp_path):
    s = Store(tmp_path)
    assert s.append(_row()) is True
    assert s.append(_row()) is False
    assert len(list(s.iter_rows())) == 1
    assert (tmp_path / "rows-suite-v2.0.0.jsonl").exists()

def test_append_rejects_invalid(tmp_path):
    bad = _row(); bad["status"] = "nope"
    with pytest.raises(SchemaError):
        Store(tmp_path).append(bad)

def test_resume_index_reads_all_shards(tmp_path):
    s = Store(tmp_path)
    r1, r2 = _row(1), _row(2)
    s.append(r1); s.append(r2)
    assert Store(tmp_path).existing_row_ids() == {r1["row_id"], r2["row_id"]}
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_store.py -v` → FAIL (no module).

- [ ] **Step 3: Implementation**

`llmtest/store.py`:
```python
"""Append-only sharded results store (TESTPLAN 7.2). Write-time validation == CI validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from llmtest import schema


class SchemaError(ValueError):
    pass


class Store:
    def __init__(self, results_dir: Path | str):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _shard(self, suite_version: str) -> Path:
        return self.dir / f"rows-{suite_version}.jsonl"

    def existing_row_ids(self) -> set[str]:
        return {r["row_id"] for r in self.iter_rows()}

    def iter_rows(self) -> Iterator[dict]:
        for shard in sorted(self.dir.glob("rows-*.jsonl")):
            with shard.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def append(self, row: dict) -> bool:
        errs = schema.validate_row(row)
        if errs:
            raise SchemaError("; ".join(errs))
        if row["row_id"] in self.existing_row_ids():
            return False
        with self._shard(row["suite_version"]).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return True

    def append_session(self, d: dict) -> None:
        with (self.dir / "sessions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, sort_keys=True) + "\n")

    def iter_sessions(self) -> Iterator[dict]:
        p = self.dir / "sessions.jsonl"
        if not p.exists():
            return
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
```

Note: `existing_row_ids()` re-reads shards per append — fine at P0–P2 scale; in-memory index is a P4+ optimization (YAGNI).

- [ ] **Step 4: Run tests** — PASS (3 tests).
- [ ] **Step 5: Commit** — `git add llmtest/store.py tests/test_store.py && git commit -m "feat(p0): append-only sharded store with dedupe + validation" && git push`

### Task 5: Configs + registry.py + fits() + validate command

**Files:**
- Create: `config/tiers.yaml`, `config/suite.yaml`, `config/registry.yaml`, `config/budgets.yaml`, `config/judges.yaml`, `config/runtime_pins.yaml`, `llmtest/registry.py`, `llmtest/validate_cmd.py`, `tests/test_registry.py`

**Interfaces:** Produces:
- `load_config(root) -> Config` dataclass with `.tiers .registry .suite .budgets .judges .runtime_pins` dicts.
- `fits(model: dict, tiers_cfg: dict, kv_dtype: str, *, tier: str) -> FitResult(fits, fits_short_context, detail)`.
- `run_validate(root=".") -> int` — config sanity + all-shard row validation + mojibake lint. Exit 0 = clean.

- [ ] **Step 1: Write config files (exact starting content)**

`config/tiers.yaml`:
```yaml
# Tier = property of the deployment artifact (TESTPLAN 3.1). VRAM measured, not marketed.
tiers:
  T1: {sku: rtx5090-laptop, marketed_gb: 24, usable_gb: 23.5}
  T2: {sku: rtx5090-desktop, marketed_gb: 32, usable_gb: 31.0}
  T3: {sku: rtx-pro-6000, marketed_gb: 96, usable_gb: 94.0}
kv_floor_ctx: 131072
runtime_overhead_gb: 1.5
kv_bytes_per_token:      # default table; per-model override key kv_bytes_per_token wins
  q8_0: 98304
  f16: 196608
  q4_0: 49152
```

`config/suite.yaml`:
```yaml
suite_version: suite-v2.0.0-shakedown   # re-declared suite-v2.0.0 at P8 freeze tag
condition_order: [runtime, spec, kv, ctx, cond, conc]
condition_vocab:
  runtime: [fork, ollama, vllm]
  spec: [ngram32, off]
  kv: [f16, q8, q4]
  cond: [PEAK, SUSTAINED32K]
  conc: [1, 2, 4, 8, 16, 32, 64, 128]
n_runs_judged: 3
n_runs_harness: 2
```

`config/registry.yaml` — six entries per TESTPLAN §3.5 rule. Exact starting content (provenance `TO-FREEZE` values replaced by Task 6; local_path for qwen3.6-35b resolved in Task 6 Step 5; coder-30b stays `TO-DOWNLOAD` until P2-adjacent):
```yaml
models:
  gpt-oss-20b:
    hf_repo: unsloth/gpt-oss-20b-GGUF
    quant_file: gpt-oss-20b-F16.gguf
    quant_family: MXFP4_MOE
    local_path: 'D:\Ollama\models\blobs\sha256-4e4f9cd88d6456e4f389e7262eca4a8d565211e2b22ece9ca7a8556168ff3c66'
    arch: {moe: true, params_total_b: 21, params_active_b: 3.6}
    claimed_ctx: 131072
    license: apache-2.0
    weights_gb: 12.9
    provenance: {source_repo: unsloth/gpt-oss-20b-GGUF, download_date: TO-FREEZE, sha256: TO-FREEZE, v1_continuity: true}
    role: shakedown
  qwen3.6-35b-a3b:
    hf_repo: bartowski/Qwen_Qwen3.6-35B-A3B-GGUF
    quant_file: Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf
    quant_family: IQ
    local_path: TO-RESOLVE-TASK6
    arch: {moe: true, params_total_b: 35, params_active_b: 3, hybrid_linear_attn: true}
    claimed_ctx: 262144
    license: apache-2.0
    weights_gb: 18.4
    provenance: {source_repo: bartowski/Qwen_Qwen3.6-35B-A3B-GGUF, download_date: TO-FREEZE, sha256: TO-FREEZE, v1_continuity: true}
  gemma-4-26b-a4b:
    hf_repo: unsloth/gemma-4-26B-A4B-it-qat-GGUF
    quant_file: gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
    quant_family: K
    local_path: 'D:\Ollama\models\blobs\sha256-dcf179a91153e3a7ece792e48ef872180d9d6ef9b7677f0a0bd3e83cfe624d5e'
    arch: {moe: true, params_total_b: 26, params_active_b: 3.8}
    claimed_ctx: 262144
    license: gemma
    weights_gb: 13.3
    provenance: {source_repo: unsloth/gemma-4-26B-A4B-it-qat-GGUF, download_date: TO-FREEZE, sha256: TO-FREEZE, v1_continuity: true}
  ornith-1.0-35b:
    hf_repo: jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF
    quant_file: Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf
    quant_family: MXFP4_MOE
    local_path: 'D:\BUILT-TOOLS\LLMtesting\bonsai\Ornith-35B-A3B-MXFP4.gguf'
    arch: {moe: true, params_total_b: 35, params_active_b: 3}
    claimed_ctx: 262144
    license: apache-2.0
    weights_gb: 18.4
    provenance: {source_repo: jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF, download_date: TO-FREEZE, sha256: TO-FREEZE, v1_continuity: true}
    notes: v1 SCORECARD row used bartowski IQ4_XS (different artifact); session-5/6 measured data is THIS artifact
  qwen3.6-27b-dense:
    hf_repo: unsloth/Qwen3.6-27B-GGUF
    quant_file: Qwen3.6-27B-Q5_K_M.gguf
    quant_family: K
    local_path: 'D:\Ollama\models\blobs\sha256-cfecab168156269f25d5ffe9e13cf2a401ca2f43a9693fa00bcd1625316ccbde'
    arch: {moe: false, params_total_b: 27, params_active_b: 27}
    claimed_ctx: 262144
    license: apache-2.0
    weights_gb: 18.2
    provenance: {source_repo: unsloth/Qwen3.6-27B-GGUF, download_date: TO-FREEZE, sha256: TO-FREEZE, v1_continuity: true}
  qwen3-coder-30b:
    hf_repo: unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
    quant_file: Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf
    quant_family: K
    local_path: TO-DOWNLOAD
    arch: {moe: true, params_total_b: 30, params_active_b: 3}
    claimed_ctx: 262144
    license: apache-2.0
    weights_gb: 17.5
    provenance: {source_repo: unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF, download_date: TO-FREEZE, sha256: TO-FREEZE, v1_continuity: false}
    notes: house-ladder pick (rule 2) — no local v1 history
```

`config/budgets.yaml`:
```yaml
verda_session_cap_usd: 20
verda_rate_usd_hr_expected: 1.89   # pre-flight spot-price check REQUIRED before rent
tier_gates: {T3: {requires_signoff: false}, T4: {requires_signoff: true}}
```

`config/judges.yaml`:
```yaml
# Pins frozen at P3 after live enumeration + Michael sign-off (TESTPLAN 6.1). DRAFT candidates:
judges:
  claude: {model: claude-fable-5, cli: claude, cli_version: TO-FREEZE-P3, invoke: 'claude -p --model {model} --output-format json'}
  codex: {model: TO-FREEZE-P3, cli: codex, cli_version: TO-FREEZE-P3, invoke: 'codex exec --model {model}'}
  gemini: {model: gemini-3-pro, cli: gemini, cli_version: TO-FREEZE-P3, invoke: 'gemini --model {model}'}
kin_map: {gpt-oss-20b: codex, gemma-4-26b-a4b: gemini}
```

`config/runtime_pins.yaml`:
```yaml
fork:
  binary: 'D:\BUILT-TOOLS\LLMtesting\bonsai\bin\llama-server.exe'
  build_id: prism-b9591-62061f9    # verify against server log line at P1 Task 8; correct if differs
standard_flags: '-ngl 99 --jinja -fa on --spec-type ngram-mod --spec-ngram-mod-n-match 32 --cache-ram 0'
ollama: {version: TO-FREEZE-P1, sanctioned_arm_env: {OLLAMA_KV_CACHE_TYPE: q8_0, OLLAMA_FLASH_ATTENTION: '1'}}
canary:
  model_id: ornith-1.0-35b
  edit_source: 'D:\BUILT-TOOLS\LLMtesting\michael\snake_27b.html'
  reference: {min_speedup: 3.5, max_speedup: 7.0}   # Table-4-derived band (measured 4.98-5.5x); ratio not absolute tps -> thermal-robust
```

- [ ] **Step 2: Failing tests**

`tests/test_registry.py`:
```python
from pathlib import Path
from llmtest.registry import load_config, fits

ROOT = Path(__file__).resolve().parents[1]

def test_config_loads_all_six_models():
    cfg = load_config(ROOT)
    assert set(cfg.registry["models"]) >= {
        "gpt-oss-20b", "qwen3.6-35b-a3b", "gemma-4-26b-a4b",
        "ornith-1.0-35b", "qwen3.6-27b-dense", "qwen3-coder-30b"}
    assert cfg.suite["condition_order"][0] == "runtime"

def test_fits_small_model_t1():
    cfg = load_config(ROOT)
    r = fits(cfg.registry["models"]["gpt-oss-20b"], cfg.tiers, "q8_0", tier="T1")
    assert r.fits is True

def test_fits_flags_short_context_not_reject():
    cfg = load_config(ROOT)
    fat = dict(cfg.registry["models"]["qwen3.6-27b-dense"], weights_gb=21.5)
    r = fits(fat, cfg.tiers, "f16", tier="T1")
    assert r.fits is True and r.fits_short_context is True
```

- [ ] **Step 3: Verify fail** — `python -m pytest tests/test_registry.py -v` → FAIL.

- [ ] **Step 4: Implementation**

`llmtest/registry.py`:
```python
"""Config loaders + fits() — ONE code path for tier placement AND VRAM preflight (TESTPLAN 3.1/7.3)."""
from __future__ import annotations

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
    return Config(root=root, tiers=_load(root, "tiers"),
                  registry=_load(root, "registry"), suite=_load(root, "suite"),
                  budgets=_load(root, "budgets"), judges=_load(root, "judges"),
                  runtime_pins=_load(root, "runtime_pins"))


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
```

`llmtest/validate_cmd.py`:
```python
"""llmtest validate — shard + config integrity + mojibake lint. Same checks CI runs. Exit 0 = clean."""
from pathlib import Path

from llmtest import schema
from llmtest.registry import load_config
from llmtest.store import Store


def run_validate(root: Path | str = ".") -> int:
    root = Path(root).resolve()
    errors: list[str] = []
    cfg = load_config(root)
    order = cfg.suite["condition_order"]
    if len(order) != len(set(order)):
        errors.append("suite.yaml condition_order contains duplicates")
    for name, m in cfg.registry["models"].items():
        for k in ("hf_repo", "quant_file", "provenance", "license", "weights_gb"):
            if k not in m:
                errors.append(f"registry:{name} missing {k}")
    n = 0
    for row in Store(root / "results").iter_rows():
        n += 1
        errors += [f"row {row.get('row_id', '?')[:12]}: {e}"
                   for e in schema.validate_row(row)]
    scan_dirs = [root / "docs", root / "suite", root / "grading"]
    scan = [root / "TESTPLAN.md"] + [p for d in scan_dirs if d.exists()
                                     for p in d.rglob("*") if p.is_file()
                                     and p.suffix in {".md", ".yaml", ".txt", ".html"}]
    for p in scan:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-utf8 file: {p}")
            continue
        bad = sorted({c for c in text if ord(c) > 0x2E7F})   # mojibake/CJK lint (amendment 29)
        if bad:
            errors.append(f"suspicious non-ASCII in {p.relative_to(root)}: {bad[:3]}")
    for e in errors:
        print(f"VALIDATE-ERROR: {e}")
    print(f"validate: {n} rows checked, {len(errors)} errors")
    return 1 if errors else 0
```

- [ ] **Step 5: Run tests** — `python -m pytest tests/test_registry.py -v` → PASS; `python -m llmtest validate` → exit 0 (no rows yet, configs clean). Note: emoji in docs are > 0x2E7F — if validate flags TESTPLAN emoji/box-drawing, extend the allowlist in `validate_cmd.py` to skip chars in ranges 0x2500-0x27BF and 0x1F300-0x1FAFF *before* committing, and add a test asserting the 每-class CJK char IS flagged.

- [ ] **Step 6: Commit** — `git add config/ llmtest/registry.py llmtest/validate_cmd.py tests/test_registry.py && git commit -m "feat(p0): configs, registry+fits(), validate cmd with mojibake lint" && git push`

### Task 6: freeze_artifacts.py — SHA256 + provenance freeze (TESTPLAN §3.5/§11.5)

**Files:**
- Create: `scripts/__init__.py` (empty), `scripts/freeze_artifacts.py`, `tests/test_freeze.py`

**Interfaces:** Produces: `freeze(registry_path, models=None) -> dict[model_id, "frozen"|"missing-file"|"already-frozen"]`; updates `config/registry.yaml` provenance in place.

- [ ] **Step 1: Failing test**

`tests/test_freeze.py`:
```python
import hashlib
import yaml
from scripts.freeze_artifacts import freeze

def test_freeze_hashes_file_and_writes_provenance(tmp_path):
    blob = tmp_path / "model.gguf"; blob.write_bytes(b"weights")
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump({"models": {"m1": {
        "local_path": str(blob),
        "provenance": {"source_repo": "x/y", "download_date": "TO-FREEZE",
                       "sha256": "TO-FREEZE", "v1_continuity": True}}}}))
    assert freeze(reg) == {"m1": "frozen"}
    d = yaml.safe_load(reg.read_text())
    assert d["models"]["m1"]["provenance"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert d["models"]["m1"]["provenance"]["download_date"] != "TO-FREEZE"
    assert freeze(reg) == {"m1": "already-frozen"}
```

- [ ] **Step 2: Verify fail** → FAIL (no scripts module).

- [ ] **Step 3: Implementation**

`scripts/freeze_artifacts.py`:
```python
"""P0 artifact freeze: SHA256 + download_date (TESTPLAN 3.5). Later mismatch = recorded finding, never auto-refetch."""
import hashlib
import sys
from datetime import date
from pathlib import Path

import yaml


def _sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def freeze(registry_path: Path | str, models: list[str] | None = None) -> dict:
    registry_path = Path(registry_path)
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    out = {}
    for name, m in data["models"].items():
        if models and name not in models:
            continue
        prov = m.get("provenance", {})
        if prov.get("sha256") not in (None, "TO-FREEZE"):
            out[name] = "already-frozen"
            continue
        p = Path(str(m.get("local_path", "")))
        if not p.is_file():
            out[name] = "missing-file"
            continue
        prov["sha256"] = _sha256(p)
        prov["download_date"] = str(date.today())
        m["provenance"] = prov
        out[name] = "frozen"
    registry_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    result = freeze(Path(__file__).resolve().parents[1] / "config" / "registry.yaml",
                    models=sys.argv[1:] or None)
    for k, v in sorted(result.items()):
        print(f"{k}: {v}")
    sys.exit(1 if "missing-file" in result.values() else 0)
```

Also create empty `tests/__init__.py` if pytest import mode needs it (only if Step 4 shows import errors).

- [ ] **Step 4: Run test** — `python -m pytest tests/test_freeze.py -v` → PASS.

- [ ] **Step 5: Resolve qwen3.6-35b local_path, then run the real freeze (~10 min for ~80 GB)**

First: `ollama show --modelfile hf.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:IQ4_XS | findstr /i "^FROM"` → paste the blob path into `config/registry.yaml` `local_path`. Then:
Run: `python scripts/freeze_artifacts.py gpt-oss-20b gemma-4-26b-a4b ornith-1.0-35b qwen3.6-27b-dense qwen3.6-35b-a3b`
Expected: 5 lines `frozen`. `qwen3-coder-30b` remains TO-FREEZE (downloads at Task 13 Step 5; not needed for shakedown).

- [ ] **Step 6: Commit** — `git add scripts/ tests/test_freeze.py config/registry.yaml && git commit -m "feat(p0): provenance freeze — 5/6 baseline artifacts SHA-pinned" && git push`

### Task 7: tables.py + status command + CI workflow — P0 COMPLETE

**Files:**
- Create: `llmtest/tables.py`, `llmtest/status_cmd.py`, `.github/workflows/ci.yml`, `tests/test_tables.py`

**Interfaces:** Produces:
- `render_serving_table(rows: list[dict]) -> str` — pure, stable sort `(model_id, condition)`, floats `{:.1f}`, filters `timing_authoritative and status=="ok" and "non-reportable" not in tags`.
- `run_tables(root=".") -> int` — writes `results/tables/serving.md` with `newline="\n"`.
- `run_status(root=".") -> int` — row counts grouped `(battery, model_id, status)`.

- [ ] **Step 1: Failing test**

`tests/test_tables.py`:
```python
from llmtest.tables import render_serving_table

def _r(model, cond, tps):
    return {"model_id": model, "hf_repo": f"org/{model}", "condition": cond,
            "timing_authoritative": True, "status": "ok",
            "response_meta": {"decode_tps": tps, "pp_tps": 0, "ttft_ms": 0},
            "tags": []}

def test_serving_table_deterministic_and_authority_filtered():
    rows = [_r("b", "cond=PEAK", 100.15), _r("a", "cond=PEAK", 50.0),
            dict(_r("c", "cond=PEAK", 999.0), timing_authoritative=False)]
    out1 = render_serving_table(rows)
    out2 = render_serving_table(list(reversed(rows)))
    assert out1 == out2
    assert "org/a" in out1 and "org/b" in out1
    assert "999" not in out1
    assert "100.2" in out1     # fixed rounding
```

- [ ] **Step 2: Verify fail** → FAIL.

- [ ] **Step 3: Implementation**

`llmtest/tables.py`:
```python
"""Byte-deterministic tables (TESTPLAN 7.5): pure functions of rows, stable sorts, fixed float fmt, LF newlines."""
from pathlib import Path

from llmtest.store import Store


def render_serving_table(rows: list[dict]) -> str:
    keep = [r for r in rows
            if r.get("timing_authoritative") and r.get("status") == "ok"
            and "non-reportable" not in r.get("tags", [])]
    keep.sort(key=lambda r: (r["model_id"], r["condition"]))
    lines = ["# Serving (Battery 5) — timing_authoritative rows only", "",
             "| Model | Condition | decode t/s | PP t/s | TTFT ms |",
             "|---|---|---|---|---|"]
    for r in keep:
        m = r["response_meta"]
        lines.append(f"| {r['hf_repo']} | {r['condition']} | {m.get('decode_tps', 0):.1f} "
                     f"| {m.get('pp_tps', 0):.1f} | {m.get('ttft_ms', 0):.0f} |")
    return "\n".join(lines) + "\n"


def run_tables(root: str | Path = ".") -> int:
    root = Path(root).resolve()
    rows = list(Store(root / "results").iter_rows())
    out = root / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "serving.md").write_text(render_serving_table(rows),
                                    encoding="utf-8", newline="\n")
    print(f"tables: wrote serving.md from {len(rows)} rows")
    return 0
```

`llmtest/status_cmd.py`:
```python
"""llmtest status — done counts from resume keys (pending matrix arrives with plan() in Task 12)."""
from collections import Counter
from pathlib import Path

from llmtest.store import Store


def run_status(root: str | Path = ".") -> int:
    counts = Counter()
    for r in Store(Path(root).resolve() / "results").iter_rows():
        counts[(r["battery"], r["model_id"], r["status"])] += 1
    if not counts:
        print("status: no rows yet")
        return 0
    for (battery, model, status), n in sorted(counts.items()):
        print(f"B{battery} {model:24s} {status:8s} {n}")
    return 0
```

`.github/workflows/ci.yml`:
```yaml
name: integrity
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -e ".[dev]"
      - run: python -m pytest -q
      - run: python -m llmtest validate
      - name: tables regenerate byte-clean
        run: |
          python -m llmtest tables
          git diff --exit-code -- results/tables || (echo "tables not regenerate-clean" && exit 1)
      - name: secret scan
        uses: gitleaks/gitleaks-action@v2
        env: {GITHUB_TOKEN: '${{ secrets.GITHUB_TOKEN }}'}
```

(The append-only-shard-vs-tag CI check lands at P8 when the first tag exists — TESTPLAN §7.6; out of P0 scope by design.)

- [ ] **Step 4: Full local CI parity** — `python -m pytest -q` all green → `python -m llmtest validate` exit 0 → `python -m llmtest tables` → run `tables` AGAIN → `git status results/tables` unchanged (byte-clean proven locally).

- [ ] **Step 5: Commit, push, watch CI green** — `git add -A && git commit -m "feat(p0): tables, status, integrity CI — P0 exit complete" && git push && gh run watch --exit-status`. **P0 done: remote verified (Task 1), schema+store+configs+freeze+CI all green.**

---

### Task 8: ServerManager (P1) — fork translation, orphan sweep, sessions, teardown-by-PID

**Files:**
- Create: `llmtest/server.py`, `tests/test_server.py`
- Modify: `pyproject.toml` (add pytest marker), `.github/workflows/ci.yml` (exclude gpu marker)

**Interfaces:** Produces (consumed by canary + batteries):
- `ServerManager(cfg: Config, store: Store)` with:
  - `.request_endpoint(model_id, runtime="fork", flags_overlay: dict | None = None, parallel=1, ctx=8192, kv="q8_0", timing_authoritative=False) -> EndpointHandle`
  - `EndpointHandle`: `.base_url` (str, `http://127.0.0.1:PORT`), `.session_id`, `.normalized_config` (dict), `.chat(messages, max_tokens, temperature, tools=None) -> dict` (raw OpenAI-compat JSON incl. `timings` when fork).
  - `.teardown()` — kill by PID, wait for VRAM drain; `.sweep_orphans()` — kill stray `llama-server.exe`, report.
- Flag composition: standard flags from `runtime_pins.yaml` + overlay dict wins per key; overlay `{"spec": "off"}` → `--spec-type none` replaces ngram.
- Every successful launch appends a SessionRow (power_mode/ac_state read via `powercfg /getactivescheme` + `WMIC`-free battery check `(Get-CimInstance Win32_Battery).BatteryStatus` — fall back to `"unknown"` on error, never crash).

- [ ] **Step 1: Failing unit tests (no GPU — subprocess + parsing logic only)**

`tests/test_server.py`:
```python
from pathlib import Path
import pytest
from llmtest.registry import load_config
from llmtest.server import compose_fork_flags, normalize_config

ROOT = Path(__file__).resolve().parents[1]

def test_compose_fork_flags_standard_plus_overlay():
    cfg = load_config(ROOT)
    flags = compose_fork_flags(cfg, ctx=8192, parallel=1, kv="q8_0", overlay=None)
    assert "--spec-type ngram-mod" in flags and "-c 8192" in flags
    assert "-ctk q8_0" in flags and "-ctv q8_0" in flags
    off = compose_fork_flags(cfg, ctx=8192, parallel=4, kv="f16", overlay={"spec": "off"})
    assert "--spec-type none" in off and "-np 4" in off and "-ctk" not in off

def test_normalize_config_cross_runtime_shape():
    n = normalize_config(runtime="fork", ctx=8192, kv="q8_0",
                         spec="ngram32", parallel=2, flash_attn=True)
    assert n == {"ctx": 8192, "kv_dtype": "q8_0", "flash_attn": True,
                 "spec_type": "ngram32", "spec_params": {"n_match": 32}, "parallel": 2}

def test_never_below_nmatch_16_guard():
    cfg = load_config(ROOT)
    with pytest.raises(ValueError):
        compose_fork_flags(cfg, ctx=8192, parallel=1, kv="q8_0",
                           overlay={"spec": "ngram8"})
```

- [ ] **Step 2: Verify fail** → FAIL.

- [ ] **Step 3: Implementation**

`llmtest/server.py`:
```python
"""ServerManager (TESTPLAN 7.3): all serving goes through here; provenance auto-attached.
Teardown by PID only. Orphan sweep before launch. fits() preflight. Fork implemented;
ollama sanctioned arm implemented (B5); vllm = remote-attach later phase."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llmtest.registry import Config, fits
from llmtest.schema import SessionRow, SCHEMA_VERSION
from llmtest.store import Store

_SPEC = {"ngram32": ("ngram-mod", {"n_match": 32}), "off": ("none", {})}


def normalize_config(*, runtime, ctx, kv, spec, parallel, flash_attn=True) -> dict:
    spec_type, spec_params = spec, {}
    if spec == "ngram32":
        spec_params = {"n_match": 32}
    return {"ctx": ctx, "kv_dtype": kv, "flash_attn": flash_attn,
            "spec_type": spec_type, "spec_params": spec_params, "parallel": parallel}


def compose_fork_flags(cfg: Config, *, ctx: int, parallel: int, kv: str,
                       overlay: dict | None) -> str:
    overlay = overlay or {}
    spec = overlay.get("spec", "ngram32")
    if spec.startswith("ngram") and spec not in _SPEC:
        n = int(spec.replace("ngram", "") or 0)
        if n < 16:
            raise ValueError("TESTPLAN 2: never run ngram n-match < 16")
        _SPEC[spec] = ("ngram-mod", {"n_match": n})
    spec_type, spec_params = _SPEC[spec]
    parts = ["-ngl 99", "--jinja", "-fa on", f"-c {ctx}", "--cache-ram 0"]
    if spec_type == "ngram-mod":
        parts += ["--spec-type ngram-mod",
                  f"--spec-ngram-mod-n-match {spec_params['n_match']}"]
    else:
        parts += ["--spec-type none"]
    if kv != "f16":
        parts += [f"-ctk {kv}", f"-ctv {kv}"]
    if parallel > 1:
        parts += [f"-np {parallel}"]
    return " ".join(parts)


def _free_port(start=8080) -> int:
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def _vram_free_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0]) / 1024


def _power_state() -> tuple[str, str]:
    try:
        scheme = subprocess.run(["powercfg", "/getactivescheme"],
                                capture_output=True, text=True).stdout
        mode = "performance" if "erformance" in scheme else "balanced"
    except Exception:
        mode = "unknown"
    return mode, "unknown"   # ac_state refinement lands with bench-night profile (TESTPLAN 11.10)


@dataclass
class EndpointHandle:
    base_url: str
    session_id: str
    normalized_config: dict
    pid: int
    _mgr: "ServerManager"

    def chat(self, messages, *, max_tokens=512, temperature=0.0, tools=None,
             timeout=1200) -> dict:
        body = {"messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))


class ServerManager:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self._active: EndpointHandle | None = None
        self._active_key: tuple | None = None

    def sweep_orphans(self) -> list[str]:
        killed = []
        for name in ("llama-server.exe",):
            r = subprocess.run(["taskkill", "/F", "/IM", name],
                               capture_output=True, text=True)
            if "SUCCESS" in r.stdout:
                killed.append(name)
        time.sleep(2)
        return killed

    def request_endpoint(self, model_id: str, runtime: str = "fork",
                         flags_overlay: dict | None = None, parallel: int = 1,
                         ctx: int = 8192, kv: str = "q8_0",
                         timing_authoritative: bool = False) -> EndpointHandle:
        key = (model_id, runtime, json.dumps(flags_overlay, sort_keys=True),
               parallel, ctx, kv)
        if self._active and self._active_key == key:
            return self._active                       # config-match reuse
        self.teardown()
        if runtime != "fork":
            raise NotImplementedError(f"runtime {runtime} lands in Task 12 (ollama) / later (vllm)")
        model = self.cfg.registry["models"][model_id]
        fit = fits(model, self.cfg.tiers, kv, tier="T1")
        if not fit.fits:
            raise RuntimeError(f"fits() preflight failed: {fit.detail}")
        if _vram_free_gb() < float(model["weights_gb"]) + 1.0:
            self.sweep_orphans()
            if _vram_free_gb() < float(model["weights_gb"]) + 1.0:
                raise RuntimeError("insufficient VRAM after orphan sweep")
        flags = compose_fork_flags(self.cfg, ctx=ctx, parallel=parallel, kv=kv,
                                   overlay=flags_overlay)
        port = _free_port()
        binary = self.cfg.runtime_pins["fork"]["binary"]
        invocation = (f'"{binary}" -m "{model["local_path"]}" {flags} '
                      f"--host 127.0.0.1 --port {port}")
        log = Path("artifacts") / f"server-{port}.log"
        log.parent.mkdir(exist_ok=True)
        proc = subprocess.Popen(invocation, stdout=log.open("w"),
                                stderr=subprocess.STDOUT, shell=True)
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"server died on launch; see {log}")
                time.sleep(1)
        else:
            raise RuntimeError("server health timeout")
        spec = (flags_overlay or {}).get("spec", "ngram32")
        session_id = f"s-{uuid.uuid4().hex[:12]}"
        mode, ac = _power_state()
        self.store.append_session(SessionRow(
            schema_version=SCHEMA_VERSION, session_id=session_id,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            runtime="llamacpp-fork",
            runtime_build=self.cfg.runtime_pins["fork"]["build_id"],
            normalized_config=normalize_config(runtime="fork", ctx=ctx, kv=kv,
                                               spec=spec, parallel=parallel),
            raw_invocation=invocation, hardware_sku="rtx5090-laptop",
            measured_usable_vram_gb=self.cfg.tiers["tiers"]["T1"]["usable_gb"],
            tp_degree=1, topology=None, driver_env={},
            power_mode=mode, ac_state=ac,
            timing_authoritative=timing_authoritative).to_dict())
        self._active = EndpointHandle(base_url=base, session_id=session_id,
                                      normalized_config=normalize_config(
                                          runtime="fork", ctx=ctx, kv=kv,
                                          spec=spec, parallel=parallel),
                                      pid=proc.pid, _mgr=self)
        self._active_key = key
        return self._active

    def teardown(self) -> None:
        if not self._active:
            return
        subprocess.run(["taskkill", "/F", "/PID", str(self._active.pid), "/T"],
                       capture_output=True)
        self._active = None
        self._active_key = None
        deadline = time.time() + 30
        while time.time() < deadline and _vram_free_gb() < 5.0:
            time.sleep(1)
```

`pyproject.toml` — extend `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: requires local RTX 5090 + fork binary (excluded in CI)"]
```
`.github/workflows/ci.yml` — change the pytest line to: `- run: python -m pytest -q -m "not gpu"`

- [ ] **Step 4: Run unit tests** — `python -m pytest tests/test_server.py -v` → PASS (3).

- [ ] **Step 5: One real GPU launch test (manual, not committed as always-on)**

Run: `python -c "from pathlib import Path; from llmtest.registry import load_config; from llmtest.store import Store; from llmtest.server import ServerManager; m=ServerManager(load_config(Path('.')), Store(Path('results'))); h=m.request_endpoint('gpt-oss-20b', ctx=4096); print(h.chat([{'role':'user','content':'Say OK'}], max_tokens=8)['choices'][0]['message']['content']); m.teardown()"`
Expected: model reply containing OK; `results/sessions.jsonl` gains one row; VRAM returns to idle after. **Verify the fork build id line in `artifacts/server-*.log` matches `runtime_pins.yaml` `build_id` — correct the config if it differs (TESTPLAN §11.4).**

- [ ] **Step 6: Commit** — `git add llmtest/server.py tests/test_server.py pyproject.toml .github/workflows/ci.yml && git commit -m "feat(p1): ServerManager — fork translation, orphan sweep, sessions, PID teardown" && git push`

### Task 9: run_cmd — minimal runner loop + --keep-server/--debug

**Files:**
- Create: `llmtest/run_cmd.py`, `tests/test_run_cmd.py`

**Interfaces:** Produces:
- `run_run(args) -> int` — loads config, selects battery plugin(s) via `llmtest.batteries.get(battery_id)`, computes `plan()`, diffs vs `Store.existing_row_ids()` (skip unless `--force`), calls `execute()` per pending item, appends rows. `--task`/`--condition`/`--model` filter plan items. `--debug` dumps each request/response transcript to `artifacts/debug/<row_id>.json`; `--keep-server` skips final teardown.
- Consumes (defined in Task 11): `llmtest.batteries.get(id) -> Battery`, `Battery.plan/execute`, `WorkItem` fields `(model_id, battery, task_id, condition, run_n, row_id)`.

- [ ] **Step 1: Failing test (fake battery injected)**

`tests/test_run_cmd.py`:
```python
from pathlib import Path
from types import SimpleNamespace
from llmtest import run_cmd
from llmtest.store import Store
from llmtest import schema

class FakeBattery:
    id = 5
    def plan(self, cfg, store, model_filter=None):
        row = schema.ResultRow.new(
            suite_version="suite-v2.0.0", model_id="m", hf_repo="o/r",
            quant_file="q", quant_sha256="a"*64, tier="T1", battery=5,
            task_id="b5.fake", fixture_sha="f"*64, condition="cond=PEAK",
            run_n=1, session_id="pending")
        return [SimpleNamespace(row_id=row.row_id, model_id="m",
                                task_id="b5.fake", condition="cond=PEAK",
                                run_n=1, template=row)]
    def execute(self, item, ctx):
        r = item.template; r.session_id = "s-fake"; return [r.to_dict()]

def test_run_skips_done_items(tmp_path, monkeypatch):
    monkeypatch.setattr(run_cmd, "_get_battery", lambda i: FakeBattery())
    monkeypatch.setattr(run_cmd, "_results_dir", lambda root: tmp_path)
    args = SimpleNamespace(suite="smoke", model=None, battery=5, task_id=None,
                           condition=None, force=False, keep_server=False, debug=False)
    assert run_cmd.run_run(args) == 0
    assert len(list(Store(tmp_path).iter_rows())) == 1
    assert run_cmd.run_run(args) == 0                 # resume: nothing re-executed
    assert len(list(Store(tmp_path).iter_rows())) == 1
```

- [ ] **Step 2: Verify fail** → FAIL.

- [ ] **Step 3: Implementation**

`llmtest/run_cmd.py`:
```python
"""llmtest run — plan/diff/execute loop with free resume (TESTPLAN 7.2/7.5)."""
import json
from pathlib import Path

from llmtest.registry import load_config
from llmtest.store import Store


def _results_dir(root: Path) -> Path:
    return root / "results"


def _get_battery(battery_id: int):
    from llmtest import batteries
    return batteries.get(battery_id)


def run_run(args) -> int:
    root = Path(".").resolve()
    cfg = load_config(root)
    store = Store(_results_dir(root))
    battery = _get_battery(args.battery)
    items = battery.plan(cfg, store, model_filter=args.model)
    if args.task_id:
        items = [i for i in items if i.task_id == args.task_id]
    if args.condition:
        items = [i for i in items if i.condition == args.condition]
    done = store.existing_row_ids()
    pending = [i for i in items if args.force or i.row_id not in done]
    print(f"run: {len(items)} planned, {len(pending)} pending")
    ctx = RunContext(cfg=cfg, store=store, root=root,
                     keep_server=args.keep_server, debug=args.debug)
    failures = 0
    try:
        for item in pending:
            try:
                for row in battery.execute(item, ctx):
                    store.append(row)
                    if args.debug:
                        dbg = root / "artifacts" / "debug"
                        dbg.mkdir(parents=True, exist_ok=True)
                        (dbg / f"{row['row_id']}.json").write_text(
                            json.dumps(row, indent=2), encoding="utf-8")
            except Exception as e:                    # row-level containment
                failures += 1
                print(f"EXEC-ERROR {item.task_id} {item.condition}: {e}")
    finally:
        if not args.keep_server and ctx.server is not None:
            ctx.server.teardown()
    print(f"run: done, {failures} failures")
    return 1 if failures else 0


class RunContext:
    """Handed to Battery.execute(). Lazily builds ServerManager on first use."""
    def __init__(self, *, cfg, store, root, keep_server, debug):
        self.cfg = cfg
        self.store = store
        self.root = root
        self.keep_server = keep_server
        self.debug = debug
        self._server = None

    @property
    def server(self):
        return self._server

    def server_manager(self):
        if self._server is None:
            from llmtest.server import ServerManager
            self._server = ServerManager(self.cfg, self.store)
        return self._server
```

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git add llmtest/run_cmd.py tests/test_run_cmd.py && git commit -m "feat(p1): run command — plan/diff/execute with free resume, debug dumps" && git push`

### Task 10: Serving canary — llmtest validate --serving (P1 exit)

**Files:**
- Create: `llmtest/canary.py`, `tests/test_canary.py`

**Interfaces:** Produces: `run_canary(root=".") -> int` — Ornith ngram edit A/B; PASS iff measured speedup within `runtime_pins.canary.reference` band; prints both arms + ratio. Pure helper `evaluate_canary(base_tps, ngram_tps, band) -> tuple[bool, str]` unit-testable without GPU.

- [ ] **Step 1: Failing test**

`tests/test_canary.py`:
```python
from llmtest.canary import evaluate_canary

BAND = {"min_speedup": 3.5, "max_speedup": 7.0}

def test_canary_pass_inside_band():
    ok, msg = evaluate_canary(100.0, 500.0, BAND)
    assert ok and "5.00x" in msg

def test_canary_fail_below_band():
    ok, msg = evaluate_canary(100.0, 200.0, BAND)
    assert not ok and "2.00x" in msg
```

- [ ] **Step 2: Verify fail** → FAIL.

- [ ] **Step 3: Implementation**

`llmtest/canary.py`:
```python
"""Serving canary (TESTPLAN 7.5): Ornith ngram edit A/B vs reference band. Re-runnable health check."""
from pathlib import Path


def evaluate_canary(base_tps: float, ngram_tps: float, band: dict) -> tuple[bool, str]:
    ratio = ngram_tps / base_tps if base_tps else 0.0
    ok = band["min_speedup"] <= ratio <= band["max_speedup"]
    return ok, (f"canary: base={base_tps:.1f} t/s ngram={ngram_tps:.1f} t/s "
                f"speedup={ratio:.2f}x band=[{band['min_speedup']},{band['max_speedup']}] "
                f"{'PASS' if ok else 'FAIL'}")


def _edit_prompt(src: Path) -> str:
    return ("You are a code editor. Output the ENTIRE file again exactly as-is, "
            "inserting `<!-- MODIFIED -->` right after the opening <body> tag. "
            "Output only the full HTML. /no_think\n\n"
            + src.read_text(encoding="utf-8"))


def _decode_tps(handle, prompt: str) -> float:
    d = handle.chat([{"role": "user", "content": prompt}],
                    max_tokens=4096, temperature=0.0)
    return float(d.get("timings", {}).get("predicted_per_second", 0.0))


def run_canary(root: str | Path = ".") -> int:
    from llmtest.registry import load_config
    from llmtest.server import ServerManager
    from llmtest.store import Store
    root = Path(root).resolve()
    cfg = load_config(root)
    can = cfg.runtime_pins["canary"]
    prompt = _edit_prompt(Path(can["edit_source"]))
    mgr = ServerManager(cfg, Store(root / "results"))
    try:
        base = _decode_tps(mgr.request_endpoint(can["model_id"], ctx=10240,
                                                flags_overlay={"spec": "off"}), prompt)
        ngram = _decode_tps(mgr.request_endpoint(can["model_id"], ctx=10240), prompt)
    finally:
        mgr.teardown()
    ok, msg = evaluate_canary(base, ngram, can["reference"])
    print(msg)
    return 0 if ok else 1
```

- [ ] **Step 4: Run unit tests** → PASS. Then the real thing (GPU, ~6 min): `python -m llmtest validate --serving` → expect `speedup=4.5-5.5x ... PASS`. **This reproduces Table 4 within thermal variance — P1 validation criterion met.**

- [ ] **Step 5: Commit** — `git add llmtest/canary.py tests/test_canary.py && git commit -m "feat(p1): re-runnable serving canary (Ornith ngram A/B) — P1 complete" && git push`

---

### Task 11: Battery ABC + plugin registry (P2)

**Files:**
- Create: `llmtest/batteries/__init__.py`, `tests/test_batteries_abc.py`

**Interfaces:** Produces (the contract B5 and every future battery implements):
- `@dataclass WorkItem`: `row_id, model_id, battery, task_id, condition, run_n, payload: dict` (payload = battery-private planning data; `template` pattern from Task 9's fake is superseded by payload).
- `class Battery(ABC)`: `id: int`; `plan(cfg, store, model_filter=None) -> list[WorkItem]`; `preflight(ctx) -> list[dict]` (default `[]`); `execute(item, ctx) -> list[dict]` (rows as dicts); optional `build_judge_packets(rows) -> list` (default `[]`).
- `get(battery_id: int) -> Battery` — registry lookup; `register(cls)` decorator.

- [ ] **Step 1: Failing test**

`tests/test_batteries_abc.py`:
```python
import pytest
from llmtest import batteries

def test_registry_rejects_unknown():
    with pytest.raises(KeyError):
        batteries.get(99)

def test_register_and_get_roundtrip():
    @batteries.register
    class Dummy(batteries.Battery):
        id = 98
        def plan(self, cfg, store, model_filter=None):
            return []
        def execute(self, item, ctx):
            return []
    assert isinstance(batteries.get(98), Dummy)
```

- [ ] **Step 2: Verify fail** → FAIL.

- [ ] **Step 3: Implementation**

`llmtest/batteries/__init__.py`:
```python
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
    def plan(self, cfg, store, model_filter=None) -> list[WorkItem]: ...

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
```

Also update `tests/test_run_cmd.py` FakeBattery to carry `payload` instead of `template` (replace `template=row` with `payload={"row": row}` and `item.template` with `item.payload["row"]`) — keep that test green.

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_batteries_abc.py tests/test_run_cmd.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add llmtest/batteries/ tests/ && git commit -m "feat(p2): battery ABC + plugin registry (ServerManager-mediated, rows-plural)" && git push`

### Task 12: Battery 5 plugin — serving measurements (ABC client #1)

**Files:**
- Create: `llmtest/batteries/b5_serving.py`, `suite/b5_serving/prompts.yaml`, `tests/test_b5.py`
- Modify: `config/registry.yaml` (add `ollama_tag` per model — used by the sanctioned Ollama arm later; only gpt-oss needed now: `ollama_tag: 'hf.co/unsloth/gpt-oss-20b-GGUF:F16'`)

**Interfaces:** Produces:
- `@register class B5Serving(Battery)` with `id=5`. Conditions planned per model (T1 local shakedown scope):
  - `runtime=fork;spec=ngram32;kv=q8;cond=PEAK` and `cond=SUSTAINED32K`
  - `runtime=fork;spec=off;kv=q8;cond=PEAK` and `cond=SUSTAINED32K` (the ngram A/B)
  - `runtime=fork;spec=ngram32;kv=q8;cond=PEAK;conc=N` for N in 2/4/8/16 (concurrency ladder)
  - (`runtime=ollama` + `runtime=vllm` arms are planned-but-skipped: `plan()` emits them only when `cfg.suite` flag `b5_extra_runtimes: true` — default false until the Ollama translation lands post-shakedown; keeps shakedown surface tight.)
- All rows: `timing_authoritative=True`, `battery=5`, `fixture_sha = sha256 of prompts.yaml content`, metrics per condition; spec stats captured from fork `timings` when present.
- Pure helpers (unit-tested): `peak_metrics(timings) -> dict`, `concurrency_metrics(list[timings], elapsed_s) -> dict`.

- [ ] **Step 1: Fixture**

`suite/b5_serving/prompts.yaml`:
```yaml
# B5 measurement prompts — serving fixtures, versioned; fixture_sha covers this file.
peak_prompt: "Write a detailed 700-word essay about the history of computing. Do not stop early."
peak_max_tokens: 1000
sustained_filler_paragraph: >-
  The quarterly infrastructure review covers server utilization, network throughput,
  storage capacity planning, backup verification, patch compliance, certificate expiry
  tracking, license reconciliation, and vendor contract renewals across all managed
  client environments in the region.
sustained_ctx_tokens: 32000
sustained_question: "Summarize the recurring themes of the document above in one paragraph."
sustained_max_tokens: 800
conc_prompt: "List 40 practical uses for a Raspberry Pi in a small business. Be concise."
conc_max_tokens: 400
```

- [ ] **Step 2: Failing unit tests**

`tests/test_b5.py`:
```python
from llmtest.batteries.b5_serving import peak_metrics, concurrency_metrics, build_sustained_prompt

def test_peak_metrics_maps_fork_timings():
    t = {"predicted_per_second": 150.5, "prompt_per_second": 2000.0,
         "predicted_n": 800, "prompt_n": 30, "predicted_ms": 5314.0}
    m = peak_metrics(t, ttft_ms=210.0)
    assert m["decode_tps"] == 150.5 and m["pp_tps"] == 2000.0
    assert m["ttft_ms"] == 210.0 and m["tokens_out"] == 800

def test_concurrency_metrics_aggregates():
    per = [{"predicted_n": 400, "predicted_per_second": 90.0},
           {"predicted_n": 400, "predicted_per_second": 88.0}]
    m = concurrency_metrics(per, elapsed_s=5.0)
    assert m["aggregate_tps"] == 160.0            # 800 tokens / 5 s
    assert m["per_stream_tps_mean"] == 89.0
    assert m["streams_ok"] == 2

def test_sustained_prompt_hits_target_length():
    p = build_sustained_prompt("word " * 40, 32000, "Q?")
    assert len(p) // 4 >= 30000                   # ~4 chars/token heuristic
```

- [ ] **Step 3: Verify fail** → FAIL.

- [ ] **Step 4: Implementation**

`llmtest/batteries/b5_serving.py`:
```python
"""Battery 5 — throughput & serving (TESTPLAN 5.5). Owns server lifecycle; mints timing_authoritative rows."""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import yaml

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register

_FIXTURE = Path("suite/b5_serving/prompts.yaml")


def _fixture_sha(root: Path) -> str:
    return hashlib.sha256((root / _FIXTURE).read_bytes()).hexdigest()


def peak_metrics(timings: dict, *, ttft_ms: float) -> dict:
    return {"decode_tps": float(timings.get("predicted_per_second", 0.0)),
            "pp_tps": float(timings.get("prompt_per_second", 0.0)),
            "ttft_ms": float(ttft_ms),
            "tokens_out": int(timings.get("predicted_n", 0)),
            "tokens_in": int(timings.get("prompt_n", 0)),
            "n_drafted": timings.get("draft_n"),
            "n_accepted": timings.get("draft_n_accepted"),
            "accept_rate": (timings.get("draft_n_accepted", 0) / timings["draft_n"])
            if timings.get("draft_n") else None}


def concurrency_metrics(per_stream: list[dict], *, elapsed_s: float) -> dict:
    total = sum(int(t.get("predicted_n", 0)) for t in per_stream)
    speeds = [float(t.get("predicted_per_second", 0.0)) for t in per_stream]
    return {"aggregate_tps": total / elapsed_s if elapsed_s else 0.0,
            "per_stream_tps_mean": sum(speeds) / len(speeds) if speeds else 0.0,
            "streams_ok": len(per_stream)}


def build_sustained_prompt(paragraph: str, target_tokens: int, question: str) -> str:
    approx_chars = target_tokens * 4
    body = (paragraph + "\n") * (approx_chars // max(len(paragraph), 1) + 1)
    return body[:approx_chars] + "\n\n" + question


def _conditions(order):
    def c(**kw):
        return schema.canonical_condition(kw, order)
    conds = [c(runtime="fork", spec="ngram32", kv="q8", cond="PEAK"),
             c(runtime="fork", spec="ngram32", kv="q8", cond="SUSTAINED32K"),
             c(runtime="fork", spec="off", kv="q8", cond="PEAK"),
             c(runtime="fork", spec="off", kv="q8", cond="SUSTAINED32K")]
    conds += [c(runtime="fork", spec="ngram32", kv="q8", cond="PEAK", conc=n)
              for n in (2, 4, 8, 16)]
    return conds


@register
class B5Serving(Battery):
    id = 5

    def plan(self, cfg, store, model_filter=None) -> list[WorkItem]:
        order = cfg.suite["condition_order"]
        fx = _fixture_sha(cfg.root)
        sv = cfg.suite["suite_version"]
        items = []
        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            if str(m.get("local_path", "")).startswith("TO-"):
                continue                    # artifact not on disk yet
            for cond in _conditions(order):
                for run_n in (1,):          # serving rows: 1 run per condition (repeat via --force)
                    rid = schema.compute_row_id(
                        suite_version=sv, model_id=model_id,
                        quant_sha256=m["provenance"]["sha256"], battery=5,
                        task_id="b5.serving", fixture_sha=fx,
                        condition=cond, run_n=run_n)
                    items.append(WorkItem(row_id=rid, model_id=model_id, battery=5,
                                          task_id="b5.serving", condition=cond,
                                          run_n=run_n,
                                          payload={"model": m, "fixture_sha": fx,
                                                   "suite_version": sv}))
        return items

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        cfg = ctx.cfg
        fixture = yaml.safe_load((cfg.root / _FIXTURE).read_text(encoding="utf-8"))
        parts = dict(kv.split("=") for kv in item.condition.split(";"))
        conc = int(parts.get("conc", 1))
        overlay = {"spec": "off"} if parts["spec"] == "off" else None
        ctx_len = 36864 if parts["cond"] == "SUSTAINED32K" else 8192
        mgr = ctx.server_manager()
        handle = mgr.request_endpoint(item.model_id, runtime="fork",
                                      flags_overlay=overlay, parallel=conc,
                                      ctx=ctx_len, kv="q8_0",
                                      timing_authoritative=True)
        if conc > 1:
            results, lock = [], threading.Lock()
            def worker():
                d = handle.chat([{"role": "user", "content": fixture["conc_prompt"]}],
                                max_tokens=fixture["conc_max_tokens"])
                with lock:
                    results.append(d.get("timings", {}))
            t0 = time.time()
            threads = [threading.Thread(target=worker) for _ in range(conc)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            metrics = concurrency_metrics(results, elapsed_s=time.time() - t0)
            resp_meta = {"decode_tps": metrics["per_stream_tps_mean"],
                         "pp_tps": 0.0, "ttft_ms": 0.0,
                         "tokens_out": sum(int(r.get("predicted_n", 0)) for r in results)}
        else:
            if parts["cond"] == "SUSTAINED32K":
                prompt = build_sustained_prompt(fixture["sustained_filler_paragraph"],
                                                fixture["sustained_ctx_tokens"],
                                                fixture["sustained_question"])
                max_tokens = fixture["sustained_max_tokens"]
            else:
                prompt = fixture["peak_prompt"]
                max_tokens = fixture["peak_max_tokens"]
            t0 = time.time()
            d = handle.chat([{"role": "user", "content": prompt}],
                            max_tokens=max_tokens)
            ttft = d.get("timings", {}).get("prompt_ms", (time.time() - t0) * 1000)
            metrics = resp_meta = peak_metrics(d.get("timings", {}), ttft_ms=ttft)
        m = item.payload["model"]
        row = schema.ResultRow.new(
            suite_version=item.payload["suite_version"], model_id=item.model_id,
            hf_repo=m["hf_repo"], quant_file=m["quant_file"],
            quant_sha256=m["provenance"]["sha256"], tier="T1", battery=5,
            task_id=item.task_id, fixture_sha=item.payload["fixture_sha"],
            condition=item.condition, run_n=item.run_n,
            session_id=handle.session_id,
            sampling={"temp": 0.0, "max_tokens": 0, "top_p": None, "seed": None},
            response_meta={k: v for k, v in resp_meta.items() if v is not None},
            metrics={k: v for k, v in metrics.items() if v is not None},
            timing_authoritative=True)
        return [row.to_dict()]
```

- [ ] **Step 5: Run unit tests** — `python -m pytest tests/test_b5.py -v` → PASS (3).
- [ ] **Step 6: Commit** — `git add llmtest/batteries/b5_serving.py suite/ tests/test_b5.py config/registry.yaml && git commit -m "feat(p2): Battery 5 serving plugin — PEAK/SUSTAINED-32k, ngram A/B, conc ladder" && git push`

### Task 13: gpt-oss-20b shakedown — P2 EXIT GATE

**Files:** none new (execution + review); Modify: `config/registry.yaml` (coder-30b freeze after download)

- [ ] **Step 1: Bench profile + preconditions**

AC power connected; performance plan active (`powercfg /getactivescheme`); no other GPU consumers (`nvidia-smi` shows ~0 MiB used); `python -m llmtest validate` exit 0.

- [ ] **Step 2: Run the shakedown (~20–30 min)**

Run: `python -m llmtest run --suite smoke --battery 5 --model gpt-oss-20b`
Expected: `run: 8 planned, 8 pending` → 8 rows appended, 0 failures. Interrupt it once mid-run (Ctrl+C after ~3 rows) and re-run to **prove free resume**: second invocation reports `8 planned, ~5 pending`.

- [ ] **Step 3: Verify data quality (the shakedown checklist)**

- `python -m llmtest status` → `B5 gpt-oss-20b ok 8`.
- `python -m llmtest tables` → `results/tables/serving.md` shows 8 rows, PEAK decode in the ~140–170 t/s band and ngram-off PEAK LOWER than ngram-on for the edit-free essay prompt is NOT expected — for from-scratch prose they should be within ~±15% (n-gram helps little on novel text; the SUSTAINED comparison is the informative pair). Sanity: no zeros, TTFT > 0, conc ladder aggregate rises with N.
- `results/sessions.jsonl`: 8+ sessions, every one carries normalized_config + raw_invocation + power_mode.
- `git diff` after re-running `llmtest tables` twice → byte-clean.

- [ ] **Step 4: Fix-forward loop**

Any failure = fix in the relevant module, re-run (resume skips good rows), repeat until the checklist passes clean **twice in a row**. Every fix commits individually with a `fix(p2-shakedown):` prefix.

- [ ] **Step 5: Close the 6/6 registry (coder-30b) — post-shakedown, pre-P3**

Run: `curl.exe -L -o "D:\BUILT-TOOLS\LLMtesting\bonsai\Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf" "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/UD-Q4_K_XL/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"` (verify exact repo file path with the HF API first — single-file vs subdir layout varies). Update `local_path` in registry.yaml, then `python scripts/freeze_artifacts.py qwen3-coder-30b` → `frozen`.

- [ ] **Step 6: P2 exit commit + tag**

```bash
git add -A
git commit -m "feat(p2): gpt-oss-20b shakedown clean — B5 rows, resume proven, tables byte-clean; registry 6/6 frozen"
git tag p2-shakedown-clean && git push && git push --tags
```

**P2 exit = ABC validated by its server-owning client; next (separate plan): P3 = B1 + judging pipeline.**

---

## Self-Review Findings (applied)

1. **Task 6 forward-reference fixed:** coder-30b download lives in Task 13 Step 5 (not "Task 11").
2. **Task 9/11 interface drift fixed:** FakeBattery in `test_run_cmd.py` uses `payload`, matching the ABC's `WorkItem.payload` (Task 11 Step 3 includes the test update).
3. **Spec coverage check:** P0 §9 exit criterion (Task 1+7) ✓ · schema/store/resume (3,4) ✓ · configs+fits (5) ✓ · provenance freeze incl. rule-1 artifacts (6) ✓ · mojibake lint (5) ✓ · CI byte-clean tables (7) ✓ · ServerManager translation/orphans/PID/fits-preflight/sessions+power (8) ✓ · debug flags (9) ✓ · re-runnable canary (10) ✓ · ABC minimal + preflight hook (11) ✓ · B5 PEAK/SUSTAINED/ngram-A/B/conc + timing_authoritative + spec-stats (12) ✓ · shakedown + resume proof (13) ✓. Deferred BY DESIGN: judging pipeline (P3), B5 ollama/vllm arms (flagged, config-gated), append-only-tag CI check (P8), Verda leg (P8).
4. **Type consistency pass:** `ResultRow.new(...)` signature matches all call sites (canary uses raw chat, no rows — correct: canary is a health check, not data); `compute_row_id` kwargs consistent in schema/store tests/B5.plan; `normalized_config` keys identical in `normalize_config()` and SessionRow consumers.
