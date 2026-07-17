"""Battery 7 — harness/config sensitivity matrix. Probe-task fixture loader.

Mirrors b1_fixtures.py's shape (Task dataclass + a loader that fails loud on
malformed fixtures, per-task fixture_sha as a content hash of the YAML
bytes). B7's probe set is FLAT and FIXED (no unit/industry/difficulty
distribution rule -- TESTPLAN 5.7 calls for a small fixed probe set run
across the matrix, not a business-unit corpus), so the loader is simpler
than B1's: one directory, one naming pattern (`probe-<NN>.yaml`).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_RESPONSE_FORMATS = {"text", "json"}
_VALID_SIGNAL_TYPES = {"contains", "regex", "numeric"}


@dataclass
class Task:
    """B7 probe task representation."""
    id: str
    prompt: str
    signals: list[dict]
    expects_tool_call: bool
    tool_schema: dict | None
    expected_tool_name: str | None
    response_format: str
    fixture_sha: str
    path: Path


def load_probe_tasks(root: Path, probes_dir: str = "suite/b7_harnessmatrix/probes") -> list[Task]:
    """Load the fixed B7 probe set.

    Args:
        root: Repository root
        probes_dir: Repo-relative path to the probes directory
            (config-overridable via suite.yaml b7.probes_dir)

    Returns:
        List of Task objects, sorted by id

    Raises:
        ValueError: If a fixture file is malformed and cannot be parsed
    """
    tasks_dir = root / probes_dir
    if not tasks_dir.exists():
        return []

    tasks = []
    for task_file in sorted(tasks_dir.glob("probe-*.yaml")):
        try:
            data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
            fixture_sha = hashlib.sha256(task_file.read_bytes()).hexdigest()

            for key in ("id", "prompt", "signals"):
                if key not in data:
                    raise ValueError(f"missing required key: {key}")

            expects_tool_call = bool(data.get("expects_tool_call", False))
            tool_schema = data.get("tool_schema")
            expected_tool_name = data.get("expected_tool_name")
            if expects_tool_call:
                if not tool_schema:
                    raise ValueError("expects_tool_call=true requires tool_schema")
                if not expected_tool_name:
                    raise ValueError("expects_tool_call=true requires expected_tool_name")

            response_format = data.get("response_format", "text")
            if response_format not in _RESPONSE_FORMATS:
                raise ValueError(
                    f"response_format must be one of {sorted(_RESPONSE_FORMATS)}: {response_format}")

            task = Task(
                id=data["id"],
                prompt=data["prompt"],
                signals=data.get("signals", []),
                expects_tool_call=expects_tool_call,
                tool_schema=tool_schema,
                expected_tool_name=expected_tool_name,
                response_format=response_format,
                fixture_sha=fixture_sha,
                path=task_file,
            )
            tasks.append(task)
        except Exception as e:
            # Fail loud on malformed fixtures (mirror b1_fixtures.load_unit_tasks)
            raise ValueError(f"malformed fixture {task_file}: {e}") from e

    return sorted(tasks, key=lambda t: t.id)


def lint_probe_tasks(tasks: list[Task]) -> list[str]:
    """Lint an already-loaded probe set: id format, uniqueness, signal shape.

    Smaller-scope sibling of validate_cmd's B1 fixture lint -- B7 has no
    unit/industry distribution rule since it's a fixed flat probe set, not a
    per-unit business-task corpus, so there's nothing analogous to lint
    there. Returns a list of human-readable error strings (empty = clean).
    """
    errs: list[str] = []
    seen_ids: set[str] = set()
    for t in tasks:
        if not re.match(r"^probe-\d{2}$", t.id):
            errs.append(f"{t.path}: id '{t.id}' doesn't match pattern 'probe-<NN>'")
        if t.id in seen_ids:
            errs.append(f"{t.path}: duplicate id '{t.id}'")
        seen_ids.add(t.id)
        if not t.signals:
            errs.append(f"{t.path}: no signals defined")
        for idx, sig in enumerate(t.signals):
            sig_type = sig.get("type")
            if sig_type not in _VALID_SIGNAL_TYPES:
                errs.append(f"{t.path}: signal {idx} unknown type '{sig_type}'")
            if sig.get("value") is None:
                errs.append(f"{t.path}: signal {idx} missing 'value'")
            elif sig_type == "regex":
                try:
                    re.compile(sig["value"])
                except re.error as e:
                    errs.append(f"{t.path}: signal {idx} regex failed to compile: {e}")
    return errs
