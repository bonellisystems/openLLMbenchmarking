"""Battery 1 — business tasks fixture loader and signal checker."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Unicode dash/hyphen variants folded to ASCII "-" before signal matching.
_DASH_VARIANTS = (
    "‐"  # HYPHEN
    "‑"  # NON-BREAKING HYPHEN
    "‒"  # FIGURE DASH
    "–"  # EN DASH
    "—"  # EM DASH
    "―"  # HORIZONTAL BAR
    "−"  # MINUS SIGN
)

# Unicode "fancy" quote variants folded to their ASCII equivalents.
_QUOTE_FOLD = {
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
}

_DASH_TABLE = {ord(c): "-" for c in _DASH_VARIANTS}
_QUOTE_TABLE = {ord(c): repl for c, repl in _QUOTE_FOLD.items()}


def _normalize_typography(text: str) -> str:
    """Normalize answer text so ASCII-authored signals tolerate typographic
    substitution (Unicode dashes, curly quotes) that real models produce.

    Applies NFKC normalization, then folds hyphen/dash variants to ASCII
    "-" and fancy quote variants to ASCII quotes.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASH_TABLE)
    text = text.translate(_QUOTE_TABLE)
    return text


@dataclass
class Task:
    """Fixture task representation."""
    id: str
    unit: str
    difficulty: str
    cls: str
    industry: str
    prompt: str
    signals: list[dict]
    fixture_sha: str
    path: Path


def load_unit_tasks(root: Path, unit: str) -> list[Task]:
    """Load all task fixtures for a given unit.

    Args:
        root: Repository root
        unit: Unit name (e.g., 'cybersecurity')

    Returns:
        List of Task objects, sorted by id

    Raises:
        ValueError: If a fixture file is malformed and cannot be parsed
    """
    tasks_dir = root / "suite" / "b1_business" / unit
    if not tasks_dir.exists():
        return []

    tasks = []
    for task_file in sorted(tasks_dir.glob("task-*.yaml")):
        try:
            data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
            fixture_sha = hashlib.sha256(task_file.read_bytes()).hexdigest()
            # Require industry field
            if "industry" not in data:
                raise ValueError("missing required key: industry")
            task = Task(
                id=data["id"],
                unit=data["unit"],
                difficulty=data["difficulty"],
                cls=data["class"],
                industry=data["industry"],
                prompt=data["prompt"],
                signals=data.get("signals", []),
                fixture_sha=fixture_sha,
                path=task_file
            )
            tasks.append(task)
        except Exception as e:
            # Fail loud on malformed fixtures
            raise ValueError(f"malformed fixture {task_file}: {e}") from e

    return sorted(tasks, key=lambda t: t.id)


def check_signals(text: str, signals: list[dict]) -> dict:
    """Check if text passes all signal checks.

    Args:
        text: The text to check against signals
        signals: List of signal dicts with type, value, and optional tolerance

    Returns:
        Dict mapping signal name (e.g. "contains-0") to result dict with 'pass' key
    """
    text = _normalize_typography(text)
    results = {}

    for idx, sig in enumerate(signals):
        sig_type = sig.get("type")
        sig_value = sig.get("value")
        sig_key = f"{sig_type}-{idx}"

        if sig_type == "contains":
            results[sig_key] = {"pass": sig_value in text}
        elif sig_type == "regex":
            try:
                results[sig_key] = {"pass": bool(re.search(sig_value, text))}
            except re.error as e:
                # Defensive: regex compile fails at runtime (shouldn't happen after lint)
                results[sig_key] = {"pass": False, "error": f"regex compile error: {e}"}
        elif sig_type == "numeric":
            tolerance = sig.get("tolerance", 0.01)
            results[sig_key] = {"pass": _check_numeric(text, sig_value, tolerance)}
        else:
            results[sig_key] = {"pass": False}

    return results


def _check_numeric(text: str, target: float | int, tolerance: float) -> bool:
    """Check if text contains a number within tolerance of target.

    Strips commas and dollar signs from digit groups before parsing.

    Args:
        text: The text to search for numbers
        target: The target number
        tolerance: The relative tolerance (0.01 = 1%)

    Returns:
        True if any number in text is within tolerance of target
    """
    # Strip commas and dollar signs from digit groups
    cleaned = re.sub(r'\$([0-9,]+)', r'\1', text)  # Remove $ before numbers
    cleaned = re.sub(r'([0-9]),([0-9])', r'\1\2', cleaned)  # Remove commas between digits

    # Find all numbers in the cleaned text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned)

    if not numbers:
        return False

    for num_str in numbers:
        try:
            num = float(num_str)
            # Check absolute relative difference
            if target != 0:
                rel_diff = abs((num - target) / target)
            else:
                rel_diff = abs(num - target)
            if rel_diff <= tolerance:
                return True
        except ValueError:
            continue

    return False
