"""Battery 4 -- long-context fixture loader, document builder, and signal checker.

Mirrors b1_fixtures.py's shape (Task dataclass + loader + fail-loud lint), but the
"document" a B4 task scores against is never stored as a literal 256k-token file.
Fixtures store a SEED: a short filler paragraph that gets repeated/padded to the
sweep's target length at execute() time, plus one or more planted "needles" at
declared depth percentages, a question, and deterministic signals (TESTPLAN 5.4:
NIAH single + multi-needle, multi-hop, position-sweep / lost-in-the-middle).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from llmtest.batteries.b1_fixtures import _check_numeric, _normalize_typography

_VALID_KINDS = {"single_needle", "multi_needle", "multi_hop", "distractor"}


@dataclass
class LongContextTask:
    """Fixture task representation for Battery 4."""
    id: str
    kind: str
    filler_template: str
    needles: list[dict]     # [{depth_pct, text}, ...]
    question: str
    signals: list[dict]
    fixture_sha: str
    path: Path


def load_longcontext_tasks(root: Path) -> list[LongContextTask]:
    """Load all B4 long-context task fixtures from suite/b4_longcontext/.

    Args:
        root: Repository root

    Returns:
        List of LongContextTask objects, sorted by id

    Raises:
        ValueError: If a fixture file is malformed and cannot be parsed
    """
    tasks_dir = root / "suite" / "b4_longcontext"
    if not tasks_dir.exists():
        return []

    tasks = []
    for task_file in sorted(tasks_dir.glob("task-*.yaml")):
        try:
            data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
            fixture_sha = hashlib.sha256(task_file.read_bytes()).hexdigest()
            for key in ("id", "kind", "filler_template", "needles", "question", "signals"):
                if key not in data:
                    raise ValueError(f"missing required key: {key}")
            if data["kind"] not in _VALID_KINDS:
                raise ValueError(f"unknown kind '{data['kind']}' (must be one of {_VALID_KINDS})")
            if not data["needles"]:
                raise ValueError("needles must be a non-empty list")
            for idx, needle in enumerate(data["needles"]):
                if "depth_pct" not in needle or "text" not in needle:
                    raise ValueError(f"needle {idx} missing depth_pct or text")
                if not (0 <= needle["depth_pct"] <= 100):
                    raise ValueError(f"needle {idx} depth_pct out of 0-100 range")
            task = LongContextTask(
                id=data["id"],
                kind=data["kind"],
                filler_template=data["filler_template"],
                needles=data["needles"],
                question=data["question"],
                signals=data.get("signals", []),
                fixture_sha=fixture_sha,
                path=task_file,
            )
            tasks.append(task)
        except Exception as e:
            raise ValueError(f"malformed fixture {task_file}: {e}") from e

    return sorted(tasks, key=lambda t: t.id)


def build_document(filler_template: str, target_tokens: int, needles: list[dict],
                   question: str) -> str:
    """Pad filler_template to ~target_tokens (4 chars/token heuristic, matching
    b5_serving.build_sustained_prompt), plant each needle at its depth_pct offset
    (snapped to the next line boundary for readability), and append the question.

    Needles are inserted in DESCENDING depth order so each insertion's offset,
    computed against the original unpadded body, is never shifted by an earlier
    insertion -- this keeps depth_pct placement deterministic and independent of
    needle list order.
    """
    approx_chars = target_tokens * 4
    unit = filler_template if filler_template.endswith("\n") else filler_template + "\n"
    reps = approx_chars // max(len(unit), 1) + 1
    body = (unit * reps)[:approx_chars]

    for needle in sorted(needles, key=lambda n: n["depth_pct"], reverse=True):
        offset = max(0, min(len(body), int(len(body) * needle["depth_pct"] / 100)))
        nl = body.find("\n", offset)
        if nl == -1:
            nl = len(body)
        body = body[:nl] + f"\n{needle['text']}\n" + body[nl:]

    instructions = ("Answer the question using ONLY information found in the document "
                    "below. Be precise and concise.\n\n")
    return f"{instructions}<document>\n{body}\n</document>\n\nQuestion: {question}\nAnswer:"


def check_needle_signals(text: str, signals: list[dict]) -> dict:
    """Deterministic scoring for B4 answers: retrieval is checkable, never judged
    (TESTPLAN 5.4). Extends b1_fixtures' contains/regex/numeric vocabulary with
    'not_contains' (distractor-rejection check) while reusing its Unicode
    typography normalization and numeric-tolerance matching.
    """
    norm = _normalize_typography(text)
    results = {}

    for idx, sig in enumerate(signals):
        sig_type = sig.get("type")
        sig_value = sig.get("value")
        key = f"{sig_type}-{idx}"

        if sig_type == "contains":
            results[key] = {"pass": sig_value in norm}
        elif sig_type == "not_contains":
            results[key] = {"pass": sig_value not in norm}
        elif sig_type == "regex":
            try:
                results[key] = {"pass": bool(re.search(sig_value, norm))}
            except re.error as e:
                results[key] = {"pass": False, "error": f"regex compile error: {e}"}
        elif sig_type == "numeric":
            tolerance = sig.get("tolerance", 0.01)
            results[key] = {"pass": _check_numeric(norm, sig_value, tolerance)}
        else:
            results[key] = {"pass": False}

    return results
