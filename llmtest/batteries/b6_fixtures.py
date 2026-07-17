"""Battery 6 -- agentic coding fixture loader, code extraction, and signal checking.

Mirrors b1_fixtures.py's shape (Task dataclass + loader + check_signals) but adds
the code-specific pieces B1 doesn't need: fenced-code-block extraction and a
compile()-only syntax check.

SAFETY (TESTPLAN 5.6 / build contract): model-generated code is NEVER executed by
this module. `compile_check()` calls the builtin `compile()` with mode="exec" to
produce a code object -- this parses and byte-compiles the source but does not run
it (no `exec()`/`eval()` call exists anywhere in this file). Scoring is otherwise
static string/regex signal matching against the extracted code text.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_TASKS_DIR = "b6_agenticcoding"

_VALID_TRACKS = {"scratch", "bugfix"}
_VALID_LANGUAGES = {"python", "bash", "sql", "js"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_SIGNAL_TYPES = {"contains", "regex", "absent"}

# Same typographic-normalization idea as b1_fixtures._normalize_typography,
# kept local/minimal here to keep the two battery modules independently
# readable (small enough that shared-util extraction isn't worth the coupling).
_DASH_TABLE = {ord(c): "-" for c in "‐‑‒–—―−"}
_QUOTE_TABLE = {ord(c): "'" for c in "‘’"} | {ord(c): '"' for c in "“”"}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DASH_TABLE)
    text = text.translate(_QUOTE_TABLE)
    return text


@dataclass
class Task:
    """B6 fixture task: either a from-scratch build (track="scratch") or a
    planted-bug self-correction task (track="bugfix")."""
    id: str
    track: str
    language: str
    difficulty: str
    prompt: str
    required_signals: list[dict] = field(default_factory=list)
    fix_signals: list[dict] = field(default_factory=list)
    regression_signals: list[dict] = field(default_factory=list)
    buggy_code: str | None = None
    symptom: str | None = None
    fixture_sha: str = ""
    path: Path | None = None


def load_tasks(root: Path) -> list[Task]:
    """Load all B6 task fixtures from suite/b6_agenticcoding/task-*.yaml.

    Fail-loud on malformed fixtures (mirrors b1_fixtures.load_unit_tasks): a
    missing required key, an unknown track/language, or a bugfix task missing
    buggy_code / regression_signals raises ValueError rather than silently
    skipping -- a silently-dropped bugfix fixture would mean nothing scores
    the model's actual self-correction ability on that bug.
    """
    tasks_dir = root / "suite" / _TASKS_DIR
    if not tasks_dir.exists():
        return []

    tasks: list[Task] = []
    for task_file in sorted(tasks_dir.glob("task-*.yaml")):
        try:
            data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
            fixture_sha = hashlib.sha256(task_file.read_bytes()).hexdigest()

            for key in ("id", "track", "language", "difficulty", "prompt"):
                if key not in data:
                    raise ValueError(f"missing required key: {key}")

            track = data["track"]
            if track not in _VALID_TRACKS:
                raise ValueError(f"invalid track: {track!r} (must be one of {_VALID_TRACKS})")

            language = data["language"]
            if language not in _VALID_LANGUAGES:
                raise ValueError(f"invalid language: {language!r} (must be one of {_VALID_LANGUAGES})")

            difficulty = data["difficulty"]
            if difficulty not in _VALID_DIFFICULTIES:
                raise ValueError(f"invalid difficulty: {difficulty!r} (must be one of {_VALID_DIFFICULTIES})")

            regression_signals = data.get("regression_signals", [])
            if track == "bugfix":
                if "buggy_code" not in data or not data["buggy_code"]:
                    raise ValueError("bugfix task missing buggy_code")
                if not regression_signals:
                    raise ValueError(
                        "bugfix task missing regression_signals "
                        "(the root-cause/no-op discriminator; required for every bugfix task)")

            task = Task(
                id=data["id"], track=track, language=language, difficulty=difficulty,
                prompt=data["prompt"],
                required_signals=data.get("required_signals", []),
                fix_signals=data.get("fix_signals", []),
                regression_signals=regression_signals,
                buggy_code=data.get("buggy_code"),
                symptom=data.get("symptom"),
                fixture_sha=fixture_sha, path=task_file)
            tasks.append(task)
        except Exception as e:
            raise ValueError(f"malformed fixture {task_file}: {e}") from e

    return sorted(tasks, key=lambda t: t.id)


def check_code_signals(code: str, signals: list[dict], prefix: str) -> dict:
    """Evaluate a list of signal dicts against extracted code text.

    Signal types:
      - contains: passes if value is a literal substring of code.
      - regex:    passes if re.search(value, code) matches.
      - absent:   passes if value is NOT a literal substring of code (the
                  "did the fix actually touch the buggy line" / no-op detector).

    Keys are namespaced by `prefix` (e.g. "required", "fix", "regression") so
    required/fix/regression signal lists never collide when merged into one
    det_checks dict, each restarting its own index at 0.
    """
    code_n = _normalize(code)
    results = {}
    for idx, sig in enumerate(signals or []):
        sig_type = sig.get("type")
        sig_value = sig.get("value")
        key = f"{prefix}.{sig_type}-{idx}"
        if sig_type == "contains":
            results[key] = {"pass": sig_value in code_n}
        elif sig_type == "regex":
            try:
                results[key] = {"pass": bool(re.search(sig_value, code_n))}
            except re.error as e:
                results[key] = {"pass": False, "error": f"regex compile error: {e}"}
        elif sig_type == "absent":
            results[key] = {"pass": sig_value not in code_n}
        else:
            results[key] = {"pass": False, "error": f"unknown signal type: {sig_type!r}"}
    return results


_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)

_LANG_ALIASES = {
    "python": {"python", "py", "python3"},
    "bash": {"bash", "sh", "shell", "zsh"},
    "sql": {"sql", "sqlite"},
    "js": {"javascript", "js", "node", "jsx"},
}


def extract_code_block(text: str, language: str | None = None) -> str | None:
    """Extract a fenced code block from model output text. NEVER executes
    anything -- pure string/regex extraction.

    Prefers a fence tagged with a known alias of `language`; falls back to an
    untagged fence, then to the first fenced block found at all; returns None
    if no fenced block exists anywhere in the text.
    """
    matches = _FENCE_RE.findall(text)
    if not matches:
        return None
    if language:
        aliases = _LANG_ALIASES.get(language, {language})
        for tag, body in matches:
            if tag.lower() in aliases:
                return body.strip("\n")
        for tag, body in matches:
            if tag.strip() == "":
                return body.strip("\n")
    return matches[0][1].strip("\n")


def compile_check(code: str) -> dict:
    """Static syntax-only check for Python code via compile(source, ..., "exec").

    compile() parses and byte-compiles source into a code object; it does NOT
    run any of the code's statements (that only happens on a subsequent
    exec()/eval() of the resulting code object, which this function never
    performs). Safe to call on arbitrary, untrusted, model-generated text --
    even code that would raise, exit the process, or have side effects if it
    were ever executed is merely parsed here.
    """
    if not code.strip():
        return {"pass": False, "error": "no code to compile"}
    try:
        compile(code, "<b6-fixture>", "exec")
        return {"pass": True}
    except SyntaxError as e:
        return {"pass": False, "error": f"SyntaxError: {e}"}
    except Exception as e:
        return {"pass": False, "error": f"{type(e).__name__}: {e}"}
