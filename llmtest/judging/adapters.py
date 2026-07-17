"""Judge CLI adapters (TESTPLAN 6.1) -- stdin invocation, strict reply parser, FakeJudge.

Three headless CLI judges (claude / codex / gemini) are driven identically:
the blinded packet body goes in on **stdin** (Windows argv-length limits kill
a 10k-token argument), stdout is captured whole, and `parse_reply` extracts
the first balanced top-level JSON object from whatever prose/fences the CLI
wrapped it in. Validation is total -- a partially-valid reply (missing
letter, wrong type, non-permutation ranking) is treated as fully invalid;
the runner (Task 7) owns the one-retry-then-error policy.

Pins in `config/judges.yaml` are still DRAFT ("TO-FREEZE-P3") as of this
module -- live enumeration and the G1/G3 human-gated freeze are a separate,
non-code controller step. Nothing in this module invokes a real judge CLI;
`make_adapter` only builds argv from whatever `invoke` template a config
entry carries.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class JudgeReply:
    raw: str
    parsed: dict | None
    error: str | None


def _scan_balanced_object(text: str, start: int) -> str | None:
    """Return text[start:end+1] for the balanced {...} beginning at `start`,
    or None if the braces starting there never balance. Braces inside JSON
    string literals (including escaped quotes) are not counted, so a reason
    string containing literal '{'/'}' cannot desync the depth counter."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _extract_first_json_object(text: str) -> str | None:
    """Find the first balanced top-level JSON object anywhere in `text`
    (arbitrary prose/markdown fences allowed around it). Tries each '{' in
    turn -- an unbalanced one (e.g. a stray brace in prose before the real
    object) is skipped rather than giving up."""
    idx = text.find("{")
    while idx != -1:
        obj = _scan_balanced_object(text, idx)
        if obj is not None:
            return obj
        idx = text.find("{", idx + 1)
    return None


def parse_reply(stdout: str, expected_letters: list[str]) -> tuple[dict | None, str | None]:
    """Extract + strictly validate a judge's reply from raw CLI stdout.

    Returns (parsed_dict, None) on success, else (None, "specific reason").
    Validation is ALL-or-nothing: `scores` must have every expected letter
    mapped to an int 0-10 (bool explicitly rejected -- it's a subclass of
    int in Python but never a valid score), `reasons` must have every
    expected letter mapped to a string, and `ranking` must be exactly a
    permutation of expected_letters (every letter once, nothing extra).
    """
    obj_text = _extract_first_json_object(stdout)
    if obj_text is None:
        return None, "no JSON object found in reply"

    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"

    if not isinstance(obj, dict):
        return None, "top-level JSON is not an object"

    scores = obj.get("scores")
    reasons = obj.get("reasons")
    ranking = obj.get("ranking")

    if not isinstance(scores, dict):
        return None, "missing or invalid 'scores' field"
    if not isinstance(reasons, dict):
        return None, "missing or invalid 'reasons' field"
    if not isinstance(ranking, list):
        return None, "missing or invalid 'ranking' field"

    expected_set = set(expected_letters)

    missing_scores = sorted(expected_set - set(scores))
    if missing_scores:
        return None, f"scores missing letters: {missing_scores}"
    for letter in expected_letters:
        val = scores[letter]
        if isinstance(val, bool):
            return None, f"score for {letter!r} is a bool, not an int"
        if not isinstance(val, int):
            return None, f"score for {letter!r} is not an int: {val!r}"
        if not (0 <= val <= 10):
            return None, f"score for {letter!r} out of range 0-10: {val}"

    missing_reasons = sorted(expected_set - set(reasons))
    if missing_reasons:
        return None, f"reasons missing letters: {missing_reasons}"
    for letter in expected_letters:
        if not isinstance(reasons[letter], str):
            return None, f"reason for {letter!r} is not a string"

    if not all(isinstance(x, str) for x in ranking):
        return None, "ranking must be a list of letter strings"
    if sorted(ranking) != sorted(expected_letters):
        return None, f"ranking is not a permutation of expected letters: {ranking}"

    return obj, None


class BaseAdapter:
    """Subprocess-based judge adapter: packet body on stdin, stdout parsed.

    `argv` is the FINAL, already-resolved command (no `{model}` placeholders
    left) -- subclasses do their own templating in __init__ before calling
    super().__init__().
    """

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv: list[str]):
        self.judge_id = judge_id
        self.model_pin = model_pin
        self.cli_version = cli_version
        self.argv = list(argv)

    def invoke(self, packet_text: str, expected_letters: list[str],
               timeout: int = 300) -> JudgeReply:
        try:
            proc = subprocess.run(
                self.argv, input=packet_text, capture_output=True, text=True,
                encoding="utf-8", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return JudgeReply(raw="", parsed=None, error="timeout")

        stdout = proc.stdout or ""
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            return JudgeReply(raw=stdout, parsed=None,
                               error=f"nonzero exit {proc.returncode}: {stderr}")

        parsed, error = self._parse_stdout(stdout, expected_letters)
        return JudgeReply(raw=stdout, parsed=parsed, error=error)

    def _parse_stdout(self, stdout: str,
                       expected_letters: list[str]) -> tuple[dict | None, str | None]:
        """Hook for CLI-specific envelope unwrapping. Default: stdout IS the
        reply text."""
        return parse_reply(stdout, expected_letters)


def _substitute_argv(argv_template: list[str], model_pin: str) -> list[str]:
    """Replace the literal '{model}' placeholder in each token. Tokens with
    no placeholder (flags, the bare '-'/'-p' stdin sentinel, etc) pass
    through unchanged."""
    return [tok.replace("{model}", model_pin) for tok in argv_template]


class ClaudeAdapter(BaseAdapter):
    """`claude -p --model <pin> --output-format json`.

    Claude's `--output-format json` wraps the actual reply text in a CLI
    envelope: `{"result": "<reply text>", ...}`. The reply text itself is
    parsed for the scores/reasons/ranking object.
    """

    DEFAULT_ARGV_TEMPLATE = ["claude", "-p", "--model", "{model}",
                              "--output-format", "json"]

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv_template: list[str] | None = None):
        argv_template = argv_template or self.DEFAULT_ARGV_TEMPLATE
        super().__init__(judge_id, model_pin, cli_version,
                          _substitute_argv(argv_template, model_pin))

    def _parse_stdout(self, stdout: str,
                       expected_letters: list[str]) -> tuple[dict | None, str | None]:
        try:
            envelope = json.loads(stdout)
            result = envelope["result"]
            if not isinstance(result, str):
                raise TypeError("envelope 'result' field is not a string")
        except Exception:
            # Defensive fallback: envelope didn't load as expected -- maybe
            # stdout IS the raw reply already (e.g. CLI flag changed, or a
            # test/older CLI version without the envelope wrapper).
            return parse_reply(stdout, expected_letters)
        return parse_reply(result, expected_letters)


class CodexAdapter(BaseAdapter):
    """`codex exec --model <pin> -` -- stdin sentinel confirmed at freeze
    time (Task 6 Step 4); argv_template lets the freeze step adjust the
    exact syntax without touching this code."""

    DEFAULT_ARGV_TEMPLATE = ["codex", "exec", "--model", "{model}", "-"]

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv_template: list[str] | None = None):
        argv_template = argv_template or self.DEFAULT_ARGV_TEMPLATE
        super().__init__(judge_id, model_pin, cli_version,
                          _substitute_argv(argv_template, model_pin))


class GeminiAdapter(BaseAdapter):
    """`gemini -m <pin> -p` (stdin) -- same freeze-time-adjustable template
    mechanism as CodexAdapter."""

    DEFAULT_ARGV_TEMPLATE = ["gemini", "-m", "{model}", "-p"]

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv_template: list[str] | None = None):
        argv_template = argv_template or self.DEFAULT_ARGV_TEMPLATE
        super().__init__(judge_id, model_pin, cli_version,
                          _substitute_argv(argv_template, model_pin))


_ADAPTER_CLASSES = {
    "claude": ClaudeAdapter,
    "codex": CodexAdapter,
    "gemini": GeminiAdapter,
}


def make_adapter(judge_id: str, judges_cfg_entry: dict) -> BaseAdapter:
    """Build a concrete adapter from one config/judges.yaml entry, e.g.
    `{model: claude-fable-5, cli: claude, cli_version: ..., invoke: 'claude -p --model {model} --output-format json'}`.

    The `invoke` string is split on whitespace into an argv template (still
    carrying the literal '{model}' placeholder token(s); a trailing '-' or
    '-p' stdin sentinel token has no placeholder so it just passes through
    unchanged) and handed to the judge_id-matched adapter class, which does
    the actual substitution.
    """
    try:
        adapter_cls = _ADAPTER_CLASSES[judge_id]
    except KeyError:
        raise ValueError(f"unknown judge_id: {judge_id!r}") from None

    invoke_template = judges_cfg_entry["invoke"]
    argv_template = invoke_template.split()
    model_pin = judges_cfg_entry["model"]
    cli_version = judges_cfg_entry.get("cli_version")

    return adapter_cls(judge_id, model_pin, cli_version, argv_template=argv_template)


class FakeJudgeAdapter:
    """Deterministic offline judge -- no subprocess at all. Used by tests and
    by `llmtest judge --pending --fake` (shakedown without burning quota).

    `scores_fn(letters) -> {letter: int}` supplies the scores; reasons and a
    best-first ranking are derived deterministically. The resulting reply is
    run through the SAME `parse_reply` validation a real adapter's output
    would face, so a misbehaving scores_fn (wrong type, out-of-range score)
    surfaces the identical error a real judge's bad reply would.
    """

    def __init__(self, scores_fn):
        self.judge_id = "fake"
        self.model_pin = "fake"
        self.cli_version = "fake"
        self.scores_fn = scores_fn

    def invoke(self, packet_text: str, expected_letters: list[str],
               timeout: int = 300) -> JudgeReply:
        scores = self.scores_fn(expected_letters)
        reasons = {letter: f"fake reason for {letter}" for letter in expected_letters}
        ranking = sorted(expected_letters,
                          key=lambda l: (-scores.get(l, 0), l))
        raw = json.dumps({"scores": scores, "reasons": reasons, "ranking": ranking})
        parsed, error = parse_reply(raw, expected_letters)
        return JudgeReply(raw=raw, parsed=parsed, error=error)
