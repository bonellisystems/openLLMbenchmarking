"""Judge CLI adapters (TESTPLAN 6.1) -- stdin/file invocation, strict reply parser, FakeJudge.

Three headless CLI judges (claude / codex / gemini) are driven per the
`delivery:` key in `config/judges.yaml`:

- `delivery: stdin` (claude, codex) -- the blinded packet body goes in on
  **stdin** (Windows argv-length limits kill a 10k-token argument), stdout
  is captured whole, and `parse_reply` extracts the first balanced
  top-level JSON object from whatever prose/fences the CLI wrapped it in.
- `delivery: file` (gemini, via the AntiGravity `agy` CLI -- gemini-cli
  itself is deprecated) -- `--print` does not forward stdin to the model
  and argv caps at 32k chars, so the packet body is delivered as a file
  path (already under `artifacts/packets/`, matching agy's `--add-dir`
  grant) embedded into an instruction string; nothing goes on stdin. See
  `FileDeliveryAdapter`.

Validation is total -- a partially-valid reply (missing letter, wrong type,
non-permutation ranking) is treated as fully invalid; the runner
(`llmtest/judging/runner.py`) owns the one-retry-then-error policy.

Pins in `config/judges.yaml` are FROZEN (G1/G3 signed off 2026-07-17:
claude-fable-5 / gpt-5.6-sol / Gemini 3.1 Pro (High) via agy) -- see the
`frozen:` key in that file. `make_adapter` builds argv (and, for file
delivery, the per-call instruction) from whatever `invoke` template the
config entry carries; nothing in this module invokes a real judge CLI on
import -- subprocess only runs inside `invoke()`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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

    @staticmethod
    def _resolve_argv0(argv: list[str]) -> list[str]:
        """Resolve argv[0] via shutil.which() before handing argv to
        subprocess.run().

        Windows quirk (surfaced live in the Task 12 quota dry-run): many CLI
        tools (codex -- and any other npm-global install) are .cmd/.bat
        shims. subprocess.run(argv, shell=False) launches via CreateProcess
        directly, which does NOT apply PATHEXT resolution the way an
        interactive shell (or `where`) does -- a bare "codex" argv[0] raises
        WinError 2 ("cannot find the file specified") even though the CLI is
        genuinely on PATH and works fine when typed at a prompt.
        shutil.which() performs that same PATHEXT resolution, so this makes
        subprocess.run() see what the shell would have. A token
        shutil.which() can't resolve (already a full path, a POSIX binary
        subprocess can launch directly, or a genuinely missing binary) passes
        through unchanged -- this must never mask a real "binary not found"
        error into something else.
        """
        if not argv:
            return argv
        resolved = shutil.which(argv[0])
        if resolved is None:
            return argv
        return [resolved, *argv[1:]]

    def invoke(self, packet_text: str, expected_letters: list[str],
               timeout: int = 300, packet_path: Path | str | None = None) -> JudgeReply:
        argv, stdin_input = self._build_invocation(packet_text, packet_path)
        argv = self._resolve_argv0(argv)
        run_kwargs: dict = dict(capture_output=True, text=True, encoding="utf-8",
                                 timeout=timeout)
        if stdin_input is None:
            # File delivery (and any other no-stdin caller): explicit DEVNULL
            # rather than leaving stdin unset -- unset would inherit the
            # harness's real stdin, letting a CLI that misbehaves (or changes
            # behavior) block reading it or see data it has no business
            # seeing.
            run_kwargs["stdin"] = subprocess.DEVNULL
        else:
            run_kwargs["input"] = stdin_input
        try:
            proc = subprocess.run(argv, **run_kwargs)
        except subprocess.TimeoutExpired:
            return JudgeReply(raw="", parsed=None, error="timeout")
        except (OSError, subprocess.SubprocessError) as exc:
            # Nonexistent binary (FileNotFoundError), permissions, or any
            # other launch-time subprocess failure -- contained the same way
            # a bad reply is, so the runner's retry-then-error path handles
            # it instead of the whole judging run aborting.
            return JudgeReply(raw="", parsed=None, error=f"subprocess: {exc}")

        stdout = proc.stdout or ""
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            return JudgeReply(raw=stdout, parsed=None,
                               error=f"nonzero exit {proc.returncode}: {stderr}")

        parsed, error = self._parse_stdout(stdout, expected_letters)
        return JudgeReply(raw=stdout, parsed=parsed, error=error)

    def _build_invocation(self, packet_text: str,
                           packet_path: Path | str | None) -> tuple[list[str], str | None]:
        """Input-side hook, symmetric to `_parse_stdout` -- builds the final
        argv + stdin payload for one call. Default (stdin delivery): argv is
        unchanged, packet_text goes on stdin. `FileDeliveryAdapter` overrides
        this to embed `packet_path` into the argv's `{instruction}` token
        and send nothing on stdin."""
        return self.argv, packet_text

    def _reply_text(self, stdout: str) -> str:
        """Envelope-unwrap hook, symmetric to `_parse_stdout` but reusable
        by a caller that needs the raw reply TEXT rather than a parsed
        {scores,reasons,ranking} dict -- `llmtest.harness.failure_class.
        _invoke_classifier` (the B8 first-failure-classification panel,
        task-b8classify) is exactly that caller: it runs a classifier's
        reply through `parse_categorical_reply` (a different schema, `
        {"label": ...}`), not `parse_reply`, so it cannot go through
        `_parse_stdout` -- it needs the UNWRAPPED text this method returns.
        Default: stdout IS the reply text (no envelope to unwrap).
        Overridden by `ClaudeAdapter`, whose CLI wraps the reply in an
        outer `{"result": "..."}` envelope -- both `_parse_stdout` (numeric
        judge path) and `_invoke_classifier` (categorical path) call this
        one method, so the envelope-unwrap logic lives in exactly one
        place."""
        return stdout

    def _parse_stdout(self, stdout: str,
                       expected_letters: list[str]) -> tuple[dict | None, str | None]:
        """Hook for CLI-specific envelope unwrapping. Default: stdout IS the
        reply text (via `_reply_text`)."""
        return parse_reply(self._reply_text(stdout), expected_letters)


class FileDeliveryAdapter(BaseAdapter):
    """Mixin overriding the input side of `invoke()` for CLIs that read the
    packet from a file path rather than stdin (Task 6 gemini/agy handoff:
    `--print` doesn't forward stdin and argv caps at 32k chars).

    Combined with a concrete adapter class (see `make_adapter`'s
    `_file_delivery_variant`) so the CLI-specific `_parse_stdout` override
    (e.g. Claude's envelope unwrap) still applies if a future judge needs
    both; `_build_invocation` is the only thing this mixin overrides.
    """

    INSTRUCTION_TEMPLATE = (
        "Read the file at {path} and follow its instructions exactly. "
        "Reply with ONLY the JSON object it specifies."
    )

    def _build_invocation(self, packet_text: str,
                           packet_path: Path | str | None) -> tuple[list[str], str | None]:
        if packet_path is None:
            raise ValueError(
                f"{type(self).__name__} requires packet_path (file delivery, no stdin)")
        instruction = self.INSTRUCTION_TEMPLATE.format(path=packet_path)
        argv = [tok.replace("{instruction}", instruction) for tok in self.argv]
        return argv, None


def _substitute_argv(argv_template: list[str], model_pin: str,
                      cli: str | None = None) -> list[str]:
    """Replace the literal '{model}' and '{cli}' placeholders in each token
    (cli only when provided). The '{instruction}' placeholder, when present,
    is deliberately left untouched here -- it's per-call (file delivery only
    knows the packet path at invoke time), substituted by
    `FileDeliveryAdapter._build_invocation` instead. Tokens with no
    placeholder (flags, the bare '-'/'-p' stdin sentinel, etc) pass through
    unchanged."""
    def _sub(tok: str) -> str:
        tok = tok.replace("{model}", model_pin)
        if cli is not None:
            tok = tok.replace("{cli}", cli)
        return tok
    return [_sub(tok) for tok in argv_template]


class ClaudeAdapter(BaseAdapter):
    """`claude -p --model <pin> --output-format json`.

    Claude's `--output-format json` wraps the actual reply text in a CLI
    envelope: `{"result": "<reply text>", ...}`. The reply text itself is
    parsed for the scores/reasons/ranking object.
    """

    DEFAULT_ARGV_TEMPLATE = ["claude", "-p", "--model", "{model}",
                              "--output-format", "json"]

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv_template: list[str] | None = None, cli: str | None = None):
        argv_template = argv_template or self.DEFAULT_ARGV_TEMPLATE
        super().__init__(judge_id, model_pin, cli_version,
                          _substitute_argv(argv_template, model_pin, cli=cli))

    def _reply_text(self, stdout: str) -> str:
        """Unwrap the CLI's outer `{"result": "..."}` envelope, returning
        the nested reply text -- falls back to `stdout` unchanged if it
        doesn't parse as the expected envelope shape (e.g. an older CLI
        version, or a test's stdout that's already unwrapped). See
        `BaseAdapter._reply_text`'s docstring for why this is a separate
        method from `_parse_stdout` rather than inlined into it."""
        try:
            envelope = json.loads(stdout)
            result = envelope["result"]
            if not isinstance(result, str):
                raise TypeError("envelope 'result' field is not a string")
        except Exception:
            # Defensive fallback: envelope didn't load as expected -- maybe
            # stdout IS the raw reply already (e.g. CLI flag changed, or a
            # test/older CLI version without the envelope wrapper).
            return stdout
        return result


class CodexAdapter(BaseAdapter):
    """`codex exec --model <pin> -` -- stdin sentinel confirmed at freeze
    time (Task 6 Step 4); argv_template lets the freeze step adjust the
    exact syntax without touching this code."""

    DEFAULT_ARGV_TEMPLATE = ["codex", "exec", "--model", "{model}", "-"]

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv_template: list[str] | None = None, cli: str | None = None):
        argv_template = argv_template or self.DEFAULT_ARGV_TEMPLATE
        super().__init__(judge_id, model_pin, cli_version,
                          _substitute_argv(argv_template, model_pin, cli=cli))


class GeminiAdapter(BaseAdapter):
    """`gemini -m <pin> -p` (stdin) -- same freeze-time-adjustable template
    mechanism as CodexAdapter."""

    DEFAULT_ARGV_TEMPLATE = ["gemini", "-m", "{model}", "-p"]

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv_template: list[str] | None = None, cli: str | None = None):
        argv_template = argv_template or self.DEFAULT_ARGV_TEMPLATE
        super().__init__(judge_id, model_pin, cli_version,
                          _substitute_argv(argv_template, model_pin, cli=cli))


class KimiAdapter(BaseAdapter):
    """Moonshot Kimi (K3) agent CLI (`kimi -p {instruction} --yolo`).

    Kimi narrates its reasoning as `• ` bullet lines and appends a
    `To resume this session: ...` footer, and its narration can itself contain
    braces (e.g. echoing the requested JSON schema back), so the reply object is
    taken as the LAST balanced `{...}` in stdout that validates as a
    {scores,reasons,ranking} reply -- parse_reply's default first-object scan can
    otherwise lock onto a brace inside the narration. Candidate 4th judge (not in
    the frozen median-of-3 panel); driven via file delivery like gemini/agy since
    a 10k-token packet cannot ride in the `-p` argv (Windows argv cap)."""

    # NB: `-p` (non-interactive prompt) cannot be combined with --yolo/--auto
    # ("Cannot combine --prompt with --yolo"); in -p mode Kimi auto-executes
    # read-only tool actions (file reads) without an approval prompt anyway.
    DEFAULT_ARGV_TEMPLATE = ["kimi", "-p", "{instruction}",
                              "--output-format", "text"]

    def __init__(self, judge_id: str, model_pin: str, cli_version: str | None,
                 argv_template: list[str] | None = None, cli: str | None = None):
        argv_template = argv_template or self.DEFAULT_ARGV_TEMPLATE
        super().__init__(judge_id, model_pin, cli_version,
                          _substitute_argv(argv_template, model_pin, cli=cli))

    def _parse_stdout(self, stdout: str,
                       expected_letters: list[str]) -> tuple[dict | None, str | None]:
        footer = stdout.rfind("To resume this session")
        text = stdout[:footer] if footer != -1 else stdout
        last_ok, last_err = None, "no valid {scores,reasons,ranking} object in reply"
        idx = text.find("{")
        while idx != -1:
            obj = _scan_balanced_object(text, idx)
            if obj is None:
                idx = text.find("{", idx + 1)
                continue
            parsed, err = parse_reply(obj, expected_letters)
            if parsed is not None:
                last_ok = parsed
            else:
                last_err = err
            idx = text.find("{", idx + len(obj))
        return (last_ok, None) if last_ok is not None else (None, last_err)


_ADAPTER_CLASSES = {
    "claude": ClaudeAdapter,
    "codex": CodexAdapter,
    "gemini": GeminiAdapter,
    "kimi": KimiAdapter,
}

_file_delivery_variants: dict[type, type] = {}


def _file_delivery_variant(adapter_cls: type) -> type:
    """Combine `adapter_cls`'s CLI-specific `_parse_stdout` (output side)
    with `FileDeliveryAdapter`'s file-path `_build_invocation` (input side)
    via multiple inheritance -- MRO puts FileDeliveryAdapter's override
    first so it wins over BaseAdapter's stdin default, while any
    `_parse_stdout` override on `adapter_cls` (e.g. Claude's envelope
    unwrap) is untouched since FileDeliveryAdapter doesn't define one.
    Memoized so repeated `make_adapter` calls for the same judge_id don't
    keep minting new types."""
    if issubclass(adapter_cls, FileDeliveryAdapter):
        return adapter_cls
    cached = _file_delivery_variants.get(adapter_cls)
    if cached is None:
        cached = type(f"{adapter_cls.__name__}FileDelivery",
                       (FileDeliveryAdapter, adapter_cls), {})
        _file_delivery_variants[adapter_cls] = cached
    return cached


def make_adapter(judge_id: str, judges_cfg_entry: dict) -> BaseAdapter:
    """Build a concrete adapter from one config/judges.yaml entry, e.g.
    `{model: claude-fable-5, cli: claude, cli_version: ..., delivery: stdin,
    invoke: 'claude -p --model {model} --output-format json'}`.

    The `invoke` string is split on whitespace into an argv template
    (`{model}` and `{cli}` are substituted now; a trailing '-'/'-p' stdin
    sentinel token has no placeholder so it just passes through unchanged;
    `{instruction}`, if present, is left for per-call substitution) and
    handed to the judge_id-matched adapter class. The `delivery` key routes
    between stdin (default) and file delivery -- `delivery: file` wraps the
    judge_id's class in `FileDeliveryAdapter` via `_file_delivery_variant`.
    """
    try:
        adapter_cls = _ADAPTER_CLASSES[judge_id]
    except KeyError:
        raise ValueError(f"unknown judge_id: {judge_id!r}") from None

    delivery = judges_cfg_entry.get("delivery", "stdin")
    if delivery == "file":
        adapter_cls = _file_delivery_variant(adapter_cls)
    elif delivery != "stdin":
        raise ValueError(f"unknown delivery mode for judge_id {judge_id!r}: {delivery!r}")

    invoke_template = judges_cfg_entry["invoke"]
    argv_template = invoke_template.split()
    model_pin = judges_cfg_entry["model"]
    cli_version = judges_cfg_entry.get("cli_version")
    cli_path = judges_cfg_entry.get("cli")

    return adapter_cls(judge_id, model_pin, cli_version, argv_template=argv_template,
                        cli=cli_path)


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
               timeout: int = 300, packet_path: Path | str | None = None) -> JudgeReply:
        # packet_path accepted-but-ignored for call-signature parity with
        # BaseAdapter (the runner invokes every adapter, real or fake, the
        # same way).
        scores = self.scores_fn(expected_letters)
        reasons = {letter: f"fake reason for {letter}" for letter in expected_letters}
        ranking = sorted(expected_letters,
                          key=lambda l: (-scores.get(l, 0), l))
        raw = json.dumps({"scores": scores, "reasons": reasons, "ranking": ranking})
        parsed, error = parse_reply(raw, expected_letters)
        return JudgeReply(raw=raw, parsed=parsed, error=error)
