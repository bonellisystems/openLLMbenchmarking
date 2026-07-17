"""Tests for judge CLI adapters (TESTPLAN 6.1) -- reply parser + adapter plumbing.

No real judge CLIs are invoked here: BaseAdapter.invoke() plumbing is proven
via monkeypatched subprocess.run, and FakeJudgeAdapter never shells out at
all. Live enumeration / pin freeze (Steps 3-6 of the task-6 brief) are a
separate, human-gated controller step, not code under test.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from llmtest.judging.adapters import (
    BaseAdapter,
    ClaudeAdapter,
    CodexAdapter,
    FakeJudgeAdapter,
    FileDeliveryAdapter,
    GeminiAdapter,
    JudgeReply,
    _substitute_argv,
    make_adapter,
    parse_reply,
)

LETTERS = ["A", "B"]


def _valid_json(scores=None, reasons=None, ranking=None) -> str:
    scores = scores if scores is not None else {"A": 7, "B": 3}
    reasons = reasons if reasons is not None else {"A": "solid", "B": "weak"}
    ranking = ranking if ranking is not None else ["A", "B"]
    return json.dumps({"scores": scores, "reasons": reasons, "ranking": ranking})


# --- parse_reply ---


def test_parse_reply_happy_path():
    parsed, error = parse_reply(_valid_json(), LETTERS)
    assert error is None
    assert parsed["scores"] == {"A": 7, "B": 3}
    assert parsed["reasons"] == {"A": "solid", "B": "weak"}
    assert parsed["ranking"] == ["A", "B"]


def test_parse_reply_json_buried_in_prose():
    stdout = "Sure, here is my scoring:\n\n" + _valid_json() + "\n\nHope that helps!"
    parsed, error = parse_reply(stdout, LETTERS)
    assert error is None
    assert parsed["scores"]["A"] == 7


def test_parse_reply_json_in_fenced_code_block():
    stdout = "```json\n" + _valid_json() + "\n```"
    parsed, error = parse_reply(stdout, LETTERS)
    assert error is None
    assert parsed["ranking"] == ["A", "B"]


def test_parse_reply_extracts_first_balanced_object_ignoring_braces_in_strings():
    # A reason containing literal braces must not desync the brace counter.
    stdout = _valid_json(reasons={"A": "uses {curly} braces", "B": "fine"})
    parsed, error = parse_reply(stdout, LETTERS)
    assert error is None
    assert parsed["reasons"]["A"] == "uses {curly} braces"


def test_parse_reply_missing_letter_in_scores():
    bad = json.dumps({"scores": {"A": 7}, "reasons": {"A": "x", "B": "y"},
                       "ranking": ["A", "B"]})
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert error is not None and "B" in error


def test_parse_reply_float_score_rejected():
    bad = _valid_json(scores={"A": 7.5, "B": 3})
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert error is not None


def test_parse_reply_bool_score_explicitly_rejected():
    # bool is a subclass of int in Python -- must be explicitly excluded.
    bad = _valid_json(scores={"A": True, "B": 3})
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert error is not None


def test_parse_reply_score_out_of_range():
    bad = _valid_json(scores={"A": 11, "B": 3})
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert error is not None


def test_parse_reply_ranking_not_a_permutation_duplicate():
    bad = _valid_json(ranking=["A", "A"])
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert error is not None


def test_parse_reply_ranking_missing_letter():
    bad = _valid_json(ranking=["A"])
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert error is not None


def test_parse_reply_ranking_extra_unknown_letter():
    bad = _valid_json(ranking=["A", "B", "C"])
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert error is not None


def test_parse_reply_no_json_object_in_stdout():
    parsed, error = parse_reply("I refuse to answer in JSON.", LETTERS)
    assert parsed is None
    assert error is not None


def test_parse_reply_missing_reason_for_letter():
    bad = json.dumps({"scores": {"A": 7, "B": 3}, "reasons": {"A": "x"},
                       "ranking": ["A", "B"]})
    parsed, error = parse_reply(bad, LETTERS)
    assert parsed is None
    assert "B" in error


# --- BaseAdapter.invoke plumbing (subprocess mocked, never a real CLI) ---


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_base_adapter_invoke_happy_path(monkeypatch):
    captured = {}

    def fake_run(argv, input, capture_output, text, encoding, timeout):
        captured["argv"] = argv
        captured["input"] = input
        captured["timeout"] = timeout
        return _FakeCompletedProcess(returncode=0, stdout=_valid_json())

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = BaseAdapter("t", "pin-x", "v1", argv=["some-cli", "--flag"])
    reply = adapter.invoke("packet body", LETTERS, timeout=99)

    assert isinstance(reply, JudgeReply)
    assert reply.error is None
    assert reply.parsed["scores"]["A"] == 7
    assert captured["argv"] == ["some-cli", "--flag"]
    assert captured["input"] == "packet body"
    assert captured["timeout"] == 99


def test_base_adapter_invoke_nonzero_exit_is_error(monkeypatch):
    def fake_run(argv, input, capture_output, text, encoding, timeout):
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = BaseAdapter("t", "pin-x", "v1", argv=["some-cli"])
    reply = adapter.invoke("packet body", LETTERS)

    assert reply.parsed is None
    assert reply.error is not None and "boom" in reply.error


def test_base_adapter_invoke_timeout(monkeypatch):
    def fake_run(argv, input, capture_output, text, encoding, timeout):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = BaseAdapter("t", "pin-x", "v1", argv=["some-cli"])
    reply = adapter.invoke("packet body", LETTERS)

    assert reply == JudgeReply(raw="", parsed=None, error="timeout")


# --- ClaudeAdapter: JSON envelope unwrap ---


def test_claude_adapter_unwraps_envelope_result_field():
    envelope = json.dumps({"result": _valid_json()})
    adapter = ClaudeAdapter("claude", "claude-fable-5", "v1")
    parsed, error = adapter._parse_stdout(envelope, LETTERS)
    assert error is None
    assert parsed["scores"]["A"] == 7


def test_claude_adapter_falls_back_to_raw_stdout_on_bad_envelope():
    # Envelope load fails entirely -- defensively fall back to parsing stdout
    # directly, as if it were the raw reply.
    adapter = ClaudeAdapter("claude", "claude-fable-5", "v1")
    parsed, error = adapter._parse_stdout(_valid_json(), LETTERS)
    assert error is None
    assert parsed["scores"]["A"] == 7


def test_claude_adapter_default_argv():
    adapter = ClaudeAdapter("claude", "claude-fable-5", "v1")
    assert adapter.argv == [
        "claude", "-p", "--model", "claude-fable-5", "--output-format", "json",
    ]


def test_claude_adapter_invoke_end_to_end_through_envelope(monkeypatch):
    def fake_run(argv, input, capture_output, text, encoding, timeout):
        return _FakeCompletedProcess(returncode=0,
                                      stdout=json.dumps({"result": _valid_json()}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = ClaudeAdapter("claude", "claude-fable-5", "v1")
    reply = adapter.invoke("packet body", LETTERS)
    assert reply.error is None
    assert reply.parsed["scores"]["A"] == 7


# --- CodexAdapter / GeminiAdapter: templated argv ---


def test_codex_adapter_default_argv_substitutes_model():
    adapter = CodexAdapter("codex", "o-pin", "v2")
    assert adapter.argv == ["codex", "exec", "--model", "o-pin", "-"]


def test_codex_adapter_custom_argv_template():
    adapter = CodexAdapter("codex", "o-pin", "v2",
                            argv_template=["codex", "exec", "{model}"])
    assert adapter.argv == ["codex", "exec", "o-pin"]


def test_gemini_adapter_default_argv_substitutes_model():
    adapter = GeminiAdapter("gemini", "gemini-3-pro", "v3")
    assert adapter.argv == ["gemini", "-m", "gemini-3-pro", "-p"]


# --- FakeJudgeAdapter ---


def test_fake_judge_round_trip():
    def scores_fn(letters):
        return {l: (8 if l == "A" else 4) for l in letters}

    adapter = FakeJudgeAdapter(scores_fn)
    reply = adapter.invoke("ignored packet text", LETTERS)

    assert reply.error is None
    assert reply.parsed["scores"] == {"A": 8, "B": 4}
    assert set(reply.parsed["reasons"]) == {"A", "B"}
    assert sorted(reply.parsed["ranking"]) == sorted(LETTERS)
    assert reply.parsed["ranking"][0] == "A"  # higher score ranks first


def test_fake_judge_invalid_scores_surface_as_error():
    # A misbehaving scores_fn (e.g. float scores) must surface the SAME
    # validation error a real adapter would produce -- FakeJudge is not a
    # blind rubber stamp.
    adapter = FakeJudgeAdapter(lambda letters: {l: 7.5 for l in letters})
    reply = adapter.invoke("ignored", LETTERS)
    assert reply.parsed is None
    assert reply.error is not None


# --- make_adapter: judges.yaml-shaped config -> adapter with substituted argv ---


def test_make_adapter_claude():
    cfg = {"model": "claude-fable-5", "cli": "claude", "cli_version": "1.2.3",
           "invoke": "claude -p --model {model} --output-format json"}
    adapter = make_adapter("claude", cfg)
    assert isinstance(adapter, ClaudeAdapter)
    assert adapter.argv == [
        "claude", "-p", "--model", "claude-fable-5", "--output-format", "json",
    ]
    assert adapter.model_pin == "claude-fable-5"
    assert adapter.cli_version == "1.2.3"


def test_make_adapter_codex_trailing_stdin_sentinel_kept_as_is():
    cfg = {"model": "codex-flagship", "cli": "codex", "cli_version": "9.9.9",
           "invoke": "codex exec --model {model} -"}
    adapter = make_adapter("codex", cfg)
    assert isinstance(adapter, CodexAdapter)
    assert adapter.argv == ["codex", "exec", "--model", "codex-flagship", "-"]


def test_make_adapter_gemini():
    cfg = {"model": "gemini-3-pro", "cli": "gemini", "cli_version": "0.1.0",
           "invoke": "gemini -m {model} -p"}
    adapter = make_adapter("gemini", cfg)
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.argv == ["gemini", "-m", "gemini-3-pro", "-p"]


def test_make_adapter_unknown_judge_id_raises():
    with pytest.raises(ValueError):
        make_adapter("nope", {"model": "x", "invoke": "nope {model}"})


# --- {cli}/{instruction} substitution + delivery routing (Task 6 gemini/agy handoff) ---


def test_substitute_argv_replaces_cli_and_model_leaves_instruction():
    argv = _substitute_argv(["{cli}", "--model", "{model}", "{instruction}", "--flag"],
                             "pin-x", cli="C:\\bin\\agy.exe")
    assert argv == ["C:\\bin\\agy.exe", "--model", "pin-x", "{instruction}", "--flag"]


def test_substitute_argv_without_cli_leaves_cli_token_untouched():
    argv = _substitute_argv(["{cli}", "--model", "{model}"], "pin-x")
    assert argv == ["{cli}", "--model", "pin-x"]


def test_make_adapter_stdin_delivery_default_is_plain_adapter():
    cfg = {"model": "claude-fable-5", "cli": "claude", "cli_version": "1.2.3",
           "delivery": "stdin", "invoke": "claude -p --model {model} --output-format json"}
    adapter = make_adapter("claude", cfg)
    assert not isinstance(adapter, FileDeliveryAdapter)


def test_make_adapter_gemini_file_delivery_routes_to_file_adapter():
    cfg = {"model": "gemini-3-pro", "cli": "C:\\bin\\agy.exe", "cli_version": "1.1.3",
           "delivery": "file",
           "invoke": "{cli} --print {instruction} --model {model} --add-dir C:\\packets"}
    adapter = make_adapter("gemini", cfg)
    assert isinstance(adapter, GeminiAdapter)
    assert isinstance(adapter, FileDeliveryAdapter)
    assert adapter.argv == ["C:\\bin\\agy.exe", "--print", "{instruction}",
                             "--model", "gemini-3-pro", "--add-dir", "C:\\packets"]


def test_make_adapter_unknown_delivery_mode_raises():
    cfg = {"model": "x", "cli": "x", "delivery": "carrier-pigeon", "invoke": "x {model}"}
    with pytest.raises(ValueError):
        make_adapter("gemini", cfg)


def test_file_delivery_adapter_embeds_packet_path_in_instruction_no_stdin(monkeypatch, tmp_path):
    captured = {}

    def fake_run(argv, input, capture_output, text, encoding, timeout):
        captured["argv"] = argv
        captured["input"] = input
        return _FakeCompletedProcess(returncode=0, stdout=_valid_json())

    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = {"model": "gemini-3-pro", "cli": "agy", "cli_version": "1.1.3",
           "delivery": "file", "invoke": "{cli} --print {instruction} --model {model}"}
    adapter = make_adapter("gemini", cfg)

    packet_path = tmp_path / "pkt.gemini.txt"
    packet_path.write_text("packet body on disk", encoding="utf-8")

    reply = adapter.invoke("ignored packet text", LETTERS, packet_path=packet_path)

    assert reply.error is None
    assert reply.parsed["scores"]["A"] == 7
    assert captured["input"] is None                # nothing on stdin
    argv = captured["argv"]
    assert argv[0] == "agy"
    instruction = argv[argv.index("--print") + 1]
    assert str(packet_path) in instruction           # packet path embedded in the instruction
    assert "Read the file at" in instruction
    assert argv[argv.index("--model") + 1] == "gemini-3-pro"


def test_file_delivery_adapter_requires_packet_path():
    adapter = FileDeliveryAdapter("gemini", "pin", "v1",
                                   argv=["agy", "--print", "{instruction}"])
    with pytest.raises(ValueError):
        adapter.invoke("text", LETTERS)               # no packet_path -- file delivery needs one


def test_base_adapter_default_delivery_ignores_packet_path(monkeypatch):
    # Stdin adapters accept the packet_path kwarg (uniform call signature
    # for the runner) but ignore it -- argv/stdin unaffected.
    captured = {}

    def fake_run(argv, input, capture_output, text, encoding, timeout):
        captured["argv"] = argv
        captured["input"] = input
        return _FakeCompletedProcess(returncode=0, stdout=_valid_json())

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = BaseAdapter("t", "pin-x", "v1", argv=["some-cli"])
    adapter.invoke("packet body", LETTERS, packet_path="/some/path.txt")

    assert captured["argv"] == ["some-cli"]
    assert captured["input"] == "packet body"
