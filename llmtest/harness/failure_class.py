"""B8 first-failure classification pipeline (Task 8) -- deterministic
detectors + a blinded classifier panel, for CATEGORIZING why a FAILED B8
run failed. This is a SEPARATE, standalone categorical pipeline: it does
NOT import `llmtest.judging.runner` or `llmtest.judging.aggregate`, and its
output never enters the numeric median-of-3 judge pipeline those modules
implement. It only reuses the *adapter* plumbing from
`llmtest.judging.adapters` (subprocess invocation shape, JSON-object
extraction) -- see `_invoke_classifier`/`parse_categorical_reply`.

LABEL SCHEMA
------------
- "a" -- schema-never-parsed: the model's tool call was never parsed as a
  valid call at all (a format/schema failure at the harness boundary).
  Deterministic -- visible directly in the `Trace`.
- "b" -- parsed-but-misused: tool calls parsed fine but were used wrongly
  (wrong tool, wrong args, wrong sequence). Requires judgment -> PANEL.
- "c" -- task-logic: tools were used correctly, but the solution LOGIC is
  wrong (the "completed-but-wrong-logic" case -- the harness thinks the run
  finished cleanly, but the oracle says the task wasn't actually done).
  Requires judgment -> PANEL.
- "d" -- harness-bug: the harness/infra failed, not the model
  (`terminal_status == "infra-error"`, or another harness-error marker in
  the trace). Deterministic wherever it's log-inferable.
- "unknown" -- the panel tied, or produced no majority (or every
  classifier abstained). Also the panel's own "we couldn't tell" verdict.
- "not_applicable" -- the run did not FAIL, so there is nothing to
  classify.

DETERMINISTIC PRECEDENCE -- (d) BEFORE (a), documented rationale
------------------------------------------------------------------
`_deterministic_failure` checks harness-bug (d) first, schema-never-parsed
(a) second, and only falls through to the panel if neither fires. This
order is deliberate, not incidental: if the HARNESS itself broke (d), the
model never got a fair shot at producing a parseable tool call in the first
place -- attributing that to the model as a schema failure (a) would be
mislabeling an infra outage as a model defect. (d) is therefore checked
first and, if it fires, wins outright regardless of what the rest of the
trace looks like. Only once the run is confirmed to be an infra-clean
model failure does (a) get to look for the model's own first unparsed tool
call. Both are scanned directly off the `Trace` (see `_is_harness_bug` /
`_first_tool_call_never_parsed`) -- no subprocess, no panel, no model
identity ever enters this path, which is also why it can run before
anything blinding-related is even built.

CONTROL FLOW (`classify_first_failure`)
----------------------------------------
1. Not-a-failure short-circuit: `completed is True` (the oracle said the
   task passed) always returns `not_applicable` outright, regardless of
   trace content -- an oracle PASS means there is nothing to classify, full
   stop. When `completed is None` (caller has no oracle verdict handy) AND
   `trace.terminal_status == "completed"` AND no deterministic failure
   marker is present either, the trace is ALSO treated as a non-failure
   (best-effort inference from the trace alone). A `completed`-terminal
   trace can still be a task FAILURE when the caller passes
   `completed=False` explicitly -- that's precisely the (c) "harness
   thinks it's done, oracle disagrees" case, and it proceeds past this
   check to step 2.
2. Deterministic detectors, in the precedence above. Either produces a
   verdict immediately (source="deterministic"), or nothing fires and
   control falls through.
3. Unresolved failure -> blinded classifier PANEL (`panel_classify`):
   render the trace+task into a neutral, model-blind presentation (the
   `Trace` itself carries no model identity -- see `render_blinded_trace`
   -- so blinding here is "don't add any", not "strip something out"), ask
   every classifier in `classifiers` to label it in {a, b, c, d}, and
   majority-vote the result (source="panel").

PANEL MECHANICS -- majority / tie / abstention
------------------------------------------------
Each classifier returns a raw label; anything not in {"a","b","c","d"}
(wrong type, blank, unrecognized word) is an ABSTENTION, not a vote for
"unknown" -- `PanelResult.abstentions` reports how many. Majority is
computed over the VALID votes only: the top label must be the UNIQUE
highest count AND strictly exceed half of the valid-vote count (a bare
plurality that doesn't clear 50% is treated the same as a tie -- both are
"no majority"). Anything short of that (an outright tie for the top spot,
or a unique top that doesn't clear the 50% bar, or zero valid votes at
all) yields "unknown". `panel_classify` is public (not `classify_first_
failure`-internal-only) because Task 9's own aggregation needs the
per-label vote breakdown and abstention count, not just the final label.

CLASSIFIER INTERFACE
---------------------
`classifiers` is a list of adapter-like objects, each satisfying ONE of:
  - `.classify(blinded_text: str) -> str` -- the simple shape (what the
    tests inject).
  - `.invoke(packet_text, expected_letters, timeout, packet_path) ->
    JudgeReply` -- the SAME shape `llmtest.judging.adapters.BaseAdapter`
    (and hence every real judge CLI adapter) already implements. This is
    the "reuse the adapter/blinding infra" path: the subprocess-invocation
    plumbing (stdin/file delivery, argv resolution, timeout handling) is
    reused as-is; only the REPLY PARSING differs -- `reply.raw` is run
    through `parse_categorical_reply` (a categorical `{"label": "a".."d"}`
    schema) instead of `adapters.parse_reply`'s numeric
    `{scores,reasons,ranking}` schema, because a first-failure label is a
    single category, not a comparative multi-model score.
No live classifier CLI is invoked anywhere in this module's own tests --
`.classify()` fakes cover the required scenarios.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from llmtest.judging.adapters import _extract_first_json_object

VALID_LABELS = {"a", "b", "c", "d"}


# -- public result type ----------------------------------------------------


@dataclass
class PanelResult:
    """Full detail of one panel vote -- `classify_first_failure` only ever
    surfaces `.label` (as the panel-sourced half of its `(label, source)`
    contract), but `.votes`/`.abstentions`/`.raw` are kept around for Task
    9's aggregation and for tests that need to assert abstentions were
    actually counted, not silently dropped."""

    label: str
    votes: dict[str, int]
    abstentions: int
    raw: list[str | None]


# -- deterministic detectors -------------------------------------------------


def _is_harness_bug(trace) -> bool:
    """(d) harness-bug: `terminal_status == "infra-error"`, or any event
    in the trace carrying a truthy `harness_error` marker in its payload
    (a hook for a harness-level failure that doesn't happen to also be
    reflected in `terminal_status`, e.g. a future adapter that wants to
    flag a harness malfunction it recovered from well enough to still
    reach a different terminal status)."""
    if trace.terminal_status == "infra-error":
        return True
    return any(bool(ev.payload.get("harness_error")) for ev in trace.events)


def _first_tool_call_never_parsed(trace) -> bool:
    """(a) schema-never-parsed: True if any `tool_call` event in the trace
    is marked `payload["parsed"] is False` -- i.e. the harness could never
    even construct a valid tool call out of the model's output (a
    format/schema failure at the harness boundary). This is distinct from
    a `tool_result` with `status == "error"`: that's a tool call that DID
    parse but then failed or was misused (the (b) "parsed-but-misused"
    case), which requires judgment and is therefore never decided here --
    only routed to the panel. Absent `"parsed"` defaults to parsed-OK
    (True), matching every existing trace producer (e.g.
    `llmtest.harness.opencode`) that doesn't yet emit this key at all."""
    return any(ev.kind == "tool_call" and ev.payload.get("parsed") is False
               for ev in trace.events)


def _deterministic_failure(trace) -> tuple[str, str] | None:
    """(d) then (a), in that precedence -- see module docstring. Returns
    `None` (not a deterministic verdict) if neither fires, so the caller
    knows to fall through to the panel."""
    if _is_harness_bug(trace):
        return "d", "deterministic"
    if _first_tool_call_never_parsed(trace):
        return "a", "deterministic"
    return None


# -- blinding ----------------------------------------------------------------


def render_blinded_trace(trace, task, *, completed: bool | None = None) -> str:
    """Render `trace` + `task.prompt` into a neutral, model-blind text
    presentation for a classifier to read. The `Trace` schema itself
    carries no model identity (see `llmtest.harness.trace`'s module
    docstring), so "blinding" here means simply never ADDING any --
    nothing about which model/harness/run produced this trace is surfaced,
    only the task prompt, the ordered interaction, and the terminal
    outcome."""
    lines: list[str] = ["## Task prompt", "", task.prompt, "",
                         "## Interaction trace", ""]
    turn_no = 0
    for ev in trace.events:
        if ev.kind == "turn":
            turn_no += 1
            lines.append(f"[turn {turn_no}]")
        elif ev.kind == "tool_call":
            lines.append(
                f"  tool_call: tool={ev.payload.get('tool')!r} "
                f"input={ev.payload.get('input')!r} "
                f"parsed={ev.payload.get('parsed', True)!r}")
        elif ev.kind == "tool_result":
            lines.append(
                f"  tool_result: status={ev.payload.get('status')!r} "
                f"output={ev.payload.get('output')!r}")
        elif ev.kind == "subagent_spawn":
            lines.append(f"  subagent_spawn: {ev.payload!r}")
        elif ev.kind == "terminal":
            lines.append(f"  terminal: {ev.payload!r}")

    lines.append("")
    lines.append(f"## Terminal status: {trace.terminal_status}")
    lines.append(f"## Oracle completed: {completed!r}")
    return "\n".join(lines)


# -- categorical reply parsing (adapter-plumbing reuse) ----------------------


def parse_categorical_reply(stdout: str) -> tuple[str | None, str | None]:
    """Extract + validate a classifier's CATEGORICAL reply out of raw CLI
    stdout: a JSON object `{"label": "a".."d", ...}` anywhere in the text
    (arbitrary surrounding prose/markdown fences tolerated). Reuses
    `llmtest.judging.adapters._extract_first_json_object` -- the same
    balanced-brace scanner the numeric judge-reply parser
    (`adapters.parse_reply`) uses -- rather than re-implementing brace
    scanning; only the SCHEMA validated afterward differs (one label, not
    {scores,reasons,ranking}).

    Returns `(label, None)` on success, else `(None, "reason")`.
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
    label = obj.get("label")
    if not isinstance(label, str) or label.strip().lower() not in VALID_LABELS:
        return None, f"missing/invalid 'label' field: {label!r}"
    return label.strip().lower(), None


def _normalize_label(raw: Any) -> str | None:
    """A raw classifier reply counts as a valid vote only if it's a string
    that, trimmed and lower-cased, is exactly one of {a,b,c,d}. Anything
    else (wrong type, blank, an unrecognized word) is not a vote -- it's an
    abstention, handled by the caller."""
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    return candidate if candidate in VALID_LABELS else None


def _invoke_classifier(classifier: Any, blinded_text: str) -> str | None:
    """Drive one classifier, either shape (see module docstring's
    "CLASSIFIER INTERFACE"), and normalize its reply to a valid label or
    `None` (abstention)."""
    if hasattr(classifier, "classify"):
        raw = classifier.classify(blinded_text)
    elif hasattr(classifier, "invoke"):
        reply = classifier.invoke(blinded_text, expected_letters=sorted(VALID_LABELS),
                                   timeout=300, packet_path=None)
        raw, _err = parse_categorical_reply(reply.raw)
    else:
        raise TypeError(
            f"classifier {classifier!r} exposes neither .classify() nor .invoke()")
    return _normalize_label(raw)


# -- panel ---------------------------------------------------------------


def panel_classify(trace, task, classifiers: list, *,
                    completed: bool | None = None) -> PanelResult:
    """Render the blinded presentation once, ask every classifier in
    `classifiers` to vote, and majority-vote the result. Majority means the
    top label is the UNIQUE highest-count label AND its count strictly
    exceeds half of the VALID (non-abstaining) votes -- a tie for the top
    spot, or a unique top that doesn't clear 50%, or zero valid votes, all
    yield "unknown" (see module docstring's "PANEL MECHANICS")."""
    blinded_text = render_blinded_trace(trace, task, completed=completed)
    raw_labels = [_invoke_classifier(c, blinded_text) for c in classifiers]

    valid = [label for label in raw_labels if label in VALID_LABELS]
    abstentions = len(raw_labels) - len(valid)
    counts = Counter(valid)

    if not counts:
        return PanelResult(label="unknown", votes={}, abstentions=abstentions, raw=raw_labels)

    top_label, top_count = counts.most_common(1)[0]
    tied_for_top = [label for label, count in counts.items() if count == top_count]
    if len(tied_for_top) == 1 and top_count * 2 > len(valid):
        winner = top_label
    else:
        winner = "unknown"

    return PanelResult(label=winner, votes=dict(counts), abstentions=abstentions, raw=raw_labels)


# -- top-level entry point ----------------------------------------------------


def classify_first_failure(trace, task, *, completed: bool | None = None,
                            classifiers: list | None = None) -> tuple[str, str]:
    """Classify the FIRST point of failure of a (possibly) failed B8 run.

    See the module docstring for the full control flow, deterministic
    precedence rationale, and panel mechanics. `classifiers` is only ever
    consulted when the run is a confirmed failure AND neither deterministic
    detector fires -- callers that pass a classifier which raises if
    invoked can use that to assert the panel was never reached.
    """
    det = _deterministic_failure(trace)

    if completed is True:
        return "not_applicable", "deterministic"
    if completed is None and trace.terminal_status == "completed" and det is None:
        return "not_applicable", "deterministic"

    if det is not None:
        return det

    result = panel_classify(trace, task, classifiers or [], completed=completed)
    return result.label, "panel"
