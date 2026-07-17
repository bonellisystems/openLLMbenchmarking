# B2 Fixture Schema

## Overview

B2 fixtures are YAML task definitions for the Tool-Calling battery (TESTPLAN
5.2) -- realistic tool-use scenarios graded primarily by DETERMINISTIC
checkers against the model's OpenAI-style `tool_calls` response, plus two
axes (5, 8) that are flagged for judging. Each fixture resides flat in
`suite/b2_toolcalling/task-<NN>.yaml` (no per-unit subdirectories -- B2 has
no business-unit axis, only scenario/axis coverage).

## The 8 axes (TESTPLAN 5.2, canonical numbering)

1. Schema adherence -- strict JSON validity rate (deterministic).
2. Correct tool selection among distractor tools (deterministic).
3. Parallel calls in one turn (deterministic).
4. Chained/dependent calls -- A's output feeds B (deterministic).
5. Error recovery -- tool returns error/garbage; retry/adapt vs fabricate (JUDGED).
6. Abstention -- no suitable tool exists; does it invent one (deterministic).
7. Tool calls at long context (32k+ of history, then a call) (deterministic).
8. Faithfulness to tool results -- final answer vs what the tool returned;
   shared with B3 (JUDGED).

Axis 1 is evaluated on every task (any tool_calls emitted must be valid
JSON / schema-conformant, regardless of which other axes the task targets).
A task's `axes:` list declares which OTHER axes it exercises; the battery
skips checkers for axes not listed. `needs_judging` is set True on a row iff
the task's axes intersect `{5, 8}` -- those two axes get a best-effort
deterministic "fabrication trap" guard (see below) but their real score
comes from the judge panel (not yet wired for B2 -- see `b2_toolcalling.py`
module docstring).

## Required Keys

- **id** (string): Unique task identifier, `<scenario-slug>-<NN>` (e.g.
  `single-tool-basic-01`). No unit prefix requirement (unlike B1).
- **scenario** (string): Category tag, snake_case (e.g. `single_tool_basic`,
  `chained_calls`, `abstention_no_tool`). Documents *why* the task exists;
  not machine-enforced against a fixed vocabulary.
- **axes** (list[int]): Subset of `[1..8]`, non-empty. Axis 1 does not need
  to be listed explicitly (it always runs) but including it is harmless.
- **industry** (string): One of `config/suite.yaml`'s `b1.industries`
  vocabulary (reused from B1 for consistency; not lint-enforced for B2 yet).
- **difficulty** (string, optional): `easy` | `medium` | `hard`. Defaults to
  `medium` if omitted.
- **tools** (list): OpenAI `/v1/chat/completions` `tools` array, verbatim --
  `[{type: function, function: {name, description, parameters: <JSON
  Schema object>}}, ...]`. Include at least one distractor tool for any task
  exercising axis 2 or 6.
- **messages** (list): Conversation turns sent verbatim to `endpoint.chat()`.
  Supports scripted multi-turn prefixes for axes 4/5/8: a `role: assistant`
  turn may carry `tool_calls` (list, `function.arguments` as a JSON
  **string**, matching the real API wire shape) instead of `content`; a
  `role: tool` turn carries `tool_call_id` + `content` (the simulated tool
  result). The FINAL message is always the live turn the model under test
  answers; everything before it is fixed/scripted.
- **expect** (dict): Deterministic scoring spec (see below).
- **rubric** (dict, optional): Judge guidance text for axes 5/8, keyed
  `axis_5` / `axis_8`. Not consumed by any code yet -- documentation for the
  eventual judge-packet wiring.
- **filler** (dict, optional): `{unit_paragraph: str, target_tokens: int}`.
  When present, any message `content` containing the literal placeholder
  `{{FILLER}}` is expanded at load time into a repeated-paragraph filler
  block sized to approximately `target_tokens` (using the same ~4
  chars/token heuristic as `b5_serving.build_sustained_prompt`). Used by the
  axis-7 long-context task so the fixture file itself stays small and
  readable instead of committing 100+ KB of literal filler text.
- **notes** (string, optional): Authoring rationale.

## `expect` block

- **tool_calls** (list, optional): `[{name, args: {...}, args_match: exact|subset}]`.
  `exact` requires the parsed argument dict's key set to equal `args`'
  key set (in addition to matching values); `subset` (default) only checks
  the listed keys/values, ignoring any extra keys the model added. Value
  comparison is `==` first, falling back to stripped-string equality (so
  `11.2` and `"11.2"` compare equal but nested dicts still compare
  structurally).
- **forbidden_tools** (list[str], optional): distractor tool names that must
  NOT appear among the emitted calls.
- **parallel_ok** (bool, optional): when true (and `tool_calls` has >=2
  entries), scores axis 3 -- all expected calls must appear as a
  multiset-match among the emitted calls (order-independent).
- **chain_check** (dict, optional): `{arg_path: str, expected_value: str}`.
  Scores axis 4 -- at least one emitted call's parsed arguments must contain
  `arg_path` (dot-path into nested objects) equal to `expected_value`. Used
  to prove the model actually read a prior scripted tool result rather than
  echoing something from the user's own prompt.
- **expect_no_call** (bool, optional): when true, scores axis 6 -- ANY tool
  call at all is a fail (invented/hallucinated a tool for a request no
  available tool actually covers, or guessed at missing required
  information instead of asking).
- **fabrication_traps** (list[str], optional): plausible-looking values that
  were NEVER returned by any (real or simulated) tool result and must not
  appear verbatim in the final answer text or in any emitted call's raw
  arguments string. Deterministic best-effort guard for axes 5/8 -- passing
  this check is necessary but not sufficient for a good axis-5/8 score; the
  real score is judged.

## Example Fixture (single-tool, axes 1+2)

```yaml
id: single-tool-basic-01
scenario: single_tool_basic
axes: [1, 2]
industry: financial_services
difficulty: easy
tools:
  - type: function
    function:
      name: get_account_balance
      description: Look up the current balance for a client account by account ID.
      parameters:
        type: object
        properties:
          account_id: {type: string, description: "Client account identifier, e.g. ACC-10293"}
        required: [account_id]
  - type: function
    function:
      name: schedule_meeting
      description: Schedule a meeting on the advisor's calendar.
      parameters:
        type: object
        properties:
          attendee: {type: string}
          datetime: {type: string}
        required: [attendee, datetime]
messages:
  - role: user
    content: "What is the current balance on account ACC-58213?"
expect:
  tool_calls:
    - {name: get_account_balance, args: {account_id: "ACC-58213"}, args_match: exact}
  forbidden_tools: [schedule_meeting]
notes: |
  Single tool + one plausible distractor. Axis 1 (schema adherence) always
  runs; axis 2 (correct selection) checks the model didn't reach for
  schedule_meeting and that account_id matches exactly.
```

## Loading Fixtures

```python
from pathlib import Path
from llmtest.batteries import b2_fixtures

root = Path(".")
tasks = b2_fixtures.load_tasks(root)
for task in tasks:
    print(f"{task.id}: axes={task.axes} scenario={task.scenario}")
```

`load_tasks` raises `ValueError` loudly on any malformed fixture file
(missing required key, `axes` outside `[1..8]`, etc.) -- mirrors
`b1_fixtures.load_unit_tasks`'s fail-loud contract exactly.

## Scoring

```python
det_checks, needs_judging, metrics = b2_fixtures.score_axes(response, task)
```

`response` is the raw dict returned by `EndpointHandle.chat(..., tools=task.tools)`
(the `/v1/chat/completions` JSON body). Returns:

- `det_checks`: `{"axis1_schema_adherence": {...}, "axis2_tool_selection": {...}, ...}`
  -- only keys for axes the task actually exercises (axis 1 always present).
- `needs_judging`: `True` iff `set(task.axes) & {5, 8}`.
- `metrics`: `{"n_tool_calls": int, "axes_applicable": [int, ...], "det_pass": bool | None}`.

## Validation

`b2_fixtures.validate_tool_schemas(tools)` and
`b2_fixtures.validate_expect_block(task)` return lists of human-readable
error strings (empty = clean). `B2ToolCalling.preflight()` runs both over
every loaded task and refuses to execute (per the Battery ABC contract) if
any task's tool schemas don't parse or its `expect` block references an
undeclared tool.
