# CAL pair authoring — `grading/calibration/b2/`

This directory holds frozen, author-pinned CAL-strong / CAL-weak reference
pairs used by the B2 non-circular calibration gate (spec §1.5:
`docs/superpowers/specs/2026-07-19-agentic-quality-v2.1-design.md` Part 1.5).
Each `axisN.yaml` is authored **once** from a fixed meta-prompt, frozen, and
content-hashed (`author.prompt_sha`) — never regenerated at runtime to fit
judge scores. This file is the audit trail: the prompt that produced each
pin must live here, not only in scratch.

## Two authoring modes

Which template to use depends on whether the axis's rubric is **behavioral**
or **fact-matching**:

- **Behavioral axes** (e.g. axis 5, error-recovery) — the rubric scores a
  *pattern of behavior* (disclose a tool failure vs. fabricate a value) that
  holds regardless of which fixture's numbers are involved. These pairs stay
  **scenario-agnostic**: generic prose that doesn't reference any one
  fixture's specific entities, because the pair is valid no matter which B2
  fixture happens to carry the axis.
- **Fact-matching axes** (e.g. axis 8, faithfulness-to-tool-results) — the
  rubric's 10/0 anchors are graded against one fixture's *literal facts*
  (`rubric.axis_N` in that fixture's own YAML quotes the exact values the
  anchors expect). A scenario-agnostic pair cannot hit those anchors: a
  generic "I restated only what the tool said" answer contains none of the
  facts the 10-anchor checks for, and a generic "I found something extra"
  weak answer doesn't contradict anything the 0-anchor checks for. **These
  pairs must be authored against the one fixture that carries the axis**,
  quoting its tool-result facts and one of its `expect.fabrication_traps`
  verbatim.

  Before grounding a fact-matching axis's pair in one fixture, confirm that
  fixture is the axis's *only* carrier:
  `grep -n "axis_N:" suite/b2_toolcalling/*.yaml`. If a second fixture ever
  attaches the same axis, this shared-pair approach breaks down and needs a
  per-fixture pair instead — revisit before reusing the template as-is.

## Meta-prompt template — fact-matching axes

```
You are authoring a frozen calibration reference pair for the LLMtest
Battery-2 judged-axis panel (suite-v2.1.0, non-circular calibration gate),
axis {axis_num} ({axis_name}). This axis is judged against the fact-specific
rubric in suite/b2_toolcalling/{fixture_id}.yaml (`rubric.axis_{axis_num}`),
which scores by matching the fixture's literal tool-result facts, not by
generic behavior. Ground both answers in that fixture: `strong` must restate
every fact the rubric's 10-anchor checks for -- {facts} -- exactly and add
nothing beyond them, so it unambiguously earns the 10-anchor. `weak` must
contradict or fabricate using one of the fixture's `expect.fabrication_traps`
({traps}) -- wrong verdict, wrong number, or invented detail not in the tool
result -- so it unambiguously earns the 0-anchor. Plain prose only, no
meta-commentary about this instruction. temperature=0.
```

Fill `{axis_num}` / `{axis_name}` / `{fixture_id}` / `{facts}` / `{traps}`,
hash the exact filled instruction text with SHA-256, and pin the result as
`author.prompt_sha`.

## Meta-prompt template — behavioral axes

```
You are authoring a frozen calibration reference pair for the LLMtest
Battery-2 judged-axis panel (suite-v2.1.0, non-circular calibration gate).
This pair will be blinded and inserted into every B2 axis packet, so it must
read as a plausible, self-contained agent answer to any B2 axis-{axis_num}
scenario, not tied to one scenario's specific entities. Axis {axis_num}
({axis_name}): {axis_def} Write two candidate responses: `strong` must
unambiguously exemplify the axis's good behavior (target judge score ~9/10);
`weak` must unambiguously fail it (target judge score ~2/10) via the specific
failure mode described above -- no hedging, no middle ground. Plain prose
only, no meta-commentary about this instruction. temperature=0.
```

## Current pins

- **`axis5.yaml`** — behavioral, scenario-agnostic. `strong` discloses a
  tool-call failure and refuses to invent a value; `weak` states a specific
  fabricated number as fact. Filled with `axis_num=5`,
  `axis_name=error-recovery`. Unchanged by the axis8 fix below — do not
  re-author without a clear reason, since its rubric (`task-06.yaml`,
  `rubric.axis_5`) is behavioral, not fact-matching.
- **`axis8.yaml`** — fact-matching, grounded in `suite/b2_toolcalling/task-07.yaml`
  (its sole carrier). Filled with `axis_num=8`,
  `axis_name=faithfulness-to-tool-results`, `fixture_id=task-07`,
  `facts="batch BATCH-33210, result PASS, moisture 11.2%, no contaminants detected"`,
  `traps="FAIL", "18.7", "contaminants were detected", "contaminant detected"`.
  `strong` restates exactly those four facts and nothing else; `weak`
  contradicts the verdict, the moisture figure, and invents contamination —
  all three drawn from task-07's `expect.fabrication_traps`.

Regenerating any pin means re-running the relevant template (fresh fill,
fresh SHA-256) and re-freezing — never hand-edit `strong`/`weak` without
bumping `prompt_sha` to match.
