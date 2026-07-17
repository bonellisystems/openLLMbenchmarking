# Knowledge Management -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
The runbook or document is testable: a newcomer could follow it step by step and reach the stated outcome without needing undocumented tribal knowledge. Prerequisites are listed up front, failure or rollback branches are included for steps that can go wrong, and a currency or ownership signal (last verified, owned by) is present when the prompt supports it.
Example character: numbered steps with prerequisites stated first, a rollback note on the one risky step, and a last-verified date.

## 7 -- Strong
The document is mostly step-by-step and usable as a runbook, but is missing one prerequisite check or a failure-branch note for a step that could plausibly fail. Nothing here requires guesswork to follow.
Example character: a clear numbered procedure that omits what to do if the third step fails.

## 5 -- Adequate
The content is a correct-gist prose description of the process rather than an executable, testable runbook -- a new reader would need to reverse-engineer the actual steps.
Example character: a paragraph explaining how the process generally works, without numbered, followable steps.

## 3 -- Weak
The document skips steps a newcomer would need, silently assumes undocumented tribal knowledge, and gives no ownership or currency signal at all.
Example character: a procedure that jumps from step one to step four with an unstated step in between.

## 0 -- Unusable
References fabricated tools, systems, or menu paths that do not exist, or gives a dangerously wrong sequence for a risky operation. Empty or off-task.

## Unit-specific red flags (deduct hard)
- Steps that silently assume undocumented tribal knowledge.
- Fabricated tool, system, or menu-path names presented as real.
- No failure or rollback branch for a step that can fail destructively.
- Prose narrative presented as a runbook instead of testable numbered steps.

## Unit-specific excellence markers
- Numbered steps a newcomer could execute unaided.
- Prerequisites listed explicitly up front.
- Failure or rollback branch given for risky steps.
- Currency or ownership signal (last-verified, owner) included when supported by the prompt.
