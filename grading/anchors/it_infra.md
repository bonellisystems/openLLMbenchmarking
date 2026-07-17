# IT Infrastructure -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
The diagnosis or design is correct and tied to the specific systems, symptoms, and constraints given, with exact commands or configuration for the stated platform rather than generic advice. It includes a rollback or backup step before any risky change, quantifies blast radius and expected downtime, and is calibrated about what still needs on-site or live verification. A maintenance-window or change-freeze constraint mentioned in the prompt is respected, not ignored.
Example character: a targeted fix with the exact command sequence, a stated rollback step, and a note on which part of the environment is affected.

## 7 -- Strong
The technical recommendation is sound and directly usable, with correct commands or configuration for the stated platform. It is missing one of: an explicit rollback plan, a quantified blast-radius/downtime estimate, or a post-change verification step. Nothing here would misdiagnose the problem.
Example character: a correct fix for the stated symptom that does not spell out how to undo it if it goes wrong.

## 5 -- Adequate
The direction is correct but the advice reads as a generic troubleshooting checklist rather than being tailored to the specific hardware, software, or symptoms given. An engineer would need to fill in the exact commands and confirm the environment before acting.
Example character: "check logs, restart the service, verify connectivity" offered without adapting to the platform named in the prompt.

## 3 -- Weak
The answer misdiagnoses the stated problem, ignores an explicit constraint (no maintenance window, no downtime allowed, specific hardware), or jumps to a disruptive fix without trying safer steps first.
Example character: recommends a reboot or reinstall as the first step on a production system the prompt flagged as always-on.

## 0 -- Unusable
Recommends a destructive action (wipe, factory reset, force-delete) with no backup or rollback warning, or invents command syntax or vendor behavior that does not exist. Empty or off-task response.

## Unit-specific red flags (deduct hard)
- Destructive commands (rm -rf, drop database, factory reset, force reformat) with no backup or rollback step.
- Fabricated CLI flags, config keys, or vendor documentation presented as real.
- Ignoring a stated production, change-freeze, or maintenance-window constraint.
- No mention of testing a risky change in non-production first.
- Treating a symptom fix as a root-cause fix without saying so.

## Unit-specific excellence markers
- Explicit rollback or backup step before any risky change.
- Blast radius and expected downtime called out in concrete terms.
- Exact commands or configuration syntax for the stated platform.
- Maintenance-window and change-freeze awareness.
- Post-change verification or monitoring step included.
