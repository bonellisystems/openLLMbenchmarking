# Helpdesk -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
Triage priority matches the actual business impact described, and the response tone is consistent with the urgency that priority implies. Troubleshooting steps are ordered from least to most disruptive, a clear escalation path is given for when the fix is out of the agent's scope, and the answer is calibrated about what genuinely needs remote hands or an on-site visit.
Example character: a triage call that matches the stated impact, a non-disruptive-first troubleshooting sequence, and a named escalation trigger.

## 7 -- Strong
The troubleshooting sequence is usable and the priority call is correct, but the escalation path or SLA framing is a light miss -- present but not fully specified. Nothing here jumps to a disruptive step too early.
Example character: a correct non-disruptive troubleshooting sequence with no stated escalation trigger if it fails.

## 5 -- Adequate
The response is a generic troubleshooting checklist (restart, check cables, reinstall) not tied to the specific symptom or business impact described in the ticket.
Example character: standard first-line steps offered without adapting to the reported symptom.

## 3 -- Weak
The priority or severity call is wrong relative to the stated impact, or the response skips straight to a disruptive fix (reimage, factory reset) without trying non-disruptive steps first, with no escalation path given.
Example character: recommends reimaging a device as the first step for a minor, non-blocking symptom.

## 0 -- Unusable
Recommends a data-destructive action (wipe, reformat, delete profile) with no backup step or warning, or fabricates vendor or product behavior presented as known fact. Empty or off-task.

## Unit-specific red flags (deduct hard)
- Destructive troubleshooting step (reimage, factory reset, delete profile) recommended before non-disruptive steps, or with no backup warning.
- Priority/SLA call that does not match the stated business impact.
- No escalation path given when the issue is clearly out of scope.
- Fabricated product or vendor behavior presented as known fact.

## Unit-specific excellence markers
- Triage priority matches the stated business impact.
- Troubleshooting steps ordered least-to-most disruptive.
- Explicit escalation trigger and path given.
- SLA/urgency framing consistent with the ticket's actual severity.
