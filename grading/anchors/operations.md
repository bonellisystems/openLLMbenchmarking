# Operations -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
The process redesign is tied concretely to the specific bottleneck and constraint stated in the prompt, with measurable KPIs that include a baseline and target, not just a goal restated as a metric. Implementation steps are sequenced with a clear first action, change-management or rollout risk is addressed, and resourcing constraints given in the prompt are respected rather than assumed away.
Example character: a redesign that names the actual bottleneck, gives a baseline-to-target KPI, and sequences the first three implementation steps.

## 7 -- Strong
The process improvement is sound and directly usable, correctly targeting the stated bottleneck. The KPI baseline or rollout sequencing is thinner than ideal, but the core recommendation would work as described.
Example character: a correct fix for the stated bottleneck with a named metric but no baseline value given.

## 5 -- Adequate
The advice is generic process-improvement guidance ("streamline the workflow," "reduce handoffs") that is directionally correct but not tied to the specific constraint described in the prompt.
Example character: general efficiency advice that would apply to almost any operational bottleneck.

## 3 -- Weak
The answer misidentifies the bottleneck, or proposes a process that ignores a stated hard constraint (headcount, budget, system limitation), with no measurable KPI anywhere in the response.
Example character: a redesign that requires more staff despite the prompt stating headcount is fixed.

## 0 -- Unusable
Cites fabricated industry-benchmark figures as if real, or recommends a change that outright violates a stated hard constraint with no acknowledgment. Empty or off-task.

## Unit-specific red flags (deduct hard)
- Fabricated industry-benchmark numbers presented as real data.
- Ignoring an explicitly stated resource or system constraint.
- No measurable KPI or success criterion anywhere in the response.
- Steps that restate goals rather than being actually sequenced and actionable.

## Unit-specific excellence markers
- Root cause or bottleneck correctly identified from the details given.
- Measurable KPI with a stated baseline and target.
- Rollout sequenced with a clear, concrete first step.
- Resourcing constraints from the prompt respected in the design.
