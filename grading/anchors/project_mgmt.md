# Project Management -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
Scope, schedule, and risk are tied concretely to the specifics given -- stated dependencies, resourcing, and deadline -- rather than generic project-plan language. Estimates show their basis (a breakdown, an assumption, or a comparable), the top risks are named with an owner and mitigation, and the critical path or a key dependency is called out explicitly.
Example character: a plan with a dated milestone list, an estimate that shows its basis, and the top risk named with an owner and mitigation.

## 7 -- Strong
The plan structure is sound and usable, correctly tied to the stated scope. Risk mitigation is thinner than ideal, or the estimate basis is not fully shown, but the schedule and scope logic hold together.
Example character: a workable schedule and scope breakdown with risks named but no owner assigned to them.

## 5 -- Adequate
The plan follows a generic template (phases labeled "planning," "execution," "closing") that is not tied to the specific scope or constraints given in the prompt.
Example character: a phase list that could describe almost any project regardless of its actual scope.

## 3 -- Weak
The plan ignores a stated hard constraint (deadline, budget, headcount), identifies no risk at all, or gives an estimate as an unexplained round number.
Example character: a timeline that assumes more people than the prompt states are available.

## 0 -- Unusable
Presents fabricated resourcing or cost figures as sourced fact, or the plan is internally inconsistent (dates that do not add up against stated dependencies). Empty or off-task.

## Unit-specific red flags (deduct hard)
- Estimate or timeline internally inconsistent with dates or dependencies stated elsewhere in the same answer.
- Fabricated resourcing or cost figures presented as fact.
- No risk register or mitigation despite a clearly risky scope.
- Ignoring an explicitly stated hard constraint (deadline, budget, headcount).

## Unit-specific excellence markers
- Estimates show their basis -- a breakdown, assumption, or comparable.
- Top risks named with an owner and mitigation.
- Critical path or a key dependency called out explicitly.
- Plan structure tied to the specific scope given, not generic phase names.
