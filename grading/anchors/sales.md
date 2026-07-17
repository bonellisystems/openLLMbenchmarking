# Sales -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
Qualification reasoning is grounded in the actual buyer signals stated in the prompt, the specific objection raised is addressed on its substance rather than deflected, and the recommended next step is concrete and time-bound. Numbers used (quota, discount, close rate, timeline) are internally consistent with what the prompt provides, and the pitch never claims a capability or guarantee the prompt does not support.
Example character: a response that names the actual objection, answers it directly, and closes with a specific next meeting or action tied to a date.

## 7 -- Strong
The pitch or qualification is usable and grounded in the prompt's details, and the objection is engaged rather than ignored. The next step is present but a little vague, or one objection is handled somewhat generically, but nothing here is manipulative or fabricated.
Example character: a solid response to the stated objection that ends with "let's follow up soon" instead of a specific date or action.

## 5 -- Adequate
The advice follows a generic sales-script structure (build rapport, handle objections, ask for the close) that is directionally correct but not adapted to the specific prospect or objection given.
Example character: stock objection-handling language that could apply to almost any deal.

## 3 -- Weak
The response misreads the buying stage or the stated objection, pitches past it instead of addressing it, or leans on high-pressure tactics presented as best practice with no concrete next step.
Example character: continues pitching product features after the prompt describes a price objection, without ever addressing price.

## 0 -- Unusable
Fabricates product capabilities, pricing, or guarantees not supported by the prompt, or manufactures false urgency/scarcity presented as genuine fact. Empty or off-task.

## Unit-specific red flags (deduct hard)
- Fabricated product capabilities, pricing, or guarantees not supported by the prompt.
- Manufactured false urgency or scarcity presented as real.
- Ignoring the specific objection stated and pitching past it.
- No concrete next step or close action anywhere in the response.
- Internally inconsistent numbers (quota, discount, timeline that contradict each other).

## Unit-specific excellence markers
- The stated objection is addressed on its actual substance, not deflected.
- Concrete, time-bound next step given.
- Qualification reasoning grounded in the specific signals in the prompt.
- Numbers used are realistic and internally consistent.
