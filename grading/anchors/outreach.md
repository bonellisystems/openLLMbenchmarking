# Outreach -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
Personalization uses the specific prospect details given in the prompt, not just a name token dropped into a template, and the message is concise and value-first rather than pitch-heavy. The sequence logic (number and spacing of follow-ups) is sound, reply-rate expectations are realistic, and a clear sender identity with an opt-out or unsubscribe path is included wherever the channel or scenario implies one is required.
Example character: a short first-touch message that references a specific detail from the prompt, followed by a defined two- or three-touch cadence with a clear opt-out.

## 7 -- Strong
The sequence is usable and reasonably personalized to the prospect described, with sound value-first framing. It is missing an explicit compliance/opt-out mention, or the follow-up cadence is implied rather than fully specified, but nothing here reads as mass-blast.
Example character: a well-targeted message with genuine personalization but no stated follow-up cadence.

## 5 -- Adequate
The message uses light mail-merge personalization ({first_name} only) with no deeper connection to the specific details given, and the cadence or compliance angle is vague or absent.
Example character: a template email that only substitutes the recipient's name and company.

## 3 -- Weak
The tone reads as mass-blast, ignoring specific personalization details available in the prompt in favor of generic boilerplate, or ignores the stated persona and channel entirely.
Example character: a generic pitch email that could be sent to any prospect regardless of the details given.

## 0 -- Unusable
Recommends deceptive tactics (misrepresented sender identity, no opt-out on a channel that clearly requires one, purchased or scraped lists presented as compliant) without any caveat, or states unrealistic reply/conversion rates as fact. Empty or off-task.

## Unit-specific red flags (deduct hard)
- No unsubscribe or opt-out path where the channel or scenario implies one is required.
- Deceptive subject lines or sender misrepresentation.
- Ignoring specific personalization details given in the prompt in favor of generic boilerplate.
- Unrealistic reply or conversion-rate claims stated as fact.

## Unit-specific excellence markers
- Personalization uses the specific details given, not just a name token.
- Sound follow-up cadence with a defined number and spacing of touches.
- Value-first framing rather than a pure pitch.
- Compliance and opt-out awareness where relevant.
