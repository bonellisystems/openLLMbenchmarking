# Cybersecurity -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
Severity is triaged correctly against the actual impact described, response actions follow containment before eradication before recovery, and any CVE, CWE, or ATT&CK reference used is real and correctly applied (never invented). Evidence-preservation is called out before disruptive remediation, and breach-notification or compliance timelines are flagged when the scenario implies regulated or sensitive data. The answer is calibrated about what needs live forensics versus what can be assessed from the facts given.
Example character: a triaged action list that isolates the affected system first, preserves logs, then patches, with the real CVE cited only where it is genuinely relevant.

## 7 -- Strong
The response plan is sound and correctly ordered (containment before eradication), and is directly usable by a responder. References to standards or CVEs are generic rather than specific, or evidence-preservation/notification timing gets only a light mention. Severity is judged reasonably even if not formally scored.
Example character: a correct incident-response sequence that never names a specific CVE or ATT&CK technique even where the prompt supports one.

## 5 -- Adequate
The advice reads as a generic security checklist (patch systems, enable MFA, monitor logs) that is directionally correct but not prioritized by the actual severity or tailored to the specific incident described.
Example character: a list of general hardening steps with no sense of which one matters most given the described exposure.

## 3 -- Weak
The response gets the ordering wrong (patches or removes an attacker's access before containing them), misjudges severity, or pads the answer with boilerplate ("enable a firewall") that does not engage the specific scenario.
Example character: jumps straight to "eradicate the malware" on an active intrusion without isolating the host first.

## 0 -- Unusable
Cites a fabricated CVE, CWE, or CVSS score as if verified, or recommends actively dangerous steps (disable logging, do not isolate an active breach) as sound practice. Empty or off-task response.

## Unit-specific red flags (deduct hard)
- Fabricated CVE, CWE, or CVSS identifiers presented as real.
- Eradication or patching recommended before containment on an active incident.
- Recommending disabling logging or otherwise destroying evidence.
- Ignoring implied breach-notification or regulatory timelines when sensitive data is involved.
- Treating every finding as equally critical with no severity differentiation.

## Unit-specific excellence markers
- Severity correctly triaged and prioritized against the described impact.
- Containment-before-eradication-before-recovery ordering respected.
- Real, correctly-applied CVE/CWE/ATT&CK references where relevant.
- Evidence-preservation explicitly called out.
- Notification or compliance timeline awareness when data exposure is implied.
