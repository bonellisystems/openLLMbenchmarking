# B1 Fixture Schema

## Overview

B1 fixtures are YAML task definitions for the Business suite — realistic MSP task scenarios graded by deterministic signal checking. Each fixture resides in `suite/b1_business/<unit>/task-<NN>.yaml`.

## Required Keys

- **id** (string): Unique task identifier in format `<unit>-<NN>` where `<unit>` matches a configured unit and `<NN>` is a two-digit number (e.g., `cybersecurity-01`, `sales-12`).
- **unit** (string): Business unit name. Must be one of the configured `b1.units_tier1` values in `config/suite.yaml`.
- **difficulty** (string): Task difficulty level. Must be one of: `easy`, `medium`, `hard`.
- **class** (string): Token class determining max output tokens. Must be one of: `short`, `standard`, `long`. Maps to `b1.max_tokens_by_class` in `config/suite.yaml`.
- **industry** (string): Industry context for the task. Must be one of the configured values in `b1.industries` in `config/suite.yaml`. Tasks are industry-diverse (not MSP-specific); the industry field describes the scenario context while work shapes remain realistic.
- **prompt** (string): The full, self-contained task prompt. Should be realistic scenario language for the chosen industry.
- **signals** (list): Deterministic checks for grading (see below).
- **notes** (string, optional): Authoring rationale, source-of-truth pointers, or placeholder markers.

## Industry Distribution Rule

For units with ≥8 tasks, the following distribution constraints apply:

- **≥5 distinct industries**: Tasks in a unit must span at least 5 different industries from the configured vocabulary.
- **≤2 tasks per industry**: No single industry can appear more than 2 times within a unit.

**Rationale:** Ensures industry diversity and prevents over-specialization in any single vertical.

**Small units exemption:** Units with <8 tasks skip the distribution check (e.g., during mid-authoring stages).

## Signal Vocabulary

Signals are checks applied to model responses to collect evidence for judging. Each signal has:

- **type** (string): One of `contains`, `regex`, or `numeric`.
- **value** (string or number): The pattern or target value.
- **tolerance** (float, optional): For `numeric` type, the relative tolerance as a decimal (e.g., 0.01 for ±1%). Defaults to 0.01.
- **weight** (string, optional): Importance hint for judges (e.g., `note`, `evidence`, `critical`). Not enforced by loader but documented for judge configuration.

### Signal Types

#### contains
Checks if the response text contains a literal substring.
```yaml
- {type: contains, value: "MFA"}
```

#### regex
Checks if the response text matches a regular expression pattern (using Python `re.search`).
```yaml
- {type: regex, value: "\\b(CVE-\\d{4}-\\d+)\\b"}
```

#### numeric
Checks if the response text contains a number within relative tolerance of the target value.
- Preprocessing: Strips `$` (dollar signs) and `,` (commas) from digit groups before number extraction.
- Matching: Any parsed number within `abs((found - target) / target) <= tolerance` passes.
- Example: Target 4200 with tolerance 0.01 passes for any number between 4158–4242.

```yaml
- {type: numeric, value: 4200, tolerance: 0.01}
```

For a response like "Cost: $4,200" or "total of 4200 dollars", the numeric checker:
1. Cleans: "Cost: 4200" and "total of 4200 dollars"
2. Extracts: [4200]
3. Compares: 4200 vs. 4200 → rel_diff = 0 ≤ 0.01 ✓

## Example Fixture

```yaml
id: cybersecurity-01
unit: cybersecurity
difficulty: easy
class: short
industry: generic_smb
prompt: |
  You are responding to a security incident at a financial services client.
  A recent scan found CVE-2026-1234 affecting their network.
  Estimate the remediation cost at $4,200 per system.
  Recommend a timeline using MFA enforcement.
signals:
  - {type: contains, value: "CVE-2026-1234", weight: critical}
  - {type: regex, value: "\\b(MFA|multi-factor)\\b", weight: evidence}
  - {type: numeric, value: 4200, tolerance: 0.01, weight: note}
notes: |
  Cybersecurity Task 01 - Vulnerability Remediation Estimate.
  Source: MSP best practices for breach response.
  Industry: generic_smb (could be financial_services in alternate version).
```

## Validation

All fixtures are validated by `python -m llmtest validate`:

1. **Syntax**: YAML parses without error.
2. **Required keys**: All listed above are present.
3. **unit**: Must be in `config/suite.yaml` `b1.units_tier1`.
4. **difficulty**: Must be one of `easy`, `medium`, `hard`.
5. **class**: Must be one of `short`, `standard`, `long`.
6. **id**: Must match format `<unit>-\d\d` where `\d\d` is two digits.
7. **signals**: Each signal must have a valid `type` in `{contains, regex, numeric}`.

Validation errors exit with code 1. See `validate_cmd.py` for implementation.

## Loading Fixtures

Use `llmtest.batteries.b1_fixtures.load_unit_tasks(root, unit)` to load all fixtures for a unit:

```python
from pathlib import Path
from llmtest.batteries import b1_fixtures

root = Path(".")
tasks = b1_fixtures.load_unit_tasks(root, "cybersecurity")
for task in tasks:
    print(f"{task.id}: {task.difficulty} {task.cls} - {task.prompt[:50]}...")
```

## Checking Signals

Use `check_signals(text, signals)` to evaluate a response:

```python
from llmtest.batteries import b1_fixtures

signals = [
    {"type": "contains", "value": "MFA"},
    {"type": "numeric", "value": 4200, "tolerance": 0.01}
]
result = b1_fixtures.check_signals("Enable MFA for $4,200.", signals)
print(result)  # {0: {"pass": True}, 1: {"pass": True}}
```

Returns a dict mapping signal index to `{"pass": bool}`.
