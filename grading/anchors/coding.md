# Coding -- scoring anchors (0-10)

Scale: integers 0-10. Anchors define 0/3/5/7/10; use 1-2/4/6/8-9 for answers between anchors.

## 10 -- Frontier
The code is correct on first read, runs or compiles as-is, and handles every edge case implied by the prompt (empty input, boundary values, failure paths) without being asked twice. It matches the requested language, framework, and version idioms, includes a verification step (tests, a worked trace, or explicit correctness reasoning), and never calls an API or library function that does not exist. Assumptions and tradeoffs (performance, complexity, security) are stated explicitly rather than silently chosen.
Example character: a function with input validation, a named edge case, and a short test or trace proving it works; any open question is flagged, not guessed at.

## 7 -- Strong
The core logic is correct and the code is usable with at most light edits. One edge case is unhandled, error handling is thin in a non-critical path, or there is no explicit verification step, but nothing here would surprise a reviewer. It stays within the requested language and framework and does not invent APIs.
Example character: working code that solves the stated problem cleanly but skips a boundary-condition check the prompt did not explicitly call out.

## 5 -- Adequate
The code addresses the stated problem at a surface level but has a real gap: missing error handling, an untested assumption, or a skeleton that needs another pass before it runs cleanly. It is a reasonable starting point but a developer must debug or extend it before shipping.
Example character: a mostly-right implementation that would fail on a null or empty-list input the prompt implied should be handled.

## 3 -- Weak
The answer misreads the actual ask -- wrong language or framework, pseudo-code presented as a finished implementation, or a solution to a simpler problem than the one posed. Boilerplate explanation pads the response without adding working substance.
Example character: a rough sketch that gestures at the right idea but would not run, with no acknowledgment that it is incomplete.

## 0 -- Unusable
The code does not run, is unrelated to the request, or is fabricated wholesale (calls to nonexistent APIs, invented syntax). Includes destructive operations presented as safe with no warning, or is an empty/refused response.

## Unit-specific red flags (deduct hard)
- Hallucinated library functions, methods, or package names presented as real.
- Hardcoded secrets, credentials, or API keys in example code.
- Silently swallowed exceptions or errors with no handling or logging.
- Claiming code was run or tested when no verification is shown or possible.
- SQL, command, or template injection vulnerabilities introduced without comment.

## Unit-specific excellence markers
- Includes a test, worked trace, or verification step, not just a final answer.
- Explicitly handles null/empty/boundary inputs the prompt implies.
- Names complexity or performance tradeoffs when they matter.
- Matches the exact language/framework/version idiom requested.
- Flags any assumption made to fill a gap in the prompt.
