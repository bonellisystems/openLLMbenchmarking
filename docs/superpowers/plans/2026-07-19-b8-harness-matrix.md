# B8 Agentic Harness Compatibility — Implementation Plan (Part 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Battery-8 — run each model through real external agent harnesses (OpenCode, Claude-Code-via-LiteLLM, Hermes/WSL2) on multi-turn coding tasks and measure whether it functions as an agent (completion, steps/tokens, first-failure class, subagent canary) — the real fix for the agentic blind spot.

**Architecture:** A `HarnessAdapter` ABC (mirrors the judge-adapter pattern) with one adapter per harness, each pointing the harness at a `ServerManager`-served llama-server endpoint (directly or via a LiteLLM proxy). Runs execute in a disposable **container sandbox** against versioned task manifests; a normalized trace schema yields deterministic completion + steps/tokens; a **separate blinded classification pipeline** labels first-failures; results are variance-tolerant (N≥5, Wilson intervals). B8 rows carry an `execution_provenance_sha` + `replicate_n`/`attempt_id`.

**Tech Stack:** Python 3.10, pytest; Docker Desktop (WSL2 backend, already present per runtime_pins) for the sandbox; LiteLLM (`vllm/vllm-openai` pin exists; LiteLLM proxy for Anthropic↔OpenAI); OpenCode CLI (to install); Hermes-agent in WSL2 (run here before).

## Global Constraints

- **Fully autonomous — ZERO human-in-the-loop.** No sign-off gates. Harness versions auto-captured; task manifests content-hashed.
- **Version boundary `suite-v2.1.0`.** B8 rows minted under v2.1.0; v2.0.0 imported by reference (P1-T7 `source_suite` machinery already reads both shards).
- **B8 is variance-tolerant, NOT byte-deterministic.** N≥5 replicates/cell; report raw outcomes + Wilson interval, never a smooth "distribution" claim.
- **Sandbox is a security boundary.** Every run in a disposable container: secret-free env, no host-credential mounts, network egress only to the model endpoint, CPU/wall/token quotas, process-tree kill on teardown, post-run cleanup verification.
- **Completion is gaming-proofed.** Hidden validators live OUTSIDE the writable workspace; protected files hash-checked; diff-constrained; behavioral tests run AFTER the harness exits.
- **Row identity:** `execution_provenance_sha` (harness version + LiteLLM version + server profile flags+template hash + full rendered prompt) is part of `condition`; `run_n` is replaced by `replicate_n` (logical) + `attempt_id` (execution), both in row identity, so nondeterministic replicates never dedup-collide.
- **Local-git-only.** Commit after each task; never push.

---

## PHASE 0 — Feasibility spikes (GATE the build; do these first)

Each spike is a task with an explicit **pass/fail** written to `docs/superpowers/notes/b8-spike-<name>.md`. If a spike fails, that harness is dropped from the matrix with the reason recorded — the build proceeds with whatever passed. **No build task in Phase 1+ starts until all three Phase-0 tasks are done.**

### Task 0.1: LiteLLM protocol spike (Claude-Code-via-LiteLLM viability)

**Files:** Create `scripts/b8_spikes/litellm_spike.py`; note `docs/superpowers/notes/b8-spike-litellm.md`.

**Interfaces produced:** a documented verdict — does `claude -p` driven against a LiteLLM proxy (Anthropic `/v1/messages`) backed by a local llama-server (OpenAI `/v1/chat/completions`) complete a trivial 1-tool-call task? Records: LiteLLM version, the proxy config, and which of {tool-result blocks, streaming, stop_reason, system prompt} survive translation.

- [ ] **Step 1: Serve a small model + start the LiteLLM proxy.** Launch llama-server for `gpt-oss-20b` (the designated small anchor) via the existing prism binary; start LiteLLM configured to map an Anthropic model name → the local OpenAI endpoint. (Config template written in the spike script; llama-server launch reuses `llmtest.server.ServerManager` flags.)
- [ ] **Step 2: Drive `claude -p` against the proxy** on a one-file "add a function + call a tool" task in a temp dir, capturing the transcript.
- [ ] **Step 3: Record the verdict** in `b8-spike-litellm.md`: PASS (task completes, tool-result round-trips) or FAIL (with the exact translation gap). Define the "unsupported" terminal result the adapter will emit if PASS is marginal.
- [ ] **Step 4: Commit** — `docs(b8): litellm protocol spike verdict`.

### Task 0.2: Server-profile spike (one server vs per-harness profiles)

**Files:** Create `scripts/b8_spikes/serverprofile_spike.py`; note `docs/superpowers/notes/b8-spike-serverprofile.md`.

**Interfaces produced:** a **server-profile matrix** — for each of {OpenCode, Hermes} (and Claude-Code if 0.1 passed), the llama-server flags/chat-template/tool-parser/stop-tokens that harness requires, and whether one config serves all or each needs its own profile.

- [ ] **Step 1: For each harness, drive one trivial tool-call task** against a `gpt-oss-20b` llama-server and record whether the harness's tool-call format is parsed (schema OK) at the default prism flags.
- [ ] **Step 2: Where it fails, vary the server profile** (chat template / `--jinja` / tool-call parser flags) until the harness's calls parse, recording the working profile + its `template_sha`.
- [ ] **Step 3: Write the server-profile matrix** to the note file: `{harness: {flags, template_sha, notes}}`.
- [ ] **Step 4: Commit** — `docs(b8): server-profile matrix spike`.

### Task 0.3: OpenCode install + version pin + Hermes/WSL reachability

**Files:** note `docs/superpowers/notes/b8-spike-harness-env.md`; `config/runtime_pins.yaml` (add `harnesses:` block).

**Interfaces produced:** confirmed installed harness CLIs + pinned versions; the WSL2↔Windows networking contract (how a WSL2 Hermes reaches a Windows-hosted `127.0.0.1:PORT` endpoint) and path-translation rules.

- [ ] **Step 1: Install OpenCode** (headless), capture `--version`; if it cannot run headless against an OpenAI-compatible endpoint, record FAIL and drop it.
- [ ] **Step 2: Confirm Hermes-agent runs in WSL2** and can reach the Windows llama-server endpoint (resolve the WSL2→host IP; document the URL rewrite). Capture the Hermes commit.
- [ ] **Step 3: Add a `harnesses:` pin block** to `config/runtime_pins.yaml` (opencode version, hermes commit, litellm version) + the WSL networking/path contract in the note.
- [ ] **Step 4: Commit** — `docs(b8): harness env + pins (opencode/hermes/litellm)`.

---

## PHASE 1 — Trace schema, adapter ABC, and the mock adapter (harness-independent core)

### Task 1: Normalized Trace schema + `HarnessAdapter` ABC + mock adapter

**Files:** Create `llmtest/harness/__init__.py`, `llmtest/harness/base.py`, `llmtest/harness/trace.py`; Test `tests/test_harness_base.py`.

**Interfaces produced:**
- `@dataclass Trace`: `events: list[TraceEvent]`, `terminal_status: str` (`"completed"|"failed-task"|"budget-exceeded"|"infra-error"|"killed"`), `steps: int`, `tokens_prompt: int`, `tokens_completion: int`, `subagent_spawned: str` (`"yes"|"no"|"not_applicable"`).
- `@dataclass TraceEvent`: `kind: str` (`"turn"|"tool_call"|"tool_result"|"subagent_spawn"|"terminal"`), `payload: dict`.
- `class HarnessAdapter(ABC)`: `setup(self, task, endpoint, workspace)`, `run(self) -> Trace`, `teardown(self)`, `version(self) -> str`. Plus `MockHarnessAdapter` (deterministic scripted Trace) for the contract test.

- [ ] **Step 1: Write the failing contract test** — `MockHarnessAdapter` yields a scripted `Trace`; assert `setup→run→teardown` sequence, that `run()` returns a `Trace` with a valid `terminal_status`, and that `steps` counts `turn` events. (Full test code: ~30 lines constructing a scripted event list and asserting the derived fields.)
- [ ] **Step 2: Run → FAIL** (module missing). `python -m pytest tests/test_harness_base.py -v`.
- [ ] **Step 3: Implement** `trace.py` (dataclasses + `steps`/token derivation from events) and `base.py` (ABC + `MockHarnessAdapter`).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(harness): Trace schema + HarnessAdapter ABC + mock adapter`.

### Task 2: Container sandbox runner (security boundary)

**Files:** Create `llmtest/harness/sandbox.py`; Test `tests/test_harness_sandbox.py`.

**Interfaces produced:** `class Sandbox`: `__enter__` builds a disposable container (from a pinned image) with the task's initial workspace mounted read-write, **no host creds**, network egress restricted to the endpoint host:port, CPU/wall/token quotas; `run_in(cmd, timeout)`; `snapshot_workspace()`; `__exit__` does process-tree kill + container removal + cleanup verification. `hidden_validate(task)` runs the task's hidden oracle from OUTSIDE the writable mount.

- [ ] **Step 1: Failing test** — using a trivial `echo`/`ls` container command (Docker required; `pytest.mark.skipif` when Docker absent), assert: workspace writes persist inside but a write outside the mount fails; a process spawned in-container is killed on `__exit__`; an attempt to reach a non-endpoint host is blocked.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `Sandbox` over the Docker Desktop/WSL2 backend (pin the base image + digest in `runtime_pins.yaml`). Egress policy via a per-container network allowing only the endpoint. Process-tree kill via container stop.
- [ ] **Step 4: Run → PASS** (skips cleanly where Docker is unavailable).
- [ ] **Step 5: Commit** — `feat(harness): disposable container sandbox with egress + cleanup guarantees`.

### Task 3: Task manifests + deterministic, anti-gaming completion oracle

**Files:** Create `suite/b8_harness/task-01..05.yaml` (5 versioned manifests); `llmtest/harness/tasks.py`; Test `tests/test_harness_tasks.py`, `suite/b8_harness/_schema.md`.

**Interfaces produced:** `@dataclass B8Task`: `id, shape, setup_repo_sha, allowed_tools, budgets, oracle, protected_shas, task_version, fixture_sha`. `load_b8_tasks(root)`; `run_oracle(task, workspace) -> (completed: bool, detail)` — runs the hidden behavioral oracle AFTER the harness exits, verifies `protected_shas` unchanged, applies the diff constraint. Five shapes: edit, multi-file, bugfix (reuse B6 planted-bug), tool-heavy, from-scratch (reuse B6 snake).

- [ ] **Step 1: Failing test** — a manifest loads with all required keys; `run_oracle` returns `completed=True` for a correct final workspace and `completed=False` when a protected file (e.g. the test file) was tampered with (protected-sha mismatch) even if the task "passes". (Test writes a tmp workspace both ways.)
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Author the 5 manifests + implement `tasks.py`.** Oracles: compile/pytest/run-artifact executed in a fresh read-only copy of the workspace outside the agent's reach; protected-file hash check; allowed-diff-path constraint.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(harness): versioned B8 task manifests + anti-gaming completion oracle`.

---

## PHASE 2 — Real harness adapters (one per Phase-0 survivor)

For EACH harness that PASSED Phase 0, one task, same shape. (If a harness failed Phase 0, skip its task and record it in the plan ledger.)

### Task 4: OpenCode adapter · Task 5: LiteLLM/Claude-Code adapter · Task 6: Hermes/WSL2 adapter

**Files (per adapter):** Create `llmtest/harness/<name>.py`; Test `tests/test_harness_<name>.py`.

**Interfaces consumed:** `HarnessAdapter` (Task 1), `Sandbox` (Task 2), `B8Task` (Task 3), the server-profile matrix (Task 0.2), `ServerManager` for the endpoint.

**Interfaces produced:** `class <Name>Adapter(HarnessAdapter)` — `setup` launches the harness in the sandbox pointed at the endpoint per its server profile (LiteLLM adapter also starts the proxy); `run` drives the task and maps the harness's native transcript/logs into the normalized `Trace` (server-side token counts from llama-server usage, NOT harness proxies); `teardown` stops the harness + proxy. `version()` from the Phase-0 pin.

- [ ] **Step 1: Failing test** — a **fault-injection contract test** per adapter (mock the harness subprocess): assert correct `terminal_status` on hang (→ `killed` at budget), malformed tool call (→ trace records `tool_call` with parse failure), proxy/endpoint disconnect (→ `infra-error`), missing usage (→ tokens fall back to server-side), and that teardown leaves no child processes. (No live harness needed for the unit test — the live smoke is Step 4.)
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the adapter + its native-log→Trace mapping.
- [ ] **Step 4: Live smoke** — run the adapter once on task-01 with `gpt-oss-20b` through the real harness; record the produced `Trace` in the report file (proves end-to-end). Assert a `Trace` with a terminal status is produced.
- [ ] **Step 5: Commit** — `feat(harness): <name> adapter with trace mapping + fault-injection tests`.

---

## PHASE 3 — Battery, classification, canary, and report

### Task 7: B8 battery + row identity (`execution_provenance_sha`, replicate_n/attempt_id) + budgets

**Files:** Create `llmtest/batteries/b8_harness.py`, `llmtest/batteries/b8_fixtures.py`; Modify `config/suite.yaml` (`b8:` block); Test `tests/test_b8.py`.

**Interfaces produced:** `@register class B8Harness(Battery)` — `plan()` yields one WorkItem per `(model, harness, task, replicate_n)` filtered by `b8.models`/`b8.harnesses`; `execute()` runs the adapter in the sandbox under per-run wall/token/step budgets, computes `execution_provenance_sha` over (harness_version, litellm_version, server-profile flags+template_sha, full rendered prompt), and emits a row with `battery=8`, `condition` encoding `(harness, task, execution_provenance_sha)`, `run_n`→`replicate_n`+`attempt_id` in the row_id preimage, `metrics={completion, steps, tokens, terminal_status, subagent_spawned}`. Deterministic-completion rows carry `needs_judging=False`; only first-failure-classification rows (Task 8) carry `needs_judging=True`.

- [ ] **Step 1: Failing tests** — `plan()` item count = `len(b8.models) × harnesses × tasks × replicates`; `execute()` (with a `MockHarnessAdapter` injected) produces a schema-valid row whose `row_id` differs across `attempt_id` for the same cell; `execution_provenance_sha` changes when the harness version changes. Reuse the `p8_gen`-style injected-endpoint pattern.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the battery + fixtures + `suite.yaml b8:` (models/harnesses/replicates≥5/budgets). Extend `schema.py`/`compute_row_id` only as needed for `attempt_id` (additive).
- [ ] **Step 4: Run → PASS + full suite.**
- [ ] **Step 5: Commit** — `feat(b8): harness battery with execution-provenance identity + budgets`.

### Task 8: First-failure classification pipeline (deterministic detectors + blinded classifier panel)

**Files:** Create `llmtest/harness/failure_class.py`; Test `tests/test_failure_class.py`.

**Interfaces produced:** `classify_first_failure(trace, task) -> (label, source)` where deterministic detectors run FIRST with stated precedence — `(a) schema-never-parsed`, `(d) harness-bug` from trace/logs — and any unresolved failed trace is routed to a **separate blinded classifier panel** (its own label schema `{a,b,c,d,unknown}`, per-trace votes, majority+tie handling, abstention reporting) that shares the adapter/blinding infra but does NOT enter the numeric median-of-3 pipeline. `not_applicable`/`unknown` allowed.

- [ ] **Step 1: Failing tests** — a trace with unparsed tool calls → `(a)` deterministically (no panel); a harness-error terminal → `(d)`; a completed-but-wrong-logic trace → routed to the classifier (mock the classifier adapter) and labeled `(c)`; a tie → `unknown`.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(b8): first-failure classification (deterministic detectors + blinded panel)`.

### Task 9: Subagent canary + B8 report section (raw outcomes + Wilson intervals + source_suite)

**Files:** Modify `scripts/p8_report.py`; `llmtest/harness/stats.py` (Wilson); Test `tests/test_report_b8.py`.

**Interfaces produced:** the canary (`subagent_spawned` per harness with a "usable result" criterion; `not_applicable` where a harness has no delegation primitive). Report B8 section: per-`(model, harness)` completion proportion with **Wilson interval**, median steps/tokens, first-failure-class distribution, and the subagent-canary headline — raw per-replicate outcomes, never a smooth distribution claim; labeled `source_suite=v2.1.0`.

- [ ] **Step 1: Failing test** — report text shows a B8 `(model,harness)` completion `k/N` + a Wilson interval, a first-failure-class breakdown, a subagent-canary line (`not_applicable` honored), and a `source_suite` label; synthetic B8 rows exercise it.
- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: Run → PASS + full suite.**
- [ ] **Step 5: Commit** — `feat(report): B8 harness section with Wilson intervals + subagent canary`.

---

## Self-Review

- **Spec coverage:** §2.0 spikes → Phase 0 (Tasks 0.1–0.3); §2.1 adapter ABC → Task 1; §2.2 provenance/identity → Task 7; §2.3 sandbox+anti-gaming → Tasks 2 + 3; §2.4 task manifests → Task 3; §2.5 normalized metrics/server-side tokens → Tasks 1 + 4-6; §2.6 first-failure classification → Task 8; §2.7 subagent canary → Task 9; §2.8 N≥5 + Wilson → Tasks 7 + 9; §2.9 budgets/terminal-status → Tasks 1 + 7; §2.10 config knobs → Task 7. All covered.
- **Gating honesty:** Phase 1+ tasks assume the Phase-0 survivors; a harness that fails its spike is dropped and its Phase-2 adapter task skipped (recorded in the ledger). No build task fabricates a harness API the spike hasn't confirmed.
- **Determinism boundary:** B8 rows are explicitly variance-tolerant (`replicate_n`+`attempt_id`, Wilson intervals) — no byte-deterministic assertion anywhere in B8 tests; deterministic tests use `MockHarnessAdapter` or synthetic traces.
- **Type consistency:** `Trace`/`TraceEvent`/`HarnessAdapter`/`B8Task`/`Sandbox`/`classify_first_failure` signatures used consistently across tasks.

## Follow-ons / out of scope
Extending the subagent canary to OpenCode/Hermes native delegation modes; a shared container-runtime reused by other batteries; promoting Continue/Pi/Maki harnesses (spec's post-baseline list).
