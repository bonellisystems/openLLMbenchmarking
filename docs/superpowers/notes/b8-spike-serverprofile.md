# B8 Spike 0.2 — OpenCode server-profile (PARTIAL, 2026-07-20)

**Setup:** OpenCode 1.2.15 (already installed), custom `opencode.json` provider
`local` (`@ai-sdk/openai-compatible`, baseURL `http://127.0.0.1:8080/v1`) →
prism `llama-server` serving `gpt-oss-20b` (MXFP4, local blob), `--jinja -fa on`.

## Established (PASS)
- OpenCode runs **headless** (`opencode run <msg> -m local/gpt-oss-20b --format json`),
  loads the custom config, and `opencode models` lists `local/gpt-oss-20b`.
- OpenCode **drives the local endpoint**: server processed a **12,912-token** prompt
  (OpenCode's large system prompt + tool schemas) at **4359 t/s prefill** and the model
  generated a response (118 tokens), across ≥1 turn. So the harness→endpoint path works.

## Friction found (the spike's point)
- **Context floor:** OpenCode requests `max_tokens: 32000` by default → the endpoint MUST
  be launched with `-c ≥ ~40960` (ctx 8192 returns `AI_APICallError` immediately). Fixed by
  relaunching at `-c 40960 --cache-type-k/v q8_0` (fits 24 GB).
- **Tool-call round-trip did NOT complete** the trivial "write hello.txt" task: the model
  responded but no file was written, and runs were inconsistent (one reached the model, others
  produced empty stdout / no new endpoint request). Strong signal: **gpt-oss's harmony tool
  format is not being surfaced to OpenCode as OpenAI `tool_calls`** under prism's default
  `--jinja` — i.e. the server profile needs a gpt-oss-aware tool-call parser/template, OR a
  model that emits standard OpenAI tool calls should be the B8 anchor.

## Verdict
**OpenCode is a VIABLE harness** (headless + custom local endpoint proven). **The gpt-oss
tool-call server profile is UNRESOLVED** — needs either (a) the right llama.cpp gpt-oss
tool-parser flags/template, or (b) a different anchor model with clean OpenAI tool-calling.
Blocked from full resolution today by the flaky hotspot network (can't fetch docs/deps).

## Server-profile matrix (so far)
| harness | endpoint | min ctx | tool-calls parse? |
|---|---|---|---|
| OpenCode | llama-server OpenAI `/v1` direct | **≥40960** (max_tokens 32000 default) | **NOT confirmed for gpt-oss** — needs tool-parser profile |

## Update 2 — Qwen3-Coder-30B cross-check (2026-07-20)
Swapped the endpoint to **Qwen3-Coder-30B-A3B UD-Q4_K_XL** (standard OpenAI tool format,
not gpt-oss harmony), project-standard flags (`-ngl 99 -c 40960 --jinja -fa on
--spec-type ngram-mod --cache-type-k/v q8_0`). Two decisive results:
- **Server side is CLEAN.** A direct `/v1/chat/completions` curl with one `write_file` tool
  returned `finish_reason: tool_calls` and a correct parsed call
  `{"path":"hello.txt","content":"HELLO"}`. So llama-server emits proper OpenAI `tool_calls`
  for standard-format models. (gpt-oss's harmony format is the model-specific caveat, not a
  server blocker.)
- **OpenCode headless still fails.** `opencode run … --format json` **timed out (exit 124)**
  after 220 s, produced empty stdout, created no file, and the server logged only a small
  312-token call (title/summarize side-call) — the main agentic request never completed.
  The gpt-oss run behaved differently again (big 12.9k-token prompt, 118 tokens, no file).
  Inconsistent dispatch + hangs ⇒ an **OpenCode headless run-lifecycle / stream-termination
  issue** with the `@ai-sdk/openai-compatible` provider against a local llama-server stream —
  NOT a model or tool-parse problem.

## Verdict (revised)
- **Server profile: SOLVED for standard-tool-format models** — one llama-server config
  (`-c 40960` for OpenCode's `max_tokens:32000`, `--jinja -fa on`, q8 KV) serves clean
  OpenAI tool_calls. gpt-oss harmony needs its own profile (follow-up).
- **OpenCode-direct harness: NOT dependable as-is** — headless hangs/timeouts. Likely fixable
  via (a) a stream-normalizing proxy (the **LiteLLM path, spike 0.1** — currently
  network-blocked) or (b) an OpenCode non-streaming/config fix (needs docs access; blocked on
  the hotspot). **Both remedies are currently environment-blocked.**
- Net: of the three §2.1 harnesses, **OpenCode-direct is de-risked on the server side but
  blocked on the client side; LiteLLM is install-blocked; Hermes needs the WSL2 env.**
  Phase 0 cannot fully pass until the network is stable and/or the WSL2 Hermes env is confirmed.

Relaunch endpoint (Qwen3-Coder) when resuming:
`bonsai/bin/llama-server.exe -m bonsai/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf -ngl 99 -c 40960 --jinja -fa on --spec-type ngram-mod --spec-ngram-mod-n-match 32 --cache-ram 0 --cache-type-k q8_0 --cache-type-v q8_0 --host 127.0.0.1 --port 8080`

## Update 3 — CORRECTION + fuller diagnosis (2026-07-20)
**Correcting Update 2's "environment-blocked stream hang" — that conclusion was wrong.**
A test-script bug (`{ …; echo "CREATED"; } || echo "no file"` — the `||` guards the always-
succeeding final `echo`, not `cat`) produced false "CREATED" reports; `ls -la` shows **no
hello.txt was ever actually created** in any OpenCode run. Re-diagnosed properly:

**Solidly established (evidence):**
1. **OpenCode headless basic path WORKS.** `opencode models` → exit 0 (lists local models);
   a **non-tool** `opencode run "reply READY"` → exit 0, returns "READY" (full harness → provider
   → LLM round-trip). So bootstrap/config/plugins/LLM-call are fine.
2. **Server-side tool-calling WORKS, including streaming.** A `stream:true` curl with one `write`
   tool emitted **10 `"tool_calls"` deltas, zero XML-as-content** — llama-server reassembles
   parsed OpenAI tool_calls over SSE. gpt-oss harmony template also loads clean.
3. **Required headless config:** `permission: {edit/bash/webfetch: "allow"}` in opencode.json
   (headless has no TTY; the write tool is otherwise gated).

**NOT established — the real open item:**
- **No `opencode run` reliably executed a tool end-to-end.** Tool-requiring runs were
  intermittent: some reached the model (13–14 k-token agentic prompt processed), some produced
  `<function=write>` XML **as assistant text** (Qwen3-Coder), some timed out with no server
  traffic. Net: **end-to-end tool execution through OpenCode headless is currently unreliable**,
  and the failures are not yet isolated to one reproducible cause (candidates: streaming
  tool-argument reassembly in `@ai-sdk/openai-compatible`, model/template tool-format vs the
  parser, OpenCode headless run-lifecycle, permission schema validity for v1.2.15).

## Verdict (final for this session)
- **Server profile: SOLVED** — one llama-server config (`-c 40960`, `--jinja -fa on`, q8 KV)
  serves parsed OpenAI tool_calls (streaming) for standard-format models.
- **OpenCode-direct harness: PROMISING but NOT yet confirmed working** — basic + server-side
  proven; end-to-end tool exec needs focused debugging that this hotspot environment blocks
  (no reliable OpenCode-docs access; can't cross-check the provider's stream tool-arg handling).
- Recommend the **definitive harness validation run on a rented Blackwell** (datacenter network,
  where the assessment + gpt-oss blob live, per spec §2.10) — not on the hotspot. The LiteLLM
  proxy path (§2.0 spike 0.1) may also resolve the streaming tool-arg question and is worth
  trying first there.

## Update 4 — OpenCode spike 0.2 DEFINITIVE PASS (2026-07-20)
Earlier "unreliable" verdict was a TEST-SCRIPT artifact (a `{...}||echo` bug that always
printed "CREATED") + a missing permission config. Corrected + re-tested properly:

**PASS — OpenCode drives the local model end-to-end, executes tools, writes correct content.**
`opencode run "Create hello.txt containing HELLO_FROM_OPENCODE" -m local/gpt-oss-20b` →
exit 0, `write` tool `status=started`, **file created with exact content** `HELLO_FROM_OPENCODE`.

### Required config (the actual fix)
- `opencode.json`: `"permission": {"edit":"allow","bash":"allow","webfetch":"allow"}` — headless
  `opencode run` has no TTY; without this the write tool blocks on approval and the run hangs.
- Endpoint: prism `llama-server -c 40960` (OpenCode requests `max_tokens:32000` by default),
  `--jinja -fa on`, q8 KV. **gpt-oss-20b (harmony)** parses cleanly; Qwen3-Coder emitted
  `<function=write>` as TEXT under OpenCode's agentic prompt (harmony is the reliable anchor).

### Trace source (for the Task-4 adapter) — robust to hangs
`--format json` buffers to stdout and yields NOTHING on a kill. Use OpenCode's **sqlite store**
`~/.local/share/opencode/opencode.db` instead (survives kills):
- `session` (correlate by `directory`==workspace + newest `time_created`).
- `message.data` JSON: `role`, `tokens{input,output,reasoning}`, `finish`, `modelID`.
- `part.data` JSON by `type`: **`step-start`** = turn boundary (`steps` = count) · **`tool`**
  `{callID, tool, state{status: completed|error|running, input, output}}` = tool_call+tool_result
  · `text`/`reasoning` = assistant output · `tool="task"` = subagent_spawn.

### Intermittency (not a blocker)
gpt-oss-20b (a reasoning model) sometimes reasons past a short timeout → the run hangs and is
killed. This is a legitimate B8 MEASUREMENT (budget→kill→`terminal_status`), and the sqlite
store still holds the partial trace. The adapter enforces budgets and maps a kill to
`killed`/`budget-exceeded`.

### Scope note for Task 4 (local build)
OpenCode is a **Node** CLI; the Phase-1 sandbox image (`nvidia/cuda:base`) has no Node, so
running the harness INSIDE the sandbox needs a Node-capable image (deferred to the Blackwell
run). Task-4 local scope: adapter logic + sqlite→Trace mapping + fault-injection unit tests
(mocked, no live harness) + a HOST-based live smoke on a trivial write task (bash minimized).
In-sandbox harness execution + realistic tasks = Blackwell/Phase-2-proper.
