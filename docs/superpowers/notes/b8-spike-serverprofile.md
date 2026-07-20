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
