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
