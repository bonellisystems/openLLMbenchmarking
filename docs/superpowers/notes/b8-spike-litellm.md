# B8 Spike 0.1 — LiteLLM protocol (DEFERRED, 2026-07-20)

**Status: BLOCKED — could not run.** `pip install litellm` fails on the current
hotspot network (pypi read-timeout / build failure, retried with 120s timeout + 8 retries).
LiteLLM is a small pip package, not a model, but the flaky network blocks the fetch.

**To complete when network is stable:** `pip install litellm`; start a LiteLLM proxy
mapping an Anthropic model → the local `http://127.0.0.1:8080/v1` endpoint; run `claude -p`
with `ANTHROPIC_BASE_URL` pointed at the proxy on a trivial 1-tool task; record whether
tool-result blocks / streaming / stop_reason survive the Anthropic↔OpenAI translation.
