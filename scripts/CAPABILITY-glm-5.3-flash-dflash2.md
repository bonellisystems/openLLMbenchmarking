# Capability profile — glm-5.3-flash-dflash2 (2026-08-31)

Endpoint: `http://192.168.0.74:8888/v1` (spark1, TP=2 SGLang, worker spark2)
Served id: `glm-5.3-flash-dflash2`
Recipe: beastllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark + #36596 ModelOpt exclude patch
Weights: LibertAIDAI/GLM-5.3-Flash-NVFP4 snapshot `11d73216` + incoai/GLM-5.3-Flash-DFlash2 D=5
KV: bfloat16  Image: `glm53-flash-dflash:c4d5d45e5-gb10tile-mopt`
hardware_sku: dgx-spark-gb10  suite: suite-v2.3.0-spark
Results: `/home/michaeldeblok/llmtest-spark/out` (NOT llmtest-v2/results)

| Question | Measured |
|---|---|
| OpenAI-compatible | yes, `choices[0].message.content` |
| Auth | none |
| Model id | `glm-5.3-flash-dflash2` (required) |
| usage | prompt/completion/reasoning_tokens |
| timings | none (not llama.cpp) |
| reasoning_content | **yes** (thinking on by default) |
| tools array | **yes** — `finish_reason=tool_calls`, glm47 parser |
| streaming | not required; server supports it |
| max_model_len advertised | **65536** (this boot; was 16384) |
| concurrency | max-running-requests=8 (mamba pool 48) |

Traps:
- Default thinking ON: PING OK in content, 38 reasoning tokens.
- `enable_thinking: false` leaked chain-of-thought into content and did **not** say exactly PING OK. Do not disable thinking for instruction batteries.
- Empty content + tool_calls is normal for tool turns, not a length-budget miss.

B4 arms > 64k: not-run.
B5 spec A/B: DFLASH is the live spec; off-arm requires a second boot without `--speculative-algorithm DFLASH`.
