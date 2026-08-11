# Local LLM Model Guide

Measured benchmark results for locally-servable LLMs, plus the exact command to run each one. Written to be read by an LLM that needs to choose a model for a workload and then start it.

**How to read this file.** Every score below is a MEASURED result, not an estimate or a vendor claim. `not run` means the cell was never measured and you must not treat it as a zero or as a weak result - where a cell is blank for a known reason, the reason is in *Known gaps* at the end. Sample sizes are small (n is given per battery), so differences under ~5 points are usually ties rather than rankings.

Generated from `llmtest-v2 results shards + labelled session constants`.

## Picking a model

Match the battery to the job, then read that column:

| If the workload is... | Read this battery | It measures |
|---|---|---|
| General business writing, analysis, drafting | **B1** | quality across 15 business departments, judged 0-10 |
| Calling tools / driving an API or agent loop | **B2** | whether it forms a valid tool call at all |
| Writing code from a spec | **B3** | code generation correctness |
| Long documents / large context | **B4** | retrieval accuracy at long context |
| Throughput-sensitive serving | **B5** | decode tokens/sec |
| Fixing or editing existing code | **B6** | bugfix and edit correctness |
| Autonomous multi-step coding (agentic) | **B8** | end-to-end task completion in a real harness with a hidden oracle |
| Generating a working app / game in one shot | **B9** | does the generated program actually run |
| Refusing unsafe requests | **B10** | safety behaviour under adversarial prompts |
| Multi-turn tool use with real filesystem effects | **B11** | agentic tool use scored from the filesystem, not from narration |

## What each battery measures

- **B1 - Business Scorecard** (/10): 15 business units x 8 tasks x 3 reps, scored 0-10 by a blinded 3-judge panel (Claude Fable-5 / GPT-5.6-sol / Gemini 3.1 Pro) against per-unit rubrics with CAL-strong/CAL-weak calibration anchors in every packet.
- **B2 - Tool Calling** (%): Can the model emit a well-formed tool call: correct schema, right tool selected, argument shapes valid. This is a FORMATION floor, not agentic skill - most models score ~100% and it cannot detect delegation failures.
- **B3 - Hallucination Resistance** (%): Unanswerable / trick / false-premise prompts. Scores the 'correct' signal: did it refuse or hedge instead of fabricating. Note the 300-token answer budget starves reasoning models, which spend it on hidden thinking.
- **B4 - Long-Context Retrieval** (%): Needle-in-a-haystack recall across a context-length sweep (16k -> 256k). Arms are pruned per model by VRAM fit, so 100B+ models legitimately get zero arms on a single card.
- **B5 - Serving Throughput** (t/s): Decode tokens/sec on the datacenter box, reported from the spec-decode OFF arm so every model's number is the same measurement. IMPORTANT: the n-gram ON arm did NOT engage for the 20 models measured before 2026-08-11 - they all report a speedup of exactly 1.00x, which is the flag missing at serve time rather than a result. This was previously explained as 'fresh text, where n-gram cannot help'; qwen3.6-27b-fable-fusion disproved that on this very battery with 6.79x (482 vs 71 t/s). Treat these figures as UNACCELERATED baselines: an edit-heavy workload with n-gram on runs several times faster, per the Speculative Decoding panel.
- **B6 - Agentic Coding** (%): 10 tasks: 5 from-scratch (is_prime, word-count CLI, backup.sh, debounce, SQL) and 5 planted-bug fixes. Deterministic checks only - the judged quality axis is built but not yet run. Does NOT include the game builds (see Game Builds panel).
- **B7 - Reproducibility** (%): Same prompt across a config matrix (system prompt / temperature / tool format / spec-decode). Reports how often the deterministic signals agree with the baseline cell - i.e. how much the answer moves when harness settings move.
- **B8 - Agentic Harness** (%): Real OpenCode agent in a disposable container: 23 sealed Python tasks across break-fix / cross-module / feature / stateful / build / robustness, scored by a hidden oracle. SINGLE-AGENT ONLY - no task requires spawning a subagent.
- **B9 - Game Builds** (%): One-shot browser games from a bare one-line prompt (snake, tetris, arkanoid, flappy, doodle jump, asteroids, roguelike, and a fly.pieter-style 3D flight sim), scored by DRIVING each build in headless Chrome: does it load, paint, animate, wire up keys, survive a key burst. Gameplay quality is human-graded in the explorer - a browser cannot tell 'the snake moved' from 'a particle blinked'.
- **B10 - Security Review** (score): Authorised-pentest code review on vulnerable/patched PAIRS plus safe-but-alarming decoys. Headline is a usable-finding score = whole-chain recall x specificity, because sensitivity is ~100% for every model and the real discriminator is not inventing defects in already-fixed code. Includes a hard tier of multi-defect chains graded on how much of the chain is found.
- **B11 - Tool Loop** (%): Can the model actually DRIVE a harness: emit a structured tool call, read the result, act on it. The client advertises schemas and owns execution, because llama.cpp's --tools never tells the model the tools exist. Scored from the filesystem, so narrating a command you never ran scores zero.

## Scorecard

Higher is better in every column. Units differ per battery (see above): B1 is /10, B5 is tokens/sec, the rest are percentages unless noted.

| Model | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | B11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `abl-gemma-4-31b` | 6.0 | 80% | 21% | 60% | 56 t/s | 100% | 89% | 95% | 71% | 23 | 100% |
| `abl-opus-35b-a3b` | 5.3 | 97% | 31% | 99% | 227 t/s | 87% | 80% | 94% | 54% | 50 | 100% |
| `abl-qwen3.6-27b` | not run | 100% | 49% | 40% | 69 t/s | 97% | 97% | 93% | 83% | 69 | 100% |
| `agents-a1-35b` | 6.7 | 100% | 62% | 19% | 208 t/s | 97% | 62% | 91% | 71% | 61 | 100% |
| `bonsai-ternary-27b` | 6.8 | 100% | 74% | 37% | 110 t/s | 93% | 72% | 86% | 46% | 44 | 100% |
| `gemma-4-26b-a4b` | 7.2 | 97% | 62% | 43% | 203 t/s | 87% | 94% | 76% | 71% | 50 | 92% |
| `gemma-4-31b-dense` | 7.4 | 100% | 77% | 68% | 57 t/s | 93% | 97% | 96% | 79% | 42 | 100% |
| `glm-4.5-air` | 6.3 | 100% | 38% | 74% | 92 t/s | 97% | 98% | 87% | 58% | 33 | 75% |
| `gpt-oss-120b` | 6.6 | 81% | 38% | 100% | 177 t/s | 73% | 86% | 90% | 62% | 67 | 92% |
| `gpt-oss-20b` | 6.0 | 90% | 33% | 95% | 244 t/s | 83% | 83% | 87% | 58% | 69 | 100% |
| `granite-4.1-30b` | 6.4 | 100% | 51% | 100% | 61 t/s | 100% | 91% | 69% | 42% | 25 | 75% |
| `laguna-s-2.1` | 6.1 | 90% | 44% | 100% | 134 t/s | 100% | 92% | 82% | 38% | 83 | 100% |
| `llama-4-scout` | 5.6 | 33% | 56% | 87% | 85 t/s | 90% | 97% | not run | 38% | 50 | 0% |
| `nemotron-3-nano-30b` | 5.9 | 100% | 33% | 95% | 292 t/s | 83% | 88% | 88% | 4% | 61 | 83% |
| `ornith-1.0-35b` | 7.4 | 100% | 69% | 38% | 210 t/s | 93% | 92% | 91% | 62% | 39 | 100% |
| `ornith-1.0-9b` | 6.7 | 93% | 74% | 24% | 130 t/s | 97% | 81% | 87% | 46% | 65 | 75% |
| `qwen3-235b` | 7.2 | 100% | 79% | 100% | 58 t/s | 100% | 94% | not run | not run | not run | not run |
| `qwen3-coder-30b` | 5.0 | 80% | 62% | 100% | 207 t/s | 100% | 97% | 73% | 67% | 6 | 100% |
| `qwen3.6-27b-dense` | 7.6 | 100% | 72% | 40% | 59 t/s | 100% | 80% | 95% | 58% | 21 | 100% |
| `qwen3.6-27b-fable-fusion` | 7.0 | 100% | 67% | 19% | 69 t/s | 97% | 100% | 90% | 71% | 62 | 100% |
| `qwen3.6-35b-a3b` | 7.3 | 100% | 69% | 42% | 226 t/s | 83% | 83% | 88% | 54% | 38 | 100% |

## Running a model

All commands use **llama.cpp** (`llama-server`), which exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1/chat/completions`.

### The two modes

**Normal** - plain decoding. Use when generating fresh text with little overlap with the prompt.

**n-gram speculative decoding** - adds `--spec-type ngram-mod --spec-ngram-mod-n-match 32`. It drafts tokens by matching n-grams already in the context, so it is fastest exactly when the output repeats the input.

It is **lossless**: at temperature 0 the output is byte-identical to normal decoding. It costs **no extra VRAM**. There is no quality tradeoff to weigh - the only question is whether your workload benefits.

| Workload | Typical speedup |
|---|---|
| Editing / rewriting a file (output largely repeats input) | **2x - 12x** |
| Refactoring, applying a diff, reformatting | 3x - 8x |
| Writing new code from scratch | ~1.0x - 1.6x |
| Free-form prose with no context overlap | ~1.0x (no harm) |

**Turn it on by default for coding and editing work.** For from-scratch generation it neither helps much nor hurts.

#### How this relates to the B5 column - read this before quoting throughput

**Every B5 number in the scorecard is an UNACCELERATED baseline.** It is reported from the spec-decode OFF arm, deliberately, so all 21 models are the same measurement.

The suite also runs an n-gram ON arm, and for the 20 models measured before 2026-08-11 that arm returned a speedup of exactly **1.00x** across the board - 59.3 vs 59.5, 264.1 vs 264.3, 60.3 vs 60.3, and so on. That is not a result about n-gram. It is the flag missing at serve time: the row recorded `spec=ngram32` in its condition while the server ran without it. It was previously explained away as 'this arm generates fresh text, where n-gram cannot help'. `qwen3.6-27b-fable-fusion` disproved that on the same battery, scoring **6.79x (482 vs 71 t/s)** once the flag actually applied.

Practical consequence for choosing a model:

- The B5 ranking between models is still sound - all 21 were measured the same (unaccelerated) way.
- The ABSOLUTE numbers understate what you will see on edit-heavy work. A model listed at 70 t/s can serve an edit workload several times faster with the n-gram flags above.
- Do NOT read the 1.00x arm as evidence that speculative decoding is not worth enabling. The standalone measurements in the table above, and the one B5 run where the flag really applied, both say the opposite.

#### Tuning `--spec-ngram-mod-n-match`

| n-match | Measured speedup | Note |
|---|---|---|
| 8 | SLOWER | SLOWER than no spec-decode - never go below 16 |
| 16 | 3.90x | floor |
| 24 | 4.98x | default |
| 32 | 5.46x | best for edit-heavy work |
| 48 | 5.40x | plateau |

**Never set it below 16** - at 8 it runs slower than no speculative decoding at all. 32 is the best measured value for edit-heavy work.

#### Measured per-model, edit-heavy workload

| Model | Normal (t/s) | With n-gram (t/s) | Speedup |
|---|---|---|---|
| `granite-4.1-30b` | 31.9 | 385.8 | **12.09x** |
| `qwen3.6-27b-dense` | 31.0 | 273.0 | **8.81x** |
| `gemma-4-26b-a4b` | 125.0 | 682.0 | **5.46x** |
| `gpt-oss-20b` | 153.0 | 655.0 | **4.28x** |
| `ornith-1.0-35b` | 153.0 | 596.0 | **3.90x** |
| `nemotron-3-nano-30b` | 194.5 | 621.6 | **3.20x** |
| `qwen3.6-35b-a3b` | 174.0 | 340.0 | **1.95x** |

Note the pattern: the SLOWEST models gain the most, because each accepted draft token saves a full forward pass. A slow dense model can end up faster than a fast MoE once n-gram is on.

## Per-model reference

> bonsai-ternary-27b ONLY: its Q2_0 is a prism-ml custom quantisation. Stock llama.cpp exits with 'failed to load model'. Use the prism fork (prebuilt Windows binary, or build the Docker image from deploy/blackwell/Dockerfile.prism).

### `abl-gemma-4-31b`

- **HF repo**: `huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-GGUF`
- **Quant file**: `Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf`
- **Weights**: 18.7 GB  (needs roughly 18.7 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: apache-2.0  |  **Claimed context**: 131072
- **Scores**: B1 6.0, B2 80%, B3 21%, B4 60%, B5 56 t/s, B6 100%, B7 89%, B8 95%, B9 71%, B10 23, B11 100%

Download:

```bash
huggingface-cli download huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-GGUF Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf --local-dir ./models/abl-gemma-4-31b
```

Run (normal):

```bash
llama-server -m ./models/abl-gemma-4-31b/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/abl-gemma-4-31b/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `abl-opus-35b-a3b`

- **HF repo**: `huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF`
- **Quant file**: `Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf`
- **Weights**: 17.2 GB  (needs roughly 17.2 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 5.3, B2 97%, B3 31%, B4 99%, B5 227 t/s, B6 87%, B7 80%, B8 94%, B9 54%, B10 50, B11 100%

Download:

```bash
huggingface-cli download huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf --local-dir ./models/abl-opus-35b-a3b
```

Run (normal):

```bash
llama-server -m ./models/abl-opus-35b-a3b/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/abl-opus-35b-a3b/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `abl-qwen3.6-27b`

- **HF repo**: `huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF`
- **Quant file**: `Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf`
- **Weights**: 16.8 GB  (needs roughly 16.8 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 not run, B2 100%, B3 49%, B4 40%, B5 69 t/s, B6 97%, B7 97%, B8 93%, B9 83%, B10 69, B11 100%

Download:

```bash
huggingface-cli download huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf --local-dir ./models/abl-qwen3.6-27b
```

Run (normal):

```bash
llama-server -m ./models/abl-qwen3.6-27b/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/abl-qwen3.6-27b/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `agents-a1-35b`

- **HF repo**: `jashepp/Agents-A1-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF`
- **Quant file**: `Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf`
- **Weights**: 18.4 GB  (needs roughly 18.4 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: MXFP4_MOE  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 6.7, B2 100%, B3 62%, B4 19%, B5 208 t/s, B6 97%, B7 62%, B8 91%, B9 71%, B10 61, B11 100%

Download:

```bash
huggingface-cli download jashepp/Agents-A1-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf --local-dir ./models/agents-a1-35b
```

Run (normal):

```bash
llama-server -m ./models/agents-a1-35b/Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/agents-a1-35b/Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `bonsai-ternary-27b`

- **HF repo**: `prism-ml/Ternary-Bonsai-27B-gguf`
- **Quant file**: `Ternary-Bonsai-27B-Q2_0.gguf`
- **Weights**: 6.7 GB  (needs roughly 6.7 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: TERNARY  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 6.8, B2 100%, B3 74%, B4 37%, B5 110 t/s, B6 93%, B7 72%, B8 86%, B9 46%, B10 44, B11 100%

Download:

```bash
huggingface-cli download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf --local-dir ./models/bonsai-ternary-27b
```

Run (normal):

```bash
llama-server -m ./models/bonsai-ternary-27b/Ternary-Bonsai-27B-Q2_0.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080  # prism fork build
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/bonsai-ternary-27b/Ternary-Bonsai-27B-Q2_0.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080  # prism fork build
```


### `gemma-4-26b-a4b`

- **HF repo**: `unsloth/gemma-4-26B-A4B-it-qat-GGUF`
- **Quant file**: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`
- **Weights**: 13.3 GB  (needs roughly 13.3 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: gemma  |  **Claimed context**: 262144
- **Scores**: B1 7.2, B2 97%, B3 62%, B4 43%, B5 203 t/s, B6 87%, B7 94%, B8 76%, B9 71%, B10 50, B11 92%

Download:

```bash
huggingface-cli download unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --local-dir ./models/gemma-4-26b-a4b
```

Run (normal):

```bash
llama-server -m ./models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `gemma-4-31b-dense`

- **HF repo**: `unsloth/gemma-4-31B-it-qat-GGUF`
- **Quant file**: `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`
- **Weights**: 16.5 GB  (needs roughly 16.5 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: gemma  |  **Claimed context**: 131072
- **Scores**: B1 7.4, B2 100%, B3 77%, B4 68%, B5 57 t/s, B6 93%, B7 97%, B8 96%, B9 79%, B10 42, B11 100%

Download:

```bash
huggingface-cli download unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf --local-dir ./models/gemma-4-31b-dense
```

Run (normal):

```bash
llama-server -m ./models/gemma-4-31b-dense/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/gemma-4-31b-dense/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `glm-4.5-air`

- **HF repo**: `unsloth/GLM-4.5-Air-GGUF`
- **Quant file**: `GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf`
- **Weights**: 68 GB  (needs roughly 68 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: -  |  **Claimed context**: 131072
- **Scores**: B1 6.3, B2 100%, B3 38%, B4 74%, B5 92 t/s, B6 97%, B7 98%, B8 87%, B9 58%, B10 33, B11 75%

Download:

```bash
huggingface-cli download unsloth/GLM-4.5-Air-GGUF GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf --local-dir ./models/glm-4.5-air
```

Run (normal):

```bash
llama-server -m ./models/glm-4.5-air/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/glm-4.5-air/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `gpt-oss-120b`

- **HF repo**: `unsloth/gpt-oss-120b-GGUF`
- **Quant file**: `gpt-oss-120b-F16.gguf`
- **Weights**: 61 GB  (needs roughly 61 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: -  |  **Claimed context**: 131072
- **Scores**: B1 6.6, B2 81%, B3 38%, B4 100%, B5 177 t/s, B6 73%, B7 86%, B8 90%, B9 62%, B10 67, B11 92%

Download:

```bash
huggingface-cli download unsloth/gpt-oss-120b-GGUF gpt-oss-120b-F16.gguf --local-dir ./models/gpt-oss-120b
```

Run (normal):

```bash
llama-server -m ./models/gpt-oss-120b/gpt-oss-120b-F16.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/gpt-oss-120b/gpt-oss-120b-F16.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `gpt-oss-20b`

- **HF repo**: `unsloth/gpt-oss-20b-GGUF`
- **Quant file**: `gpt-oss-20b-F16.gguf`
- **Weights**: 12.9 GB  (needs roughly 12.9 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: MXFP4_MOE  |  **License**: apache-2.0  |  **Claimed context**: 131072
- **Scores**: B1 6.0, B2 90%, B3 33%, B4 95%, B5 244 t/s, B6 83%, B7 83%, B8 87%, B9 58%, B10 69, B11 100%

Download:

```bash
huggingface-cli download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-F16.gguf --local-dir ./models/gpt-oss-20b
```

Run (normal):

```bash
llama-server -m ./models/gpt-oss-20b/gpt-oss-20b-F16.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/gpt-oss-20b/gpt-oss-20b-F16.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `granite-4.1-30b`

- **HF repo**: `unsloth/granite-4.1-30b-GGUF`
- **Quant file**: `granite-4.1-30b-UD-Q4_K_XL.gguf`
- **Weights**: 16.5 GB  (needs roughly 16.5 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: apache-2.0  |  **Claimed context**: 131072
- **Scores**: B1 6.4, B2 100%, B3 51%, B4 100%, B5 61 t/s, B6 100%, B7 91%, B8 69%, B9 42%, B10 25, B11 75%

Download:

```bash
huggingface-cli download unsloth/granite-4.1-30b-GGUF granite-4.1-30b-UD-Q4_K_XL.gguf --local-dir ./models/granite-4.1-30b
```

Run (normal):

```bash
llama-server -m ./models/granite-4.1-30b/granite-4.1-30b-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/granite-4.1-30b/granite-4.1-30b-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `laguna-s-2.1`

- **HF repo**: `unsloth/Laguna-S-2.1-GGUF`
- **Quant file**: `UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf`
- **Weights**: 57.6 GB  (needs roughly 57.6 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: IQ  |  **License**: openmdw-1.1  |  **Claimed context**: 1048576
- **Scores**: B1 6.1, B2 90%, B3 44%, B4 100%, B5 134 t/s, B6 100%, B7 92%, B8 82%, B9 38%, B10 83, B11 100%

Download:

```bash
huggingface-cli download unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf --local-dir ./models/laguna-s-2.1
```

Run (normal):

```bash
llama-server -m ./models/laguna-s-2.1/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/laguna-s-2.1/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `llama-4-scout`

- **HF repo**: `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF`
- **Quant file**: `Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf`
- **Weights**: 62 GB  (needs roughly 62 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: -  |  **Claimed context**: 131072
- **Scores**: B1 5.6, B2 33%, B3 56%, B4 87%, B5 85 t/s, B6 90%, B7 97%, B8 not run, B9 38%, B10 50, B11 0%

Download:

```bash
huggingface-cli download unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf --local-dir ./models/llama-4-scout
```

Run (normal):

```bash
llama-server -m ./models/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `nemotron-3-nano-30b`

- **HF repo**: `unsloth/Nemotron-3-Nano-30B-A3B-GGUF`
- **Quant file**: `Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf`
- **Weights**: 21.3 GB  (needs roughly 21.3 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: nvidia-open-model-license  |  **Claimed context**: 131072
- **Scores**: B1 5.9, B2 100%, B3 33%, B4 95%, B5 292 t/s, B6 83%, B7 88%, B8 88%, B9 4%, B10 61, B11 83%

Download:

```bash
huggingface-cli download unsloth/Nemotron-3-Nano-30B-A3B-GGUF Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --local-dir ./models/nemotron-3-nano-30b
```

Run (normal):

```bash
llama-server -m ./models/nemotron-3-nano-30b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/nemotron-3-nano-30b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `ornith-1.0-35b`

- **HF repo**: `jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF`
- **Quant file**: `Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf`
- **Weights**: 18.4 GB  (needs roughly 18.4 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: MXFP4_MOE  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 7.4, B2 100%, B3 69%, B4 38%, B5 210 t/s, B6 93%, B7 92%, B8 91%, B9 62%, B10 39, B11 100%

Download:

```bash
huggingface-cli download jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf --local-dir ./models/ornith-1.0-35b
```

Run (normal):

```bash
llama-server -m ./models/ornith-1.0-35b/Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/ornith-1.0-35b/Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `ornith-1.0-9b`

- **HF repo**: `jashepp/Ornith-1.0-9B-MXFP4_Hybrid-Imatrix-GGUF`
- **Quant file**: `Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf`
- **Weights**: 8.9 GB  (needs roughly 8.9 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: MXFP4  |  **License**: mit  |  **Claimed context**: 262144
- **Scores**: B1 6.7, B2 93%, B3 74%, B4 24%, B5 130 t/s, B6 97%, B7 81%, B8 87%, B9 46%, B10 65, B11 75%

Download:

```bash
huggingface-cli download jashepp/Ornith-1.0-9B-MXFP4_Hybrid-Imatrix-GGUF Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf --local-dir ./models/ornith-1.0-9b
```

Run (normal):

```bash
llama-server -m ./models/ornith-1.0-9b/Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/ornith-1.0-9b/Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `qwen3-235b`

- **HF repo**: `unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF`
- **Quant file**: `Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf`
- **Weights**: 134 GB  (needs roughly 134 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: -  |  **Claimed context**: 262144
- **Scores**: B1 7.2, B2 100%, B3 79%, B4 100%, B5 58 t/s, B6 100%, B7 94%, B8 not run, B9 not run, B10 not run, B11 not run

Download:

```bash
huggingface-cli download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf --local-dir ./models/qwen3-235b
```

Run (normal):

```bash
llama-server -m ./models/qwen3-235b/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/qwen3-235b/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `qwen3-coder-30b`

- **HF repo**: `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`
- **Quant file**: `Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf`
- **Weights**: 17.5 GB  (needs roughly 17.5 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 5.0, B2 80%, B3 62%, B4 100%, B5 207 t/s, B6 100%, B7 97%, B8 73%, B9 67%, B10 6, B11 100%

Download:

```bash
huggingface-cli download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf --local-dir ./models/qwen3-coder-30b
```

Run (normal):

```bash
llama-server -m ./models/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `qwen3.6-27b-dense`

- **HF repo**: `unsloth/Qwen3.6-27B-GGUF`
- **Quant file**: `Qwen3.6-27B-Q5_K_M.gguf`
- **Weights**: 18.2 GB  (needs roughly 18.2 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 7.6, B2 100%, B3 72%, B4 40%, B5 59 t/s, B6 100%, B7 80%, B8 95%, B9 58%, B10 21, B11 100%

Download:

```bash
huggingface-cli download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q5_K_M.gguf --local-dir ./models/qwen3.6-27b-dense
```

Run (normal):

```bash
llama-server -m ./models/qwen3.6-27b-dense/Qwen3.6-27B-Q5_K_M.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/qwen3.6-27b-dense/Qwen3.6-27B-Q5_K_M.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `qwen3.6-27b-fable-fusion`

- **HF repo**: `DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF`
- **Quant file**: `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q4_K_M.gguf`
- **Weights**: 18.0 GB  (needs roughly 18.0 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: K  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 7.0, B2 100%, B3 67%, B4 19%, B5 69 t/s, B6 97%, B7 100%, B8 90%, B9 71%, B10 62, B11 100%

Download:

```bash
huggingface-cli download DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q4_K_M.gguf --local-dir ./models/qwen3.6-27b-fable-fusion
```

Run (normal):

```bash
llama-server -m ./models/qwen3.6-27b-fable-fusion/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q4_K_M.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/qwen3.6-27b-fable-fusion/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q4_K_M.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


### `qwen3.6-35b-a3b`

- **HF repo**: `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
- **Quant file**: `Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf`
- **Weights**: 18.4 GB  (needs roughly 18.4 GB VRAM plus KV cache; quantised KV as below keeps that small)
- **Quant family**: IQ  |  **License**: apache-2.0  |  **Claimed context**: 262144
- **Scores**: B1 7.3, B2 100%, B3 69%, B4 42%, B5 226 t/s, B6 83%, B7 83%, B8 88%, B9 54%, B10 38, B11 100%

Download:

```bash
huggingface-cli download bartowski/Qwen_Qwen3.6-35B-A3B-GGUF Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf --local-dir ./models/qwen3.6-35b-a3b
```

Run (normal):

```bash
llama-server -m ./models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --host 127.0.0.1 --port 8080
```

Run (with n-gram speculative decoding):

```bash
llama-server -m ./models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf \
    -ngl 99 -c 32768 --jinja -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
    --host 127.0.0.1 --port 8080
```


## Known gaps and caveats

Read these before quoting any number.

- **[HIGH] B8 cannot be measured on a Docker-less box - five models' 0% was an artifact** - B8's completion oracle (run_oracle) validates the agent's work inside a container. vast.ai instances have no Docker, so the runs were made with b8.sandbox.enabled=false: the AGENT runs fine on the host, but the ORACLE still shells out to a container, fails setup, and completion is never credited. Every row measured that way carries oracle.detail = "hidden_validate setup failed: FileNotFoundError(...)" instead of a PASS/FAIL verdict with a stage. Scored naively this produced a flat 0% for all five models run that way - abl-gemma-4-31b, abl-opus-35b-a3b, glm-4.5-air, gpt-oss-120b and laguna-s-2.1 - including gpt-oss-120b, which every other signal says is among the strongest agentic models in the roster. Those rows are now excluded as MISSING measurements rather than failed ones, and the cells read untested.
- **[LOW] bonsai-ternary-27b: RESOLVED 2026-08-10 - all four cells now measured** - Its Q2_0 is a prism-ml custom quantization and only the prism llama.cpp fork has the kernels for it; on the official ggml image the server exited before serving. The prism image itself then failed to build, which deferred these cells twice. Root cause was never the CUDA version the Dockerfile blamed: ggml links the CUDA driver api for its VMM allocator, the devel image ships that only as a stub whose SONAME is libcuda.so.1, and nothing on the link path provides that name - so every driver symbol came back undefined. A second defect followed it: the runtime image has no libgomp1, so the binary built clean and died on first exec. Both were found and fixed locally on an RTX 5090 Laptop, which reports the same sm_120 compute capability as the PRO 6000, at zero cost. B8 86%, B9 46%, B10 44, B11 100%, with 115/115 B8 rows and no infra errors.
- **[HIGH] B5's n-gram arm never actually engaged for 20 of 21 models** - Every model measured before 2026-08-11 reports a spec-decode speedup of exactly 1.00x - abl-gemma 59.3 vs 59.5, gpt-oss-20b 264.1 vs 264.3, qwen3.6-27b-dense 60.3 vs 60.3, and so on down the roster. That is not a finding about n-gram; it is the arm recording spec=ngram32 in its condition while the server ran without the flag. qwen3.6-27b-fable-fusion, measured through a different serving path on 2026-08-11, is the first whose ngram arm genuinely fired: 482 vs 71 t/s, 6.79x, in line with the 2-12x the standalone n-gram measurements show on edit-heavy work. So B5's whole column is effectively a spec=OFF measurement, and the headline now takes the spec=off arm explicitly so all 21 numbers mean the same thing. Reading the old ngram arm as evidence that speculative decoding does not help would be exactly backwards.
- **[HIGH] abl-qwen3.6-27b cannot produce a B1 answer - it never stops thinking** - Measured 2026-08-10 on a PRO 6000: 23 of 23 B1 generations returned an EMPTY answer (chars=0, artifact sha256 = the hash of the empty string). This is not the small-budget artifact the suite already corrects for - a single probe with the full 16000-token budget spent all 16000 on reasoning_content and emitted zero characters of content, finish_reason=length. Suppressing thinking via chat_template_kwargs enable_thinking=false does not rescue it either: the model then emits degenerate output ('3333333...') for the whole budget. The model serves normally for this suite's SHORTER batteries - its B2/B3/B6/B8/B9/B10/B11 cells are all populated - so this is specific to long-form generation on this quant (huihui-ai Q4_K, an MTP GGUF served without its draft head).
- **[HIGH] qwen3-235b is held out by choice - its 4 cells are real gaps, not results** - Excluded on 2026-07-30 pending a dedicated large-model pass. At 134GB it does not fit the 96GB card, so --cpu-moe streams its experts over PCIe and every row costs roughly 8x: B8-B11 priced at ~11.3h / ~$13.54, which was 46% of the remaining budget for 14% of the remaining cells. The exclusion is recorded in scripts/build_run_manifest.py EXCLUDED and is one dict entry to reverse.
- **[HIGH] B4 has only ever run 7 of its 8 tasks - the classic single-needle probe is missing** - b4.single-needle-01 has ZERO rows for all 16 roster models; the other seven B4 tasks have 49 each. build_document() sizes the filler with a 4-chars-per-token heuristic, and that task's filler is dense operational log text (timestamps, asset IDs, digit groups) that really tokenizes at ~2.97 chars/token - a 1.35x overshoot. Every arm therefore overflows its own tier and the server rejects the request: 20716 vs 16384, 86644 vs 65536, 174563 vs 131072, 350408 vs 262144. No row is written, so the loss is invisible unless task-level completeness is checked. The missing task is the canonical 'lost in the middle' needle-in-a-haystack probe at depth 50%, which is the single most standard thing B4 claims to measure.
- **[HIGH] Coverage is ragged across the three newest batteries** - B9 (games) ran for 12 models but 4 of those have partial rows, and the four largest models plus laguna have none at all - 96 completed rows were lost when a rented box was left idle, ran out of credit and could not be restarted. B10 (security) covers 6 models of 20. B11 (tool loop) covers 1. Every blank cell in the matrix is genuinely unrun, never a zero - but the newer columns are far thinner than B1-B7 and should not be read as a roster-wide ranking yet.
- **[HIGH] Subagent delegation is deliberately unscored - and the canary never fires** - TESTPLAN 5.7 excludes the subagent axis from scoring ON PURPOSE - 'documented 0% with local models - non-differentiating' - keeping it as one unscored canary. The consequence still matters: every B8 row records subagent_spawned = 'no', so B8 is SINGLE-AGENT completion only. B11 now covers the related question (can the model drive a tool loop at all) and the answer is yes, but that is a client-owned loop, not model-initiated delegation.
- **[HIGH] B10's hard tier reverses its own base tier - so neither alone is safe to quote** - On single-defect textbook cases every model scores at or near 100% and the base tier cannot separate them. On multi-defect chains the ordering changes outright: abl-gemma-4-31b is perfect on the base tier and worst on chains (25% whole-chain), while abl-qwen3.6-27b leads. Quoting the base tier alone would have produced - and did produce - the wrong recommendation.
- **[MEDIUM] The abliteration A/Bs are confounded by quantisation** - Abliterated builds beat their bases in both families (qwen3.6-27b 25%->75% whole-chain, gemma-4-31b 66%->100% base specificity), which is the opposite of the usual assumption. But the abliterated files are Q4_K while the bases are Q5_K_M, so part of that delta could be quantisation rather than abliteration.
- **[MEDIUM] Small samples - most rankings are statistical ties** - B2/B6 run n=30 per model, B8's sweep n=69, B10's hard tier n=12 per model. Wilson intervals on whole-chain recall span roughly +/-25 points, so abl-qwen3.6-27b vs gpt-oss-120b is NOT a separated result. The matrix marks ties with a tilde.
- **[MEDIUM] Judged axes built and never run** - B6's 0-10 code-quality axis has 510 generated rows waiting and is not wired into JUDGED_BATTERIES; B2's error-recovery and faithfulness axes are wired but have never been executed. Both would add discrimination to batteries currently sitting at a ceiling, and both need judge quota rather than GPU.
- **[MEDIUM] Games are scored for 'runs clean', not for being good games** - The browser oracle can prove a build loads, paints, animates, wires keys and survives input. It CANNOT tell that the snake advanced - validated the hard way: a frozen snake whose particle layer animates changes more of the board than a working one. Gameplay quality is therefore human-graded in the explorer.
- **[MEDIUM] The suite under-reports speculative decoding** - B5's spec-decode arm measures ~1.00x for every model because it generates fresh text, where n-gram drafting almost never hits. On edit/rewrite work - what agentic coding actually does - the same feature is worth 1.95x to 12.08x. Laguna also ran with no acceleration at all, and its purpose-built DFlash draft could not be loaded by upstream llama.cpp (wrong tensor count - it needs poolside's fork).
- **[LOW] The quant-format A/B was never actually run** - gemma-4-26b-a4b-mxfp4 exists as a controlled quant arm ('runs B5 + B2 + B6') but produced B8 rows only. Worse, that B8 data is what the roster model's agentic score uses, so gemma-4-26b-a4b's B8 is the MXFP4 quant while its B1-B7 are UD-Q4_K_XL - one row mixing two quants.
- **[LOW] Judges agree with each other only 35% of the time** - Across 6,120 judged answers the 3-seat panel lands within 1 point of itself on just 35.1%; mean spread is 2.45 points. A B1 gap of a few tenths is inside judge noise. Gemini scores its own family +0.53 higher than others; codex scores its own -0.67.
- **[LOW] Laguna has no B5 / B7 / B8, and its B1 is a rescaled incremental wave** - B5 and B7 were skipped (box-specific / needs the fork's spec arm) and B8 postdates its peer group. Its B1 6.1 comes from a 3-letter incremental packet rescaled through the CAL anchors rather than a full-roster packet - defensible, one step further from the frozen sixteen.

### Methodology notes

- **hardware**: ONE hardware standard: RTX PRO 6000 (Blackwell), one model per card. Hardware is NOT interchangeable - re-running one model on an A100 (Ampere) moved its deterministic scores by up to 13 points (B6 87 -> 100, B2 97 -> 90) at temperature 0, because batching and GPU numerics shift borderline outputs. A 2026-08-03 provenance audit found cells measured elsewhere (an RTX 5090 Laptop; one rented RTX 5090 32GB session) and every one of them has been WITHDRAWN from this page pending PRO-6000 re-runs - they show as 'not run', never as a number from the wrong machine. The withdrawal list is committed in config/superseded.yaml; replacements land under suite-v2.2.0.
- **ngram_workload**: n-gram speculative decoding is lossless (temp-0 output is byte-identical) and costs no VRAM, but its speedup depends entirely on how much the output repeats the context. Edit/rewrite: 2-12x. Fresh generation: ~1.0-1.6x.
- **b1_incremental**: Models added after the frozen 16 are judged INCREMENTALLY - only the new model plus the two calibration anchors, leaving the frozen 16 untouched, because re-judging the whole roster to add one model costs the entire panel again. A small packet is a more lenient instrument, and the anchors measure exactly how much: the same two fixed answers score 7.62/0.89 in the 18-letter cohort, 8.43/1.36 in a 4-letter one and 8.74/1.36 in a 3-letter one. Every incremental score on this page is therefore mapped back onto the frozen scale through those two anchors, overall and per department, so a newcomer is never ranked against the 16 on a looser ruler. Uncorrected, the abliterated pair would have read about six ranks too high. Laguna keeps its originally published 6.1, which used rounded anchor constants; measured anchors put it at 6.0. See scripts/b1_rescale.py.

