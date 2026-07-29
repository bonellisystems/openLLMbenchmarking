#!/bin/bash
# Fetch every model that still has gaps.
#
# Files are pulled CONCURRENTLY because Hugging Face throttles a SINGLE transfer
# (measured: -x16 collapsing to one connection at ~21MB/s); parallelism has to be
# across files, where the same box reaches ~80MB/s aggregate.
#
# But concurrency is CAPPED. Firing all 26 files at -x16 would open ~416
# connections at once, which HF rate-limits, and it would also fill the disk in
# nondeterministic order so a truncated download could belong to any model. JOBS
# files at a time keeps the aggregate rate without either problem.
set -u
M=/root/models
JOBS=6
mkdir -p $M
get(){ # dir repo path
  mkdir -p "$M/$1"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
    --retry-wait=5 --max-tries=5 --auto-file-renaming=false \
    -d "$M/$1" -o "$(basename "$3")" \
    "https://huggingface.co/$2/resolve/main/$3" >> "/root/dl_$1.log" 2>&1 \
    || echo "FAIL $1 $3" >> /root/dl_fail
}
gate(){ while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done; }
gate; get abl-qwen3.6-27b huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf &
gate; get agents-a1-35b jashepp/Agents-A1-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf &
gate; get bonsai-ternary-27b prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf &
gate; get gemma-4-26b-a4b unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf &
gate; get gemma-4-31b-dense unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf &
gate; get glm-4.5-air unsloth/GLM-4.5-Air-GGUF UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf &
gate; get glm-4.5-air unsloth/GLM-4.5-Air-GGUF UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00002-of-00002.gguf &
gate; get gpt-oss-120b unsloth/gpt-oss-120b-GGUF gpt-oss-120b-F16.gguf &
gate; get gpt-oss-20b unsloth/gpt-oss-20b-GGUF gpt-oss-20b-F16.gguf &
gate; get granite-4.1-30b unsloth/granite-4.1-30b-GGUF granite-4.1-30b-UD-Q4_K_XL.gguf &
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf &
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00002-of-00003.gguf &
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00003-of-00003.gguf &
gate; get llama-4-scout unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf &
gate; get llama-4-scout unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00002-of-00002.gguf &
gate; get nemotron-3-nano-30b unsloth/Nemotron-3-Nano-30B-A3B-GGUF Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf &
gate; get ornith-1.0-35b jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf &
gate; get ornith-1.0-9b jashepp/Ornith-1.0-9B-MXFP4_Hybrid-Imatrix-GGUF Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf &
gate; get qwen3-235b unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q4_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf &
gate; get qwen3-235b unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q4_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00002-of-00003.gguf &
gate; get qwen3-235b unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q4_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00003-of-00003.gguf &
gate; get qwen3-coder-30b unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf &
gate; get qwen3.6-27b-dense unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q5_K_M.gguf &
gate; get qwen3.6-35b-a3b bartowski/Qwen_Qwen3.6-35B-A3B-GGUF Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf &
gate; get abl-gemma-4-31b huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-GGUF Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf &
gate; get abl-opus-35b-a3b huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf &
wait
echo DL_DONE > /root/dl_done
df -h /root | tail -1
du -sh $M
# Report anything that failed AND anything suspiciously small, so a truncated shard is
# caught here rather than as a "wrong number of tensors" load error an hour later.
echo "--- download failures ---"; cat /root/dl_fail 2>/dev/null || echo none
echo "--- files under 100MB (suspect) ---"
find $M -name '*.gguf' -size -100M -printf '%p %s\n' 2>/dev/null || true
