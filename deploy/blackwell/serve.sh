#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve one model with llama.cpp CUDA, on the HOST network so the OpenCode
# containers reach it at host.docker.internal:8080 (host-gateway) exactly as
# the OpenCodeAdapter already assumes -- no adapter change needed on Linux.
#
#     bash serve.sh <stable-model-name.gguf>     # e.g. gpt-oss-20b.gguf
#     bash serve.sh stop                          # stop + remove the server
#
# THE KV-EXHAUSTION FIX (why this differs from the CLAUDE.md template):
#   --ctx-checkpoints 0  -> disable the per-slot context-checkpoint cache that
#                           accumulated ~13k-token OpenCode prompts across runs
#                           until the KV cache OOM'd and the server crashed
#                           (GGML_ASSERT logits!=nullptr). This was THE root
#                           cause of the escalating infra-errors on the local
#                           Windows box.
#   --parallel 1         -> single slot; no cross-request KV contention.
# ngram spec-decode is intentionally ABSENT: it's a prism-fork-only flag and
# only affects DECODE SPEED, never completion/oracle outcome, so the ranking
# is valid without it (just slower). Build the prism fork on-box if you want
# the 2-9x edit-decode speedup to cut GPU-hours.
# ---------------------------------------------------------------------------
set -euo pipefail
B8_ROOT="${B8_ROOT:-/opt/b8}"
MODELS="$B8_ROOT/models"
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
NAME=llama-b8
CTX="${CTX:-40960}"          # OpenCode needs max_tokens 32000 -> ctx >= ~40k
PORT="${PORT:-8080}"

if [ "${1:-}" = "stop" ]; then
  docker rm -f "$NAME" 2>/dev/null || true; echo "stopped $NAME"; exit 0
fi
GGUF="${1:?usage: serve.sh <model.gguf> | stop}"

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --network host \
    -v "$MODELS":/models:ro \
    "$LLAMA_IMAGE" \
    --model "/models/$GGUF" -ngl 99 -c "$CTX" --jinja -fa on \
    --ctx-checkpoints 0 --parallel 1 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 0.0.0.0 --port "$PORT" >/dev/null

echo -n "waiting for $GGUF to load"
for i in $(seq 1 120); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo " -> healthy (~$((i*2))s)"; exit 0
  fi
  sleep 2; echo -n "."
done
echo " !! TIMEOUT -- last server log:"; docker logs --tail 30 "$NAME"; exit 1
