#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve one model with llama.cpp CUDA on the HOST network so OpenCode
# containers reach it at host.docker.internal:<port> (host-gateway), exactly
# as the OpenCodeAdapter assumes -- no adapter change needed on Linux.
#
#     bash serve.sh <model.gguf> [gpu] [port]     # gpu: "all"|0|1 ; port: 8080
#     bash serve.sh stop                           # stop+remove ALL b8 servers
#
# THE KV-EXHAUSTION FIX (why this differs from the CLAUDE.md template):
#   --ctx-checkpoints 0  -> disable the per-slot context-checkpoint cache that
#                           accumulated OpenCode's ~13k-token prompts across
#                           runs until the KV OOM'd and the server CRASHED
#                           (GGML_ASSERT logits!=nullptr) -- the local root cause.
#   --parallel 1         -> single slot; no cross-request KV contention.
# ngram spec-decode is intentionally ABSENT (prism-fork-only flag; affects
# DECODE SPEED only, never completion/oracle outcome -> ranking stays valid).
# ---------------------------------------------------------------------------
set -euo pipefail
B8_ROOT="${B8_ROOT:-/opt/b8}"
MODELS="$B8_ROOT/models"
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
CTX="${CTX:-40960}"          # OpenCode needs max_tokens 32000 -> ctx >= ~40k

if [ "${1:-}" = "stop" ]; then
  docker ps -a --filter 'name=llama-b8-' -q | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "stopped all llama-b8-* servers"; exit 0
fi
GGUF="${1:?usage: serve.sh <model.gguf> [gpu] [port] | stop}"
GPU="${2:-all}"              # "all" | "0" | "1"
PORT="${3:-8080}"
NAME="llama-b8-$PORT"
if [ "$GPU" = "all" ]; then GPUFLAG=(--gpus all); else GPUFLAG=(--gpus "device=$GPU"); fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
# EXTRA_FLAGS: capability-gated additions from the caller (the campaign runner sets
# "--spec-type ngram-mod --spec-ngram-mod-n-match 32" iff the image's --help advertises
# ngram-mod - it landed upstream, the old "prism-fork-only" note here was stale). Lossless
# at temp 0; affects decode SPEED only, never completion/oracle outcome.
# shellcheck disable=SC2086
docker run -d --name "$NAME" "${GPUFLAG[@]}" --network host \
    -v "$MODELS":/models:ro \
    "$LLAMA_IMAGE" \
    --model "/models/$GGUF" -ngl 99 -c "$CTX" --jinja -fa on \
    --ctx-checkpoints 0 --parallel 1 \
    --cache-type-k q8_0 --cache-type-v q8_0 ${EXTRA_FLAGS:-} \
    --host 0.0.0.0 --port "$PORT" >/dev/null

echo -n "waiting for $GGUF (gpu=$GPU port=$PORT) to load"
for i in $(seq 1 150); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo " -> healthy (~$((i*2))s)"; exit 0
  fi
  sleep 2; echo -n "."
done
echo " !! TIMEOUT -- last server log:"; docker logs --tail 30 "$NAME"; exit 1
