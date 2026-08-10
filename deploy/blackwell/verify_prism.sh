#!/usr/bin/env bash
# Prove a built prism-llama image can actually serve bonsai BEFORE renting a box.
#
# WHY THIS EXISTS: bonsai-ternary-27b is the only model that needs this image at
# all -- its Q2_0 is a prism custom quantization the official ggml image refuses
# outright ("failed to load model", measured 2026-07-30, 14 minutes of a rented
# box). The image build itself then failed on the Korea box and the four bonsai
# cells have been deferred ever since. Renting a GPU to discover the image is
# still broken costs money; this script costs nothing.
#
# The four checks, in the order a failure would bite:
#   1. the image exists and the binary runs at all
#   2. CUDA is compiled in AND a device is visible
#   3. the model LOADS (the thing the official image cannot do)
#   4. layers actually went to the GPU and a completion comes back
#
# Check 4 is not redundant with 3. The Dockerfile pins
# -DCMAKE_CUDA_ARCHITECTURES=120 precisely because a mismatched arch produces a
# binary that loads and silently runs on the CPU: the model answers, slowly, and
# the run reads as "this model is slow" instead of "this build is wrong". A
# 27B ternary model on a Blackwell card decodes at ~68 t/s; single-digit t/s
# means the GPU never got the layers.
#
#     bash deploy/blackwell/verify_prism.sh [gguf_dir] [gguf] [image]
set -uo pipefail
DIR="${1:-/d/BUILT-TOOLS/LLMtesting/bonsai}"
GGUF="${2:-Ternary-Bonsai-27B-Q2_0.gguf}"
IMAGE="${3:-prism-llama:1}"
PORT="${PORT:-8099}"
NAME="prism-verify-$PORT"
# Git Bash rewrites POSIX-looking argv into Windows paths, so --entrypoint
# /app/llama-server reaches docker as "C:/Program Files/Git/app/llama-server".
# Harmless on the Linux box this ships to; fatal when verifying here.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'
FAIL=0
say(){ printf '\n=== %s ===\n' "$*"; }
ok(){ printf '  PASS  %s\n' "$*"; }
no(){ printf '  FAIL  %s\n' "$*"; FAIL=1; }

say "1. image + binary"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  no "image $IMAGE not present"; exit 1
fi
ok "image $IMAGE present ($(docker image inspect "$IMAGE" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}'))"
# --gpus all even for --version: the binary links libcuda.so.1, which only exists
# once the NVIDIA container runtime injects the real driver. Without it this
# reports a missing-library error that looks like a broken build.
VER=$(docker run --rm --gpus all --entrypoint /app/llama-server "$IMAGE" --version 2>&1 | head -3)
echo "$VER" | sed 's/^/      /'

say "2. CUDA compiled in, device visible"
DEVS=$(docker run --rm --gpus all --entrypoint /app/llama-server "$IMAGE" --list-devices 2>&1 | tail -8)
echo "$DEVS" | sed 's/^/      /'
echo "$DEVS" | grep -qi "CUDA" && ok "CUDA device enumerated" || no "no CUDA device in --list-devices"

say "3. model load"
if [ ! -f "$DIR/$GGUF" ]; then no "missing $DIR/$GGUF"; exit 1; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all -p "$PORT:$PORT" -v "$DIR":/models:ro "$IMAGE" \
  --model "/models/$GGUF" -ngl 99 -c 4096 --jinja -fa on \
  --host 0.0.0.0 --port "$PORT" >/dev/null
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  ok "model loaded and /health is up"
else
  no "model never became healthy - last 25 log lines:"
  docker logs "$NAME" 2>&1 | tail -25 | sed 's/^/      /'
  docker rm -f "$NAME" >/dev/null 2>&1; exit 1
fi

say "4. GPU offload + a real completion"
docker logs "$NAME" 2>&1 | grep -Ei "offload|CUDA[0-9]|buffer size" | tail -6 | sed 's/^/      /'
OFF=$(docker logs "$NAME" 2>&1 | grep -Eo "offloaded [0-9]+/[0-9]+ layers" | tail -1)
[ -n "$OFF" ] && ok "$OFF" || echo "      (no explicit offload line; relying on t/s below)"
RESP=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: BONSAI OK"}],"max_tokens":512,"temperature":0}')
python - "$RESP" <<'PY'
import json,sys
try: d=json.loads(sys.argv[1])
except Exception as e: print(f"  FAIL  unparseable reply: {e}"); sys.exit(1)
txt=(d.get("choices") or [{}])[0].get("message",{}).get("content","")
t=d.get("timings") or {}
tps=t.get("predicted_per_second")
fin=(d.get("choices") or [{}])[0].get("finish_reason")
print(f"      reply: {txt.strip()[:60]!r}  finish={fin}  predicted_n={t.get('predicted_n')}")
print(f"      decode: {tps:.1f} t/s" if tps else "      decode: (no timings)")
if not txt.strip():
    # Bonsai is a reasoning model: a small max_tokens is spent entirely on hidden
    # thinking and returns content:"" with finish_reason:"length". That is a
    # starved budget, not a broken image - measured 24 tokens -> empty, 512 -> "BONSAI OK".
    hint = " (budget spent on reasoning - raise max_tokens)" if fin == "length" else ""
    print(f"  FAIL  empty completion{hint}"); sys.exit(1)
print("  PASS  completion returned")
# A 27B ternary on Blackwell decodes ~68 t/s. Single digits = CPU fallback,
# which is the exact failure a wrong -DCMAKE_CUDA_ARCHITECTURES produces.
if tps is not None and tps < 20:
    print(f"  FAIL  {tps:.1f} t/s is CPU-class - GPU offload did NOT happen"); sys.exit(1)
if tps is not None: print(f"  PASS  {tps:.1f} t/s is GPU-class")
PY
[ $? -ne 0 ] && FAIL=1

docker rm -f "$NAME" >/dev/null 2>&1 || true
say "RESULT"
[ "$FAIL" -eq 0 ] && echo "  ALL CHECKS PASSED - image is fit to ship to a rented box" \
                  || echo "  FAILURES ABOVE - do NOT rent until resolved"
exit $FAIL
