#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download the two B8 model GGUFs into $B8_ROOT/models and symlink each to a
# STABLE name (gpt-oss-20b.gguf, gemma-4-26b-a4b.gguf) that serve.sh expects.
# Datacenter HF pull is fast (~200+ MB/s) -- but PROBE first (a slow host has
# broken HF throughput ~4 MB/s; see project memory). ~27 GB total.
#
# Model choices (both natively MXFP4, the local FP4 llama.cpp reads directly):
#   gpt-oss-20b : official openai/gpt-oss-20b  (matches Ollama's gpt-oss:20b)
#   gemma-4-26b : unsloth/gemma-4-26B-A4B-it-qat-GGUF (canonical, 493K dl)
# Override any REPO/PATTERN below via env if you want a different quant.
# ---------------------------------------------------------------------------
set -euo pipefail
B8_ROOT="${B8_ROOT:-/opt/b8}"
MODELS="$B8_ROOT/models"
HF="${HF:-$B8_ROOT/venv/bin/hf}"   # huggingface_hub 1.x CLI (huggingface-cli download is deprecated -> prints a hint, downloads nothing)
mkdir -p "$MODELS"

GPT_OSS_REPO="${GPT_OSS_REPO:-ggml-org/gpt-oss-20b-GGUF}"   # GGUF repo (llama.cpp org); openai/gpt-oss-20b is safetensors, NOT gguf
GPT_OSS_PATTERN="${GPT_OSS_PATTERN:-*.gguf}"     # single file gpt-oss-20b-MXFP4.gguf (hf --include is CASE-SENSITIVE, so *mxfp4* misses it)
GEMMA_REPO="${GEMMA_REPO:-unsloth/gemma-4-26B-A4B-it-qat-GGUF}"
GEMMA_PATTERN="${GEMMA_PATTERN:-gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"   # the MAIN model (UD-Q4_K_XL); repo also has MTP-draft + mmproj .gguf that must NOT be picked

# link_largest <subdir> <stable-name>: point <stable-name> at the biggest .gguf
link_largest () {
  local dir="$1" name="$2"
  local big
  big="$(find "$dir" -name '*.gguf' -printf '%s\t%p\n' | sort -rn | head -1 | cut -f2)"
  [ -n "$big" ] || { echo "!! no .gguf found under $dir" >&2; return 1; }
  ln -sf "$big" "$MODELS/$name"
  echo "   $name -> $big ($(du -h "$big" | cut -f1))"
}

echo "== gpt-oss-20b  ($GPT_OSS_REPO :: $GPT_OSS_PATTERN) =="
"$HF" download "$GPT_OSS_REPO" --include "$GPT_OSS_PATTERN" \
    --local-dir "$MODELS/gpt-oss-20b-src" || \
  "$HF" download "$GPT_OSS_REPO" --include "*.gguf" --local-dir "$MODELS/gpt-oss-20b-src"
link_largest "$MODELS/gpt-oss-20b-src" gpt-oss-20b.gguf

echo "== gemma-4-26b-a4b  ($GEMMA_REPO :: $GEMMA_PATTERN) =="
"$HF" download "$GEMMA_REPO" --include "$GEMMA_PATTERN" \
    --local-dir "$MODELS/gemma-4-26b-src" || \
  "$HF" download "$GEMMA_REPO" --include "*UD-Q4*.gguf" --local-dir "$MODELS/gemma-4-26b-src"  # fallback: a UD quant, NOT the MTP/mmproj .gguf
link_largest "$MODELS/gemma-4-26b-src" gemma-4-26b-a4b.gguf

echo "models ready:"; ls -lL "$MODELS"/*.gguf
