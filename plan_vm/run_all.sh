#!/bin/bash
# Hardware-consistency campaign: every wrong-hardware cell re-measured on THIS card.
set -u
export B8_ROOT=/opt/b8
REPO=$B8_ROOT/llmtest-v2
PY=$B8_ROOT/venv/bin/python
OUT=$B8_ROOT/out
M=$B8_ROOT/models
EP="--endpoint-url http://127.0.0.1:8080"
mkdir -p "$OUT" "$M"
cd "$REPO"
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $B8_ROOT/run.log; }

NGRAM=0; grep -q '^ngram=1' $B8_ROOT/caps 2>/dev/null && NGRAM=1
EXTRA_FLAGS=""
[ "$NGRAM" = "1" ] && EXTRA_FLAGS="--spec-type ngram-mod --spec-ngram-mod-n-match 32"
export EXTRA_FLAGS

run_step(){ # $1 model  $2 battery  $3.. command  (real exit codes -> $B8_ROOT/steps)
  mid="$1"; bat="$2"; shift 2
  log "  $mid $bat start"
  "$@" > $B8_ROOT/last_step.log 2>&1
  rc=$?
  tail -3 $B8_ROOT/last_step.log | tee -a $B8_ROOT/run.log
  if [ "$rc" -eq 0 ]; then echo "$mid $bat ok" >> $B8_ROOT/steps
  else log "  $mid $bat FAILED rc=$rc"; echo "$mid $bat fail rc=$rc" >> $B8_ROOT/steps
       cat $B8_ROOT/last_step.log >> $B8_ROOT/step_failures.log; fi
  return $rc
}

JOBS=4
gate(){ while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done; }
get(){ # dir repo path
  mkdir -p "$M/$1"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
    --retry-wait=5 --max-tries=5 --auto-file-renaming=false \
    -d "$M/$1" -o "$(basename "$3")" \
    "https://huggingface.co/$2/resolve/main/$3" >> "$B8_ROOT/dl_$1.log" 2>&1 \
    || echo "FAIL $1 $3" >> $B8_ROOT/dl_fail
}
release(){ du -sh "$M/$1" 2>/dev/null | tee -a $B8_ROOT/run.log; rm -rf "${M:?}/$1"
           log "  released $1 ; free: $(df -h $B8_ROOT | awk 'NR==2{print $4}')"; }
serve_model(){ # $1 gguf-relpath  $2 image
  LLAMA_IMAGE="$2" bash deploy/blackwell/serve.sh "$1"
}
stop_server(){ bash deploy/blackwell/serve.sh stop; }

# --- HF throughput probe: die loudly on a host HF serves at ~4MB/s -----------------
MIN_MBPS=25
rm -f /tmp/probe.bin
timeout 60 aria2c -x8 -s8 -k1M --file-allocation=none --console-log-level=error \
  -d /tmp -o probe.bin \
  "https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_0.gguf" >/dev/null 2>&1 || true
PSZ=$(stat -c %s /tmp/probe.bin 2>/dev/null || echo 0); rm -f /tmp/probe.bin
RATE=$(( PSZ / 60 / 1000000 ))
log "HF throughput probe: ~${RATE} MB/s (floor ${MIN_MBPS})"
if [ "$RATE" -lt "$MIN_MBPS" ]; then
  log "ABORT: HF too slow on this host - destroy the box, rent another."
  echo "DL_ABORT rate=${RATE}" > $B8_ROOT/dl_abort; exit 1
fi
echo PROBE_OK > $B8_ROOT/dl_done

# ---------- abl-opus-35b-a3b : B8 (17.2 GB) ----------
log "===== abl-opus-35b-a3b : fetching 17.2 GB ====="
gate; get abl-opus-35b-a3b huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf &
wait
GG="abl-opus-35b-a3b/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "abl-opus-35b-a3b" B8 "$PY" scripts/run_b8_local.py $EP --model "abl-opus-35b-a3b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_abl-opus-35b-a3b
    echo "abl-opus-35b-a3b" >> $B8_ROOT/models_done
    log "abl-opus-35b-a3b complete"
  else
    log "abl-opus-35b-a3b SERVE-FAIL"; echo "abl-opus-35b-a3b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "abl-opus-35b-a3b SKIP (fetch failed)"; echo "abl-opus-35b-a3b missing-gguf" >> $B8_ROOT/failures
fi
release "abl-opus-35b-a3b"


# ---------- laguna-s-2.1 : B8 (57.6 GB) ----------
log "===== laguna-s-2.1 : fetching 57.6 GB ====="
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf &
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00002-of-00003.gguf &
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00003-of-00003.gguf &
wait
GG="laguna-s-2.1/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "laguna-s-2.1" B8 "$PY" scripts/run_b8_local.py $EP --model "laguna-s-2.1" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_laguna-s-2.1
    echo "laguna-s-2.1" >> $B8_ROOT/models_done
    log "laguna-s-2.1 complete"
  else
    log "laguna-s-2.1 SERVE-FAIL"; echo "laguna-s-2.1 serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "laguna-s-2.1 SKIP (fetch failed)"; echo "laguna-s-2.1 missing-gguf" >> $B8_ROOT/failures
fi
release "laguna-s-2.1"


# ---------- llama-4-scout : B8 (62.0 GB) ----------
log "===== llama-4-scout : fetching 62.0 GB ====="
gate; get llama-4-scout unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf &
gate; get llama-4-scout unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00002-of-00002.gguf &
wait
GG="llama-4-scout/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    # llama-4-scout went 100%% infra-error in every prior B8 attempt: probe ONE task
    # with the real sandbox before paying for the full 115-run sweep.
    if ! run_step "llama-4-scout" B8-probe "$PY" scripts/run_b8_local.py $EP --model "llama-4-scout" --task py-bugfix-01 --limit 1 --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_probe_llama-4-scout; then
      log "  llama-4-scout B8 SKIPPED: probe produced no eligible row - harness cannot drive this model; documented exclusion, not a model score"
      echo "llama-4-scout B8 skip probe-failed" >> $B8_ROOT/steps
    else
      run_step "llama-4-scout" B8 "$PY" scripts/run_b8_local.py $EP --model "llama-4-scout" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_llama-4-scout
    fi
    echo "llama-4-scout" >> $B8_ROOT/models_done
    log "llama-4-scout complete"
  else
    log "llama-4-scout SERVE-FAIL"; echo "llama-4-scout serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "llama-4-scout SKIP (fetch failed)"; echo "llama-4-scout missing-gguf" >> $B8_ROOT/failures
fi
release "llama-4-scout"


# ---------- gpt-oss-120b : B8 (65.4 GB) ----------
log "===== gpt-oss-120b : fetching 65.4 GB ====="
gate; get gpt-oss-120b unsloth/gpt-oss-120b-GGUF gpt-oss-120b-F16.gguf &
wait
GG="gpt-oss-120b/gpt-oss-120b-F16.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "gpt-oss-120b" B8 "$PY" scripts/run_b8_local.py $EP --model "gpt-oss-120b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_gpt-oss-120b
    echo "gpt-oss-120b" >> $B8_ROOT/models_done
    log "gpt-oss-120b complete"
  else
    log "gpt-oss-120b SERVE-FAIL"; echo "gpt-oss-120b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "gpt-oss-120b SKIP (fetch failed)"; echo "gpt-oss-120b missing-gguf" >> $B8_ROOT/failures
fi
release "gpt-oss-120b"


# ---------- glm-4.5-air : B8 (67.7 GB) ----------
log "===== glm-4.5-air : fetching 67.7 GB ====="
gate; get glm-4.5-air unsloth/GLM-4.5-Air-GGUF UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf &
gate; get glm-4.5-air unsloth/GLM-4.5-Air-GGUF UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00002-of-00002.gguf &
wait
GG="glm-4.5-air/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "glm-4.5-air" B8 "$PY" scripts/run_b8_local.py $EP --model "glm-4.5-air" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_glm-4.5-air
    echo "glm-4.5-air" >> $B8_ROOT/models_done
    log "glm-4.5-air complete"
  else
    log "glm-4.5-air SERVE-FAIL"; echo "glm-4.5-air serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "glm-4.5-air SKIP (fetch failed)"; echo "glm-4.5-air missing-gguf" >> $B8_ROOT/failures
fi
release "glm-4.5-air"


# ---------- gpt-oss-20b : B8,B9 (13.8 GB) ----------
log "===== gpt-oss-20b : fetching 13.8 GB ====="
gate; get gpt-oss-20b unsloth/gpt-oss-20b-GGUF gpt-oss-20b-F16.gguf &
wait
GG="gpt-oss-20b/gpt-oss-20b-F16.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "gpt-oss-20b" B9 "$PY" scripts/run_games.py $EP --model "gpt-oss-20b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "gpt-oss-20b" B8 "$PY" scripts/run_b8_local.py $EP --model "gpt-oss-20b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_gpt-oss-20b
    echo "gpt-oss-20b" >> $B8_ROOT/models_done
    log "gpt-oss-20b complete"
  else
    log "gpt-oss-20b SERVE-FAIL"; echo "gpt-oss-20b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "gpt-oss-20b SKIP (fetch failed)"; echo "gpt-oss-20b missing-gguf" >> $B8_ROOT/failures
fi
release "gpt-oss-20b"


# ---------- gemma-4-26b-a4b : B8,B9 (14.2 GB) ----------
log "===== gemma-4-26b-a4b : fetching 14.2 GB ====="
gate; get gemma-4-26b-a4b unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf &
wait
GG="gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "gemma-4-26b-a4b" B9 "$PY" scripts/run_games.py $EP --model "gemma-4-26b-a4b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "gemma-4-26b-a4b" B8 "$PY" scripts/run_b8_local.py $EP --model "gemma-4-26b-a4b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_gemma-4-26b-a4b
    echo "gemma-4-26b-a4b" >> $B8_ROOT/models_done
    log "gemma-4-26b-a4b complete"
  else
    log "gemma-4-26b-a4b SERVE-FAIL"; echo "gemma-4-26b-a4b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "gemma-4-26b-a4b SKIP (fetch failed)"; echo "gemma-4-26b-a4b missing-gguf" >> $B8_ROOT/failures
fi
release "gemma-4-26b-a4b"


# ---------- qwen3-coder-30b : B8,B9 (17.7 GB) ----------
log "===== qwen3-coder-30b : fetching 17.7 GB ====="
gate; get qwen3-coder-30b unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf &
wait
GG="qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "qwen3-coder-30b" B9 "$PY" scripts/run_games.py $EP --model "qwen3-coder-30b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "qwen3-coder-30b" B8 "$PY" scripts/run_b8_local.py $EP --model "qwen3-coder-30b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_qwen3-coder-30b
    echo "qwen3-coder-30b" >> $B8_ROOT/models_done
    log "qwen3-coder-30b complete"
  else
    log "qwen3-coder-30b SERVE-FAIL"; echo "qwen3-coder-30b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "qwen3-coder-30b SKIP (fetch failed)"; echo "qwen3-coder-30b missing-gguf" >> $B8_ROOT/failures
fi
release "qwen3-coder-30b"


# ---------- qwen3.6-35b-a3b : B8,B9 (19.7 GB) ----------
log "===== qwen3.6-35b-a3b : fetching 19.7 GB ====="
gate; get qwen3.6-35b-a3b bartowski/Qwen_Qwen3.6-35B-A3B-GGUF Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf &
wait
GG="qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "qwen3.6-35b-a3b" B9 "$PY" scripts/run_games.py $EP --model "qwen3.6-35b-a3b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "qwen3.6-35b-a3b" B8 "$PY" scripts/run_b8_local.py $EP --model "qwen3.6-35b-a3b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_qwen3.6-35b-a3b
    echo "qwen3.6-35b-a3b" >> $B8_ROOT/models_done
    log "qwen3.6-35b-a3b complete"
  else
    log "qwen3.6-35b-a3b SERVE-FAIL"; echo "qwen3.6-35b-a3b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "qwen3.6-35b-a3b SKIP (fetch failed)"; echo "qwen3.6-35b-a3b missing-gguf" >> $B8_ROOT/failures
fi
release "qwen3.6-35b-a3b"


# ---------- agents-a1-35b : B8,B9 (19.8 GB) ----------
log "===== agents-a1-35b : fetching 19.8 GB ====="
gate; get agents-a1-35b jashepp/Agents-A1-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf &
wait
GG="agents-a1-35b/Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "agents-a1-35b" B9 "$PY" scripts/run_games.py $EP --model "agents-a1-35b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "agents-a1-35b" B8 "$PY" scripts/run_b8_local.py $EP --model "agents-a1-35b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_agents-a1-35b
    echo "agents-a1-35b" >> $B8_ROOT/models_done
    log "agents-a1-35b complete"
  else
    log "agents-a1-35b SERVE-FAIL"; echo "agents-a1-35b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "agents-a1-35b SKIP (fetch failed)"; echo "agents-a1-35b missing-gguf" >> $B8_ROOT/failures
fi
release "agents-a1-35b"


# ---------- ornith-1.0-35b : B8,B9 (19.8 GB) ----------
log "===== ornith-1.0-35b : fetching 19.8 GB ====="
gate; get ornith-1.0-35b jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf &
wait
GG="ornith-1.0-35b/Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "ornith-1.0-35b" B9 "$PY" scripts/run_games.py $EP --model "ornith-1.0-35b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "ornith-1.0-35b" B8 "$PY" scripts/run_b8_local.py $EP --model "ornith-1.0-35b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_ornith-1.0-35b
    echo "ornith-1.0-35b" >> $B8_ROOT/models_done
    log "ornith-1.0-35b complete"
  else
    log "ornith-1.0-35b SERVE-FAIL"; echo "ornith-1.0-35b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "ornith-1.0-35b SKIP (fetch failed)"; echo "ornith-1.0-35b missing-gguf" >> $B8_ROOT/failures
fi
release "ornith-1.0-35b"


# ---------- nemotron-3-nano-30b : B8,B9 (22.8 GB) ----------
log "===== nemotron-3-nano-30b : fetching 22.8 GB ====="
gate; get nemotron-3-nano-30b unsloth/Nemotron-3-Nano-30B-A3B-GGUF Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf &
wait
GG="nemotron-3-nano-30b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "nemotron-3-nano-30b" B9 "$PY" scripts/run_games.py $EP --model "nemotron-3-nano-30b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "nemotron-3-nano-30b" B8 "$PY" scripts/run_b8_local.py $EP --model "nemotron-3-nano-30b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_nemotron-3-nano-30b
    echo "nemotron-3-nano-30b" >> $B8_ROOT/models_done
    log "nemotron-3-nano-30b complete"
  else
    log "nemotron-3-nano-30b SERVE-FAIL"; echo "nemotron-3-nano-30b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "nemotron-3-nano-30b SKIP (fetch failed)"; echo "nemotron-3-nano-30b missing-gguf" >> $B8_ROOT/failures
fi
release "nemotron-3-nano-30b"


# ---------- abl-gemma-4-31b : B8 (18.7 GB) ----------
log "===== abl-gemma-4-31b : fetching 18.7 GB ====="
gate; get abl-gemma-4-31b huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-GGUF Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf &
wait
GG="abl-gemma-4-31b/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "abl-gemma-4-31b" B8 "$PY" scripts/run_b8_local.py $EP --model "abl-gemma-4-31b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_abl-gemma-4-31b
    echo "abl-gemma-4-31b" >> $B8_ROOT/models_done
    log "abl-gemma-4-31b complete"
  else
    log "abl-gemma-4-31b SERVE-FAIL"; echo "abl-gemma-4-31b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "abl-gemma-4-31b SKIP (fetch failed)"; echo "abl-gemma-4-31b missing-gguf" >> $B8_ROOT/failures
fi
release "abl-gemma-4-31b"


# ---------- ornith-1.0-9b : B8,B9 (9.5 GB) ----------
log "===== ornith-1.0-9b : fetching 9.5 GB ====="
gate; get ornith-1.0-9b jashepp/Ornith-1.0-9B-MXFP4_Hybrid-Imatrix-GGUF Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf &
wait
GG="ornith-1.0-9b/Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "ornith-1.0-9b" B9 "$PY" scripts/run_games.py $EP --model "ornith-1.0-9b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "ornith-1.0-9b" B8 "$PY" scripts/run_b8_local.py $EP --model "ornith-1.0-9b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_ornith-1.0-9b
    echo "ornith-1.0-9b" >> $B8_ROOT/models_done
    log "ornith-1.0-9b complete"
  else
    log "ornith-1.0-9b SERVE-FAIL"; echo "ornith-1.0-9b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "ornith-1.0-9b SKIP (fetch failed)"; echo "ornith-1.0-9b missing-gguf" >> $B8_ROOT/failures
fi
release "ornith-1.0-9b"


# ---------- gemma-4-31b-dense : B8,B9 (17.3 GB) ----------
log "===== gemma-4-31b-dense : fetching 17.3 GB ====="
gate; get gemma-4-31b-dense unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf &
wait
GG="gemma-4-31b-dense/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "gemma-4-31b-dense" B9 "$PY" scripts/run_games.py $EP --model "gemma-4-31b-dense" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "gemma-4-31b-dense" B8 "$PY" scripts/run_b8_local.py $EP --model "gemma-4-31b-dense" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_gemma-4-31b-dense
    echo "gemma-4-31b-dense" >> $B8_ROOT/models_done
    log "gemma-4-31b-dense complete"
  else
    log "gemma-4-31b-dense SERVE-FAIL"; echo "gemma-4-31b-dense serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "gemma-4-31b-dense SKIP (fetch failed)"; echo "gemma-4-31b-dense missing-gguf" >> $B8_ROOT/failures
fi
release "gemma-4-31b-dense"


# ---------- granite-4.1-30b : B8,B9 (17.7 GB) ----------
log "===== granite-4.1-30b : fetching 17.7 GB ====="
gate; get granite-4.1-30b unsloth/granite-4.1-30b-GGUF granite-4.1-30b-UD-Q4_K_XL.gguf &
wait
GG="granite-4.1-30b/granite-4.1-30b-UD-Q4_K_XL.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "granite-4.1-30b" B9 "$PY" scripts/run_games.py $EP --model "granite-4.1-30b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "granite-4.1-30b" B8 "$PY" scripts/run_b8_local.py $EP --model "granite-4.1-30b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_granite-4.1-30b
    echo "granite-4.1-30b" >> $B8_ROOT/models_done
    log "granite-4.1-30b complete"
  else
    log "granite-4.1-30b SERVE-FAIL"; echo "granite-4.1-30b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "granite-4.1-30b SKIP (fetch failed)"; echo "granite-4.1-30b missing-gguf" >> $B8_ROOT/failures
fi
release "granite-4.1-30b"


# ---------- qwen3.6-27b-dense : B8,B9 (19.5 GB) ----------
log "===== qwen3.6-27b-dense : fetching 19.5 GB ====="
gate; get qwen3.6-27b-dense unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q5_K_M.gguf &
wait
GG="qwen3.6-27b-dense/Qwen3.6-27B-Q5_K_M.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "qwen3.6-27b-dense" B9 "$PY" scripts/run_games.py $EP --model "qwen3.6-27b-dense" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "qwen3.6-27b-dense" B8 "$PY" scripts/run_b8_local.py $EP --model "qwen3.6-27b-dense" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_qwen3.6-27b-dense
    echo "qwen3.6-27b-dense" >> $B8_ROOT/models_done
    log "qwen3.6-27b-dense complete"
  else
    log "qwen3.6-27b-dense SERVE-FAIL"; echo "qwen3.6-27b-dense serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "qwen3.6-27b-dense SKIP (fetch failed)"; echo "qwen3.6-27b-dense missing-gguf" >> $B8_ROOT/failures
fi
release "qwen3.6-27b-dense"


# ---------- abl-qwen3.6-27b : B2,B3,B6,B8,B11 (16.8 GB) ----------
log "===== abl-qwen3.6-27b : fetching 16.8 GB ====="
gate; get abl-qwen3.6-27b huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf &
wait
GG="abl-qwen3.6-27b/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    mkdir -p $B8_ROOT/agentws; run_step "abl-qwen3.6-27b" B11 "$PY" scripts/run_tools_agent.py $EP --model "abl-qwen3.6-27b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --workspace $B8_ROOT/agentws --out $OUT/tools
    run_step "abl-qwen3.6-27b" B2 "$PY" scripts/bigmodel_gen.py --model "abl-qwen3.6-27b" --batteries 2 $EP --results-dir $OUT/suite
    run_step "abl-qwen3.6-27b" B3 "$PY" scripts/bigmodel_gen.py --model "abl-qwen3.6-27b" --batteries 3 $EP --results-dir $OUT/suite
    run_step "abl-qwen3.6-27b" B6 "$PY" scripts/bigmodel_gen.py --model "abl-qwen3.6-27b" --batteries 6 $EP --results-dir $OUT/suite
    # abl-qwen3.6-27b went 100%% infra-error in every prior B8 attempt: probe ONE task
    # with the real sandbox before paying for the full 115-run sweep.
    if ! run_step "abl-qwen3.6-27b" B8-probe "$PY" scripts/run_b8_local.py $EP --model "abl-qwen3.6-27b" --task py-bugfix-01 --limit 1 --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_probe_abl-qwen3.6-27b; then
      log "  abl-qwen3.6-27b B8 SKIPPED: probe produced no eligible row - harness cannot drive this model; documented exclusion, not a model score"
      echo "abl-qwen3.6-27b B8 skip probe-failed" >> $B8_ROOT/steps
    else
      run_step "abl-qwen3.6-27b" B8 "$PY" scripts/run_b8_local.py $EP --model "abl-qwen3.6-27b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_abl-qwen3.6-27b
    fi
    echo "abl-qwen3.6-27b" >> $B8_ROOT/models_done
    log "abl-qwen3.6-27b complete"
  else
    log "abl-qwen3.6-27b SERVE-FAIL"; echo "abl-qwen3.6-27b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "abl-qwen3.6-27b SKIP (fetch failed)"; echo "abl-qwen3.6-27b missing-gguf" >> $B8_ROOT/failures
fi
release "abl-qwen3.6-27b"


# ---------- bonsai-ternary-27b : B8,B9,B10,B11 (7.2 GB, PRISM fork serve) ----------
log "===== bonsai-ternary-27b : fetching 7.2 GB ====="
gate; get bonsai-ternary-27b prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf &
wait
GG="bonsai-ternary-27b/Ternary-Bonsai-27B-Q2_0.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "prism-llama:1"; then
    mkdir -p $B8_ROOT/agentws; run_step "bonsai-ternary-27b" B11 "$PY" scripts/run_tools_agent.py $EP --model "bonsai-ternary-27b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --workspace $B8_ROOT/agentws --out $OUT/tools
    run_step "bonsai-ternary-27b" B10 "$PY" scripts/run_security.py $EP --model "bonsai-ternary-27b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/security
    run_step "bonsai-ternary-27b" B9 "$PY" scripts/run_games.py $EP --model "bonsai-ternary-27b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
    run_step "bonsai-ternary-27b" B8 "$PY" scripts/run_b8_local.py $EP --model "bonsai-ternary-27b" --hardware-sku rtx-pro-6000-vm --results-dir $OUT/b8_bonsai-ternary-27b
    echo "bonsai-ternary-27b" >> $B8_ROOT/models_done
    log "bonsai-ternary-27b complete"
  else
    log "bonsai-ternary-27b SERVE-FAIL"; echo "bonsai-ternary-27b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "bonsai-ternary-27b SKIP (fetch failed)"; echo "bonsai-ternary-27b missing-gguf" >> $B8_ROOT/failures
fi
release "bonsai-ternary-27b"

stop_server
echo ALL_DONE > $B8_ROOT/run_all_done
log "EVERYTHING DONE: $(grep -c ' ok$' $B8_ROOT/steps 2>/dev/null || echo 0) ok, \
$(grep -c ' fail' $B8_ROOT/steps 2>/dev/null || echo 0) failed, \
$(grep -c ' skip' $B8_ROOT/steps 2>/dev/null || echo 0) skipped"
