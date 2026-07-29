#!/bin/bash
# Close every gap, one model at a time so each GGUF is loaded once.
set -u
cd /root/llmtest-v2
export LD_LIBRARY_PATH=/app
export LLMTEST_ROOT=/root/llmtest-v2
export LLMTEST_OUT=/root/out
export LLMTEST_BIN=/app/llama-server
export LLMTEST_LIBS=/app
BIN=/app/llama-server
OUT=/root/out
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*" | tee -a /root/run.log; }

# Does this binary support ngram spec-decode? setup.sh probed --help and wrote the
# answer here. B5/B7 are timing-authoritative at spec=ngram32 and must SKIP rather than
# silently produce timings from a different serving config.
NGRAM=0
[ -f /root/caps ] && grep -q '^ngram=1' /root/caps && NGRAM=1
SPEC=""
[ "$NGRAM" = "1" ] && SPEC="--spec-type ngram-mod --spec-ngram-mod-n-match 32"

stop_server(){ pkill -f "llama-server" 2>/dev/null; sleep 5; }

serve(){ # $1 gguf  $2 extra flags -- the SHARED endpoint for B1/B2/B3/B6/B8-B11
  stop_server
  # shellcheck disable=SC2086
  nohup $BIN -m "$1" -ngl 99 -c 49152 --parallel 1 --jinja -fa on \
    -ctk q8_0 -ctv q8_0 $SPEC $2 \
    --host 127.0.0.1 --port 8080 --no-webui > /root/serve.log 2>&1 &
  for i in $(seq 1 200); do
    curl -s -m3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && return 0
    sleep 4
  done
  return 1
}

# Run one battery and RECORD ITS REAL EXIT CODE. The previous shape was
# `python3 ... 2>&1 | tail -3`, where the pipe discards the runner's status and the
# model was then appended to models_done unconditionally - so a battery that crashed on
# row 1 looked identical to one that completed. That is exactly how a YAML bug once
# produced zero rows across six models while the run reported healthy.
run_step(){ # $1 model  $2 battery  $3.. command
  mid="$1"; bat="$2"; shift 2
  log "  $mid $bat start"
  "$@" > /root/last_step.log 2>&1
  rc=$?
  tail -3 /root/last_step.log | tee -a /root/run.log
  if [ "$rc" -eq 0 ]; then
    echo "$mid $bat ok" >> /root/steps
  else
    log "  $mid $bat FAILED rc=$rc"
    echo "$mid $bat fail rc=$rc" >> /root/steps
    cat /root/last_step.log >> /root/step_failures.log
  fi
}

# B4/B7 relaunch the server per serving-config group; B5 controls its own launch. They
# take --gpu0/--gpu1 because P8 had two cards - this box has one, so gpu1 is empty and
# that worker thread exits immediately.
run_serving(){ # $1 model  $2 battery-number
  run_step "$1" "B$2" python3 scratchpad/p8_gen_serving.py --battery "$2" \
    --gpu0 "$1" --gpu1 ""
}


# ---------- abl-opus-35b-a3b : B1,B2,B3,B4,B5,B6,B7,B8,B9,B11 ----------
GG="/root/models/abl-opus-35b-a3b/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf"
if [ -f "$GG" ]; then
  log "===== abl-opus-35b-a3b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "abl-opus-35b-a3b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "abl-opus-35b-a3b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "abl-opus-35b-a3b" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "abl-opus-35b-a3b" --reps 3 --out $OUT/games --chrome ""
    run_step "abl-opus-35b-a3b" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "abl-opus-35b-a3b" --results-dir $OUT/b8_abl-opus-35b-a3b
    run_step "abl-opus-35b-a3b" B2+B3+B6+B1 python3 scripts/bigmodel_gen.py --model "abl-opus-35b-a3b" --batteries 2,3,6,1 --endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite
  else
    log "abl-opus-35b-a3b SERVE-FAIL (phase A)"; echo "abl-opus-35b-a3b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  run_serving "abl-opus-35b-a3b" 4
  if [ "$NGRAM" = "1" ]; then
    run_serving "abl-opus-35b-a3b" 7
  else
    log "  abl-opus-35b-a3b B7 SKIP - binary has no --spec-type ngram-mod, and B7 is timing-authoritative at spec=ngram32"
    echo "abl-opus-35b-a3b B7 skip no-ngram" >> /root/steps
  fi
  if [ "$NGRAM" = "1" ]; then
    run_step "abl-opus-35b-a3b" B5 python3 scratchpad/p8_gen_b5.py --gpu0 "abl-opus-35b-a3b" --gpu1 ""
  else
    log "  abl-opus-35b-a3b B5 SKIP - binary has no --spec-type ngram-mod, and B5 is timing-authoritative at spec=ngram32"
    echo "abl-opus-35b-a3b B5 skip no-ngram" >> /root/steps
  fi
  echo "abl-opus-35b-a3b" >> /root/models_done
  log "abl-opus-35b-a3b complete"
else
  log "abl-opus-35b-a3b SKIP (missing $GG)"; echo "abl-opus-35b-a3b missing-gguf" >> /root/failures
fi


# ---------- abl-gemma-4-31b : B1,B2,B3,B4,B5,B6,B7,B8,B9,B11 ----------
GG="/root/models/abl-gemma-4-31b/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
if [ -f "$GG" ]; then
  log "===== abl-gemma-4-31b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "abl-gemma-4-31b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "abl-gemma-4-31b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "abl-gemma-4-31b" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "abl-gemma-4-31b" --reps 3 --out $OUT/games --chrome ""
    run_step "abl-gemma-4-31b" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "abl-gemma-4-31b" --results-dir $OUT/b8_abl-gemma-4-31b
    run_step "abl-gemma-4-31b" B2+B3+B6+B1 python3 scripts/bigmodel_gen.py --model "abl-gemma-4-31b" --batteries 2,3,6,1 --endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite
  else
    log "abl-gemma-4-31b SERVE-FAIL (phase A)"; echo "abl-gemma-4-31b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  run_serving "abl-gemma-4-31b" 4
  if [ "$NGRAM" = "1" ]; then
    run_serving "abl-gemma-4-31b" 7
  else
    log "  abl-gemma-4-31b B7 SKIP - binary has no --spec-type ngram-mod, and B7 is timing-authoritative at spec=ngram32"
    echo "abl-gemma-4-31b B7 skip no-ngram" >> /root/steps
  fi
  if [ "$NGRAM" = "1" ]; then
    run_step "abl-gemma-4-31b" B5 python3 scratchpad/p8_gen_b5.py --gpu0 "abl-gemma-4-31b" --gpu1 ""
  else
    log "  abl-gemma-4-31b B5 SKIP - binary has no --spec-type ngram-mod, and B5 is timing-authoritative at spec=ngram32"
    echo "abl-gemma-4-31b B5 skip no-ngram" >> /root/steps
  fi
  echo "abl-gemma-4-31b" >> /root/models_done
  log "abl-gemma-4-31b complete"
else
  log "abl-gemma-4-31b SKIP (missing $GG)"; echo "abl-gemma-4-31b missing-gguf" >> /root/failures
fi


# ---------- laguna-s-2.1 : B4,B5,B7,B8,B9,B10,B11 ----------
GG="/root/models/laguna-s-2.1/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf"
if [ -f "$GG" ]; then
  log "===== laguna-s-2.1 ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "laguna-s-2.1" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "laguna-s-2.1" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --reps 3 --out $OUT/security
    run_step "laguna-s-2.1" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --reps 3 --out $OUT/games --chrome ""
    run_step "laguna-s-2.1" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --results-dir $OUT/b8_laguna-s-2.1
  else
    log "laguna-s-2.1 SERVE-FAIL (phase A)"; echo "laguna-s-2.1 serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  run_serving "laguna-s-2.1" 4
  if [ "$NGRAM" = "1" ]; then
    run_serving "laguna-s-2.1" 7
  else
    log "  laguna-s-2.1 B7 SKIP - binary has no --spec-type ngram-mod, and B7 is timing-authoritative at spec=ngram32"
    echo "laguna-s-2.1 B7 skip no-ngram" >> /root/steps
  fi
  if [ "$NGRAM" = "1" ]; then
    run_step "laguna-s-2.1" B5 python3 scratchpad/p8_gen_b5.py --gpu0 "laguna-s-2.1" --gpu1 ""
  else
    log "  laguna-s-2.1 B5 SKIP - binary has no --spec-type ngram-mod, and B5 is timing-authoritative at spec=ngram32"
    echo "laguna-s-2.1 B5 skip no-ngram" >> /root/steps
  fi
  echo "laguna-s-2.1" >> /root/models_done
  log "laguna-s-2.1 complete"
else
  log "laguna-s-2.1 SKIP (missing $GG)"; echo "laguna-s-2.1 missing-gguf" >> /root/failures
fi


# ---------- abl-qwen3.6-27b : B1,B4,B5,B7,B8,B9 ----------
GG="/root/models/abl-qwen3.6-27b/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf"
if [ -f "$GG" ]; then
  log "===== abl-qwen3.6-27b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    run_step "abl-qwen3.6-27b" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "abl-qwen3.6-27b" --reps 3 --out $OUT/games --chrome ""
    run_step "abl-qwen3.6-27b" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "abl-qwen3.6-27b" --results-dir $OUT/b8_abl-qwen3.6-27b
    run_step "abl-qwen3.6-27b" B1 python3 scripts/bigmodel_gen.py --model "abl-qwen3.6-27b" --batteries 1 --endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite
  else
    log "abl-qwen3.6-27b SERVE-FAIL (phase A)"; echo "abl-qwen3.6-27b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  run_serving "abl-qwen3.6-27b" 4
  if [ "$NGRAM" = "1" ]; then
    run_serving "abl-qwen3.6-27b" 7
  else
    log "  abl-qwen3.6-27b B7 SKIP - binary has no --spec-type ngram-mod, and B7 is timing-authoritative at spec=ngram32"
    echo "abl-qwen3.6-27b B7 skip no-ngram" >> /root/steps
  fi
  if [ "$NGRAM" = "1" ]; then
    run_step "abl-qwen3.6-27b" B5 python3 scratchpad/p8_gen_b5.py --gpu0 "abl-qwen3.6-27b" --gpu1 ""
  else
    log "  abl-qwen3.6-27b B5 SKIP - binary has no --spec-type ngram-mod, and B5 is timing-authoritative at spec=ngram32"
    echo "abl-qwen3.6-27b B5 skip no-ngram" >> /root/steps
  fi
  echo "abl-qwen3.6-27b" >> /root/models_done
  log "abl-qwen3.6-27b complete"
else
  log "abl-qwen3.6-27b SKIP (missing $GG)"; echo "abl-qwen3.6-27b missing-gguf" >> /root/failures
fi


# ---------- llama-4-scout : B8,B9,B10,B11 ----------
GG="/root/models/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$GG" ]; then
  log "===== llama-4-scout ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "llama-4-scout" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "llama-4-scout" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --reps 3 --out $OUT/security
    run_step "llama-4-scout" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --reps 3 --out $OUT/games --chrome ""
    run_step "llama-4-scout" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --results-dir $OUT/b8_llama-4-scout
  else
    log "llama-4-scout SERVE-FAIL (phase A)"; echo "llama-4-scout serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "llama-4-scout" >> /root/models_done
  log "llama-4-scout complete"
else
  log "llama-4-scout SKIP (missing $GG)"; echo "llama-4-scout missing-gguf" >> /root/failures
fi


# ---------- glm-4.5-air : B8,B9,B10,B11 ----------
GG="/root/models/glm-4.5-air/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$GG" ]; then
  log "===== glm-4.5-air ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "glm-4.5-air" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "glm-4.5-air" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --reps 3 --out $OUT/security
    run_step "glm-4.5-air" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --reps 3 --out $OUT/games --chrome ""
    run_step "glm-4.5-air" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --results-dir $OUT/b8_glm-4.5-air
  else
    log "glm-4.5-air SERVE-FAIL (phase A)"; echo "glm-4.5-air serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "glm-4.5-air" >> /root/models_done
  log "glm-4.5-air complete"
else
  log "glm-4.5-air SKIP (missing $GG)"; echo "glm-4.5-air missing-gguf" >> /root/failures
fi


# ---------- qwen3-235b : B8,B9,B10,B11 ----------
GG="/root/models/qwen3-235b/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3-235b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" "--cpu-moe"; then
    mkdir -p /root/agentws; run_step "qwen3-235b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "qwen3-235b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --reps 3 --out $OUT/security
    run_step "qwen3-235b" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --reps 3 --out $OUT/games --chrome ""
    run_step "qwen3-235b" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --results-dir $OUT/b8_qwen3-235b
  else
    log "qwen3-235b SERVE-FAIL (phase A)"; echo "qwen3-235b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "qwen3-235b" >> /root/models_done
  log "qwen3-235b complete"
else
  log "qwen3-235b SKIP (missing $GG)"; echo "qwen3-235b missing-gguf" >> /root/failures
fi


# ---------- gpt-oss-120b : B8,B9,B11 ----------
GG="/root/models/gpt-oss-120b/gpt-oss-120b-F16.gguf"
if [ -f "$GG" ]; then
  log "===== gpt-oss-120b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "gpt-oss-120b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-120b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "gpt-oss-120b" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-120b" --reps 3 --out $OUT/games --chrome ""
    run_step "gpt-oss-120b" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-120b" --results-dir $OUT/b8_gpt-oss-120b
  else
    log "gpt-oss-120b SERVE-FAIL (phase A)"; echo "gpt-oss-120b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "gpt-oss-120b" >> /root/models_done
  log "gpt-oss-120b complete"
else
  log "gpt-oss-120b SKIP (missing $GG)"; echo "gpt-oss-120b missing-gguf" >> /root/failures
fi


# ---------- bonsai-ternary-27b : B10,B11 ----------
GG="/root/models/bonsai-ternary-27b/Ternary-Bonsai-27B-Q2_0.gguf"
if [ -f "$GG" ]; then
  log "===== bonsai-ternary-27b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "bonsai-ternary-27b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "bonsai-ternary-27b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "bonsai-ternary-27b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "bonsai-ternary-27b" --reps 3 --out $OUT/security
  else
    log "bonsai-ternary-27b SERVE-FAIL (phase A)"; echo "bonsai-ternary-27b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "bonsai-ternary-27b" >> /root/models_done
  log "bonsai-ternary-27b complete"
else
  log "bonsai-ternary-27b SKIP (missing $GG)"; echo "bonsai-ternary-27b missing-gguf" >> /root/failures
fi


# ---------- ornith-1.0-9b : B10,B11 ----------
GG="/root/models/ornith-1.0-9b/Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
  log "===== ornith-1.0-9b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "ornith-1.0-9b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-9b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "ornith-1.0-9b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-9b" --reps 3 --out $OUT/security
  else
    log "ornith-1.0-9b SERVE-FAIL (phase A)"; echo "ornith-1.0-9b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "ornith-1.0-9b" >> /root/models_done
  log "ornith-1.0-9b complete"
else
  log "ornith-1.0-9b SKIP (missing $GG)"; echo "ornith-1.0-9b missing-gguf" >> /root/failures
fi


# ---------- gpt-oss-20b : B10,B11 ----------
GG="/root/models/gpt-oss-20b/gpt-oss-20b-F16.gguf"
if [ -f "$GG" ]; then
  log "===== gpt-oss-20b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "gpt-oss-20b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-20b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "gpt-oss-20b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-20b" --reps 3 --out $OUT/security
  else
    log "gpt-oss-20b SERVE-FAIL (phase A)"; echo "gpt-oss-20b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "gpt-oss-20b" >> /root/models_done
  log "gpt-oss-20b complete"
else
  log "gpt-oss-20b SKIP (missing $GG)"; echo "gpt-oss-20b missing-gguf" >> /root/failures
fi


# ---------- gemma-4-26b-a4b : B10,B11 ----------
GG="/root/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== gemma-4-26b-a4b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "gemma-4-26b-a4b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gemma-4-26b-a4b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "gemma-4-26b-a4b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "gemma-4-26b-a4b" --reps 3 --out $OUT/security
  else
    log "gemma-4-26b-a4b SERVE-FAIL (phase A)"; echo "gemma-4-26b-a4b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "gemma-4-26b-a4b" >> /root/models_done
  log "gemma-4-26b-a4b complete"
else
  log "gemma-4-26b-a4b SKIP (missing $GG)"; echo "gemma-4-26b-a4b missing-gguf" >> /root/failures
fi


# ---------- granite-4.1-30b : B10,B11 ----------
GG="/root/models/granite-4.1-30b/granite-4.1-30b-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== granite-4.1-30b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "granite-4.1-30b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "granite-4.1-30b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "granite-4.1-30b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "granite-4.1-30b" --reps 3 --out $OUT/security
  else
    log "granite-4.1-30b SERVE-FAIL (phase A)"; echo "granite-4.1-30b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "granite-4.1-30b" >> /root/models_done
  log "granite-4.1-30b complete"
else
  log "granite-4.1-30b SKIP (missing $GG)"; echo "granite-4.1-30b missing-gguf" >> /root/failures
fi


# ---------- qwen3-coder-30b : B10,B11 ----------
GG="/root/models/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3-coder-30b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "qwen3-coder-30b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-coder-30b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "qwen3-coder-30b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-coder-30b" --reps 3 --out $OUT/security
  else
    log "qwen3-coder-30b SERVE-FAIL (phase A)"; echo "qwen3-coder-30b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "qwen3-coder-30b" >> /root/models_done
  log "qwen3-coder-30b complete"
else
  log "qwen3-coder-30b SKIP (missing $GG)"; echo "qwen3-coder-30b missing-gguf" >> /root/failures
fi


# ---------- qwen3.6-35b-a3b : B10,B11 ----------
GG="/root/models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3.6-35b-a3b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "qwen3.6-35b-a3b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3.6-35b-a3b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "qwen3.6-35b-a3b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "qwen3.6-35b-a3b" --reps 3 --out $OUT/security
  else
    log "qwen3.6-35b-a3b SERVE-FAIL (phase A)"; echo "qwen3.6-35b-a3b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "qwen3.6-35b-a3b" >> /root/models_done
  log "qwen3.6-35b-a3b complete"
else
  log "qwen3.6-35b-a3b SKIP (missing $GG)"; echo "qwen3.6-35b-a3b missing-gguf" >> /root/failures
fi


# ---------- agents-a1-35b : B10,B11 ----------
GG="/root/models/agents-a1-35b/Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
  log "===== agents-a1-35b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "agents-a1-35b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "agents-a1-35b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "agents-a1-35b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "agents-a1-35b" --reps 3 --out $OUT/security
  else
    log "agents-a1-35b SERVE-FAIL (phase A)"; echo "agents-a1-35b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "agents-a1-35b" >> /root/models_done
  log "agents-a1-35b complete"
else
  log "agents-a1-35b SKIP (missing $GG)"; echo "agents-a1-35b missing-gguf" >> /root/failures
fi


# ---------- ornith-1.0-35b : B10,B11 ----------
GG="/root/models/ornith-1.0-35b/Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
  log "===== ornith-1.0-35b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "ornith-1.0-35b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-35b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "ornith-1.0-35b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-35b" --reps 3 --out $OUT/security
  else
    log "ornith-1.0-35b SERVE-FAIL (phase A)"; echo "ornith-1.0-35b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "ornith-1.0-35b" >> /root/models_done
  log "ornith-1.0-35b complete"
else
  log "ornith-1.0-35b SKIP (missing $GG)"; echo "ornith-1.0-35b missing-gguf" >> /root/failures
fi


# ---------- nemotron-3-nano-30b : B10,B11 ----------
GG="/root/models/nemotron-3-nano-30b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== nemotron-3-nano-30b ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "nemotron-3-nano-30b" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "nemotron-3-nano-30b" --reps 3 --workspace /root/agentws --out $OUT/tools
    run_step "nemotron-3-nano-30b" B10 python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "nemotron-3-nano-30b" --reps 3 --out $OUT/security
  else
    log "nemotron-3-nano-30b SERVE-FAIL (phase A)"; echo "nemotron-3-nano-30b serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "nemotron-3-nano-30b" >> /root/models_done
  log "nemotron-3-nano-30b complete"
else
  log "nemotron-3-nano-30b SKIP (missing $GG)"; echo "nemotron-3-nano-30b missing-gguf" >> /root/failures
fi


# ---------- gemma-4-31b-dense : B11 ----------
GG="/root/models/gemma-4-31b-dense/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== gemma-4-31b-dense ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "gemma-4-31b-dense" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gemma-4-31b-dense" --reps 3 --workspace /root/agentws --out $OUT/tools
  else
    log "gemma-4-31b-dense SERVE-FAIL (phase A)"; echo "gemma-4-31b-dense serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "gemma-4-31b-dense" >> /root/models_done
  log "gemma-4-31b-dense complete"
else
  log "gemma-4-31b-dense SKIP (missing $GG)"; echo "gemma-4-31b-dense missing-gguf" >> /root/failures
fi


# ---------- qwen3.6-27b-dense : B11 ----------
GG="/root/models/qwen3.6-27b-dense/Qwen3.6-27B-Q5_K_M.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3.6-27b-dense ====="
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    mkdir -p /root/agentws; run_step "qwen3.6-27b-dense" B11 python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3.6-27b-dense" --reps 3 --workspace /root/agentws --out $OUT/tools
  else
    log "qwen3.6-27b-dense SERVE-FAIL (phase A)"; echo "qwen3.6-27b-dense serve-fail" >> /root/failures
  fi
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
  :   # no own-server batteries for this model
  echo "qwen3.6-27b-dense" >> /root/models_done
  log "qwen3.6-27b-dense complete"
else
  log "qwen3.6-27b-dense SKIP (missing $GG)"; echo "qwen3.6-27b-dense missing-gguf" >> /root/failures
fi


stop_server
echo ALL_DONE > /root/run_all_done
log "EVERYTHING DONE"
log "step results: $(grep -c ' ok$' /root/steps 2>/dev/null || echo 0) ok, \
$(grep -c ' fail' /root/steps 2>/dev/null || echo 0) failed"
