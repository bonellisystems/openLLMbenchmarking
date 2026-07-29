#!/bin/bash
# Close every gap, one model at a time so each GGUF is loaded once.
set -u
cd /root/llmtest-v2
export LD_LIBRARY_PATH=/app
BIN=/app/llama-server
OUT=/root/out
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*" | tee -a /root/run.log; }

serve(){ # $1 gguf  $2 extra flags
  pkill -f "app/llama-server" 2>/dev/null; sleep 4
  # shellcheck disable=SC2086
  nohup $BIN -m "$1" -ngl 99 -c 32768 --parallel 1 --jinja -fa on $2 \
    --host 127.0.0.1 --port 8080 --no-webui > /root/serve.log 2>&1 &
  for i in $(seq 1 200); do
    curl -s -m3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && return 0
    sleep 4
  done
  return 1
}


# ---------- abl-opus-35b-a3b : B1,B2,B3,B4,B5,B6,B7,B8,B9,B11 ----------
GG="/root/models/abl-opus-35b-a3b/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf"
if [ -f "$GG" ]; then
  log "===== abl-opus-35b-a3b ====="
  if serve "$GG" ""; then
    log "  abl-opus-35b-a3b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "abl-opus-35b-a3b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  abl-opus-35b-a3b B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "abl-opus-35b-a3b" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  abl-opus-35b-a3b B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "abl-opus-35b-a3b" --results-dir $OUT/b8_abl-opus-35b-a3b 2>&1 | tail -3
    log "  abl-opus-35b-a3b suite batteries 2,3,6,7,4,5,1"; python3 scripts/bigmodel_gen.py --model "abl-opus-35b-a3b" --batteries 2,3,6,7,4,5,1 --endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite 2>&1 | tail -5
    echo "abl-opus-35b-a3b" >> /root/models_done
    log "abl-opus-35b-a3b complete"
  else
    log "abl-opus-35b-a3b SERVE-FAIL"; echo "abl-opus-35b-a3b serve-fail" >> /root/failures
  fi
else
  log "abl-opus-35b-a3b SKIP (missing $GG)"; echo "abl-opus-35b-a3b missing-gguf" >> /root/failures
fi


# ---------- abl-gemma-4-31b : B1,B2,B3,B4,B5,B6,B7,B8,B9,B11 ----------
GG="/root/models/abl-gemma-4-31b/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
if [ -f "$GG" ]; then
  log "===== abl-gemma-4-31b ====="
  if serve "$GG" ""; then
    log "  abl-gemma-4-31b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "abl-gemma-4-31b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  abl-gemma-4-31b B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "abl-gemma-4-31b" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  abl-gemma-4-31b B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "abl-gemma-4-31b" --results-dir $OUT/b8_abl-gemma-4-31b 2>&1 | tail -3
    log "  abl-gemma-4-31b suite batteries 2,3,6,7,4,5,1"; python3 scripts/bigmodel_gen.py --model "abl-gemma-4-31b" --batteries 2,3,6,7,4,5,1 --endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite 2>&1 | tail -5
    echo "abl-gemma-4-31b" >> /root/models_done
    log "abl-gemma-4-31b complete"
  else
    log "abl-gemma-4-31b SERVE-FAIL"; echo "abl-gemma-4-31b serve-fail" >> /root/failures
  fi
else
  log "abl-gemma-4-31b SKIP (missing $GG)"; echo "abl-gemma-4-31b missing-gguf" >> /root/failures
fi


# ---------- laguna-s-2.1 : B4,B5,B7,B8,B9,B10,B11 ----------
GG="/root/models/laguna-s-2.1/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf"
if [ -f "$GG" ]; then
  log "===== laguna-s-2.1 ====="
  if serve "$GG" ""; then
    log "  laguna-s-2.1 B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  laguna-s-2.1 B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --reps 3 --out $OUT/security 2>&1 | tail -3
    log "  laguna-s-2.1 B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  laguna-s-2.1 B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "laguna-s-2.1" --results-dir $OUT/b8_laguna-s-2.1 2>&1 | tail -3
    log "  laguna-s-2.1 suite batteries 7,4,5"; python3 scripts/bigmodel_gen.py --model "laguna-s-2.1" --batteries 7,4,5 --endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite 2>&1 | tail -5
    echo "laguna-s-2.1" >> /root/models_done
    log "laguna-s-2.1 complete"
  else
    log "laguna-s-2.1 SERVE-FAIL"; echo "laguna-s-2.1 serve-fail" >> /root/failures
  fi
else
  log "laguna-s-2.1 SKIP (missing $GG)"; echo "laguna-s-2.1 missing-gguf" >> /root/failures
fi


# ---------- abl-qwen3.6-27b : B1,B4,B5,B7,B8,B9 ----------
GG="/root/models/abl-qwen3.6-27b/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf"
if [ -f "$GG" ]; then
  log "===== abl-qwen3.6-27b ====="
  if serve "$GG" ""; then
    log "  abl-qwen3.6-27b B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "abl-qwen3.6-27b" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  abl-qwen3.6-27b B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "abl-qwen3.6-27b" --results-dir $OUT/b8_abl-qwen3.6-27b 2>&1 | tail -3
    log "  abl-qwen3.6-27b suite batteries 7,4,5,1"; python3 scripts/bigmodel_gen.py --model "abl-qwen3.6-27b" --batteries 7,4,5,1 --endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite 2>&1 | tail -5
    echo "abl-qwen3.6-27b" >> /root/models_done
    log "abl-qwen3.6-27b complete"
  else
    log "abl-qwen3.6-27b SERVE-FAIL"; echo "abl-qwen3.6-27b serve-fail" >> /root/failures
  fi
else
  log "abl-qwen3.6-27b SKIP (missing $GG)"; echo "abl-qwen3.6-27b missing-gguf" >> /root/failures
fi


# ---------- llama-4-scout : B8,B9,B10,B11 ----------
GG="/root/models/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$GG" ]; then
  log "===== llama-4-scout ====="
  if serve "$GG" ""; then
    log "  llama-4-scout B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  llama-4-scout B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --reps 3 --out $OUT/security 2>&1 | tail -3
    log "  llama-4-scout B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  llama-4-scout B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "llama-4-scout" --results-dir $OUT/b8_llama-4-scout 2>&1 | tail -3
    echo "llama-4-scout" >> /root/models_done
    log "llama-4-scout complete"
  else
    log "llama-4-scout SERVE-FAIL"; echo "llama-4-scout serve-fail" >> /root/failures
  fi
else
  log "llama-4-scout SKIP (missing $GG)"; echo "llama-4-scout missing-gguf" >> /root/failures
fi


# ---------- glm-4.5-air : B8,B9,B10,B11 ----------
GG="/root/models/glm-4.5-air/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$GG" ]; then
  log "===== glm-4.5-air ====="
  if serve "$GG" ""; then
    log "  glm-4.5-air B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  glm-4.5-air B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --reps 3 --out $OUT/security 2>&1 | tail -3
    log "  glm-4.5-air B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  glm-4.5-air B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "glm-4.5-air" --results-dir $OUT/b8_glm-4.5-air 2>&1 | tail -3
    echo "glm-4.5-air" >> /root/models_done
    log "glm-4.5-air complete"
  else
    log "glm-4.5-air SERVE-FAIL"; echo "glm-4.5-air serve-fail" >> /root/failures
  fi
else
  log "glm-4.5-air SKIP (missing $GG)"; echo "glm-4.5-air missing-gguf" >> /root/failures
fi


# ---------- qwen3-235b : B8,B9,B10,B11 ----------
GG="/root/models/qwen3-235b/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3-235b ====="
  if serve "$GG" "--cpu-moe"; then
    log "  qwen3-235b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  qwen3-235b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --reps 3 --out $OUT/security 2>&1 | tail -3
    log "  qwen3-235b B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  qwen3-235b B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-235b" --results-dir $OUT/b8_qwen3-235b 2>&1 | tail -3
    echo "qwen3-235b" >> /root/models_done
    log "qwen3-235b complete"
  else
    log "qwen3-235b SERVE-FAIL"; echo "qwen3-235b serve-fail" >> /root/failures
  fi
else
  log "qwen3-235b SKIP (missing $GG)"; echo "qwen3-235b missing-gguf" >> /root/failures
fi


# ---------- gpt-oss-120b : B8,B9,B11 ----------
GG="/root/models/gpt-oss-120b/gpt-oss-120b-F16.gguf"
if [ -f "$GG" ]; then
  log "===== gpt-oss-120b ====="
  if serve "$GG" ""; then
    log "  gpt-oss-120b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-120b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  gpt-oss-120b B9 games"; python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-120b" --reps 3 --out $OUT/games --chrome "" 2>&1 | tail -3
    log "  gpt-oss-120b B8 agentic harness (host mode)"; python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-120b" --results-dir $OUT/b8_gpt-oss-120b 2>&1 | tail -3
    echo "gpt-oss-120b" >> /root/models_done
    log "gpt-oss-120b complete"
  else
    log "gpt-oss-120b SERVE-FAIL"; echo "gpt-oss-120b serve-fail" >> /root/failures
  fi
else
  log "gpt-oss-120b SKIP (missing $GG)"; echo "gpt-oss-120b missing-gguf" >> /root/failures
fi


# ---------- bonsai-ternary-27b : B10,B11 ----------
GG="/root/models/bonsai-ternary-27b/Ternary-Bonsai-27B-Q2_0.gguf"
if [ -f "$GG" ]; then
  log "===== bonsai-ternary-27b ====="
  if serve "$GG" ""; then
    log "  bonsai-ternary-27b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "bonsai-ternary-27b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  bonsai-ternary-27b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "bonsai-ternary-27b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "bonsai-ternary-27b" >> /root/models_done
    log "bonsai-ternary-27b complete"
  else
    log "bonsai-ternary-27b SERVE-FAIL"; echo "bonsai-ternary-27b serve-fail" >> /root/failures
  fi
else
  log "bonsai-ternary-27b SKIP (missing $GG)"; echo "bonsai-ternary-27b missing-gguf" >> /root/failures
fi


# ---------- ornith-1.0-9b : B10,B11 ----------
GG="/root/models/ornith-1.0-9b/Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
  log "===== ornith-1.0-9b ====="
  if serve "$GG" ""; then
    log "  ornith-1.0-9b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-9b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  ornith-1.0-9b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-9b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "ornith-1.0-9b" >> /root/models_done
    log "ornith-1.0-9b complete"
  else
    log "ornith-1.0-9b SERVE-FAIL"; echo "ornith-1.0-9b serve-fail" >> /root/failures
  fi
else
  log "ornith-1.0-9b SKIP (missing $GG)"; echo "ornith-1.0-9b missing-gguf" >> /root/failures
fi


# ---------- gpt-oss-20b : B10,B11 ----------
GG="/root/models/gpt-oss-20b/gpt-oss-20b-F16.gguf"
if [ -f "$GG" ]; then
  log "===== gpt-oss-20b ====="
  if serve "$GG" ""; then
    log "  gpt-oss-20b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-20b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  gpt-oss-20b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "gpt-oss-20b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "gpt-oss-20b" >> /root/models_done
    log "gpt-oss-20b complete"
  else
    log "gpt-oss-20b SERVE-FAIL"; echo "gpt-oss-20b serve-fail" >> /root/failures
  fi
else
  log "gpt-oss-20b SKIP (missing $GG)"; echo "gpt-oss-20b missing-gguf" >> /root/failures
fi


# ---------- gemma-4-26b-a4b : B10,B11 ----------
GG="/root/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== gemma-4-26b-a4b ====="
  if serve "$GG" ""; then
    log "  gemma-4-26b-a4b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gemma-4-26b-a4b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  gemma-4-26b-a4b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "gemma-4-26b-a4b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "gemma-4-26b-a4b" >> /root/models_done
    log "gemma-4-26b-a4b complete"
  else
    log "gemma-4-26b-a4b SERVE-FAIL"; echo "gemma-4-26b-a4b serve-fail" >> /root/failures
  fi
else
  log "gemma-4-26b-a4b SKIP (missing $GG)"; echo "gemma-4-26b-a4b missing-gguf" >> /root/failures
fi


# ---------- granite-4.1-30b : B10,B11 ----------
GG="/root/models/granite-4.1-30b/granite-4.1-30b-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== granite-4.1-30b ====="
  if serve "$GG" ""; then
    log "  granite-4.1-30b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "granite-4.1-30b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  granite-4.1-30b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "granite-4.1-30b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "granite-4.1-30b" >> /root/models_done
    log "granite-4.1-30b complete"
  else
    log "granite-4.1-30b SERVE-FAIL"; echo "granite-4.1-30b serve-fail" >> /root/failures
  fi
else
  log "granite-4.1-30b SKIP (missing $GG)"; echo "granite-4.1-30b missing-gguf" >> /root/failures
fi


# ---------- qwen3-coder-30b : B10,B11 ----------
GG="/root/models/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3-coder-30b ====="
  if serve "$GG" ""; then
    log "  qwen3-coder-30b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-coder-30b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  qwen3-coder-30b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "qwen3-coder-30b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "qwen3-coder-30b" >> /root/models_done
    log "qwen3-coder-30b complete"
  else
    log "qwen3-coder-30b SERVE-FAIL"; echo "qwen3-coder-30b serve-fail" >> /root/failures
  fi
else
  log "qwen3-coder-30b SKIP (missing $GG)"; echo "qwen3-coder-30b missing-gguf" >> /root/failures
fi


# ---------- qwen3.6-35b-a3b : B10,B11 ----------
GG="/root/models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3.6-35b-a3b ====="
  if serve "$GG" ""; then
    log "  qwen3.6-35b-a3b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3.6-35b-a3b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  qwen3.6-35b-a3b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "qwen3.6-35b-a3b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "qwen3.6-35b-a3b" >> /root/models_done
    log "qwen3.6-35b-a3b complete"
  else
    log "qwen3.6-35b-a3b SERVE-FAIL"; echo "qwen3.6-35b-a3b serve-fail" >> /root/failures
  fi
else
  log "qwen3.6-35b-a3b SKIP (missing $GG)"; echo "qwen3.6-35b-a3b missing-gguf" >> /root/failures
fi


# ---------- agents-a1-35b : B10,B11 ----------
GG="/root/models/agents-a1-35b/Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
  log "===== agents-a1-35b ====="
  if serve "$GG" ""; then
    log "  agents-a1-35b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "agents-a1-35b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  agents-a1-35b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "agents-a1-35b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "agents-a1-35b" >> /root/models_done
    log "agents-a1-35b complete"
  else
    log "agents-a1-35b SERVE-FAIL"; echo "agents-a1-35b serve-fail" >> /root/failures
  fi
else
  log "agents-a1-35b SKIP (missing $GG)"; echo "agents-a1-35b missing-gguf" >> /root/failures
fi


# ---------- ornith-1.0-35b : B10,B11 ----------
GG="/root/models/ornith-1.0-35b/Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
  log "===== ornith-1.0-35b ====="
  if serve "$GG" ""; then
    log "  ornith-1.0-35b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-35b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  ornith-1.0-35b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "ornith-1.0-35b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "ornith-1.0-35b" >> /root/models_done
    log "ornith-1.0-35b complete"
  else
    log "ornith-1.0-35b SERVE-FAIL"; echo "ornith-1.0-35b serve-fail" >> /root/failures
  fi
else
  log "ornith-1.0-35b SKIP (missing $GG)"; echo "ornith-1.0-35b missing-gguf" >> /root/failures
fi


# ---------- nemotron-3-nano-30b : B10,B11 ----------
GG="/root/models/nemotron-3-nano-30b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== nemotron-3-nano-30b ====="
  if serve "$GG" ""; then
    log "  nemotron-3-nano-30b B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "nemotron-3-nano-30b" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    log "  nemotron-3-nano-30b B10 security"; python3 scripts/run_security.py --endpoint-url http://127.0.0.1:8080 --model "nemotron-3-nano-30b" --reps 3 --out $OUT/security 2>&1 | tail -3
    echo "nemotron-3-nano-30b" >> /root/models_done
    log "nemotron-3-nano-30b complete"
  else
    log "nemotron-3-nano-30b SERVE-FAIL"; echo "nemotron-3-nano-30b serve-fail" >> /root/failures
  fi
else
  log "nemotron-3-nano-30b SKIP (missing $GG)"; echo "nemotron-3-nano-30b missing-gguf" >> /root/failures
fi


# ---------- gemma-4-31b-dense : B11 ----------
GG="/root/models/gemma-4-31b-dense/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
  log "===== gemma-4-31b-dense ====="
  if serve "$GG" ""; then
    log "  gemma-4-31b-dense B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "gemma-4-31b-dense" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    echo "gemma-4-31b-dense" >> /root/models_done
    log "gemma-4-31b-dense complete"
  else
    log "gemma-4-31b-dense SERVE-FAIL"; echo "gemma-4-31b-dense serve-fail" >> /root/failures
  fi
else
  log "gemma-4-31b-dense SKIP (missing $GG)"; echo "gemma-4-31b-dense missing-gguf" >> /root/failures
fi


# ---------- qwen3.6-27b-dense : B11 ----------
GG="/root/models/qwen3.6-27b-dense/Qwen3.6-27B-Q5_K_M.gguf"
if [ -f "$GG" ]; then
  log "===== qwen3.6-27b-dense ====="
  if serve "$GG" ""; then
    log "  qwen3.6-27b-dense B11 tool loop"; mkdir -p /root/agentws && python3 scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 --model "qwen3.6-27b-dense" --reps 3 --workspace /root/agentws --out $OUT/tools 2>&1 | tail -3
    echo "qwen3.6-27b-dense" >> /root/models_done
    log "qwen3.6-27b-dense complete"
  else
    log "qwen3.6-27b-dense SERVE-FAIL"; echo "qwen3.6-27b-dense serve-fail" >> /root/failures
  fi
else
  log "qwen3.6-27b-dense SKIP (missing $GG)"; echo "qwen3.6-27b-dense missing-gguf" >> /root/failures
fi


pkill -f "app/llama-server" 2>/dev/null
echo ALL_DONE > /root/run_all_done
log "EVERYTHING DONE"
