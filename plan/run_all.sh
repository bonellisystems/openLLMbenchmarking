#!/bin/bash
# Close every gap, one model at a time: FETCH -> RUN ITS BATTERIES -> DELETE.
#
# Why interleaved rather than "download all 638.7GB, then run":
#  * PEAK DISK becomes the largest single model (~134GB) instead of the whole corpus, so
#    the disk floor drops 780GB -> 300GB. At 780GB, 15 of the available RTX PRO 6000
#    offers were disqualified on disk alone and some hours none qualified at all.
#  * IT COSTS NOTHING. The two phases were already sequential - download.sh ran to
#    completion before run_all.sh started - so there was never any overlap to lose.
#  * ROWS START ARRIVING IN MINUTES instead of after a ~2.2h download, and a box that
#    dies early has produced COMPLETE models rather than a full disk and no results.
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

M=/root/models
mkdir -p $M
# Files within a model are fetched CONCURRENTLY (HF throttles a single transfer:
# measured -x16 collapsing to one connection at ~21MB/s), but capped - firing everything
# at once opens hundreds of connections and gets rate-limited.
JOBS=4
gate(){ while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done; }
get(){ # dir repo path
  mkdir -p "$M/$1"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
    --retry-wait=5 --max-tries=5 --auto-file-renaming=false \
    -d "$M/$1" -o "$(basename "$3")" \
    "https://huggingface.co/$2/resolve/main/$3" >> "/root/dl_$1.log" 2>&1 \
    || echo "FAIL $1 $3" >> /root/dl_fail
}
# Free the weights once a model's batteries are done. Without this the interleaving
# buys nothing: the disk fills exactly as it did before, just more slowly.
release(){
  du -sh "$M/$1" 2>/dev/null | tee -a /root/run.log
  rm -rf "$M/$1"
  log "  released $1 ; free: $(df -h /root | awk 'NR==2{print $4}')"
}

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

# --- throughput probe: FAIL FAST --------------------------------------------------
# A host's advertised inet_down is its link speed, NOT the rate Hugging Face serves it.
# Cheap hosts have measured ~4MB/s while advertising gigabits; at that rate this corpus
# alone is ~44 hours and the budget is gone before a single row exists. Die here instead.
MIN_MBPS=25
rm -f /tmp/probe.bin
timeout 60 aria2c -x8 -s8 -k1M --file-allocation=none --console-log-level=error \
  -d /tmp -o probe.bin \
  "https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_0.gguf" >/dev/null 2>&1
PSZ=$(stat -c %s /tmp/probe.bin 2>/dev/null || echo 0)
rm -f /tmp/probe.bin
RATE=$(( PSZ / 60 / 1000000 ))
log "HF throughput probe: ~${RATE} MB/s (floor ${MIN_MBPS} MB/s)"
if [ "$RATE" -lt "$MIN_MBPS" ]; then
  log "ABORT: Hugging Face serves this host too slowly - the download would cost more"
  log "than the compute. Destroy this box and rent another."
  echo "DL_ABORT rate=${RATE}" > /root/dl_abort
  exit 1
fi
# The watcher keys its stage off this marker; with interleaved fetching there is no
# separate download phase, so declare it once the probe has passed.
echo PROBE_OK > /root/dl_done


# ---------- gemma-4-31b-dense : B11 (17.3 GB) ----------
log "===== gemma-4-31b-dense : fetching 17.3 GB ====="
gate; get gemma-4-31b-dense unsloth/gemma-4-31B-it-qat-GGUF gemma-4-31B-it-qat-UD-Q4_K_XL.gguf &
wait
GG="/root/models/gemma-4-31b-dense/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
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
  log "gemma-4-31b-dense SKIP (fetch failed, no $GG)"; echo "gemma-4-31b-dense missing-gguf" >> /root/failures
fi
stop_server
release "gemma-4-31b-dense"


# ---------- qwen3.6-27b-dense : B11 (19.5 GB) ----------
log "===== qwen3.6-27b-dense : fetching 19.5 GB ====="
gate; get qwen3.6-27b-dense unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q5_K_M.gguf &
wait
GG="/root/models/qwen3.6-27b-dense/Qwen3.6-27B-Q5_K_M.gguf"
if [ -f "$GG" ]; then
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
  log "qwen3.6-27b-dense SKIP (fetch failed, no $GG)"; echo "qwen3.6-27b-dense missing-gguf" >> /root/failures
fi
stop_server
release "qwen3.6-27b-dense"


# ---------- abl-opus-35b-a3b : B4,B5,B7 (17.2 GB) ----------
log "===== abl-opus-35b-a3b : fetching 17.2 GB ====="
gate; get abl-opus-35b-a3b huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-MTP-GGUF Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf &
wait
GG="/root/models/abl-opus-35b-a3b/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-ggml-model-Q3_K.gguf"
if [ -f "$GG" ]; then
  # Phase A - batteries that share one endpoint.
  :   # no shared-endpoint batteries for this model
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
  log "abl-opus-35b-a3b SKIP (fetch failed, no $GG)"; echo "abl-opus-35b-a3b missing-gguf" >> /root/failures
fi
stop_server
release "abl-opus-35b-a3b"


# ---------- gpt-oss-20b : B10,B11 (13.8 GB) ----------
log "===== gpt-oss-20b : fetching 13.8 GB ====="
gate; get gpt-oss-20b unsloth/gpt-oss-20b-GGUF gpt-oss-20b-F16.gguf &
wait
GG="/root/models/gpt-oss-20b/gpt-oss-20b-F16.gguf"
if [ -f "$GG" ]; then
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
  log "gpt-oss-20b SKIP (fetch failed, no $GG)"; echo "gpt-oss-20b missing-gguf" >> /root/failures
fi
stop_server
release "gpt-oss-20b"


# ---------- gemma-4-26b-a4b : B10,B11 (14.2 GB) ----------
log "===== gemma-4-26b-a4b : fetching 14.2 GB ====="
gate; get gemma-4-26b-a4b unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf &
wait
GG="/root/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
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
  log "gemma-4-26b-a4b SKIP (fetch failed, no $GG)"; echo "gemma-4-26b-a4b missing-gguf" >> /root/failures
fi
stop_server
release "gemma-4-26b-a4b"


# ---------- qwen3-coder-30b : B10,B11 (17.7 GB) ----------
log "===== qwen3-coder-30b : fetching 17.7 GB ====="
gate; get qwen3-coder-30b unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf &
wait
GG="/root/models/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
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
  log "qwen3-coder-30b SKIP (fetch failed, no $GG)"; echo "qwen3-coder-30b missing-gguf" >> /root/failures
fi
stop_server
release "qwen3-coder-30b"


# ---------- qwen3.6-35b-a3b : B10,B11 (19.7 GB) ----------
log "===== qwen3.6-35b-a3b : fetching 19.7 GB ====="
gate; get qwen3.6-35b-a3b bartowski/Qwen_Qwen3.6-35B-A3B-GGUF Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf &
wait
GG="/root/models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf"
if [ -f "$GG" ]; then
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
  log "qwen3.6-35b-a3b SKIP (fetch failed, no $GG)"; echo "qwen3.6-35b-a3b missing-gguf" >> /root/failures
fi
stop_server
release "qwen3.6-35b-a3b"


# ---------- agents-a1-35b : B10,B11 (19.8 GB) ----------
log "===== agents-a1-35b : fetching 19.8 GB ====="
gate; get agents-a1-35b jashepp/Agents-A1-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf &
wait
GG="/root/models/agents-a1-35b/Agents-A1-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
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
  log "agents-a1-35b SKIP (fetch failed, no $GG)"; echo "agents-a1-35b missing-gguf" >> /root/failures
fi
stop_server
release "agents-a1-35b"


# ---------- ornith-1.0-35b : B10,B11 (19.8 GB) ----------
log "===== ornith-1.0-35b : fetching 19.8 GB ====="
gate; get ornith-1.0-35b jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf &
wait
GG="/root/models/ornith-1.0-35b/Ornith-1.0-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
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
  log "ornith-1.0-35b SKIP (fetch failed, no $GG)"; echo "ornith-1.0-35b missing-gguf" >> /root/failures
fi
stop_server
release "ornith-1.0-35b"


# ---------- nemotron-3-nano-30b : B10,B11 (22.8 GB) ----------
log "===== nemotron-3-nano-30b : fetching 22.8 GB ====="
gate; get nemotron-3-nano-30b unsloth/Nemotron-3-Nano-30B-A3B-GGUF Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf &
wait
GG="/root/models/nemotron-3-nano-30b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
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
  log "nemotron-3-nano-30b SKIP (fetch failed, no $GG)"; echo "nemotron-3-nano-30b missing-gguf" >> /root/failures
fi
stop_server
release "nemotron-3-nano-30b"


# ---------- laguna-s-2.1 : B4,B5,B7 (57.6 GB) ----------
log "===== laguna-s-2.1 : fetching 57.6 GB ====="
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf &
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00002-of-00003.gguf &
gate; get laguna-s-2.1 unsloth/Laguna-S-2.1-GGUF UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00003-of-00003.gguf &
wait
GG="/root/models/laguna-s-2.1/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf"
if [ -f "$GG" ]; then
  # Phase A - batteries that share one endpoint.
  :   # no shared-endpoint batteries for this model
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
  log "laguna-s-2.1 SKIP (fetch failed, no $GG)"; echo "laguna-s-2.1 missing-gguf" >> /root/failures
fi
stop_server
release "laguna-s-2.1"


# ---------- gpt-oss-120b : B8,B9,B11 (65.4 GB) ----------
log "===== gpt-oss-120b : fetching 65.4 GB ====="
gate; get gpt-oss-120b unsloth/gpt-oss-120b-GGUF gpt-oss-120b-F16.gguf &
wait
GG="/root/models/gpt-oss-120b/gpt-oss-120b-F16.gguf"
if [ -f "$GG" ]; then
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
  log "gpt-oss-120b SKIP (fetch failed, no $GG)"; echo "gpt-oss-120b missing-gguf" >> /root/failures
fi
stop_server
release "gpt-oss-120b"


# ---------- abl-gemma-4-31b : B4,B5,B7 (18.7 GB) ----------
log "===== abl-gemma-4-31b : fetching 18.7 GB ====="
gate; get abl-gemma-4-31b huihui-ai/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-GGUF Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf &
wait
GG="/root/models/abl-gemma-4-31b/Huihui-gemma-4-31B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
if [ -f "$GG" ]; then
  # Phase A - batteries that share one endpoint.
  :   # no shared-endpoint batteries for this model
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
  log "abl-gemma-4-31b SKIP (fetch failed, no $GG)"; echo "abl-gemma-4-31b missing-gguf" >> /root/failures
fi
stop_server
release "abl-gemma-4-31b"


# ---------- bonsai-ternary-27b : B10,B11 (7.2 GB) ----------
log "===== bonsai-ternary-27b : fetching 7.2 GB ====="
gate; get bonsai-ternary-27b prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf &
wait
GG="/root/models/bonsai-ternary-27b/Ternary-Bonsai-27B-Q2_0.gguf"
if [ -f "$GG" ]; then
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
  log "bonsai-ternary-27b SKIP (fetch failed, no $GG)"; echo "bonsai-ternary-27b missing-gguf" >> /root/failures
fi
stop_server
release "bonsai-ternary-27b"


# ---------- ornith-1.0-9b : B10,B11 (9.5 GB) ----------
log "===== ornith-1.0-9b : fetching 9.5 GB ====="
gate; get ornith-1.0-9b jashepp/Ornith-1.0-9B-MXFP4_Hybrid-Imatrix-GGUF Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf &
wait
GG="/root/models/ornith-1.0-9b/Ornith-1.0-9B-MXFP4_Q8_0-Imatrix.gguf"
if [ -f "$GG" ]; then
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
  log "ornith-1.0-9b SKIP (fetch failed, no $GG)"; echo "ornith-1.0-9b missing-gguf" >> /root/failures
fi
stop_server
release "ornith-1.0-9b"


# ---------- granite-4.1-30b : B10,B11 (17.7 GB) ----------
log "===== granite-4.1-30b : fetching 17.7 GB ====="
gate; get granite-4.1-30b unsloth/granite-4.1-30b-GGUF granite-4.1-30b-UD-Q4_K_XL.gguf &
wait
GG="/root/models/granite-4.1-30b/granite-4.1-30b-UD-Q4_K_XL.gguf"
if [ -f "$GG" ]; then
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
  log "granite-4.1-30b SKIP (fetch failed, no $GG)"; echo "granite-4.1-30b missing-gguf" >> /root/failures
fi
stop_server
release "granite-4.1-30b"


# ---------- llama-4-scout : B8,B9,B10,B11 (62.0 GB) ----------
log "===== llama-4-scout : fetching 62.0 GB ====="
gate; get llama-4-scout unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf &
gate; get llama-4-scout unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00002-of-00002.gguf &
wait
GG="/root/models/llama-4-scout/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$GG" ]; then
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
  log "llama-4-scout SKIP (fetch failed, no $GG)"; echo "llama-4-scout missing-gguf" >> /root/failures
fi
stop_server
release "llama-4-scout"


# ---------- glm-4.5-air : B8,B9,B10,B11 (67.7 GB) ----------
log "===== glm-4.5-air : fetching 67.7 GB ====="
gate; get glm-4.5-air unsloth/GLM-4.5-Air-GGUF UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf &
gate; get glm-4.5-air unsloth/GLM-4.5-Air-GGUF UD-Q4_K_XL/GLM-4.5-Air-UD-Q4_K_XL-00002-of-00002.gguf &
wait
GG="/root/models/glm-4.5-air/GLM-4.5-Air-UD-Q4_K_XL-00001-of-00002.gguf"
if [ -f "$GG" ]; then
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
  log "glm-4.5-air SKIP (fetch failed, no $GG)"; echo "glm-4.5-air missing-gguf" >> /root/failures
fi
stop_server
release "glm-4.5-air"


# ---------- abl-qwen3.6-27b : B4,B5,B7,B8,B9 (16.8 GB) ----------
log "===== abl-qwen3.6-27b : fetching 16.8 GB ====="
gate; get abl-qwen3.6-27b huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf &
wait
GG="/root/models/abl-qwen3.6-27b/Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf"
if [ -f "$GG" ]; then
  # Phase A - batteries that share one endpoint.
  if serve "$GG" ""; then
    run_step "abl-qwen3.6-27b" B9 python3 scripts/run_games.py --endpoint-url http://127.0.0.1:8080 --model "abl-qwen3.6-27b" --reps 3 --out $OUT/games --chrome ""
    run_step "abl-qwen3.6-27b" B8 python3 scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 --model "abl-qwen3.6-27b" --results-dir $OUT/b8_abl-qwen3.6-27b
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
  log "abl-qwen3.6-27b SKIP (fetch failed, no $GG)"; echo "abl-qwen3.6-27b missing-gguf" >> /root/failures
fi
stop_server
release "abl-qwen3.6-27b"


# ---------- qwen3-235b : B8,B9,B10,B11 (134.3 GB) ----------
log "===== qwen3-235b : fetching 134.3 GB ====="
gate; get qwen3-235b unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q4_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf &
gate; get qwen3-235b unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q4_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00002-of-00003.gguf &
gate; get qwen3-235b unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF UD-Q4_K_XL/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00003-of-00003.gguf &
wait
GG="/root/models/qwen3-235b/Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf"
if [ -f "$GG" ]; then
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
  log "qwen3-235b SKIP (fetch failed, no $GG)"; echo "qwen3-235b missing-gguf" >> /root/failures
fi
stop_server
release "qwen3-235b"


stop_server
echo ALL_DONE > /root/run_all_done
log "EVERYTHING DONE"
log "step results: $(grep -c ' ok$' /root/steps 2>/dev/null || echo 0) ok, \
$(grep -c ' fail' /root/steps 2>/dev/null || echo 0) failed"
