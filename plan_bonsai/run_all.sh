#!/bin/bash
# Serve bonsai-ternary-27b from the prism fork and close B10 + B11.
set -u
cd /root/llmtest-v2
BIN=/root/prism-llama/build/bin/llama-server
export LD_LIBRARY_PATH=/root/prism-llama/build/bin
OUT=/root/out
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*" | tee -a /root/run.log; }

if [ -f /root/build_failed ]; then
  log "PRISM BUILD FAILED: $(cat /root/build_failed) - cannot serve Q2_0, aborting"
  echo "bonsai-ternary-27b prism-build-failed" >> /root/failures
  echo ALL_DONE > /root/run_all_done
  exit 1
fi

run_step(){ # $1 model $2 battery $3.. command
  mid="$1"; bat="$2"; shift 2
  log "  $mid $bat start"
  "$@" > /root/last_step.log 2>&1
  rc=$?
  tail -3 /root/last_step.log | tee -a /root/run.log
  if [ "$rc" -eq 0 ]; then echo "$mid $bat ok" >> /root/steps
  else log "  $mid $bat FAILED rc=$rc"; echo "$mid $bat fail rc=$rc" >> /root/steps
       cat /root/last_step.log >> /root/step_failures.log; fi
}

M=/root/models/bonsai-ternary-27b
mkdir -p $M
log "fetching Ternary-Bonsai-27B-Q2_0.gguf"
aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
  -d $M -o "Ternary-Bonsai-27B-Q2_0.gguf" "https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_0.gguf" \
  >> /root/dl.log 2>&1 || echo "FAIL fetch" >> /root/dl_fail

GG="$M/Ternary-Bonsai-27B-Q2_0.gguf"
if [ ! -f "$GG" ]; then
  log "bonsai-ternary-27b SKIP (fetch failed)"; echo "bonsai-ternary-27b missing-gguf" >> /root/failures
  echo ALL_DONE > /root/run_all_done; exit 1
fi

log "===== bonsai-ternary-27b (prism fork, Q2_0) ====="
pkill -f "prism-llama/build/bin/llama-server" 2>/dev/null; sleep 3
nohup $BIN -m "$GG" -ngl 99 -c 49152 --parallel 1 --jinja -fa on \
  -ctk q8_0 -ctv q8_0 --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
  --host 127.0.0.1 --port 8080 --no-webui > /root/serve.log 2>&1 &
ok=0
for i in $(seq 1 200); do
  curl -s -m3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && { ok=1; break; }
  sleep 4
done
if [ "$ok" != "1" ]; then
  log "bonsai-ternary-27b SERVE-FAIL even on the prism fork - see /root/serve.log"
  echo "bonsai-ternary-27b serve-fail-on-prism" >> /root/failures
else
  mkdir -p /root/agentws
  run_step "bonsai-ternary-27b" B11 python3 scripts/run_tools_agent.py \
    --endpoint-url http://127.0.0.1:8080 --model "bonsai-ternary-27b" --reps 3 \
    --workspace /root/agentws --out $OUT/tools
  run_step "bonsai-ternary-27b" B10 python3 scripts/run_security.py \
    --endpoint-url http://127.0.0.1:8080 --model "bonsai-ternary-27b" --reps 3 \
    --out $OUT/security
  echo "bonsai-ternary-27b" >> /root/models_done
fi
pkill -f "prism-llama/build/bin/llama-server" 2>/dev/null
echo ALL_DONE > /root/run_all_done
log "EVERYTHING DONE"
