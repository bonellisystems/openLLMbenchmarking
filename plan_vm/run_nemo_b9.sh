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
  # --lowest-speed-limit: abandon a WEDGED connection instead of crawling on it.
  # abl-gemma-4-31b's fetch collapsed to 894 KiB/s on one stuck CDN edge while fresh
  # curls to the same repo pulled 16 MB/s - 40 minutes of rented GPU idling for 2 GB.
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
    --lowest-speed-limit=2M \
    --retry-wait=5 --max-tries=10 --auto-file-renaming=false \
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
  "https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/resolve/main/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf" >/dev/null 2>&1 || true
PSZ=$(stat -c %s /tmp/probe.bin 2>/dev/null || echo 0); rm -f /tmp/probe.bin
RATE=$(( PSZ / 60 / 1000000 ))
log "HF throughput probe: ~${RATE} MB/s (floor ${MIN_MBPS})"
if [ "$RATE" -lt "$MIN_MBPS" ]; then
  log "ABORT: HF too slow on this host - destroy the box, rent another."
  echo "DL_ABORT rate=${RATE}" > $B8_ROOT/dl_abort; exit 1
fi
echo PROBE_OK > $B8_ROOT/dl_done

# ---------- nemotron-3-nano-30b : B9 (22.8 GB) ----------
log "===== nemotron-3-nano-30b : fetching 22.8 GB ====="
gate; get nemotron-3-nano-30b unsloth/Nemotron-3-Nano-30B-A3B-GGUF Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf &
wait
GG="nemotron-3-nano-30b/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "ghcr.io/ggml-org/llama.cpp:server-cuda"; then
    run_step "nemotron-3-nano-30b" B9 "$PY" scripts/run_games.py $EP --model "nemotron-3-nano-30b" --reps 3 --suite-version suite-v2.2.0 --hardware-sku rtx-pro-6000-vm --out $OUT/games --chrome ""
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

stop_server
echo ALL_DONE > $B8_ROOT/run_all_done
log "EVERYTHING DONE: $(grep -c ' ok$' $B8_ROOT/steps 2>/dev/null || echo 0) ok, \
$(grep -c ' fail' $B8_ROOT/steps 2>/dev/null || echo 0) failed, \
$(grep -c ' skip' $B8_ROOT/steps 2>/dev/null || echo 0) skipped"
