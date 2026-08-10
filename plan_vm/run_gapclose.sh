#!/usr/bin/env bash
# Follow-on queue: the two GAP models, run after bonsai's four withdrawn cells.
#
# These are not hardware-consistency re-runs - they are cells that were never
# measured at all:
#   abl-qwen3.6-27b          B1 only (every other battery is already complete)
#   qwen3.6-27b-fable-fusion B1,B2,B3,B6,B10,B11 (B8/B9 landed mid-campaign)
#
# WHY B1 NEEDS A CUSTOM RUNNER: scripts/emit_vm_plan.py strips B1 from every
# plan it emits ("B1 is judging-bound, not GPU-bound"). That is true of the
# JUDGING half and false of the half that matters here - abl-qwen3.6-27b has
# zero B1 rows, and the answers have to be GENERATED on a GPU before any judge
# can score them. bigmodel_gen.py drives B1 directly (its default battery set is
# literally 1,2,3,6) and is the same driver the laguna run used, so this reaches
# for it rather than teaching the emitter a new trick mid-campaign.
#
# Ordered cheapest-useful-first so a credit shortfall costs the least: the model
# needing ONE cell goes first, then the model needing six.
set -u
export B8_ROOT=/opt/b8
REPO=$B8_ROOT/llmtest-v2
PY=$B8_ROOT/venv/bin/python
OUT=$B8_ROOT/out
M=$B8_ROOT/models
EP="--endpoint-url http://127.0.0.1:8080"
mkdir -p "$OUT" "$M" "$B8_ROOT/agentws"
cd "$REPO"
log(){ echo "$(date -u +%H:%M:%S) gapclose: $*" | tee -a $B8_ROOT/run.log; }

run_step(){ mid="$1"; bat="$2"; shift 2
  log "  $mid $bat start"
  "$@" > $B8_ROOT/last_step.log 2>&1; rc=$?
  tail -3 $B8_ROOT/last_step.log | tee -a $B8_ROOT/run.log
  if [ "$rc" -eq 0 ]; then echo "$mid $bat ok" >> $B8_ROOT/steps
  else log "  $mid $bat FAILED rc=$rc"; echo "$mid $bat fail rc=$rc" >> $B8_ROOT/steps
       cat $B8_ROOT/last_step.log >> $B8_ROOT/step_failures.log; fi
  return $rc; }

get(){ mkdir -p "$M/$1"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
    --lowest-speed-limit=1M --retry-wait=10 --max-tries=5 --auto-file-renaming=false \
    -d "$M/$1" -o "$(basename "$3")" \
    "https://huggingface.co/$2/resolve/main/$3" >> "$B8_ROOT/dl_$1.log" 2>&1 \
    || echo "FAIL $1 $3" >> $B8_ROOT/dl_fail; }
release(){ du -sh "$M/$1" 2>/dev/null | tee -a $B8_ROOT/run.log; rm -rf "${M:?}/$1"
           log "  released $1 ; free: $(df -h $B8_ROOT | awk 'NR==2{print $4}')"; }
serve_model(){ LLAMA_IMAGE="${2:-ghcr.io/ggml-org/llama.cpp:server-cuda}" \
                 bash deploy/blackwell/serve.sh "$1"; }
stop_server(){ bash deploy/blackwell/serve.sh stop; }

# ONLY for the custom B9/B10/B11 runners. bigmodel_gen accepts neither flag and
# would die on an unrecognised argument: suite rows (B1-B7) take suite_version
# from config/suite.yaml through the store, and record provenance as
# session_id=bmg-<model> with hardware_sku unset - verified against the
# campaign's own abl-qwen3.6-27b B2 rows. Matching that, not improving on it.
SV="--suite-version suite-v2.2.0"
HW="--hardware-sku rtx-pro-6000-vm"

# Wait for the bonsai queue to finish before touching the GPU.
while [ ! -f $B8_ROOT/run_all_done ]; do sleep 30; done
log "bonsai queue done - starting gap models"

# ---------- abl-qwen3.6-27b : B1 generation only (~18 GB) ----------
log "===== abl-qwen3.6-27b : B1 generation ====="
AQ_REPO="huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF"
AQ_FILE="Huihui-Qwen3.6-27B-abliterated-ggml-model-Q4_K.gguf"
get abl-qwen3.6-27b "$AQ_REPO" "$AQ_FILE"
GG="abl-qwen3.6-27b/$AQ_FILE"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG"; then
    # --batteries 1 ONLY: everything else for this model is already measured on
    # this card, and re-running would write rows that dedupe against themselves.
    run_step "abl-qwen3.6-27b" B1 "$PY" scripts/bigmodel_gen.py --model abl-qwen3.6-27b \
      --batteries 1 --results-dir $OUT/suite $EP
    echo "abl-qwen3.6-27b" >> $B8_ROOT/models_done
  else
    log "abl-qwen3.6-27b SERVE-FAIL"; echo "abl-qwen3.6-27b serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "abl-qwen3.6-27b SKIP (fetch failed)"; echo "abl-qwen3.6-27b missing-gguf" >> $B8_ROOT/failures
fi
release abl-qwen3.6-27b

# ---------- qwen3.6-27b-fable-fusion : B1,B2,B3,B6 then B10,B11 (18.0 GB) ----------
log "===== qwen3.6-27b-fable-fusion : B1,B2,B3,B6,B10,B11 ====="
FF_REPO="DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF"
FF_FILE="Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q4_K_M.gguf"
get qwen3.6-27b-fable-fusion "$FF_REPO" "$FF_FILE"
GG="qwen3.6-27b-fable-fusion/$FF_FILE"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG"; then
    # B10/B11 first: minutes each, and they are whole cells. B1 is the long pole
    # (120 tasks x 3 runs) so it goes last - if credit runs out mid-B1 we lose a
    # partial generation, not two complete cells.
    run_step "qwen3.6-27b-fable-fusion" B10 "$PY" scripts/run_security.py $EP \
      --model qwen3.6-27b-fable-fusion --reps 3 $SV $HW --out $OUT/security
    run_step "qwen3.6-27b-fable-fusion" B11 "$PY" scripts/run_tools_agent.py $EP \
      --model qwen3.6-27b-fable-fusion --reps 3 $SV $HW --out $OUT/tools \
      --workspace $B8_ROOT/agentws
    run_step "qwen3.6-27b-fable-fusion" B2B3B6 "$PY" scripts/bigmodel_gen.py \
      --model qwen3.6-27b-fable-fusion --batteries 2,3,6 --results-dir $OUT/suite $EP
    run_step "qwen3.6-27b-fable-fusion" B1 "$PY" scripts/bigmodel_gen.py \
      --model qwen3.6-27b-fable-fusion --batteries 1 --results-dir $OUT/suite $EP
    echo "qwen3.6-27b-fable-fusion" >> $B8_ROOT/models_done
  else
    log "fable-fusion SERVE-FAIL"; echo "qwen3.6-27b-fable-fusion serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "fable-fusion SKIP (fetch failed)"; echo "qwen3.6-27b-fable-fusion missing-gguf" >> $B8_ROOT/failures
fi
release qwen3.6-27b-fable-fusion

stop_server
touch $B8_ROOT/all_done
log "GAP QUEUE COMPLETE: $(grep -c ' ok$' $B8_ROOT/steps 2>/dev/null || echo 0) ok, \
$(grep -c ' fail' $B8_ROOT/steps 2>/dev/null || echo 0) failed"
