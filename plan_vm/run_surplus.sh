#!/bin/bash
# Follow-on queue, run AFTER the main sweep finishes.
#
# Why this exists as a separate chain: run_all.sh writes run_all_done at its end, and
# deploy/shutdown_guard.sh powers the box off 15 minutes later. Anything still to do
# has to be sequenced here, with the guard re-pointed at the all_done flag this script
# writes (DONE_FLAG=all_done) so it protects the WHOLE session instead of cutting it
# short. The guard is never disarmed - an unguarded box idles to the credit floor.
#
# Order is value-per-minute, cheapest wins first, so a credit shortfall costs the least:
#   1. qwen3.6-27b-fable-fusion  (Michael's request, 2 new cells, ~1h)
#   2. nemotron-3-nano-30b B9    (completes a cell left at 4/24 when a game wedged
#                                 the browser; B8 for this model is already clean)
#   3. bonsai-ternary-27b        (4 cells, but needs the prism image built first -
#                                 CUDA 12.8 now, since 12.6 cannot compile sm_120)
set -u
export B8_ROOT=/opt/b8
REPO=$B8_ROOT/llmtest-v2
cd "$REPO"
log(){ echo "$(date -u +%H:%M:%S) surplus: $*" | tee -a $B8_ROOT/run.log; }

while [ ! -f $B8_ROOT/run_all_done ]; do sleep 30; done
log "main sweep complete - starting follow-on queue"

log "1/3 qwen3.6-27b-fable-fusion (requested mid-campaign)"
bash plan_vm/run_fable.sh || log "fable runner exited non-zero"

log "2/3 nemotron-3-nano-30b B9 re-run"
bash plan_vm/run_nemo_b9.sh || log "nemotron B9 runner exited non-zero"

log "3/3 bonsai: prism image, then its four cells"
# Built HERE and not during the sweep on purpose: this compile saturates CPU, and B8
# enforces a 180s per-run wall-clock budget - starving the harness would fabricate
# budget_exceeded rows. Nothing else runs now, so it is safe.
if timeout 2700 docker build -t prism-llama:1 \
     -f deploy/blackwell/Dockerfile.prism deploy/blackwell \
     > $B8_ROOT/prism_build2.log 2>&1; then
  log "prism image built - running bonsai"
  rm -f $B8_ROOT/prism_missing
  bash plan_vm/run_bonsai.sh || log "bonsai runner exited non-zero"
else
  log "PRISM BUILD FAILED AGAIN - bonsai stays deferred; see prism_build2.log"
  tail -25 $B8_ROOT/prism_build2.log | tee -a $B8_ROOT/run.log
fi

log "follow-on queue complete"
touch $B8_ROOT/all_done
