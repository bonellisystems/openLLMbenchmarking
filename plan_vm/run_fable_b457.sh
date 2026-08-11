#!/usr/bin/env bash
# The last open cells: qwen3.6-27b-fable-fusion B4, B5, B7.
#
# WHY THESE NEED THE CANONICAL RUNNER AND NOT bigmodel_gen:
# bigmodel_gen hands every battery ONE pre-launched server. That is fine for
# B1/B2/B3/B6, which only generate against a fixed config. It is wrong here:
#   B4 sweeps context length 16k -> 256k, one arm per length
#   B5 measures throughput with spec-decode ON and OFF as separate arms
#   B7 varies system prompt / temperature / tool format / spec-decode
# Each of those asks ServerManager for a DIFFERENT serving config per arm. Given
# a fixed handle they would all silently measure the same one and the rows would
# record arms that never ran - the same provenance trap that put spec=ngram32 on
# rows served without it earlier in this campaign.
#
# ServerManager was Windows-only until now (taskkill, tier pinned to the 24 GB
# laptop, a Windows binary path), which is the actual reason these three cells
# were never measured on a rented card. The three env overrides below are what
# make it work here; see llmtest/server.py.
set -u
export B8_ROOT=/opt/b8
REPO=$B8_ROOT/llmtest-v2
PY=$B8_ROOT/venv/bin/python
M=$B8_ROOT/models
MODEL=qwen3.6-27b-fable-fusion
FF_REPO="DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF"
FF_FILE="Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q4_K_M.gguf"
WANT=18047255072
cd "$REPO"
log(){ echo "$(date -u +%H:%M:%S) b457: $*" | tee -a $B8_ROOT/run.log; }

# ---- 1. a NATIVE llama-server on the box -------------------------------------
# ServerManager launches a bare binary with Popen; it does not speak Docker. The
# llama.cpp CUDA image already has one, so lift it out rather than compiling.
# libgomp1 because ggml links OpenMP and the runtime image does not ship it -
# the same defect that made the prism image build clean and die on first exec.
if [ ! -x $B8_ROOT/llama/llama-server ]; then
  log "extracting a native llama-server from the CUDA image"
  apt-get install -y -qq libgomp1 >/dev/null 2>&1 || true
  docker pull -q ghcr.io/ggml-org/llama.cpp:server-cuda >/dev/null 2>&1
  cid=$(docker create ghcr.io/ggml-org/llama.cpp:server-cuda)
  rm -rf $B8_ROOT/llama; mkdir -p $B8_ROOT/llama
  docker cp "$cid:/app/." $B8_ROOT/llama/ >/dev/null 2>&1
  docker rm -f "$cid" >/dev/null 2>&1
  chmod +x $B8_ROOT/llama/llama-server 2>/dev/null || true
fi
export LD_LIBRARY_PATH=$B8_ROOT/llama:${LD_LIBRARY_PATH:-}
if ! $B8_ROOT/llama/llama-server --version >/dev/null 2>&1; then
  log "FATAL: extracted llama-server will not run"
  $B8_ROOT/llama/llama-server --version 2>&1 | tail -5 | tee -a $B8_ROOT/run.log
  touch $B8_ROOT/final_done; exit 1
fi
log "native llama-server OK: $($B8_ROOT/llama/llama-server --version 2>&1 | head -1)"

# ---- 2. weights ---------------------------------------------------------------
mkdir -p "$M/$MODEL"
for attempt in 1 2 3; do
  GOT=$(stat -c %s "$M/$MODEL/$FF_FILE" 2>/dev/null || echo 0)
  [ "$GOT" = "$WANT" ] && break
  log "download attempt $attempt ($GOT/$WANT)"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
    --lowest-speed-limit=1M --retry-wait=10 --max-tries=5 --auto-file-renaming=false \
    -d "$M/$MODEL" -o "$FF_FILE" \
    "https://huggingface.co/$FF_REPO/resolve/main/$FF_FILE" >> $B8_ROOT/dl_b457.log 2>&1
done
GOT=$(stat -c %s "$M/$MODEL/$FF_FILE" 2>/dev/null || echo 0)
if [ "$GOT" != "$WANT" ]; then
  log "FATAL: weights incomplete ($GOT/$WANT)"; touch $B8_ROOT/final_done; exit 1
fi
log "weights size-verified"

# ---- 3. point the registry at the box's copy ----------------------------------
# Box-local edit only, never committed: local_path is inherently per-machine.
$PY - "$M/$MODEL/$FF_FILE" <<'EOS'
import re, sys, pathlib
p = pathlib.Path("config/registry.yaml"); s = p.read_text(encoding="utf-8")
new = sys.argv[1]
s = re.sub(r"(?m)^(    local_path: ).*Qwen3\.6-27B-Fable-Fus.*$", r"\1" + new, s)
p.write_text(s, encoding="utf-8")
print("local_path ->", new)
EOS

# ---- 4. run the three batteries ----------------------------------------------
export LLMTEST_TIER=T3                                  # 96 GB card, not the 24 GB laptop
export LLMTEST_FORK_BINARY=$B8_ROOT/llama/llama-server   # not the Windows pin
for B in 4 5 7; do
  log "B$B start"
  "$PY" -m llmtest run --model "$MODEL" --battery "$B" > $B8_ROOT/last_b457.log 2>&1
  rc=$?
  tail -4 $B8_ROOT/last_b457.log | tee -a $B8_ROOT/run.log
  if [ "$rc" -eq 0 ]; then echo "$MODEL B$B ok" >> $B8_ROOT/steps; log "B$B OK"
  else echo "$MODEL B$B fail rc=$rc" >> $B8_ROOT/steps; log "B$B FAILED rc=$rc"
       cat $B8_ROOT/last_b457.log >> $B8_ROOT/step_failures.log; fi
done

rm -rf "${M:?}/$MODEL"
touch $B8_ROOT/final_done
log "B4/B5/B7 COMPLETE: $(grep -c ' ok$' $B8_ROOT/steps 2>/dev/null || echo 0) ok"
