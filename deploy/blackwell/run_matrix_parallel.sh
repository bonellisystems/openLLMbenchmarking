#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Parallel 2-GPU B8 run: gpt-oss on GPU0:8080, gemma on GPU1:8081. RUNS ONLY
# after smoke.sh passes. Replicate-ROUND ordering (one --force replicate of
# every task per round) so every cell gets EVEN coverage even if the budget
# cuts the run short. Each model writes its OWN results dir (results_gpt/,
# results_gemma/) -- two parallel run_b8_local processes NEVER append to the
# same shard (interleave-corruption guard); merge per-model shards at report
# time. Appends per run; the LOCAL watch_and_teardown.sh pulls both dirs back
# every ~2 min and tears the box down at the balance floor (data already safe).
#
#     MAX_ROUNDS=5 nohup bash deploy/blackwell/run_matrix_parallel.sh &
# ---------------------------------------------------------------------------
set -uo pipefail                       # NOT -e: one task's nonzero must not abort the run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
B8_ROOT="${B8_ROOT:-/opt/b8}"
PY="${PY:-$B8_ROOT/venv/bin/python}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
STATUS="$B8_ROOT/run_status.txt"
cd "$REPO"
rm -f "$B8_ROOT/RUN_DONE"

E0=http://127.0.0.1:8080; M0=gpt-oss-20b;           G0=gpt-oss-20b.gguf;     D0="$REPO/results_gpt"
E1=http://127.0.0.1:8081; M1=gemma-4-26b-a4b-mxfp4; G1=gemma-4-26b-a4b.gguf; D1="$REPO/results_gemma"
mkdir -p "$D0" "$D1"

TASKS="$("$PY" -c 'from llmtest.registry import load_config; from pathlib import Path; print(" ".join(load_config(Path.cwd()).suite["b8"]["tasks"]))')"
NT=$(wc -w <<<"$TASKS")

log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$STATUS"; }
healthy(){ curl -s -m3 "$1/health" 2>/dev/null | grep -q '"ok"'; }
ensure(){ healthy "$4" || { log "endpoint $4 down -> restart $2"; bash "$SCRIPT_DIR/serve.sh" "$1" "$3" "${4##*:}"; }; }

log "SERVE gpt-oss(GPU0:8080) + gemma(GPU1:8081); $NT tasks x $MAX_ROUNDS rounds x 2 models"
bash "$SCRIPT_DIR/serve.sh" "$G0" 0 8080
bash "$SCRIPT_DIR/serve.sh" "$G1" 1 8081

for r in $(seq 1 "$MAX_ROUNDS"); do
  log "ROUND $r/$MAX_ROUNDS begin"
  n=0
  for T in $TASKS; do
    n=$((n+1))
    ensure "$G0" "$M0" 0 "$E0"
    ensure "$G1" "$M1" 1 "$E1"
    "$PY" scripts/run_b8_local.py --endpoint-url "$E0" --model "$M0" --task "$T" --force --results-dir "$D0" >>"$B8_ROOT/gpt.log" 2>&1 &
    p0=$!
    "$PY" scripts/run_b8_local.py --endpoint-url "$E1" --model "$M1" --task "$T" --force --results-dir "$D1" >>"$B8_ROOT/gemma.log" 2>&1 &
    p1=$!
    wait $p0; wait $p1
    log "  round $r task $n/$NT ($T) done"
  done
  log "ROUND $r/$MAX_ROUNDS complete"
done
bash "$SCRIPT_DIR/serve.sh" stop
log "ALL ROUNDS DONE"
touch "$B8_ROOT/RUN_DONE"
