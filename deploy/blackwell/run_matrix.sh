#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the FULL B8 dev-form matrix on this box: every model in suite.yaml's
# b8.models x every task in b8.tasks x b8.replicates. Serves each model once,
# then runs ONE task at a time so a dead server is caught + restarted BETWEEN
# tasks (belt-and-suspenders on top of the --ctx-checkpoints 0 fix); within a
# task, run_b8_local's own 3x infra-error retry bridges any transient blip.
#
# Produces ONLY the raw completion rows (results/rows-<suite>.jsonl + the
# persisted Traces). Classification + report run OFF-BOX (they need the judge
# panel / agy creds) -- see RUN_ON_BLACKWELL.md "Pull back + classify".
#
#     bash deploy/blackwell/run_matrix.sh              # all models, all tasks
#     MODELS_OVERRIDE="gpt-oss-20b" bash …/run_matrix.sh   # one model
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
B8_ROOT="${B8_ROOT:-/opt/b8}"
PY="${PY:-$B8_ROOT/venv/bin/python}"
ENDPOINT="http://127.0.0.1:8080"

# model-id (suite.yaml b8.models)  ->  stable gguf name (fetch_models.sh)
declare -A GGUF=(
  [gpt-oss-20b]=gpt-oss-20b.gguf
  [gemma-4-26b-a4b-mxfp4]=gemma-4-26b-a4b.gguf
)
MODELS="${MODELS_OVERRIDE:-${!GGUF[@]}}"

cd "$REPO"
TASKS="$("$PY" - <<'PY'
from llmtest.registry import load_config
from pathlib import Path
cfg = load_config(Path.cwd())
print(" ".join(cfg.suite["b8"]["tasks"]))
PY
)"
NTASK=$(wc -w <<<"$TASKS")
echo "== models: $MODELS =="
echo "== $NTASK dev-form tasks x b8.replicates each =="

healthy () { curl -s -m 3 "$ENDPOINT/health" 2>/dev/null | grep -q '"ok"'; }

for MODEL in $MODELS; do
  echo; echo "########## MODEL $MODEL (${GGUF[$MODEL]}) ##########"
  bash "$SCRIPT_DIR/serve.sh" "${GGUF[$MODEL]}"
  n=0
  for TASK in $TASKS; do
    n=$((n+1))
    if ! healthy; then
      echo "  [$n/$NTASK] endpoint DOWN before $TASK -> restarting server"
      bash "$SCRIPT_DIR/serve.sh" "${GGUF[$MODEL]}"
    fi
    echo "  [$n/$NTASK] $MODEL :: $TASK"
    "$PY" scripts/run_b8_local.py --endpoint-url "$ENDPOINT" --model "$MODEL" --task "$TASK" \
       || echo "  (run_b8_local returned nonzero for $TASK -- see rows; continuing)"
  done
  bash "$SCRIPT_DIR/serve.sh" stop
done

echo; echo "== matrix done. Row summary: =="
"$PY" - <<'PY'
import json, collections, glob, os
for f in sorted(glob.glob("results/rows-*.jsonl")):
    if "v2.1" not in f:  # B8 lives in the v2.1.x shard
        continue
    rows=[json.loads(l) for l in open(f)]
    b8=[r for r in rows if r.get("battery")==8 or str(r.get("battery_id","")).endswith("8")]
    by=collections.Counter((r["model_id"], r["metrics"]["terminal_status"]) for r in rows)
    comp=collections.Counter(r["model_id"] for r in rows if r["metrics"].get("completion"))
    print(os.path.basename(f), "rows:", len(rows))
    for (m,ts),c in sorted(by.items()):
        print(f"   {m:26} {ts:16} {c}")
    print("   completed:", dict(comp))
PY
echo "Next (OFF-BOX): pull results/ + artifacts/b8_traces back, then classify + p8_report locally."
