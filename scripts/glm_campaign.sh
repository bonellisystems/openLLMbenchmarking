#!/usr/bin/env bash
# Sequential GLM-5.3-Flash DFlash2 batteries against the live :8888 endpoint.
set -euo pipefail
export HOME=/home/michaeldeblok
export LLMTEST_HARDWARE_SKU=dgx-spark-gb10
export LLMTEST_TIER=T_GB10
export LLMTEST_RUNTIME=endpoint
export LLMTEST_SUITE_VERSION=suite-v2.3.0-spark
export LLMTEST_RESULTS_DIR="$HOME/llmtest-spark/out"
export LLMTEST_ENDPOINT_URL=http://127.0.0.1:8888
export LLMTEST_SERVED_MODEL=glm-5.3-flash-dflash2
export LLMTEST_TP=2
export LLMTEST_TOPOLOGY=2x-dgx-spark-gb10
export LLMTEST_SPEC=dflash
export PYTHONUNBUFFERED=1
export LLMTEST_ROOT="$HOME/llmtest-spark/src"
mkdir -p "$LLMTEST_RESULTS_DIR" "$LLMTEST_RESULTS_DIR/tools" "$LLMTEST_RESULTS_DIR/security" "$LLMTEST_RESULTS_DIR/games" "$LLMTEST_RESULTS_DIR/b8_glm-5.3-flash-dflash2"
# shellcheck disable=SC1091
source "$HOME/llmtest-spark/venv/bin/activate"
cd "$LLMTEST_ROOT"
LOG="$LLMTEST_RESULTS_DIR/campaign.log"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

wait_b2() {
  while pgrep -f "llmtest run --battery 2 --model glm-5.3-flash-dflash2" >/dev/null; do
    say "waiting for B2 pid"
    sleep 30
  done
}

run_llm() {
  local bat="$1"
  say "START B${bat}"
  python -m llmtest run --battery "$bat" --model glm-5.3-flash-dflash2 --keep-server \
    >> "$LLMTEST_RESULTS_DIR/b${bat}.log" 2>&1 || say "B${bat} exited $?"
  say "END B${bat}"
}

wait_b2

# B2 may already have finished via start_glm_b2.sh
if ! grep -q '"task_id": "b2.' "$LLMTEST_RESULTS_DIR/rows-suite-v2.3.0-spark.jsonl" 2>/dev/null; then
  run_llm 2
else
  say "B2 rows present, skip re-run"
fi

say "START B11"
python scripts/run_tools_agent.py \
  --endpoint-url http://127.0.0.1:8888 \
  --model glm-5.3-flash-dflash2 \
  --workspace /tmp/agentws-glm \
  --reps 3 \
  --suite-version suite-v2.3.0-spark \
  --hardware-sku dgx-spark-gb10 \
  --out "$LLMTEST_RESULTS_DIR/tools" \
  >> "$LLMTEST_RESULTS_DIR/b11.log" 2>&1 || say "B11 exited $?"
say "END B11"

run_llm 6
run_llm 3

say "START B10"
python scripts/run_security.py \
  --endpoint-url http://127.0.0.1:8888 \
  --model glm-5.3-flash-dflash2 \
  --reps 3 \
  --suite-version suite-v2.3.0-spark \
  --hardware-sku dgx-spark-gb10 \
  --out "$LLMTEST_RESULTS_DIR/security" \
  >> "$LLMTEST_RESULTS_DIR/b10.log" 2>&1 || say "B10 exited $?"
say "END B10"

say "START B9"
python scripts/run_games.py \
  --endpoint-url http://127.0.0.1:8888 \
  --model glm-5.3-flash-dflash2 \
  --reps 3 \
  --suite-version suite-v2.3.0-spark \
  --hardware-sku dgx-spark-gb10 \
  --out "$LLMTEST_RESULTS_DIR/games" \
  --chrome "" \
  >> "$LLMTEST_RESULTS_DIR/b9.log" 2>&1 || say "B9 exited $?"
say "END B9"

run_llm 7
run_llm 4
run_llm 5
run_llm 1

say "START B8 probe"
python scripts/run_b8_local.py \
  --endpoint-url http://127.0.0.1:8888 \
  --model glm-5.3-flash-dflash2 \
  --hardware-sku dgx-spark-gb10 \
  --results-dir "$LLMTEST_RESULTS_DIR/b8_glm-5.3-flash-dflash2" \
  --task py-brk-01 --limit 1 \
  >> "$LLMTEST_RESULTS_DIR/b8_probe.log" 2>&1 || say "B8 probe exited $?"
say "END B8 probe"
say "CAMPAIGN_GLM_DONE"
