#!/usr/bin/env bash
# Usage: run_spark_battery.sh <battery-id> <registry-model-id> [served-name]
set -euo pipefail
BATTERY="${1:?battery}"
MODEL="${2:?model id}"
SERVED="${3:-$MODEL}"
export HOME=/home/michaeldeblok
export LLMTEST_HARDWARE_SKU=dgx-spark-gb10
export LLMTEST_TIER=T_GB10
export LLMTEST_RUNTIME=endpoint
export LLMTEST_SUITE_VERSION=suite-v2.3.0-spark
export LLMTEST_RESULTS_DIR="$HOME/llmtest-spark/out"
export LLMTEST_ENDPOINT_URL=http://127.0.0.1:8888
export LLMTEST_SERVED_MODEL="$SERVED"
export LLMTEST_TP=2
export LLMTEST_TOPOLOGY=2x-dgx-spark-gb10
export LLMTEST_SPEC=dflash
export LLMTEST_ROOT="$HOME/llmtest-spark/src"
mkdir -p "$LLMTEST_RESULTS_DIR"
# shellcheck disable=SC1091
source "$HOME/llmtest-spark/venv/bin/activate"
cd "$LLMTEST_ROOT"
export PYTHONUNBUFFERED=1
exec python -m llmtest run --battery "$BATTERY" --model "$MODEL" --keep-server
