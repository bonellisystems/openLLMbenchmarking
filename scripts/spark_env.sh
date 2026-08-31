#!/usr/bin/env bash
# Source this on a Spark before any llmtest / B8-B11 command.
# Does not launch a model. Attach-only.
export LLMTEST_HARDWARE_SKU=dgx-spark-gb10
export LLMTEST_TIER=T_GB10
export LLMTEST_RUNTIME=endpoint
export LLMTEST_SUITE_VERSION=suite-v2.3.0-spark
export LLMTEST_RESULTS_DIR="${LLMTEST_RESULTS_DIR:-$HOME/llmtest-spark/out}"
export LLMTEST_ENDPOINT_URL="${LLMTEST_ENDPOINT_URL:-http://127.0.0.1:8888}"
export LLMTEST_TP="${LLMTEST_TP:-2}"
export LLMTEST_TOPOLOGY=2x-dgx-spark-gb10
export LLMTEST_SPEC="${LLMTEST_SPEC:-dflash}"
mkdir -p "$LLMTEST_RESULTS_DIR"
cd "${LLMTEST_ROOT:-$HOME/llmtest-spark/src}"
# shellcheck disable=SC1091
source "$HOME/llmtest-spark/venv/bin/activate"
