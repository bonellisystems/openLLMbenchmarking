#!/usr/bin/env bash
# 1-hour Spark shop card against the live OpenAI endpoint.
set -euo pipefail
ENDPOINT="${ENDPOINT:-http://127.0.0.1:8888}"
MODEL="${1:-glm-5.3-flash-dflash2}"
OUT="${2:-$HOME/llmtest-spark/out/hour-$MODEL}"
SRC="${SUITE:-$HOME/llmtest-spark/src}"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
cd "$SRC"
exec python3 -m spark_hour.run \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --out "$OUT" \
  --suite "$SRC" \
  --budget-s 3600
