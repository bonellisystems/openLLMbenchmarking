#!/usr/bin/env bash
# Rebuild progress.html every 20s and serve it on :8890.
set -euo pipefail
OUT=/home/michaeldeblok/llmtest-spark/out
SRC=/home/michaeldeblok/llmtest-spark/src
PY=/home/michaeldeblok/llmtest-spark/venv/bin/python
mkdir -p "$OUT"
pkill -f "http.server 8890" 2>/dev/null || true
pkill -f "build_spark_progress.py --watch" 2>/dev/null || true
nohup bash -c "
  while true; do
    $PY $SRC/scripts/build_spark_progress.py --root $OUT --html $OUT/progress.html --json $OUT/progress.json || true
    sleep 20
  done
" >/home/michaeldeblok/llmtest-spark/out/progress-watch.log 2>&1 &
echo "WATCH_PID $!"
cd "$OUT"
nohup $PY -m http.server 8890 --bind 0.0.0.0 >/home/michaeldeblok/llmtest-spark/out/progress-http.log 2>&1 &
echo "HTTP_PID $!"
sleep 1
$PY $SRC/scripts/build_spark_progress.py --root $OUT --html $OUT/progress.html --json $OUT/progress.json
echo "OPEN http://192.168.0.74:8890/progress.html"
