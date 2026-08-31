#!/usr/bin/env bash
set -euo pipefail
chmod +x /home/michaeldeblok/llmtest-spark/src/scripts/run_spark_battery.sh
mkdir -p /home/michaeldeblok/llmtest-spark/out
nohup bash /home/michaeldeblok/llmtest-spark/src/scripts/run_spark_battery.sh \
  2 glm-5.3-flash-dflash2 glm-5.3-flash-dflash2 \
  > /home/michaeldeblok/llmtest-spark/out/b2.log 2>&1 &
echo "B2_PID $!"
sleep 1
head -20 /home/michaeldeblok/llmtest-spark/out/b2.log || true
