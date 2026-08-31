#!/usr/bin/env bash
set -euo pipefail
chmod +x /home/michaeldeblok/llmtest-spark/src/scripts/glm_campaign.sh
nohup bash /home/michaeldeblok/llmtest-spark/src/scripts/glm_campaign.sh \
  >/home/michaeldeblok/llmtest-spark/out/campaign.nohup 2>&1 &
echo "CAMPAIGN_PID $!"
