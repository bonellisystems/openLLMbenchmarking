#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LOCAL watcher (runs on the Windows box / git-bash -- has the Verda creds +
# ssh key; the run box has NEITHER). The advisor's BLOCKING safety net:
#   - incremental result pull every ~2 min, so LOCAL always holds the latest
#     rows + traces. A $0-kill destroys the box's OS volume; without this the
#     only copy of the data dies with it (and at $0 you can't re-provision).
#   - balance-floor AUTO-TEARDOWN: stop + delete the instance while results
#     are already safe locally, instead of riding the meter to the $0-kill.
#     Michael is away; nothing else stops the meter.
# Exits (re-invoking the coordinator) on RUN_DONE or floor -> then merge +
# classify + report locally.
#
#     FLOOR=2.5 nohup bash deploy/blackwell/watch_and_teardown.sh &
# ---------------------------------------------------------------------------
set -uo pipefail
SCR="/c/Users/Michael/AppData/Local/Temp/claude/D--BUILT-TOOLS-LLMtesting/3a826912-ebcc-45c0-b4ce-abf47845c1e0/scratchpad"
REPO="/d/BUILT-TOOLS/LLMtesting/llmtest-v2"
IP="$(cat "$SCR/b8_box_ip.txt")"
IID="$(cat "$SCR/b8_instance_id.txt")"
KEY="$HOME/.ssh/$(cat "$SCR/b8_ssh_key.txt")"
FLOOR="${FLOOR:-2.5}"
SSHO="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=12"
PULL="$REPO/results_box"; mkdir -p "$PULL"

bal(){ python "$SCR/verda.py" balance 2>/dev/null | grep -oE "'amount': [0-9.]+" | grep -oE "[0-9.]+"; }
pull(){
  scp $SSHO -rq root@"$IP":/opt/b8/llmtest-v2/results_gpt   "$PULL/" 2>/dev/null || true
  scp $SSHO -rq root@"$IP":/opt/b8/llmtest-v2/results_gemma "$PULL/" 2>/dev/null || true
  scp $SSHO -rq root@"$IP":/opt/b8/llmtest-v2/artifacts/b8_traces "$PULL/" 2>/dev/null || true
}
teardown(){ echo "final pull..."; pull; echo "deleting instance $IID..."; python "$SCR/verda.py" delete "$IID"; }

echo "watcher start: IP=$IP floor=\$$FLOOR -> $PULL"
for i in $(seq 1 900); do          # 900 * 120s = 30h hard cap
  b="$(bal)"
  pull
  done_marker="$(ssh $SSHO root@"$IP" 'test -f /opt/b8/RUN_DONE && echo yes' 2>/dev/null || true)"
  ng="$(cat "$PULL"/results_gpt/*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
  nm="$(cat "$PULL"/results_gemma/*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
  echo "$(date +%H:%M:%S) bal=\$${b:-?} gpt_rows=${ng:-0} gemma_rows=${nm:-0} done=${done_marker:-no}"
  if [ "${done_marker:-}" = "yes" ]; then echo "RUN_DONE -> teardown"; teardown; exit 0; fi
  if [ -n "$b" ] && awk "BEGIN{exit !($b < $FLOOR)}"; then
    echo "BALANCE FLOOR (\$$b < \$$FLOOR) -> teardown"; teardown; exit 0
  fi
  sleep 120
done
echo "watcher 30h cap -> teardown"; teardown
