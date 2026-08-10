#!/usr/bin/env bash
# Keep attempting the rental until a box actually BOOTS, then stop.
#
# WHY: the VM-capable RTX PRO 6000 market is 1-2 offers wide and its telemetry
# flaps, so a single --go is a coin flip - not because the plan is wrong but
# because the offer list is empty this second, or the host it picked cannot pass
# its GPU through. Both are cheap to survive (rent_and_run fails fast at boot,
# destroys the instance, and cools the machine down: measured ~4 minutes and a
# few cents per bad host) and both clear on their own. What is expensive is a
# human re-running the same command forty times, which is what happened last
# campaign.
#
# Every guard rail lives in rent_and_run.py and still applies on each attempt:
# the card gate, the orphan guard (refuses --go while a live instance carries
# the label), fatal-boot fail-fast, machine cooldowns, and the two-strike ban.
# This script only supplies patience.
#
#   bash scripts/rent_until_up.sh [max_attempts] [sleep_seconds]
set -u
MAX="${1:-40}"
NAP="${2:-420}"
LABEL="${LABEL:-bonsai-close}"
PLAN="${PLAN:-plan_vm}"
HOURS="${HOURS:-12}"
cd "$(dirname "$0")/.."

# Never inherit a marker from an earlier run. rent_and_run now clears these when
# it destroys a box, but a file predating that fix (or written by an older copy)
# would be read as instant success - which is exactly how this loop once
# announced "BOOTED" off a destroyed instance id.
rm -f "$PLAN/INSTANCE" "$PLAN/ENDPOINT"

for i in $(seq 1 "$MAX"); do
  echo "=== attempt $i/$MAX  $(date -u +%H:%M:%S)Z ==="
  out=$(timeout 2400 python scripts/rent_and_run.py --go --vm \
          --est-hours "$HOURS" --label "$LABEL" --plan-dir "$PLAN" 2>&1)
  echo "$out" | tail -25
  # SUCCESS: rent_and_run writes plan_vm/INSTANCE only once a box is past the
  # boot gate, so that file - not the exit code - is the honest signal.
  # Require BOTH markers: INSTANCE is written at create time, ENDPOINT only
  # after the SSH/card/Docker gates pass, so ENDPOINT is the one that means the
  # box is genuinely usable.
  if [ -s "$PLAN/INSTANCE" ] && [ -s "$PLAN/ENDPOINT" ]; then
    echo "=== BOOTED: instance $(cat "$PLAN/INSTANCE") after $i attempt(s) ==="
    exit 0
  fi
  # A live box we failed to record would be an orphan burning money. rent_and_run
  # destroys on fatal boot, so reaching here means nothing is running - but say so
  # explicitly, because silence here is what an orphan looks like.
  echo "--- no instance recorded; nothing should be billing. sleeping ${NAP}s ---"
  sleep "$NAP"
done
echo "=== GAVE UP after $MAX attempts - market never yielded a bootable host ==="
exit 1
