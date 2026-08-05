#!/bin/bash
# Bound the cost of a dead watcher.
#
# The watcher pulls results and destroys the instance on completion, but it runs on the
# operator's machine and has now been killed twice mid-run. If it dies after the work
# finishes, nothing stops the box: it idles until the credit floor, which on this run is
# about eleven hours and roughly $14 of GPU time for zero rows. That exact shape - a box
# left running after its work completed - already cost $9.65 and 96 unpulled rows once.
#
# So the box bounds its own idle time. It cannot destroy its vast.ai instance (that needs
# the API key, which is never placed on a rented machine), but powering off ends the GPU
# billing, and the instance can still be destroyed from the API afterwards.
#
# GRACE exists so the watcher gets at least two more 5-minute polls to pull the final
# rows before the machine goes away.
# DONE_FLAG is parameterised because work can legitimately continue AFTER the main
# sweep writes run_all_done (a follow-on queue: an extra model, a re-run, a deferred
# build). Hard-coding run_all_done would power the box off in the middle of that.
set -u
GRACE=900
DONE_FLAG="${DONE_FLAG:-run_all_done}"
while true; do
  if [ -f ${B8_ROOT:-/root}/$DONE_FLAG ]; then
    echo "$(date -u +%H:%M:%S) shutdown-guard: run complete, ${GRACE}s grace for the final pull" \
      >> ${B8_ROOT:-/root}/shutdown_guard.log
    sleep "$GRACE"
    echo "$(date -u +%H:%M:%S) shutdown-guard: powering off" >> ${B8_ROOT:-/root}/shutdown_guard.log
    sync
    poweroff -f 2>/dev/null || shutdown -h now 2>/dev/null || halt -f
    exit 0
  fi
  sleep 30
done
