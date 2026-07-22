#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 1-task SMOKE GATE (advisor's blocking requirement): before the full matrix,
# prove on THIS box that the official llama.cpp:server-cuda image serves the
# GGUF, accepts --ctx-checkpoints, and that a --network host server is
# reachable from an OpenCode container via host.docker.internal ON LINUX.
# Serves gpt-oss on GPU0:8080, runs ONE task once into results_smoke/, and
# prints the row so the caller can verify completed + tokens_prompt>0 (i.e. the
# model actually ran and the oracle executed) -- a clean pass gates the full
# run; an infra-error here means the pipeline is broken, found for pennies.
#
#     bash deploy/blackwell/smoke.sh [task]        # default task py-brk-01
# ---------------------------------------------------------------------------
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
B8_ROOT="${B8_ROOT:-/opt/b8}"
PY="${PY:-$B8_ROOT/venv/bin/python}"
TASK="${1:-py-brk-01}"
cd "$REPO"

bash "$SCRIPT_DIR/serve.sh" gpt-oss-20b.gguf 0 8080 || { echo "SMOKE_FAIL: server did not become healthy"; exit 1; }
rm -rf "$REPO/results_smoke"; mkdir -p "$REPO/results_smoke"
"$PY" scripts/run_b8_local.py --endpoint-url http://127.0.0.1:8080 \
    --model gpt-oss-20b --task "$TASK" --limit 1 --results-dir "$REPO/results_smoke"

echo "=== SMOKE ROW ==="
"$PY" - <<'PY'
import json, glob
rows=[json.loads(l) for f in glob.glob("results_smoke/*.jsonl") for l in open(f)]
if not rows:
    print("SMOKE_FAIL: no row emitted"); raise SystemExit(2)
m = rows[-1]["metrics"]
print(f"terminal_status={m['terminal_status']} completion={m['completion']} "
      f"tokens_prompt={m['tokens_prompt']} tokens_completion={m['tokens_completion']} steps={m['steps']}")
# The gate requires COMPLETION=True on this trivial task (py-brk-01 = collapse
# whitespace). Reaching the endpoint (tokens_prompt>0) + a non-infra terminal
# is necessary but NOT sufficient: a run can terminate `completed` yet have
# made ZERO successful edits (e.g. the workspace mount not writable by the
# container's uid-1000 user -> every edit/write tool errors -> oracle sees the
# untouched broken repo -> completion=False for EVERY task, a silent 0%). Only
# completion=True proves the agent actually WROTE a working fix end-to-end.
reached = m['terminal_status'] != 'infra-error' and (m['tokens_prompt'] or 0) > 0
if not reached:
    print("SMOKE_FAIL: infra-error or zero prompt tokens (endpoint unreachable / pipeline broken)")
    raise SystemExit(1)
if not m['completion']:
    print("SMOKE_FAIL: reached the model but completion=False on a TRIVIAL task.")
    print("  Likely the agent's edits never landed -- inspect the newest trace's")
    print("  tool_result events: if every edit/write is \"error\", the workspace")
    print("  mount is not writable by the container user (chmod/uid). Do NOT")
    print("  launch the full matrix until a trivial task completes.")
    raise SystemExit(1)
print("SMOKE_PASS")
PY
