#!/usr/bin/env python3
"""Emit the box-side scripts that close every gap, from plan/manifest.json.

Design decisions worth knowing:

* ONE MODEL AT A TIME, ALL ITS BATTERIES. Loading a 17-134GB GGUF costs minutes;
  grouping by model instead of by battery pays that once per model rather than once
  per (model, battery). That is where the cost estimate's saving comes from.
* RESUMABLE AND IDEMPOTENT. Every runner skips rows already in its shard, and the
  driver skips a model whose batteries are all done, so a killed run resumes.
* RESULTS ARE PULLED PER MODEL, not at the end. The last box died with 96 unpulled
  rows on it; a phase marker is written after each model so the watcher can pull.
* B8 RUNS IN HOST MODE. Docker is absent on vast.ai instances, and opencode.py
  supports sandbox_image=None. That drops the container boundary, which is acceptable
  ONLY because this is a disposable box running our own task manifests - it must not
  become the pattern for an untrusted model or a persistent machine.
* qwen3-235b gets --cpu-moe. At 134GB it does not fit a 96GB card; the box needs
  enough RAM, and its throughput-sensitive batteries are expected to be slow.

    python scripts/emit_run_plan.py --manifest plan/manifest.json --out plan/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_sh(path: Path, text: str) -> None:
    """Shell scripts MUST be written with LF endings.

    Path.write_text() on Windows translates \\n to \\r\\n, and bash on the Linux box
    then chokes on the carriage returns - `for i in ...; do\\r` is a syntax error, and a
    `\\` line-continuation followed by \\r stops being a continuation at all. Every
    script emitted here is generated on Windows and executed on Linux, so the newline
    has to be pinned explicitly rather than left to the platform default.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

# how each battery is driven on the box
DRIVER = {
    "B1": ("bigmodel", "1"), "B2": ("bigmodel", "2"), "B3": ("bigmodel", "3"),
    "B4": ("bigmodel", "4"), "B5": ("bigmodel", "5"), "B6": ("bigmodel", "6"),
    "B7": ("bigmodel", "7"), "B8": ("b8", None), "B9": ("games", None),
    "B10": ("security", None), "B11": ("tools", None),
}

DOWNLOAD = """#!/bin/bash
# Fetch every model that still has gaps.
#
# Files are pulled CONCURRENTLY because Hugging Face throttles a SINGLE transfer
# (measured: -x16 collapsing to one connection at ~21MB/s); parallelism has to be
# across files, where the same box reaches ~80MB/s aggregate.
#
# But concurrency is CAPPED. Firing all %(nfiles)d files at -x16 would open ~%(nconn)d
# connections at once, which HF rate-limits, and it would also fill the disk in
# nondeterministic order so a truncated download could belong to any model. JOBS
# files at a time keeps the aggregate rate without either problem.
set -u
M=/root/models
JOBS=6
mkdir -p $M
get(){ # dir repo path
  mkdir -p "$M/$1"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \\
    --retry-wait=5 --max-tries=5 --auto-file-renaming=false \\
    -d "$M/$1" -o "$(basename "$3")" \\
    "https://huggingface.co/$2/resolve/main/$3" >> "/root/dl_$1.log" 2>&1 \\
    || echo "FAIL $1 $3" >> /root/dl_fail
}
gate(){ while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done; }
%(gets)s
wait
echo DL_DONE > /root/dl_done
df -h /root | tail -1
du -sh $M
# Report anything that failed AND anything suspiciously small, so a truncated shard is
# caught here rather than as a "wrong number of tensors" load error an hour later.
echo "--- download failures ---"; cat /root/dl_fail 2>/dev/null || echo none
echo "--- files under 100MB (suspect) ---"
find $M -name '*.gguf' -size -100M -printf '%%p %%s\\n' 2>/dev/null || true
"""

RUNNER = """#!/bin/bash
# Close every gap, one model at a time so each GGUF is loaded once.
set -u
cd /root/llmtest-v2
export LD_LIBRARY_PATH=/app
BIN=/app/llama-server
OUT=/root/out
mkdir -p $OUT
log(){ echo "$(date +%%H:%%M:%%S) $*" | tee -a /root/run.log; }

serve(){ # $1 gguf  $2 extra flags
  pkill -f "app/llama-server" 2>/dev/null; sleep 4
  # shellcheck disable=SC2086
  nohup $BIN -m "$1" -ngl 99 -c 32768 --parallel 1 --jinja -fa on $2 \\
    --host 127.0.0.1 --port 8080 --no-webui > /root/serve.log 2>&1 &
  for i in $(seq 1 200); do
    curl -s -m3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && return 0
    sleep 4
  done
  return 1
}

%(models)s

pkill -f "app/llama-server" 2>/dev/null
echo ALL_DONE > /root/run_all_done
log "EVERYTHING DONE"
"""

MODEL_BLOCK = """
# ---------- %(id)s : %(bats)s ----------
GG="%(gguf)s"
if [ -f "$GG" ]; then
  log "===== %(id)s ====="
  if serve "$GG" "%(flags)s"; then
%(steps)s
    echo "%(id)s" >> /root/models_done
    log "%(id)s complete"
  else
    log "%(id)s SERVE-FAIL"; echo "%(id)s serve-fail" >> /root/failures
  fi
else
  log "%(id)s SKIP (missing $GG)"; echo "%(id)s missing-gguf" >> /root/failures
fi
"""


def steps_for(mid: str, bats: list[str]) -> str:
    """Emit one command per battery, grouped so the cheap deterministic ones run
    before the long ones - if the box dies we keep the most columns."""
    order = ["B11", "B10", "B2", "B3", "B6", "B9", "B7", "B4", "B5", "B8", "B1"]
    bats = [b for b in order if b in bats]
    lines = []
    bigm = [DRIVER[b][1] for b in bats if DRIVER[b][0] == "bigmodel"]
    for b in bats:
        kind, arg = DRIVER[b]
        if kind == "bigmodel":
            continue
        if kind == "games":
            lines.append(f'    log "  {mid} B9 games"; python3 scripts/run_games.py '
                         f'--endpoint-url http://127.0.0.1:8080 --model "{mid}" --reps 3 '
                         f'--out $OUT/games --chrome "" 2>&1 | tail -3')
        elif kind == "security":
            lines.append(f'    log "  {mid} B10 security"; python3 scripts/run_security.py '
                         f'--endpoint-url http://127.0.0.1:8080 --model "{mid}" --reps 3 '
                         f'--out $OUT/security 2>&1 | tail -3')
        elif kind == "tools":
            lines.append(f'    log "  {mid} B11 tool loop"; mkdir -p /root/agentws && '
                         f'python3 scripts/run_tools_agent.py '
                         f'--endpoint-url http://127.0.0.1:8080 --model "{mid}" --reps 3 '
                         f'--workspace /root/agentws --out $OUT/tools 2>&1 | tail -3')
        elif kind == "b8":
            lines.append(f'    log "  {mid} B8 agentic harness (host mode)"; '
                         f'python3 scripts/run_b8_local.py '
                         f'--endpoint-url http://127.0.0.1:8080 --model "{mid}" '
                         f'--results-dir $OUT/b8_{mid} 2>&1 | tail -3')
    if bigm:
        lines.append(f'    log "  {mid} suite batteries {",".join(bigm)}"; '
                     f'python3 scripts/bigmodel_gen.py --model "{mid}" '
                     f'--batteries {",".join(bigm)} '
                     f'--endpoint-url http://127.0.0.1:8080 --results-dir $OUT/suite 2>&1 | tail -5')
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="plan/manifest.json")
    ap.add_argument("--out", default="plan")
    args = ap.parse_args()

    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    # --- download script ---
    gets = []
    for m in man["models"]:
        for f in m["files"]:
            gets.append(f'gate; get {m["id"]} {m["repo"]} {f["path"]} &')
    write_sh(outdir / "download.sh",
             DOWNLOAD % {"gets": "\n".join(gets), "nfiles": len(gets),
                         "nconn": len(gets) * 16})

    # --- runner: heaviest-gap models first so partial funding still buys the most ---
    blocks = []
    for m in sorted(man["models"], key=lambda x: (-len(x["batteries"]), x["gb"])):
        first = m["files"][0]["path"] if m["files"] else ""
        gguf = f'/root/models/{m["id"]}/{Path(first).name}'
        flags = "--cpu-moe" if m.get("fits_card") is False else ""
        blocks.append(MODEL_BLOCK % {"id": m["id"], "bats": ",".join(m["batteries"]),
                                     "gguf": gguf, "flags": flags,
                                     "steps": steps_for(m["id"], m["batteries"])})
    write_sh(outdir / "run_all.sh", RUNNER % {"models": "\n".join(blocks)})

    # --- one-time box setup (B8 host mode needs node + opencode) ---
    write_sh(outdir / "setup.sh", """#!/bin/bash
# One-time box setup. B8 runs in HOST mode because vast.ai instances have no Docker;
# opencode.py supports sandbox_image=None. Acceptable on a disposable box running our
# own manifests - not a pattern for untrusted models or a persistent machine.
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-pip curl git aria2 tmux nodejs npm >/root/setup.log 2>&1
python3 -m pip install -q --break-system-packages pyyaml playwright >>/root/setup.log 2>&1
python3 -m playwright install --with-deps chromium >>/root/setup.log 2>&1
npm install -g opencode-ai >>/root/setup.log 2>&1 || echo "opencode install failed (B8 will skip)"
# B8 host mode
python3 - <<'PY'
import yaml, io
p = "/root/llmtest-v2/config/suite.yaml"
d = yaml.safe_load(io.open(p, encoding="utf-8").read())
d.setdefault("b8", {}).setdefault("sandbox", {})["enabled"] = False
io.open(p, "w", encoding="utf-8").write(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print("b8.sandbox.enabled = False (host mode)")
PY
echo "node: $(node --version 2>/dev/null)  opencode: $(command -v opencode || echo absent)"
python3 -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); print('chromium OK'); b.close(); p.stop()"
echo SETUP_DONE > /root/setup_done
""")

    for f in ("download.sh", "run_all.sh", "setup.sh"):
        (outdir / f).chmod(0o755)

    print(f"wrote {outdir/'download.sh'}  ({len(gets)} files, {man['totals']['download_gb']} GB)")
    print(f"wrote {outdir/'run_all.sh'}   ({len(blocks)} models, {man['totals']['missing_cells']} cells)")
    print(f"wrote {outdir/'setup.sh'}")
    print("\nmodel order (heaviest gaps first):")
    for m in sorted(man["models"], key=lambda x: (-len(x["batteries"]), x["gb"]))[:8]:
        print(f"   {m['id']:24s} {len(m['batteries']):2d} cells  {m['gb']:6.1f} GB  {','.join(m['batteries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
