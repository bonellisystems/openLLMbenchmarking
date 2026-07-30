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

import yaml

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

# How each battery is driven on the box.
#
# THE SPLIT THAT MATTERS: bigmodel_gen.py serves ONE endpoint and hands every battery
# the same handle - its fake ServerManager.request_endpoint() ignores the ctx/kv/flags
# the battery asks for (see its own docstring: "B1/B2/B3/B6"). That is fine for
# batteries whose arms all share one serving config, and WRONG for the three that sweep
# serving configs:
#
#   B4 sweeps ctx 16k..256k and kv f16/q8/q4  -> needs -c and -ctk/-ctv per arm
#   B7 sweeps spec-decode (ngram-mod vs none) -> needs different --spec-type per arm
#   B5 is throughput and timing-authoritative -> needs its own controlled launch
#
# Routed through bigmodel_gen they would have produced rows LABELLED with each arm's
# condition while every one of them actually ran the single launched config: 256k needle
# prompts sent to a 32k server, and 80 B7 rows per model that are all the same
# measurement under different names. p8_gen_serving.py / p8_gen_b5.py are the P8-era
# drivers that relaunch the server per group, so those three go there instead.
DRIVER = {
    "B1": ("bigmodel", "1"), "B2": ("bigmodel", "2"), "B3": ("bigmodel", "3"),
    "B6": ("bigmodel", "6"),
    "B4": ("serving", "4"), "B7": ("serving", "7"), "B5": ("b5", None),
    "B8": ("b8", None), "B9": ("games", None),
    "B10": ("security", None), "B11": ("tools", None),
}

# Batteries that own their own server lifecycle. The shared endpoint must be stopped
# before these run or their launch collides with it on port 8080 - which either fails
# to bind or, worse, gets answered by the still-running server at the wrong ctx.
OWN_SERVER = {"B4", "B5", "B7"}

# B5 and B7 are only meaningful on a binary that supports --spec-type ngram-mod: the
# roster's conditions are spec=ngram32, and a timing row labelled ngram32 that ran
# without it is a fabricated number. setup.sh probes --help and records the answer.
NEEDS_NGRAM = {"B5", "B7"}

# B8/OpenCode needs ctx >= 40960 (recorded in the B8 local findings); the shared
# endpoint is launched above that. KV is q8_0 to match the roster's normalized_config.
SHARED_CTX = 49152

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

# --- throughput probe: FAIL FAST ---------------------------------------------------
# A host's advertised inet_down is its link speed, NOT the rate Hugging Face will serve
# it. Cheap hosts have measured ~4MB/s against HF while advertising gigabits; at that
# rate this download alone is ~44 hours and would consume the entire budget before a
# single row is generated. Better to die here, loudly, than to bleed out slowly.
MIN_MBPS=25
probe(){
  rm -f /tmp/probe.bin
  timeout 60 aria2c -x8 -s8 -k1M --file-allocation=none --console-log-level=error \\
    -d /tmp -o probe.bin --max-download-limit=0 \\
    "https://huggingface.co/%(probe_repo)s/resolve/main/%(probe_file)s" >/dev/null 2>&1
  sz=$(stat -c %%s /tmp/probe.bin 2>/dev/null || echo 0)
  rm -f /tmp/probe.bin
  echo $(( sz / 60 / 1000000 ))
}
RATE=$(probe)
echo "HF throughput probe: ~${RATE} MB/s (floor ${MIN_MBPS})" | tee /root/dl_probe
if [ "$RATE" -lt "$MIN_MBPS" ]; then
  echo "ABORT: Hugging Face throughput too low on this host - the download would cost" >> /root/dl_probe
  echo "more than the compute. Destroy this box and rent another." >> /root/dl_probe
  echo "DL_ABORT rate=${RATE}" > /root/dl_abort
  cat /root/dl_probe
  exit 1
fi

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
# Close every gap, one model at a time: FETCH -> RUN ITS BATTERIES -> DELETE.
#
# Why interleaved rather than "download all 638.7GB, then run":
#  * PEAK DISK becomes the largest single model (~134GB) instead of the whole corpus, so
#    the disk floor drops 780GB -> 300GB. At 780GB, 15 of the available RTX PRO 6000
#    offers were disqualified on disk alone and some hours none qualified at all.
#  * IT COSTS NOTHING. The two phases were already sequential - download.sh ran to
#    completion before run_all.sh started - so there was never any overlap to lose.
#  * ROWS START ARRIVING IN MINUTES instead of after a ~2.2h download, and a box that
#    dies early has produced COMPLETE models rather than a full disk and no results.
set -u
cd /root/llmtest-v2
export LD_LIBRARY_PATH=/app
export LLMTEST_ROOT=/root/llmtest-v2
export LLMTEST_OUT=/root/out
export LLMTEST_BIN=/app/llama-server
export LLMTEST_LIBS=/app
BIN=/app/llama-server
OUT=/root/out
mkdir -p $OUT
log(){ echo "$(date +%%H:%%M:%%S) $*" | tee -a /root/run.log; }

# Does this binary support ngram spec-decode? setup.sh probed --help and wrote the
# answer here. B5/B7 are timing-authoritative at spec=ngram32 and must SKIP rather than
# silently produce timings from a different serving config.
NGRAM=0
[ -f /root/caps ] && grep -q '^ngram=1' /root/caps && NGRAM=1
SPEC=""
[ "$NGRAM" = "1" ] && SPEC="--spec-type ngram-mod --spec-ngram-mod-n-match 32"

stop_server(){ pkill -f "llama-server" 2>/dev/null; sleep 5; }

M=/root/models
mkdir -p $M
# Files within a model are fetched CONCURRENTLY (HF throttles a single transfer:
# measured -x16 collapsing to one connection at ~21MB/s), but capped - firing everything
# at once opens hundreds of connections and gets rate-limited.
JOBS=4
gate(){ while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done; }
get(){ # dir repo path
  mkdir -p "$M/$1"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \\
    --retry-wait=5 --max-tries=5 --auto-file-renaming=false \\
    -d "$M/$1" -o "$(basename "$3")" \\
    "https://huggingface.co/$2/resolve/main/$3" >> "/root/dl_$1.log" 2>&1 \\
    || echo "FAIL $1 $3" >> /root/dl_fail
}
# Free the weights once a model's batteries are done. Without this the interleaving
# buys nothing: the disk fills exactly as it did before, just more slowly.
release(){
  du -sh "$M/$1" 2>/dev/null | tee -a /root/run.log
  rm -rf "$M/$1"
  log "  released $1 ; free: $(df -h /root | awk 'NR==2{print $4}')"
}

serve(){ # $1 gguf  $2 extra flags -- the SHARED endpoint for B1/B2/B3/B6/B8-B11
  stop_server
  # shellcheck disable=SC2086
  nohup $BIN -m "$1" -ngl 99 -c %(shared_ctx)d --parallel 1 --jinja -fa on \\
    -ctk q8_0 -ctv q8_0 $SPEC $2 \\
    --host 127.0.0.1 --port 8080 --no-webui > /root/serve.log 2>&1 &
  for i in $(seq 1 200); do
    curl -s -m3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && return 0
    sleep 4
  done
  return 1
}

# Run one battery and RECORD ITS REAL EXIT CODE. The previous shape was
# `python3 ... 2>&1 | tail -3`, where the pipe discards the runner's status and the
# model was then appended to models_done unconditionally - so a battery that crashed on
# row 1 looked identical to one that completed. That is exactly how a YAML bug once
# produced zero rows across six models while the run reported healthy.
run_step(){ # $1 model  $2 battery  $3.. command
  mid="$1"; bat="$2"; shift 2
  log "  $mid $bat start"
  "$@" > /root/last_step.log 2>&1
  rc=$?
  tail -3 /root/last_step.log | tee -a /root/run.log
  if [ "$rc" -eq 0 ]; then
    echo "$mid $bat ok" >> /root/steps
  else
    log "  $mid $bat FAILED rc=$rc"
    echo "$mid $bat fail rc=$rc" >> /root/steps
    cat /root/last_step.log >> /root/step_failures.log
  fi
}

# B4/B7 relaunch the server per serving-config group; B5 controls its own launch. They
# take --gpu0/--gpu1 because P8 had two cards - this box has one, so gpu1 is empty and
# that worker thread exits immediately.
run_serving(){ # $1 model  $2 battery-number
  run_step "$1" "B$2" python3 scratchpad/p8_gen_serving.py --battery "$2" \\
    --gpu0 "$1" --gpu1 ""
}

# --- throughput probe: FAIL FAST --------------------------------------------------
# A host's advertised inet_down is its link speed, NOT the rate Hugging Face serves it.
# Cheap hosts have measured ~4MB/s while advertising gigabits; at that rate this corpus
# alone is ~44 hours and the budget is gone before a single row exists. Die here instead.
MIN_MBPS=25
rm -f /tmp/probe.bin
timeout 60 aria2c -x8 -s8 -k1M --file-allocation=none --console-log-level=error \\
  -d /tmp -o probe.bin \\
  "https://huggingface.co/%(probe_repo)s/resolve/main/%(probe_file)s" >/dev/null 2>&1
PSZ=$(stat -c %%s /tmp/probe.bin 2>/dev/null || echo 0)
rm -f /tmp/probe.bin
RATE=$(( PSZ / 60 / 1000000 ))
log "HF throughput probe: ~${RATE} MB/s (floor ${MIN_MBPS} MB/s)"
if [ "$RATE" -lt "$MIN_MBPS" ]; then
  log "ABORT: Hugging Face serves this host too slowly - the download would cost more"
  log "than the compute. Destroy this box and rent another."
  echo "DL_ABORT rate=${RATE}" > /root/dl_abort
  exit 1
fi
# The watcher keys its stage off this marker; with interleaved fetching there is no
# separate download phase, so declare it once the probe has passed.
echo PROBE_OK > /root/dl_done

%(models)s

stop_server
echo ALL_DONE > /root/run_all_done
log "EVERYTHING DONE"
log "step results: $(grep -c ' ok$' /root/steps 2>/dev/null || echo 0) ok, \\
$(grep -c ' fail' /root/steps 2>/dev/null || echo 0) failed"
"""

MODEL_BLOCK = """
# ---------- %(id)s : %(bats)s (%(gb).1f GB) ----------
log "===== %(id)s : fetching %(gb).1f GB ====="
%(gets)s
wait
GG="%(gguf)s"
if [ -f "$GG" ]; then
  # Phase A - batteries that share one endpoint.
%(phase_a)s
  # Phase B - batteries that launch their own servers per arm. The shared endpoint MUST
  # be down first: these drivers bind 8080 themselves, and a surviving Phase-A server
  # either blocks the bind or answers the requests at the wrong ctx.
  stop_server
%(phase_b)s
  echo "%(id)s" >> /root/models_done
  log "%(id)s complete"
else
  log "%(id)s SKIP (fetch failed, no $GG)"; echo "%(id)s missing-gguf" >> /root/failures
fi
stop_server
release "%(id)s"
"""

PHASE_A = """  if serve "$GG" "%(flags)s"; then
%(steps)s
  else
    log "%(id)s SERVE-FAIL (phase A)"; echo "%(id)s serve-fail" >> /root/failures
  fi"""


def phase_a_block(mid: str, flags: str, steps: str) -> str:
    if not steps.strip():
        return "  :   # no shared-endpoint batteries for this model"
    return PHASE_A % {"flags": flags, "steps": steps, "id": mid}


ENDPOINT = "--endpoint-url http://127.0.0.1:8080"

# Rows per battery, and measured seconds per row (B9/B10/B11 from n=264/396/144 real runs).
ROWS = {"B1": 360, "B2": 30, "B3": 39, "B4": 21, "B5": 8, "B6": 30,
        "B7": 80, "B8": 69, "B9": 24, "B10": 66, "B11": 12}
SECS = {"B1": 12.0, "B2": 8.0, "B3": 6.0, "B4": 25.0, "B5": 30.0, "B6": 15.0,
        "B7": 8.0, "B8": 35.0, "B9": 40.8, "B10": 20.8, "B11": 10.9}
LOAD_S = 210
FETCH_MBPS = 110          # measured 119 MB/s on the Spain box; keep a margin


def est_seconds(m: dict, reg: dict) -> float:
    """Rough wall-clock for one model's remaining cells, used only for ORDERING.

    The per-row constants above were calibrated on MoE models. The first run showed a
    dense 31B taking 8.8h where they predicted 2.8h, so architecture matters more than
    the row count does: decode is memory-bandwidth-bound and a dense model pushes its
    whole weight set per token, while an A3B MoE touches ~3B. A model that does not fit
    the card at all pays far worse again, because its experts stream from system RAM.
    """
    e = reg.get(m["id"], {}) or {}
    arch = e.get("arch", {}) or {}
    if m.get("fits_card") is False:
        factor = 8.0                      # --cpu-moe: experts stream over PCIe
    elif not arch.get("moe"):
        factor = 3.0                      # dense: measured on abl-gemma-4-31b
    else:
        factor = 1.0
    compute = sum(ROWS[b] * SECS[b] for b in m["batteries"] if b in ROWS) * factor
    fetch = m["gb"] * 1000 / FETCH_MBPS
    return LOAD_S + fetch + compute


def steps_for(mid: str, bats: list[str]) -> tuple[str, str]:
    """(phase_a, phase_b) shell lines for one model.

    Cheap deterministic batteries run before long ones, so if the box dies mid-model we
    keep the most complete columns. B1 goes last: it is the biggest (360 rows) and its
    score needs a separate judging pass anyway.
    """
    order = ["B11", "B10", "B2", "B3", "B6", "B9", "B8", "B1", "B4", "B7", "B5"]
    bats = [b for b in order if b in bats]
    a, b_lines = [], []

    for bat in bats:
        kind, _ = DRIVER[bat]
        if kind in ("bigmodel", "serving", "b5"):
            continue
        if kind == "games":
            a.append(f'    run_step "{mid}" B9 python3 scripts/run_games.py {ENDPOINT} '
                     f'--model "{mid}" --reps 3 --out $OUT/games --chrome ""')
        elif kind == "security":
            a.append(f'    run_step "{mid}" B10 python3 scripts/run_security.py {ENDPOINT} '
                     f'--model "{mid}" --reps 3 --out $OUT/security')
        elif kind == "tools":
            a.append(f'    mkdir -p /root/agentws; '
                     f'run_step "{mid}" B11 python3 scripts/run_tools_agent.py {ENDPOINT} '
                     f'--model "{mid}" --reps 3 --workspace /root/agentws --out $OUT/tools')
        elif kind == "b8":
            a.append(f'    run_step "{mid}" B8 python3 scripts/run_b8_local.py {ENDPOINT} '
                     f'--model "{mid}" --results-dir $OUT/b8_{mid}')

    bigm = [DRIVER[x][1] for x in bats if DRIVER[x][0] == "bigmodel"]
    if bigm:
        a.append(f'    run_step "{mid}" B{"+B".join(bigm)} python3 scripts/bigmodel_gen.py '
                 f'--model "{mid}" --batteries {",".join(bigm)} {ENDPOINT} '
                 f'--results-dir $OUT/suite')

    # Phase B: own-server batteries. B5/B7 additionally require ngram support, or their
    # spec=ngram32 condition label would not describe what actually ran.
    for bat in bats:
        kind, arg = DRIVER[bat]
        if bat not in OWN_SERVER:
            continue
        guard_open = guard_close = ""
        if bat in NEEDS_NGRAM:
            guard_open = ('  if [ "$NGRAM" = "1" ]; then\n')
            guard_close = ('\n  else\n'
                           f'    log "  {mid} {bat} SKIP - binary has no --spec-type '
                           f'ngram-mod, and {bat} is timing-authoritative at spec=ngram32"\n'
                           f'    echo "{mid} {bat} skip no-ngram" >> /root/steps\n'
                           '  fi')
        indent = "    " if guard_open else "  "
        if kind == "serving":
            body = f'{indent}run_serving "{mid}" {arg}'
        else:  # b5
            body = (f'{indent}run_step "{mid}" B5 python3 scratchpad/p8_gen_b5.py '
                    f'--gpu0 "{mid}" --gpu1 ""')
        b_lines.append(guard_open + body + guard_close)

    if not b_lines:
        b_lines = ["  :   # no own-server batteries for this model"]
    return "\n".join(a), "\n".join(b_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="plan/manifest.json")
    ap.add_argument("--out", default="plan")
    ap.add_argument("--skip", default="",
                    help="comma-separated batteries to leave out of this run, e.g. B1. "
                         "B1 is 360 rows per model and its score does not exist until a "
                         "separate judging pass runs, so on a tight budget those hours "
                         "buy nothing usable while cheap B10/B11 cells go unclosed.")
    ap.add_argument("--skip-models", default="",
                    help="comma-separated models to leave out of THIS run because they "
                         "are handled by a different plan - not because they are unwanted. "
                         "bonsai-ternary-27b belongs here whenever the standard image is "
                         "in use: its Q2_0 needs the prism fork, so including it only "
                         "buys a fetch and a 13-minute serve timeout.")
    args = ap.parse_args()

    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    skip_models = {s.strip() for s in args.skip_models.split(",") if s.strip()}
    if skip_models:
        before = len(man["models"])
        dropped = sum(len(m["batteries"]) for m in man["models"] if m["id"] in skip_models)
        man["models"] = [m for m in man["models"] if m["id"] not in skip_models]
        man["totals"]["missing_cells"] -= dropped
        print(f"skipping models {','.join(sorted(skip_models))}: "
              f"{before - len(man['models'])} model(s), {dropped} cells handled elsewhere\n")
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    if skip:
        dropped = 0
        keep = []
        for m in man["models"]:
            n0 = len(m["batteries"])
            m["batteries"] = [b for b in m["batteries"] if b not in skip]
            dropped += n0 - len(m["batteries"])
            if m["batteries"]:
                keep.append(m)
        man["models"] = keep
        man["totals"]["missing_cells"] -= dropped
        print(f"skipping {','.join(sorted(skip))}: {dropped} cells deferred\n")
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    # --- runner: CHEAPEST MODELS FIRST -------------------------------------------------
    # Previously ordered heaviest-gap-first, on the theory that partial funding should buy
    # the most cells. It does the opposite: the first run spent 15 hours and ~$20 on three
    # of the four most expensive models and closed 16 cells, while 14 small models needing
    # only B10/B11 - about 30 minutes each - never started. Ordering by estimated cost
    # ascending maximises CELLS CLOSED PER DOLLAR, which is what matters when the budget
    # may run out mid-run. B1 sinks to the back on its own: 360 rows, and its score needs a
    # separate judging pass regardless.
    reg = yaml.safe_load((ROOT / "config" / "registry.yaml").read_text(encoding="utf-8"))["models"]
    blocks = []
    nfiles = 0
    for m in sorted(man["models"], key=lambda x: est_seconds(x, reg)):
        first = m["files"][0]["path"] if m["files"] else ""
        gguf = f'/root/models/{m["id"]}/{Path(first).name}'
        flags = "--cpu-moe" if m.get("fits_card") is False else ""
        pa, pb = steps_for(m["id"], m["batteries"])
        gets = "\n".join(f'gate; get {m["id"]} {m["repo"]} {f["path"]} &'
                         for f in m["files"])
        nfiles += len(m["files"])
        blocks.append(MODEL_BLOCK % {"id": m["id"], "bats": ",".join(m["batteries"]),
                                     "gguf": gguf, "gb": m["gb"], "gets": gets,
                                     "phase_a": phase_a_block(m["id"], flags, pa),
                                     "phase_b": pb})

    # Probe against the smallest real file in the run, so the measurement is of exactly
    # the transfer this box is about to repeat.
    probe_m = min((m for m in man["models"] if m["files"]), key=lambda x: x["gb"])
    write_sh(outdir / "run_all.sh", RUNNER % {"models": "\n".join(blocks),
                                              "shared_ctx": SHARED_CTX,
                                              "probe_repo": probe_m["repo"],
                                              "probe_file": probe_m["files"][0]["path"]})
    # download.sh is retained only so an operator can pre-stage weights by hand; the
    # run itself no longer uses it.
    write_sh(outdir / "download.sh",
             "#!/bin/bash\n"
             "# NOT USED BY THE RUN. run_all.sh fetches each model just before it runs\n"
             "# and deletes it after, which is what keeps peak disk at one model.\n"
             "echo 'see plan/run_all.sh - fetching is interleaved per model'\n")

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

# --- capability probe -------------------------------------------------------------
# --spec-type ngram-mod originated in the prism fork and has since landed upstream, so
# whether THIS image has it is an empirical question, not an assumption. B5/B7 are
# timing-authoritative at spec=ngram32 and skip themselves if the answer is no.
: > /root/caps
# LD_LIBRARY_PATH IS REQUIRED EVEN TO READ --help. Without it the binary dies with
# "libllama-server-impl.so: cannot open shared object file" and every grep below finds
# nothing - which the first version of this probe recorded as "no --spec-type flag in
# this build", silently disqualifying B5/B7 on a build that supports ngram-mod fine.
export LD_LIBRARY_PATH=/app
HELP=$(/app/llama-server --help 2>&1)
NLINES=$(printf '%s\n' "$HELP" | wc -l)
if [ "$NLINES" -lt 20 ]; then
  echo "ngram=0  # PROBE FAILED - --help produced $NLINES lines:" >> /root/caps
  printf '%s\n' "$HELP" | head -3 | sed 's/^/#   /' >> /root/caps
elif printf '%s\n' "$HELP" | grep -q 'ngram-mod'; then
  echo "ngram=1" >> /root/caps
else
  echo "ngram=0  # build has no ngram-mod spec type" >> /root/caps
fi
/app/llama-server --version >> /root/caps 2>&1 || true
echo "--- caps ---"; cat /root/caps

# --- config patches --------------------------------------------------------------
python3 - <<'PY'
import io, json, yaml
from pathlib import Path

sp = "/root/llmtest-v2/config/suite.yaml"
d = yaml.safe_load(io.open(sp, encoding="utf-8").read())

# B8 host mode: vast.ai instances have no Docker, and opencode.py supports
# sandbox_image=None. Acceptable on a disposable box running our own manifests.
d.setdefault("b8", {}).setdefault("sandbox", {})["enabled"] = False

# B4 arms are pruned by estimated physical fit. The default plan tier is T1 (the 24GB
# laptop), which on this 96GB card would prune every tier above 16k and give a 57.6GB
# model ZERO arms - contributing no B4 rows while looking like a clean run. The frozen
# roster's B4 rows were produced on this same card class (results/sessions.jsonl:
# hardware_sku=rtx-pro-6000, measured_usable_vram_gb=94.0), which is T3.
d.setdefault("b4", {})["plan_tier"] = "T3"
io.open(sp, "w", encoding="utf-8").write(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print("suite.yaml: b8.sandbox.enabled=False, b4.plan_tier=T3")

# p8_gen_serving.py / p8_gen_b5.py resolve the GGUF from registry local_path, which
# still holds the Windows authoring paths. Point it at what download.sh actually wrote.
# download.sh flattens every file to $M/<id>/<basename>, so a sharded model's shards all
# land side by side in one directory and llama.cpp resolves the set from shard 00001.
rp = "/root/llmtest-v2/config/registry.yaml"
reg = yaml.safe_load(io.open(rp, encoding="utf-8").read())
man = json.load(io.open("/root/plan_manifest.json", encoding="utf-8"))
fixed = 0
for m in man["models"]:
    if not m["files"]:
        continue
    p = Path("/root/models") / m["id"] / Path(m["files"][0]["path"]).name
    if m["id"] in reg["models"]:
        reg["models"][m["id"]]["local_path"] = str(p)
        fixed += 1
io.open(rp, "w", encoding="utf-8").write(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True))
print(f"registry.yaml: local_path repointed for {fixed} models")
PY
echo "node: $(node --version 2>/dev/null)  opencode: $(command -v opencode || echo absent)"
python3 -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); print('chromium OK'); b.close(); p.stop()"
echo SETUP_DONE > /root/setup_done
""")

    for f in ("download.sh", "run_all.sh", "setup.sh"):
        (outdir / f).chmod(0o755)

    biggest = max(m["gb"] for m in man["models"])
    print(f"wrote {outdir/'download.sh'}  (placeholder - fetching is interleaved)")
    print(f"wrote {outdir/'run_all.sh'}   ({len(blocks)} models, {nfiles} files, "
          f"{man['totals']['missing_cells']} cells)")
    print(f"  total fetched      : {man['totals']['download_gb']} GB")
    print(f"  PEAK DISK          : ~{biggest:.0f} GB (largest single model, not the corpus)")
    print(f"wrote {outdir/'setup.sh'}")
    print("\nmodel order (cheapest first = most cells per dollar), cumulative hours:")
    cum = 0.0
    for m in sorted(man["models"], key=lambda x: est_seconds(x, reg)):
        cum += est_seconds(m, reg) / 3600
        print(f"   {cum:6.1f}h  {m['id']:22s} {len(m['batteries']):2d} cells "
              f"{m['gb']:6.1f} GB  {','.join(m['batteries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
