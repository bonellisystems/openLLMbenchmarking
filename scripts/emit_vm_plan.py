#!/usr/bin/env python3
"""Emit the box-side scripts for the hardware-consistency campaign on a vast.ai VM.

This is the VM-layout sibling of emit_run_plan.py, and exists because the two runtimes
genuinely differ:

  container box (emit_run_plan)          KVM VM (this file)
  ---------------------------------      ------------------------------------------
  /app/llama-server on the host          dockerized llama via deploy/blackwell/serve.sh
  no Docker -> B8 oracle IMPOSSIBLE      Docker works -> b8.sandbox.enabled=true (the
  (that is what poisoned 554 rows)       whole point of renting a VM)
  system python                          $B8_ROOT/venv
  /root layout                           /opt/b8 layout (RUN_ON_BLACKWELL.md)

What carries over unchanged, because each rule was paid for once already: per-model
FETCH -> RUN -> RELEASE (peak disk = one model), cheapest-model-first ordering, the HF
throughput probe (abort <25MB/s), run_step's real exit codes, resumable runners.

Campaign-specific rules:
  * bonsai-ternary-27b is served from prism-llama:1 (Dockerfile.prism) - its Q2_0 does
    not load on the official image; this also finally closes its B10/B11.
  * llama-4-scout and abl-qwen3.6-27b went 100% infra-error in every previous B8
    attempt: each gets a 1-task PROBE first, and a probe with zero eligible rows skips
    that model's full B8 (documented exclusion) instead of burning 1-2h proving it 115
    times.
  * Every row is stamped suite-v2.2.0 + hardware_sku=rtx-pro-6000-vm (runners) or via
    config + --hardware-sku session row (B8), so this audit never needs ledger
    archaeology again.
  * NO seeding of canonical shards: every in-scope cell is re-run WHOLE (the mixed-cell
    decision); runners start from empty --out and resume from their own campaign shard.

    python scripts/emit_vm_plan.py            # reads plan/manifest.json -> plan_vm/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402

from emit_run_plan import est_seconds  # noqa: E402  (same cost model = same ordering)

HW_SKU = "rtx-pro-6000-vm"
B8_ROOT = "/opt/b8"
# The two habitual all-infra models get a 1-task probe before their full B8 sweep.
PROBE_MODELS = ("llama-4-scout", "abl-qwen3.6-27b")
PRISM_MODELS = ("bonsai-ternary-27b",)


def write_sh(path: Path, text: str) -> None:
    """LF endings, always - bash on the VM rejects CRLF (`do\\r`), and these files are
    generated on Windows. Same rule, same reason as emit_run_plan.write_sh."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


SETUP = """#!/bin/bash
# One-shot VM setup. Run AFTER the repo is extracted at $B8_ROOT/llmtest-v2.
set -euo pipefail
export B8_ROOT=%(b8_root)s
REPO=$B8_ROOT/llmtest-v2
VENV=$B8_ROOT/venv
cd "$REPO"

echo "== base packages the KVM image may lack =="
export DEBIAN_FRONTEND=noninteractive
# unattended-upgrades stole the dpkg lock and killed this very apt call on the
# Korea box (2026-08-04); a benchmark box wants no background apt churn at all.
systemctl mask --now unattended-upgrades >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  sleep 5
done
apt-get update -qq
apt-get install -y --no-install-recommends aria2 tmux curl jq >/dev/null

echo "== bootstrap (Docker + nvidia runtime + llama image + b8-sandbox:1 + venv) =="
bash deploy/blackwell/bootstrap.sh

echo "== B9 needs a browser: playwright chromium into the venv =="
"$VENV/bin/pip" install -q playwright
"$VENV/bin/python" -m playwright install --with-deps chromium >/dev/null

echo "== prism image for bonsai-ternary-27b (Q2_0 refuses the official image) =="
# NON-FATAL: only bonsai (1 of 19 models, last in the run order) needs this image,
# and its first build attempt died compiling on the Korea box - a prism failure must
# not hold the other 18 models hostage. run_all's serve step fails loudly for bonsai
# if the image is still missing when its turn comes, and the flag file makes the
# situation visible to the watcher.
if timeout 2700 docker build -t prism-llama:1 -f deploy/blackwell/Dockerfile.prism deploy/blackwell \
     > $B8_ROOT/prism_build.log 2>&1; then
  rm -f $B8_ROOT/prism_missing
else
  echo "PRISM BUILD FAILED (non-fatal) - bonsai will fail its serve unless the image"
  echo "is built by hand before its turn; full log: $B8_ROOT/prism_build.log"
  touch $B8_ROOT/prism_missing
fi

echo "== capability probe: ngram spec-decode in the official image =="
# LESSON ENCODED: probe INSIDE the runtime that will serve, and treat a failed probe as
# PROBE-FAILED, never as "no". (The container-era probe once ran the binary without its
# LD_LIBRARY_PATH, read the crash as 'no ngram', and nearly skipped two batteries.)
: > $B8_ROOT/caps
HELP=$(docker run --rm ghcr.io/ggml-org/llama.cpp:server-cuda --help 2>&1 || true)
if [ "$(printf '%%s\\n' "$HELP" | wc -l)" -lt 20 ]; then
  echo "ngram=0 # PROBE FAILED - --help too short" >> $B8_ROOT/caps
elif printf '%%s\\n' "$HELP" | grep -q 'ngram-mod'; then
  echo "ngram=1" >> $B8_ROOT/caps
else
  echo "ngram=0 # build has no ngram-mod" >> $B8_ROOT/caps
fi
cat $B8_ROOT/caps

echo "== hard preconditions (fail setup, not hour 6 of the sweep) =="
docker info >/dev/null                                  # a real VM, not a container
docker image inspect b8-sandbox:1 >/dev/null            # oracle containment image
# prism-llama:1 is only asserted when its build claims success - when the build
# already failed (prism_missing flag), bonsai is the accepted casualty, not setup.
if [ ! -f $B8_ROOT/prism_missing ]; then
  docker image inspect prism-llama:1 >/dev/null
fi
"$VENV/bin/python" - <<'PY'
import yaml, io, sys
d = yaml.safe_load(io.open("config/suite.yaml", encoding="utf-8").read())
assert d["suite_version"] == "suite-v2.2.0", d["suite_version"]
assert d["b8"]["sandbox"]["enabled"] is True, "B8 sandbox must stay ENABLED on the VM"
print("suite.yaml: v2.2.0, sandbox enabled - OK")
PY

echo SETUP_DONE > $B8_ROOT/setup_done
"""


RUNNER_HEAD = """#!/bin/bash
# Hardware-consistency campaign: every wrong-hardware cell re-measured on THIS card.
set -u
export B8_ROOT=%(b8_root)s
REPO=$B8_ROOT/llmtest-v2
PY=$B8_ROOT/venv/bin/python
OUT=$B8_ROOT/out
M=$B8_ROOT/models
EP="--endpoint-url http://127.0.0.1:8080"
mkdir -p "$OUT" "$M"
cd "$REPO"
log(){ echo "$(date -u +%%H:%%M:%%S) $*" | tee -a $B8_ROOT/run.log; }

NGRAM=0; grep -q '^ngram=1' $B8_ROOT/caps 2>/dev/null && NGRAM=1
EXTRA_FLAGS=""
[ "$NGRAM" = "1" ] && EXTRA_FLAGS="--spec-type ngram-mod --spec-ngram-mod-n-match 32"
export EXTRA_FLAGS

run_step(){ # $1 model  $2 battery  $3.. command  (real exit codes -> $B8_ROOT/steps)
  mid="$1"; bat="$2"; shift 2
  log "  $mid $bat start"
  "$@" > $B8_ROOT/last_step.log 2>&1
  rc=$?
  tail -3 $B8_ROOT/last_step.log | tee -a $B8_ROOT/run.log
  if [ "$rc" -eq 0 ]; then echo "$mid $bat ok" >> $B8_ROOT/steps
  else log "  $mid $bat FAILED rc=$rc"; echo "$mid $bat fail rc=$rc" >> $B8_ROOT/steps
       cat $B8_ROOT/last_step.log >> $B8_ROOT/step_failures.log; fi
  return $rc
}

JOBS=4
gate(){ while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done; }
get(){ # dir repo path
  mkdir -p "$M/$1"
  aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \\
    --retry-wait=5 --max-tries=5 --auto-file-renaming=false \\
    -d "$M/$1" -o "$(basename "$3")" \\
    "https://huggingface.co/$2/resolve/main/$3" >> "$B8_ROOT/dl_$1.log" 2>&1 \\
    || echo "FAIL $1 $3" >> $B8_ROOT/dl_fail
}
release(){ du -sh "$M/$1" 2>/dev/null | tee -a $B8_ROOT/run.log; rm -rf "${M:?}/$1"
           log "  released $1 ; free: $(df -h $B8_ROOT | awk 'NR==2{print $4}')"; }
serve_model(){ # $1 gguf-relpath  $2 image
  LLAMA_IMAGE="$2" bash deploy/blackwell/serve.sh "$1"
}
stop_server(){ bash deploy/blackwell/serve.sh stop; }

# --- HF throughput probe: die loudly on a host HF serves at ~4MB/s -----------------
MIN_MBPS=25
rm -f /tmp/probe.bin
timeout 60 aria2c -x8 -s8 -k1M --file-allocation=none --console-log-level=error \\
  -d /tmp -o probe.bin \\
  "https://huggingface.co/%(probe_repo)s/resolve/main/%(probe_file)s" >/dev/null 2>&1 || true
PSZ=$(stat -c %%s /tmp/probe.bin 2>/dev/null || echo 0); rm -f /tmp/probe.bin
RATE=$(( PSZ / 60 / 1000000 ))
log "HF throughput probe: ~${RATE} MB/s (floor ${MIN_MBPS})"
if [ "$RATE" -lt "$MIN_MBPS" ]; then
  log "ABORT: HF too slow on this host - destroy the box, rent another."
  echo "DL_ABORT rate=${RATE}" > $B8_ROOT/dl_abort; exit 1
fi
echo PROBE_OK > $B8_ROOT/dl_done
"""

RUNNER_TAIL = """
stop_server
echo ALL_DONE > $B8_ROOT/run_all_done
log "EVERYTHING DONE: $(grep -c ' ok$' $B8_ROOT/steps 2>/dev/null || echo 0) ok, \\
$(grep -c ' fail' $B8_ROOT/steps 2>/dev/null || echo 0) failed, \\
$(grep -c ' skip' $B8_ROOT/steps 2>/dev/null || echo 0) skipped"
"""

MODEL_BLOCK = """
# ---------- %(id)s : %(bats)s (%(gb).1f GB%(prism_note)s) ----------
log "===== %(id)s : fetching %(gb).1f GB ====="
%(gets)s
wait
GG="%(gguf_rel)s"
if [ -f "$M/$GG" ]; then
  if serve_model "$GG" "%(image)s"; then
%(steps)s
    echo "%(id)s" >> $B8_ROOT/models_done
    log "%(id)s complete"
  else
    log "%(id)s SERVE-FAIL"; echo "%(id)s serve-fail" >> $B8_ROOT/failures
  fi
  stop_server
else
  log "%(id)s SKIP (fetch failed)"; echo "%(id)s missing-gguf" >> $B8_ROOT/failures
fi
release "%(id)s"
"""


def steps_for(mid: str, bats: list[str]) -> str:
    """Per-battery commands, cheap deterministic first. B1 never appears (skipped at
    manifest filtering - it needs a judging pass, not GPU time)."""
    order = ["B11", "B10", "B2", "B3", "B6", "B9", "B8"]
    bats = [b for b in order if b in bats]
    L = []
    sv = "--suite-version suite-v2.2.0"
    hw = f"--hardware-sku {HW_SKU}"
    for b in bats:
        if b == "B11":
            L.append(f'    mkdir -p $B8_ROOT/agentws; run_step "{mid}" B11 "$PY" '
                     f'scripts/run_tools_agent.py $EP --model "{mid}" --reps 3 {sv} {hw} '
                     f'--workspace $B8_ROOT/agentws --out $OUT/tools')
        elif b == "B10":
            L.append(f'    run_step "{mid}" B10 "$PY" scripts/run_security.py $EP '
                     f'--model "{mid}" --reps 3 {sv} {hw} --out $OUT/security')
        elif b == "B9":
            L.append(f'    run_step "{mid}" B9 "$PY" scripts/run_games.py $EP '
                     f'--model "{mid}" --reps 3 {sv} {hw} --out $OUT/games --chrome ""')
        elif b == "B8":
            probe = ""
            if mid in PROBE_MODELS:
                probe = (
                    f'    # {mid} went 100%% infra-error in every prior B8 attempt: probe ONE task\n'
                    f'    # with the real sandbox before paying for the full 115-run sweep.\n'
                    f'    if ! run_step "{mid}" B8-probe "$PY" scripts/run_b8_local.py $EP '
                    f'--model "{mid}" --task b8.py-brk-01 --limit 1 {hw} '
                    # Task ids are namespaced "b8.*" - the old bare "py-bugfix-01"
                    # matched NOTHING, so the probe exited 0 having planned zero
                    # work items and the gate waved a broken model through. The
                    # 8-infra-error abort inside run_b8_local caught it instead
                    # (~2.5 min, not an hour), but a gate that passes by doing
                    # nothing is worse than no gate.
                    f'--results-dir $OUT/b8_probe_{mid}; then\n'
                    f'      log "  {mid} B8 SKIPPED: probe produced no eligible row - harness '
                    f'cannot drive this model; documented exclusion, not a model score"\n'
                    f'      echo "{mid} B8 skip probe-failed" >> $B8_ROOT/steps\n'
                    f'    else\n  ')
            body = (f'    run_step "{mid}" B8 "$PY" scripts/run_b8_local.py $EP '
                    f'--model "{mid}" {hw} --results-dir $OUT/b8_{mid}')
            if probe:
                L.append(probe + body + "\n    fi")
            else:
                L.append(body)
        else:  # B2/B3/B6 via bigmodel_gen (abl-qwen only in this campaign)
            n = b[1:]
            L.append(f'    run_step "{mid}" {b} "$PY" scripts/bigmodel_gen.py '
                     f'--model "{mid}" --batteries {n} $EP --results-dir $OUT/suite')
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="plan/manifest.json")
    ap.add_argument("--out", default="plan_vm")
    ap.add_argument("--only", default="",
                    help="comma-separated model ids: emit a runner for JUST these, as "
                         "<out>/run_only.sh. Used to rescue the highest-value cells when "
                         "credit will not cover the whole sweep - abl-qwen3.6-27b carries "
                         "the campaign's only non-B8 cells and is scheduled last.")
    ap.add_argument("--runner-name", default="run_all.sh")
    args = ap.parse_args()

    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    reg = yaml.safe_load((ROOT / "config" / "registry.yaml").read_text(encoding="utf-8"))["models"]
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    # B1 is judging-bound, not GPU-bound: filter it here so the plan never promises it.
    dropped_b1 = 0
    for m in man["models"]:
        n0 = len(m["batteries"])
        m["batteries"] = [b for b in m["batteries"] if b != "B1"]
        dropped_b1 += n0 - len(m["batteries"])
    man["models"] = [m for m in man["models"] if m["batteries"]]

    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        missing = want - {m["id"] for m in man["models"]}
        if missing:
            print(f"--only: unknown model id(s): {sorted(missing)}")
            return 2
        man["models"] = [m for m in man["models"] if m["id"] in want]
        args.runner_name = args.runner_name if args.runner_name != "run_all.sh" else "run_only.sh"

    blocks, nfiles = [], 0
    ordered = sorted(man["models"], key=lambda x: est_seconds(x, reg))
    for m in ordered:
        first = m["files"][0]["path"] if m["files"] else ""
        gguf_rel = f'{m["id"]}/{Path(first).name}'
        gets = "\n".join(f'gate; get {m["id"]} {m["repo"]} {f["path"]} &'
                         for f in m["files"])
        nfiles += len(m["files"])
        prism = m["id"] in PRISM_MODELS
        blocks.append(MODEL_BLOCK % {
            "id": m["id"], "bats": ",".join(m["batteries"]), "gb": m["gb"],
            "gets": gets, "gguf_rel": gguf_rel,
            "image": "prism-llama:1" if prism else "ghcr.io/ggml-org/llama.cpp:server-cuda",
            "prism_note": ", PRISM fork serve" if prism else "",
            "steps": steps_for(m["id"], m["batteries"]),
        })

    probe_m = min((m for m in man["models"] if m["files"]), key=lambda x: x["gb"])
    write_sh(outdir / "vm_setup.sh", SETUP % {"b8_root": B8_ROOT})
    write_sh(outdir / args.runner_name,
             (RUNNER_HEAD % {"b8_root": B8_ROOT,
                             "probe_repo": probe_m["repo"],
                             "probe_file": probe_m["files"][0]["path"]})
             + "\n".join(blocks) + RUNNER_TAIL)
    (outdir / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    for f in ("vm_setup.sh", "run_all.sh"):
        (outdir / f).chmod(0o755)

    cells = sum(len(m["batteries"]) for m in man["models"])
    total_h = sum(est_seconds(m, reg) for m in man["models"]) / 3600
    print(f"wrote {outdir}/vm_setup.sh + run_all.sh + manifest.json")
    print(f"  models {len(man['models'])}  GPU cells {cells}  (B1 x{dropped_b1} deferred to judging)")
    print(f"  download {sum(m['gb'] for m in man['models']):.1f} GB   est ~{total_h:.1f} h")
    print("\norder (cheapest first), cumulative hours:")
    cum = 0.0
    for m in ordered:
        cum += est_seconds(m, reg) / 3600
        print(f"  {cum:6.1f}h  {m['id']:22s} {','.join(m['batteries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
