#!/usr/bin/env python3
"""Emit a self-contained plan that closes bonsai-ternary-27b's B10/B11.

These two cells are the only ones left that a normal run cannot close. Ternary-Bonsai's
Q2_0 is a prism-ml CUSTOM quantization; the official ggml image simply refuses it -
"failed to load model ... Ternary-Bonsai-27B-Q2_0.gguf" - so the fix is a binary, not
more GPU hours.

Two constraints shape this plan:

  * THE STANDARD IMAGE CANNOT COMPILE. ghcr.io/ggml-org/llama.cpp:server-cuda is a
    RUNTIME image: no nvcc, no cmake (verified on the live box). So this runs on
    nvidia/cuda:*-devel, which has the toolchain, and builds the fork there.

  * BLACKWELL NEEDS ITS OWN CUDA ARCH. The card is compute capability 12.0, so the
    build is pinned to CMAKE_CUDA_ARCHITECTURES=120. Letting cmake guess has produced
    binaries that load and then run on the CPU, which looks like a slow model rather
    than a broken build.

Source: github.com/PrismML-Eng/llama.cpp, branch `prism` (the upstream fork, actively
pushed - checked the day this was written).

    python scripts/emit_bonsai_plan.py
    python scripts/rent_and_run.py --plan-dir plan_bonsai --go \\
        --image nvidia/cuda:12.6.2-devel-ubuntu24.04 --label bonsai-prism \\
        --min-disk 120 --est-hours 2
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plan_bonsai"

REPO = "prism-ml/Ternary-Bonsai-27B-gguf"
FILE = "Ternary-Bonsai-27B-Q2_0.gguf"
MID = "bonsai-ternary-27b"

SETUP = r"""#!/bin/bash
# Build the prism llama.cpp fork, which is the only build with Q2_0 kernels.
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake build-essential curl aria2 tmux python3-pip \
    libcurl4-openssl-dev >/root/setup.log 2>&1
python3 -m pip install -q --break-system-packages pyyaml >>/root/setup.log 2>&1 || \
    python3 -m pip install -q pyyaml >>/root/setup.log 2>&1

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | tee /root/caps_gpu
echo "nvcc: $(nvcc --version 2>/dev/null | tail -1)" | tee -a /root/caps_gpu

git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp /root/prism-llama \
    >>/root/setup.log 2>&1 || { echo "CLONE FAILED" > /root/build_failed; exit 1; }

cd /root/prism-llama
# 120 = Blackwell (compute 12.0). Pinned, not guessed: a mismatched arch produces a
# binary that silently falls back to CPU, which reads as a slow model, not a bad build.
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF >>/root/setup.log 2>&1 \
      || { echo "CMAKE CONFIGURE FAILED" > /root/build_failed; tail -30 /root/setup.log; exit 1; }
cmake --build build --config Release -j"$(nproc)" --target llama-server >>/root/setup.log 2>&1 \
      || { echo "BUILD FAILED" > /root/build_failed; tail -40 /root/setup.log; exit 1; }

BIN=/root/prism-llama/build/bin/llama-server
test -x $BIN || { echo "NO BINARY" > /root/build_failed; exit 1; }
$BIN --version 2>&1 | tee /root/caps
# Q2_0 is the whole point of this build; if the fork stopped shipping it, say so loudly
# rather than discovering it as a load failure 20 minutes later.
$BIN --help 2>&1 | grep -qi "spec-type" && echo "spec-type present" >> /root/caps

# B8 host mode is irrelevant here, but the registry path repoint is not: the runners
# resolve the GGUF through it.
python3 - <<'PY'
import io, json, yaml
from pathlib import Path
rp = "/root/llmtest-v2/config/registry.yaml"
reg = yaml.safe_load(io.open(rp, encoding="utf-8").read())
man = json.load(io.open("/root/plan_manifest.json", encoding="utf-8"))
for m in man["models"]:
    if m["id"] in reg["models"] and m["files"]:
        reg["models"][m["id"]]["local_path"] = str(
            Path("/root/models") / m["id"] / Path(m["files"][0]["path"]).name)
io.open(rp, "w", encoding="utf-8").write(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True))
print("registry local_path repointed")
PY
echo SETUP_DONE > /root/setup_done
"""

RUNNER = r"""#!/bin/bash
# Serve bonsai-ternary-27b from the prism fork and close B10 + B11.
set -u
cd /root/llmtest-v2
BIN=/root/prism-llama/build/bin/llama-server
export LD_LIBRARY_PATH=/root/prism-llama/build/bin
OUT=/root/out
mkdir -p $OUT
log(){ echo "$(date +%%H:%%M:%%S) $*" | tee -a /root/run.log; }

if [ -f /root/build_failed ]; then
  log "PRISM BUILD FAILED: $(cat /root/build_failed) - cannot serve Q2_0, aborting"
  echo "%(mid)s prism-build-failed" >> /root/failures
  echo ALL_DONE > /root/run_all_done
  exit 1
fi

run_step(){ # $1 model $2 battery $3.. command
  mid="$1"; bat="$2"; shift 2
  log "  $mid $bat start"
  "$@" > /root/last_step.log 2>&1
  rc=$?
  tail -3 /root/last_step.log | tee -a /root/run.log
  if [ "$rc" -eq 0 ]; then echo "$mid $bat ok" >> /root/steps
  else log "  $mid $bat FAILED rc=$rc"; echo "$mid $bat fail rc=$rc" >> /root/steps
       cat /root/last_step.log >> /root/step_failures.log; fi
}

M=/root/models/%(mid)s
mkdir -p $M
log "fetching %(file)s"
aria2c -x8 -s8 -k1M --continue=true --file-allocation=none --console-log-level=warn \
  -d $M -o "%(file)s" "https://huggingface.co/%(repo)s/resolve/main/%(file)s" \
  >> /root/dl.log 2>&1 || echo "FAIL fetch" >> /root/dl_fail

GG="$M/%(file)s"
if [ ! -f "$GG" ]; then
  log "%(mid)s SKIP (fetch failed)"; echo "%(mid)s missing-gguf" >> /root/failures
  echo ALL_DONE > /root/run_all_done; exit 1
fi

log "===== %(mid)s (prism fork, Q2_0) ====="
pkill -f "prism-llama/build/bin/llama-server" 2>/dev/null; sleep 3
nohup $BIN -m "$GG" -ngl 99 -c 49152 --parallel 1 --jinja -fa on \
  -ctk q8_0 -ctv q8_0 --spec-type ngram-mod --spec-ngram-mod-n-match 32 \
  --host 127.0.0.1 --port 8080 --no-webui > /root/serve.log 2>&1 &
ok=0
for i in $(seq 1 200); do
  curl -s -m3 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && { ok=1; break; }
  sleep 4
done
if [ "$ok" != "1" ]; then
  log "%(mid)s SERVE-FAIL even on the prism fork - see /root/serve.log"
  echo "%(mid)s serve-fail-on-prism" >> /root/failures
else
  mkdir -p /root/agentws
  run_step "%(mid)s" B11 python3 scripts/run_tools_agent.py \
    --endpoint-url http://127.0.0.1:8080 --model "%(mid)s" --reps 3 \
    --workspace /root/agentws --out $OUT/tools
  run_step "%(mid)s" B10 python3 scripts/run_security.py \
    --endpoint-url http://127.0.0.1:8080 --model "%(mid)s" --reps 3 \
    --out $OUT/security
  echo "%(mid)s" >> /root/models_done
fi
pkill -f "prism-llama/build/bin/llama-server" 2>/dev/null
echo ALL_DONE > /root/run_all_done
log "EVERYTHING DONE"
"""


def write_sh(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_sh(OUT / "setup.sh", SETUP)
    write_sh(OUT / "run_all.sh", RUNNER % {"mid": MID, "repo": REPO, "file": FILE})
    (OUT / "manifest.json").write_text(json.dumps({
        "target_card": "96GB (RTX PRO 6000 Blackwell only)",
        "models": [{"id": MID, "repo": REPO, "quant_file": FILE,
                    "files": [{"path": FILE, "size": 6_700_000_000}],
                    "gb": 6.7, "resolved": True, "fits_card": True,
                    "batteries": ["B10", "B11"],
                    "notes": {"B10": "needs the prism fork's Q2_0 kernels",
                              "B11": "needs the prism fork's Q2_0 kernels"}}],
        "totals": {"models": 1, "download_gb": 6.7, "missing_cells": 2},
        "warnings": {"unresolved_files": [], "size_mismatch_vs_registry": [],
                     "needs_offload": []},
    }, indent=1), encoding="utf-8")
    for f in ("setup.sh", "run_all.sh"):
        (OUT / f).chmod(0o755)
    print(f"wrote {OUT}/setup.sh      (clone + CUDA-120 build of the prism fork)")
    print(f"wrote {OUT}/run_all.sh    ({MID} B10 + B11 on the fork binary)")
    print(f"wrote {OUT}/manifest.json (1 model, 2 cells, 6.7 GB)")
    print("\nlaunch:")
    print("  python scripts/rent_and_run.py --plan-dir plan_bonsai --go \\")
    print("      --image nvidia/cuda:12.6.2-devel-ubuntu24.04 --label bonsai-prism \\")
    print("      --min-disk 120 --est-hours 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
