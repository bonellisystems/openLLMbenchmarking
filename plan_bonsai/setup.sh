#!/bin/bash
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
