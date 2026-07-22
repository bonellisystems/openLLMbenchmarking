#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot setup for a FRESH Ubuntu 24.04 + CUDA (DataCrunch/Verda) box so it
# can run the B8 agentic eval. Run it FROM the extracted repo root:
#
#     cd /opt/b8/llmtest-v2 && sudo bash deploy/blackwell/bootstrap.sh
#
# Installs: Docker (only if the image doesn't already ship it) + NVIDIA
# container toolkit, the official llama.cpp CUDA server image, the b8-sandbox:1
# containment image, and a Python venv with the repo. Idempotent enough to
# re-run. Does NOT download models (fetch_models.sh) and needs no judge creds.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
B8_ROOT="${B8_ROOT:-/opt/b8}"
VENV="$B8_ROOT/venv"
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"

echo "== repo=$REPO  B8_ROOT=$B8_ROOT  llama_image=$LLAMA_IMAGE =="
mkdir -p "$B8_ROOT/models"

echo "== 1/6 system packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
PKGS="python3 python3-pip python3-venv git curl ca-certificates tar"
# The CUDA+Docker image already ships Docker CE (docker-ce/containerd.io);
# installing docker.io on top hits "containerd.io Conflicts: containerd" and
# aborts. Only add docker.io if Docker is genuinely missing.
command -v docker >/dev/null 2>&1 || PKGS="$PKGS docker.io"
apt-get install -y --no-install-recommends $PKGS
systemctl enable --now docker 2>/dev/null || service docker start 2>/dev/null || true

echo "== 2/6 pull llama.cpp CUDA server image =="
docker pull "$LLAMA_IMAGE"

echo "== 3/6 nvidia container runtime for Docker =="
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "   installing nvidia-container-toolkit..."
  install -m0755 -d /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null || true
sleep 3
if docker run --rm --gpus all "$LLAMA_IMAGE" --version >/dev/null 2>&1; then
  echo "   GPU reachable from Docker."
else
  echo "   (GPU-in-Docker probe inconclusive via llama --version; the smoke test will confirm.)"
fi

echo "== 4/6 verify --ctx-checkpoints in image (the KV-exhaustion fix depends on it) =="
if docker run --rm "$LLAMA_IMAGE" --help 2>&1 | grep -q -- '--ctx-checkpoints'; then
  echo "   --ctx-checkpoints present."
else
  echo "!! WARNING: '$LLAMA_IMAGE' has no --ctx-checkpoints (image too old); pin a newer tag." >&2
fi

echo "== 5/6 build b8-sandbox:1 containment image =="
docker build -t b8-sandbox:1 -f "$SCRIPT_DIR/Dockerfile.sandbox" "$SCRIPT_DIR"

echo "== 6/6 python venv + repo install =="
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q pyyaml "huggingface_hub[cli]"
( cd "$REPO" && "$VENV/bin/pip" install -q -e . 2>/dev/null ) || \
    echo "   (pip install -e . skipped/failed; pyyaml alone is enough to run B8)"

cat <<EOF

bootstrap complete.
  venv:      $VENV
  images:    $(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'llama.cpp|b8-sandbox' | tr '\n' ' ')
Next:
  sudo bash deploy/blackwell/fetch_models.sh
  bash deploy/blackwell/smoke.sh          # 1-task gate BEFORE the full matrix
EOF
