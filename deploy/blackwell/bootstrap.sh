#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot setup for a FRESH Ubuntu 24.04 + CUDA (DataCrunch/Verda) box so it
# can run the B8 agentic eval. Run it FROM the extracted repo root:
#
#     cd /opt/b8/llmtest-v2 && sudo bash deploy/blackwell/bootstrap.sh
#
# Installs: Docker + NVIDIA container toolkit (GPU-in-Docker), the official
# llama.cpp CUDA server image (serving), the b8-sandbox:1 containment image
# (OpenCode runs INSIDE it), and a Python venv with the repo installed.
# Idempotent enough to re-run. Does NOT download models (see fetch_models.sh)
# and does NOT need any judge/agy creds (classification happens off-box).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"           # repo root (…/llmtest-v2)
B8_ROOT="${B8_ROOT:-/opt/b8}"
VENV="$B8_ROOT/venv"
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"

echo "== repo=$REPO  B8_ROOT=$B8_ROOT  llama_image=$LLAMA_IMAGE =="
mkdir -p "$B8_ROOT/models"

echo "== 1/5 system packages =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    docker.io python3 python3-pip python3-venv git curl ca-certificates tar
systemctl enable --now docker 2>/dev/null || service docker start || true

echo "== 2/5 NVIDIA container toolkit (skip if GPU already visible in Docker) =="
if ! docker run --rm --gpus all "$LLAMA_IMAGE" --version >/dev/null 2>&1; then
  install -m0755 -d /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker 2>/dev/null || service docker restart
else
  echo "   GPU already reachable from Docker; toolkit install skipped."
fi

echo "== 3/5 pull llama.cpp CUDA server image =="
docker pull "$LLAMA_IMAGE"
# Sanity: the KV-fix flags this run DEPENDS ON must exist in this image tag.
if ! docker run --rm "$LLAMA_IMAGE" --help 2>&1 | grep -q -- '--ctx-checkpoints'; then
  echo "!! WARNING: '$LLAMA_IMAGE' has no --ctx-checkpoints flag (image too old)."
  echo "!! The KV-exhaustion fix relies on it. Pin a NEWER server-cuda tag and re-run." >&2
fi

echo "== 4/5 build b8-sandbox:1 containment image =="
docker build -t b8-sandbox:1 -f "$SCRIPT_DIR/Dockerfile.sandbox" "$SCRIPT_DIR"

echo "== 5/5 python venv + repo install =="
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q pyyaml "huggingface_hub[cli]"
( cd "$REPO" && "$VENV/bin/pip" install -q -e . 2>/dev/null ) || \
    echo "   (pip install -e . skipped/failed; pyyaml alone is enough to run B8)"

cat <<EOF

bootstrap complete.
  venv:          $VENV
  models dir:    $B8_ROOT/models   (empty -- run fetch_models.sh next)
  images:        $(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'llama.cpp|b8-sandbox' | tr '\n' ' ')

Next:
  sudo bash deploy/blackwell/fetch_models.sh          # download the 2 GGUFs
  bash deploy/blackwell/run_matrix.sh                 # serve + run the full matrix
EOF
