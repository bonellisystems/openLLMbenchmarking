#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs LOCALLY (Windows git-bash) to tarball the repo for scp to the run box.
# The repo is local-git-only (no remote to clone), so we ship a tarball.
# Excludes the heavy/regenerated dirs -- the box produces fresh results.
#
#     bash deploy/blackwell/pack_repo.sh          # -> prints the tarball path
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/c/Users/Michael/AppData/Local/Temp/claude/D--BUILT-TOOLS-LLMtesting/3a826912-ebcc-45c0-b4ce-abf47845c1e0/scratchpad/llmtest-v2-b8.tgz}"

cd "$(dirname "$REPO")"
tar czf "$OUT" \
  --exclude='llmtest-v2/.git' \
  --exclude='llmtest-v2/results/*' \
  --exclude='llmtest-v2/artifacts/*' \
  --exclude='llmtest-v2/.superpowers' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/.pytest_cache' \
  --exclude='*/venv' \
  "$(basename "$REPO")"

echo "packed: $OUT ($(du -h "$OUT" | cut -f1))"
echo "scp it:  scp -i <key> '$OUT' root@<box-ip>:/opt/b8/"
echo "on box:  mkdir -p /opt/b8 && tar xzf /opt/b8/$(basename "$OUT") -C /opt/b8"
