#!/bin/bash
# One-shot VM setup. Run AFTER the repo is extracted at $B8_ROOT/llmtest-v2.
set -euo pipefail
export B8_ROOT=/opt/b8
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
if timeout 2700 docker build -t prism-llama:1 -f deploy/blackwell/Dockerfile.prism deploy/blackwell      > $B8_ROOT/prism_build.log 2>&1; then
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
if [ "$(printf '%s\n' "$HELP" | wc -l)" -lt 20 ]; then
  echo "ngram=0 # PROBE FAILED - --help too short" >> $B8_ROOT/caps
elif printf '%s\n' "$HELP" | grep -q 'ngram-mod'; then
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
