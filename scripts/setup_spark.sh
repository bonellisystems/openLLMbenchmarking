#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/llmtest-spark"
mkdir -p "$BASE/out" "$BASE/src"
cd "$BASE"
if [ -f "$HOME/llmtest-v2-spark.tgz" ]; then
  tar -xzf "$HOME/llmtest-v2-spark.tgz" -C "$BASE"
  rm -rf "$BASE/src"
  mv "$BASE/llmtest-v2" "$BASE/src"
fi
python3 -m venv "$BASE/venv"
# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
pip install -U pip
pip install -e "$BASE/src[dev]" PyYAML playwright
python -m playwright install chromium || true
cd "$BASE/src"
python -m pytest -q -m "not gpu" || true
python -m llmtest validate
echo SETUP_OK
