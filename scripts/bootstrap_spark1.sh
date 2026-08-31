#!/usr/bin/env bash
set -euo pipefail
export HOME=/home/michaeldeblok
cd "$HOME"
mkdir -p llmtest-spark
tar -xzf "$HOME/llmtest-v2-spark.tgz" -C "$HOME/llmtest-spark"
rm -rf "$HOME/llmtest-spark/src"
mv "$HOME/llmtest-spark/llmtest-v2" "$HOME/llmtest-spark/src"
python3 -m venv "$HOME/llmtest-spark/venv"
# shellcheck disable=SC1091
source "$HOME/llmtest-spark/venv/bin/activate"
pip install -U pip
pip install -e "$HOME/llmtest-spark/src[dev]" PyYAML playwright
python -m playwright install chromium || true
cd "$HOME/llmtest-spark/src"
python -m pytest -q -m "not gpu"
python -m llmtest validate
mkdir -p "$HOME/llmtest-spark/out"
echo SETUP_OK
