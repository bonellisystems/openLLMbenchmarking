#!/usr/bin/env bash
set -euo pipefail
cd /home/michaeldeblok/llmtest-spark/src
docker build -t b8-sandbox:1 -f deploy/blackwell/Dockerfile.sandbox deploy/blackwell
docker pull python:3.11-slim
echo B8_SANDBOX_OK
