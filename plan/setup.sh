#!/bin/bash
# One-time box setup. B8 runs in HOST mode because vast.ai instances have no Docker;
# opencode.py supports sandbox_image=None. Acceptable on a disposable box running our
# own manifests - not a pattern for untrusted models or a persistent machine.
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-pip curl git aria2 tmux nodejs npm >/root/setup.log 2>&1
python3 -m pip install -q --break-system-packages pyyaml playwright >>/root/setup.log 2>&1
python3 -m playwright install --with-deps chromium >>/root/setup.log 2>&1
npm install -g opencode-ai >>/root/setup.log 2>&1 || echo "opencode install failed (B8 will skip)"
# B8 host mode
python3 - <<'PY'
import yaml, io
p = "/root/llmtest-v2/config/suite.yaml"
d = yaml.safe_load(io.open(p, encoding="utf-8").read())
d.setdefault("b8", {}).setdefault("sandbox", {})["enabled"] = False
io.open(p, "w", encoding="utf-8").write(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print("b8.sandbox.enabled = False (host mode)")
PY
echo "node: $(node --version 2>/dev/null)  opencode: $(command -v opencode || echo absent)"
python3 -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); print('chromium OK'); b.close(); p.stop()"
echo SETUP_DONE > /root/setup_done
