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

# --- capability probe -------------------------------------------------------------
# --spec-type ngram-mod originated in the prism fork and has since landed upstream, so
# whether THIS image has it is an empirical question, not an assumption. B5/B7 are
# timing-authoritative at spec=ngram32 and skip themselves if the answer is no.
: > /root/caps
# LD_LIBRARY_PATH IS REQUIRED EVEN TO READ --help. Without it the binary dies with
# "libllama-server-impl.so: cannot open shared object file" and every grep below finds
# nothing - which the first version of this probe recorded as "no --spec-type flag in
# this build", silently disqualifying B5/B7 on a build that supports ngram-mod fine.
export LD_LIBRARY_PATH=/app
HELP=$(/app/llama-server --help 2>&1)
NLINES=$(printf '%s
' "$HELP" | wc -l)
if [ "$NLINES" -lt 20 ]; then
  echo "ngram=0  # PROBE FAILED - --help produced $NLINES lines:" >> /root/caps
  printf '%s
' "$HELP" | head -3 | sed 's/^/#   /' >> /root/caps
elif printf '%s
' "$HELP" | grep -q 'ngram-mod'; then
  echo "ngram=1" >> /root/caps
else
  echo "ngram=0  # build has no ngram-mod spec type" >> /root/caps
fi
/app/llama-server --version >> /root/caps 2>&1 || true
echo "--- caps ---"; cat /root/caps

# --- config patches --------------------------------------------------------------
python3 - <<'PY'
import io, json, yaml
from pathlib import Path

sp = "/root/llmtest-v2/config/suite.yaml"
d = yaml.safe_load(io.open(sp, encoding="utf-8").read())

# B8 host mode: vast.ai instances have no Docker, and opencode.py supports
# sandbox_image=None. Acceptable on a disposable box running our own manifests.
d.setdefault("b8", {}).setdefault("sandbox", {})["enabled"] = False

# B4 arms are pruned by estimated physical fit. The default plan tier is T1 (the 24GB
# laptop), which on this 96GB card would prune every tier above 16k and give a 57.6GB
# model ZERO arms - contributing no B4 rows while looking like a clean run. The frozen
# roster's B4 rows were produced on this same card class (results/sessions.jsonl:
# hardware_sku=rtx-pro-6000, measured_usable_vram_gb=94.0), which is T3.
d.setdefault("b4", {})["plan_tier"] = "T3"
io.open(sp, "w", encoding="utf-8").write(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
print("suite.yaml: b8.sandbox.enabled=False, b4.plan_tier=T3")

# p8_gen_serving.py / p8_gen_b5.py resolve the GGUF from registry local_path, which
# still holds the Windows authoring paths. Point it at what download.sh actually wrote.
# download.sh flattens every file to $M/<id>/<basename>, so a sharded model's shards all
# land side by side in one directory and llama.cpp resolves the set from shard 00001.
rp = "/root/llmtest-v2/config/registry.yaml"
reg = yaml.safe_load(io.open(rp, encoding="utf-8").read())
man = json.load(io.open("/root/plan_manifest.json", encoding="utf-8"))
fixed = 0
for m in man["models"]:
    if not m["files"]:
        continue
    p = Path("/root/models") / m["id"] / Path(m["files"][0]["path"]).name
    if m["id"] in reg["models"]:
        reg["models"][m["id"]]["local_path"] = str(p)
        fixed += 1
io.open(rp, "w", encoding="utf-8").write(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True))
print(f"registry.yaml: local_path repointed for {fixed} models")
PY
echo "node: $(node --version 2>/dev/null)  opencode: $(command -v opencode || echo absent)"
python3 -c "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); b=p.chromium.launch(); print('chromium OK'); b.close(); p.stop()"
echo SETUP_DONE > /root/setup_done
