#!/usr/bin/env python3
"""Provision an RTX PRO 6000 (and nothing else), ship the plan, run it, and hand off
to the watcher which pulls per model and destroys the box on completion.

HARD RULES ENCODED HERE, each because it already went wrong once:

  RTX PRO 6000 ONLY. Every offer is checked by name, and the card is re-verified ON
  the box before any work starts. An A100 (Ampere) and a Q RTX 8000 (Turing) both
  slipped through manual selection earlier.

  /root/.ssh MUST BE 0700. sshd silently refuses authorized_keys in a
  group-writable directory - that cost three destroyed boxes.

  scp TAKES -P, ssh TAKES -p. Reusing one option string broke every result pull
  three separate times, so the two are built separately.

  DESTROY ON COMPLETION, PULL PER MODEL. A previous box was left idle after its run,
  ran out of credit, could not be restarted, and took 96 unpulled rows with it.

    python scripts/rent_and_run.py --check     # verify offers + plan, rent nothing
    python scripts/rent_and_run.py --go        # provision and launch
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYFILE = Path.home() / ".config" / "vastai" / "vast_api_key"
PUBKEY = Path("C:/Users/Michael/.ssh/vast_laguna.pub")
PRIVKEY = "C:/Users/Michael/.ssh/vast_laguna"

REQUIRED_GPU = "RTX PRO 6000"
MIN_DISK_GB = 780          # 639GB of weights + results + the container image
MIN_RAM_GB = 200           # qwen3-235b needs --cpu-moe headroom

SSH_COMMON = ["-i", PRIVKEY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
              "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
              "-o", "LogLevel=ERROR"]


def api():
    from vastai import VastAI
    return VastAI(api_key=KEYFILE.read_text().strip())


def norm(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("-", "")


def onstart(pub: str) -> str:
    return ("export DEBIAN_FRONTEND=noninteractive; "
            "mkdir -p /root/.ssh; chmod 700 /root/.ssh; chmod go-w /root; "
            f"echo '{pub}' >> /root/.ssh/authorized_keys; "
            "chmod 600 /root/.ssh/authorized_keys; chown -R root:root /root/.ssh; "
            "apt-get update -qq && apt-get install -y -qq openssh-server python3 python3-pip "
            "curl git aria2 tmux >/tmp/onstart.log 2>&1; "
            "sed -i 's/#\\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config; "
            "mkdir -p /run/sshd && /usr/sbin/sshd; "
            "echo DONE > /tmp/onstart_done")


def find_offers(v, limit=6):
    q = (f"gpu_ram>=90 gpu_ram<100 num_gpus=1 rentable=true disk_space>={MIN_DISK_GB} "
         f"reliability>0.98 inet_down>=2000")
    res = v.search_offers(query=q, order="dph+")
    rows = res if isinstance(res, list) else res.get("offers", [])
    ok = []
    for o in rows:
        if norm(REQUIRED_GPU) not in norm(o.get("gpu_name")):
            continue
        if (o.get("cpu_ram") or 0) / 1024 < MIN_RAM_GB:
            continue
        ok.append(o)
        if len(ok) >= limit:
            break
    return ok


def sshx(host, port, cmd, timeout=900):
    return subprocess.run(["ssh", *SSH_COMMON, "-p", str(port), host, cmd],
                          capture_output=True, text=True, timeout=timeout)


def scp_to(host, port, local, remote, timeout=3600):
    return subprocess.run(["scp", *SSH_COMMON, "-P", str(port), str(local), f"{host}:{remote}"],
                          capture_output=True, text=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--est-hours", type=float, default=22.0)
    args = ap.parse_args()

    man = json.loads((ROOT / "plan" / "manifest.json").read_text(encoding="utf-8"))
    for f in ("download.sh", "run_all.sh", "setup.sh"):
        p = ROOT / "plan" / f
        if not p.exists():
            print(f"missing plan/{f} - run scripts/emit_run_plan.py first")
            return 2

    v = api()
    credit = v.show_user().get("credit", 0)
    offers = find_offers(v)

    print(f"plan      : {man['totals']['models']} models, {man['totals']['missing_cells']} cells, "
          f"{man['totals']['download_gb']} GB")
    print(f"credit    : ${credit:.2f}")
    print(f"card gate : {REQUIRED_GPU} ONLY, >={MIN_DISK_GB}GB disk, >={MIN_RAM_GB}GB RAM")
    if not offers:
        print("NO QUALIFYING RTX PRO 6000 OFFER - refusing to rent anything else")
        return 2
    for o in offers:
        print(f"  id={o['id']} {o.get('gpu_name')} {o.get('gpu_ram',0)/1024:.0f}GB "
              f"${o.get('dph_total',0):.3f}/h disk={o.get('disk_space',0):.0f}GB "
              f"ram={(o.get('cpu_ram') or 0)/1024:.0f}GB rel={o.get('reliability2',0):.3f} "
              f"{o.get('geolocation')}")

    pick = offers[0]
    est = pick["dph_total"] * args.est_hours
    print(f"\npick      : {pick['id']} @ ${pick['dph_total']:.3f}/h")
    print(f"estimate  : ~{args.est_hours:.0f}h -> ${est:.2f} "
          f"({'affordable' if est < credit else 'OVER BUDGET'})")

    if not args.go:
        print("\n--check only: nothing rented.")
        return 0
    if est > credit:
        print("refusing to start: estimate exceeds credit")
        return 2

    inst = v.create_instance(id=pick["id"], image="ghcr.io/ggml-org/llama.cpp:server-cuda",
                             disk=MIN_DISK_GB, label="close-all-gaps",
                             onstart_cmd=onstart(PUBKEY.read_text().strip()), runtype="ssh")
    cid = inst.get("new_contract")
    print(f"created   : {cid}  (success={inst.get('success')})")
    (ROOT / "plan" / "INSTANCE").write_text(str(cid), encoding="utf-8")

    host = port = None
    for _ in range(80):
        d = v.show_instance(id=cid)
        if d.get("actual_status") == "running" and d.get("ssh_host"):
            host, port = "root@" + d["ssh_host"], d["ssh_port"]
            break
        time.sleep(15)
    if not host:
        print("box never came up - destroying")
        v.destroy_instance(id=cid)
        return 1
    (ROOT / "plan" / "ENDPOINT").write_text(f"{host} {port}", encoding="utf-8")

    gpu = ""
    for _ in range(25):
        r = sshx(host, port, "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
        if r.returncode == 0 and r.stdout.strip():
            gpu = r.stdout.strip()
            break
        time.sleep(20)
    print("on-box GPU:", gpu or "(no answer)")
    if norm(REQUIRED_GPU) not in norm(gpu):
        print("WRONG CARD ON THE BOX - destroying immediately")
        v.destroy_instance(id=cid)
        return 3

    print("shipping repo + plan ...")
    subprocess.run(["tar", "czf", "/tmp/repo.tgz", "-C", str(ROOT.parent), ROOT.name,
                    "--exclude=" + ROOT.name + "/.git",
                    "--exclude=" + ROOT.name + "/artifacts",
                    "--exclude=" + ROOT.name + "/results_*"], check=False)
    scp_to(host, port, "/tmp/repo.tgz", "/root/repo.tgz")
    sshx(host, port, "cd /root && tar xzf repo.tgz && ls -d /root/llmtest-v2")
    for f in ("setup.sh", "download.sh", "run_all.sh"):
        scp_to(host, port, ROOT / "plan" / f, f"/root/{f}")
    sshx(host, port, "chmod +x /root/setup.sh /root/download.sh /root/run_all.sh")

    print("launching setup -> download -> run_all in tmux ...")
    sshx(host, port,
         "tmux new-session -d -s work 'bash /root/setup.sh > /root/setup_out.log 2>&1; "
         "bash /root/download.sh > /root/dl_out.log 2>&1; "
         "bash /root/run_all.sh > /root/run_out.log 2>&1'")
    print(f"\nlaunched on {cid} ({host}:{port})")
    print("now run: python scripts/watch_run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
