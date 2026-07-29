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
# qwen3-235b is 134GB and does not fit the 96GB card, so --cpu-moe parks its expert
# tensors in system RAM: ~120GB resident plus OS and page-cache headroom. 160GB clears
# that. A 200GB floor disqualified 50 of the 54 available RTX PRO 6000 boxes (the
# common configuration is 126GB) for headroom nothing in the plan uses.
MIN_RAM_GB = 160

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
    """Qualifying RTX PRO 6000 offers, cheapest first, plus a per-constraint tally of
    the rejects. "No offer" on its own is a dead end; knowing WHICH filter bound is
    what tells you whether to wait for capacity or relax a number."""
    res = v.search_offers(query="gpu_ram>=90 gpu_ram<100 num_gpus=1 rentable=true",
                          order="dph+")
    rows = res if isinstance(res, list) else res.get("offers", [])
    ok, why = [], {"wrong card": 0, "disk": 0, "ram": 0, "reliability": 0, "slow net": 0}
    for o in rows:
        if norm(REQUIRED_GPU) not in norm(o.get("gpu_name")):
            why["wrong card"] += 1
            continue
        if (o.get("disk_space") or 0) < MIN_DISK_GB:
            why["disk"] += 1
            continue
        if (o.get("cpu_ram") or 0) / 1024 < MIN_RAM_GB:
            why["ram"] += 1
            continue
        if (o.get("reliability2") or 0) < 0.98:
            why["reliability"] += 1
            continue
        if (o.get("inet_down") or 0) < 1500:
            why["slow net"] += 1
            continue
        ok.append(o)
    return ok[:limit], why


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
        # These are generated on Windows and executed on Linux. A single CR makes bash
        # fail on `do\r` and silently breaks `\` line continuations.
        if b"\r" in p.read_bytes():
            print(f"plan/{f} has CRLF endings - re-run scripts/emit_run_plan.py")
            return 2

    v = api()
    credit = v.show_user().get("credit", 0)
    offers, why = find_offers(v)

    print(f"plan      : {man['totals']['models']} models, {man['totals']['missing_cells']} cells, "
          f"{man['totals']['download_gb']} GB")
    print(f"credit    : ${credit:.2f}")
    print(f"card gate : {REQUIRED_GPU} ONLY, >={MIN_DISK_GB}GB disk, >={MIN_RAM_GB}GB RAM")
    if not offers:
        print("NO QUALIFYING RTX PRO 6000 OFFER - refusing to rent anything else")
        print("  rejected by: " + ", ".join(f"{k}={n}" for k, n in why.items() if n))
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

    # onstart installs sshd via apt, which can take 10-20 minutes on a slow host.
    # SILENCE IS "NOT READY YET", NOT "WRONG CARD" - conflating the two destroyed a
    # perfectly good box on the first attempt.
    gpu = ""
    for attempt in range(90):
        r = sshx(host, port, "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                 timeout=45)
        if r.returncode == 0 and r.stdout.strip():
            gpu = r.stdout.strip()
            break
        if attempt and attempt % 10 == 0:
            print(f"  waiting for sshd ... ({attempt*20//60} min)")
        time.sleep(20)

    if not gpu:
        print("box never answered over SSH; leaving it up for manual inspection")
        print(f"  instance {cid} at {host}:{port} - destroy with scripts/watch_run.py or the API")
        return 1
    print("on-box GPU:", gpu)
    if norm(REQUIRED_GPU) not in norm(gpu):
        print(f"WRONG CARD ({gpu}) - destroying immediately")
        v.destroy_instance(id=cid)
        return 3
    print(f"card gate PASSED: {gpu}")

    print("shipping repo + plan ...")
    # Verify each step. The tar used to run with check=False, so a failed archive
    # scp'd nothing and the run started against an empty box - which looks identical
    # to a model that would not load.
    tgz = ROOT / "plan" / "repo.tgz"
    tgz.unlink(missing_ok=True)
    t = subprocess.run(["tar", "czf", str(tgz), "-C", str(ROOT.parent),
                        "--exclude=.git", "--exclude=artifacts", "--exclude=results_*",
                        "--exclude=plan/repo.tgz", ROOT.name],
                       capture_output=True, text=True)
    if not tgz.exists() or tgz.stat().st_size < 100_000:
        print(f"tar FAILED (rc={t.returncode}): {t.stderr[:400]}")
        print(f"instance {cid} left running at {host}:{port} - fix and re-ship")
        return 1
    print(f"  archive: {tgz.stat().st_size/1e6:.1f} MB")

    if scp_to(host, port, tgz, "/root/repo.tgz").returncode != 0:
        print("scp of repo FAILED; instance left running for inspection")
        return 1
    r = sshx(host, port, "cd /root && tar xzf repo.tgz && test -f llmtest-v2/scripts/"
                         "bigmodel_gen.py && echo REPO_OK")
    if "REPO_OK" not in r.stdout:
        print("repo did not unpack on the box; instance left running:", r.stdout, r.stderr[:300])
        return 1
    for f in ("setup.sh", "download.sh", "run_all.sh"):
        scp_to(host, port, ROOT / "plan" / f, f"/root/{f}")
    # setup.sh repoints registry local_path from this manifest (the p8 drivers resolve
    # the GGUF through local_path, which still holds the Windows authoring paths).
    scp_to(host, port, ROOT / "plan" / "manifest.json", "/root/plan_manifest.json")
    # A CR in these would break bash on the box; assert none survived the trip.
    r = sshx(host, port, "chmod +x /root/*.sh; "
                         "grep -l $'\\r' /root/setup.sh /root/download.sh /root/run_all.sh "
                         "2>/dev/null || echo NO_CR")
    if "NO_CR" not in r.stdout:
        print("CRLF found in shipped scripts - refusing to launch:", r.stdout)
        return 1
    print("  scripts shipped, LF endings confirmed")

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
