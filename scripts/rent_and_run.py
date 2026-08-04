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
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYFILE = Path.home() / ".config" / "vastai" / "vast_api_key"
PUBKEY = Path("C:/Users/Michael/.ssh/vast_laguna.pub")
PRIVKEY = "C:/Users/Michael/.ssh/vast_laguna"

REQUIRED_GPU = "RTX PRO 6000"
# FLOORS MUST TRACK THE CAMPAIGN, NOT FOSSILIZE. Twice now a stale floor silently
# rejected most of the market: 200GB RAM (for a model later excluded) cut 50 of 54
# offers, and the 160GB/1500Mbps pair below was still excluding $0.80-1.06/h boxes -
# Michael spotted a $1.06 PCIe-5.0 machine (m:144787) that the RAM floor would have
# refused for headroom nothing in the current plan uses. Each floor states WHAT it
# protects; when that thing leaves the plan, the floor goes with it.
#
# Disk: peak = largest single model (glm-4.5-air, ~68GB; qwen3-235b is EXCLUDED) + the
# container image + results + fetch-ahead slack. Fetch->run->release keeps it flat.
MIN_DISK_GB = 200
# RAM: weights live in VRAM, not RAM - no --cpu-moe model remains in the plan. RAM is
# page cache + aria2 buffers + the harness; 64GB is already generous.
MIN_RAM_GB = 64
# Inet: the on-box HF probe is the REAL gate (aborts below 25MB/s before any spend).
# Michael's 2026-08-03 spec raised the pre-screen to 2Gbps: 504GB of fetches at
# 2Gbps ~= 34min total, so downloads never dominate the meter.
MIN_INET_MBPS = 2000

SSH_COMMON = ["-i", PRIVKEY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
              "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
              "-o", "LogLevel=ERROR"]


def api():
    from vastai import VastAI
    return VastAI(api_key=KEYFILE.read_text().strip())


def safe_err(e: Exception) -> str:
    """The vast SDK builds request URLs with ?api_key=... and puts the whole URL in
    HTTPError text, so printing a raw SDK exception leaks the key into logs and
    terminal scrollback. Never print one directly."""
    return re.sub(r"api_key=[A-Za-z0-9]+", "api_key=REDACTED", str(e))[:300]


def norm(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("-", "")


def onstart(pub: str) -> str:
    """Fix up /root/.ssh and get out of the way.

    DO NOT INSTALL OR START sshd HERE. vast.ai's runtype="ssh" supplies the ssh layer
    for any image, and an earlier version of this function ran
    `apt-get install openssh-server` + `sed PermitRootLogin` + `/usr/sbin/sshd`. That
    replaced the working daemon with one whose authorized_keys did not include the
    instance's attached key, and every connection got "Permission denied (publickey)"
    even though the key was registered AND attached (verified via show_ssh_keys and
    attach_ssh, which answered "already associated"). The runs that DID connect used a
    plain onstart that never touched sshd.

    What stays: 0700 on /root/.ssh and go-w off /root. sshd silently refuses
    authorized_keys in a group-writable directory, and that cost three boxes.
    Package installs belong in setup.sh, which is also why this is now fast - the
    30-minute "waiting for sshd" window was self-inflicted.
    """
    return ("mkdir -p /root/.ssh; chmod 700 /root/.ssh; chmod go-w /root; "
            f"echo '{pub}' >> /root/.ssh/authorized_keys; "
            "chmod 600 /root/.ssh/authorized_keys; chown -R root:root /root/.ssh; "
            "echo DONE > /tmp/onstart_done")


def find_offers(v, limit=6, vm=False):
    """Qualifying RTX PRO 6000 offers, cheapest first, plus a per-constraint tally of
    the rejects. "No offer" on its own is a dead end; knowing WHICH filter bound is
    what tells you whether to wait for capacity or relax a number."""
    # verified=any is ESSENTIAL for the VM path: vast's default query returns only
    # host-verified machines, and on 2026-08-03 ZERO VM-capable PRO 6000 offers were
    # verified - the filter silently emptied the market. Unverified is acceptable here
    # because every host-quality failure is gated downstream (Docker gate, HF throughput
    # probe, stall watcher, low-credit destroy); worst case is ~1-2 dollars of discovery.
    q = "gpu_ram>=90 gpu_ram<100 num_gpus=1 rentable=true"
    if vm:
        q += " verified=any"
    res = v.search_offers(query=q, order="dph+")
    rows = res if isinstance(res, list) else res.get("offers", [])
    ok, why = [], {"wrong card": 0, "not a VM host": 0, "disk": 0, "ram": 0,
                   "reliability": 0, "slow net": 0, "not pcie5": 0, "slow disk": 0}
    for o in rows:
        if norm(REQUIRED_GPU) not in norm(o.get("gpu_name")):
            why["wrong card"] += 1
            continue
        if vm and not o.get("vms_enabled"):
            why["not a VM host"] += 1
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
        if (o.get("inet_down") or 0) < MIN_INET_MBPS:
            why["slow net"] += 1
            continue
        # Michael's 2026-08-03 host spec for the campaign box: PCIe 5.0, NVMe-class
        # disk, 2 Gbps+ inet so 504GB of model fetches never dominate wall-clock.
        if vm and (o.get("pci_gen") or 0) < 5.0:
            why["not pcie5"] += 1
            continue
        if vm and (o.get("disk_bw") or 0) < 1500:
            why["slow disk"] += 1
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
    # Parameterised rather than duplicated: every rule in this file was learned the hard
    # way (card gate, 0700 on /root/.ssh, separate ssh -p / scp -P, destroy-on-complete),
    # and a second copy for a one-off run would drift away from all of them.
    ap.add_argument("--plan-dir", default="plan",
                    help="directory holding setup.sh / run_all.sh / manifest.json")
    ap.add_argument("--image", default="ghcr.io/ggml-org/llama.cpp:server-cuda",
                    help="container image. The default is a RUNTIME image with no nvcc; "
                         "a run that compiles anything needs a -devel image.")
    ap.add_argument("--label", default="close-all-gaps")
    ap.add_argument("--min-disk", type=int, default=None)
    ap.add_argument("--vm", action="store_true",
                    help="Rent a KVM VIRTUAL MACHINE, not a container. Required for the "
                         "B8 campaign: the completion oracle shells out to Docker, and "
                         "container instances have none - that exact gap poisoned 554 "
                         "rows. Filters offers to vms_enabled=true, launches "
                         "docker.io/vastai/kvm:ubuntu_terminal, verifies `docker info` "
                         "on-box, and ships to the /opt/b8 layout "
                         "(deploy/blackwell/RUN_ON_BLACKWELL.md).")
    args = ap.parse_args()
    if args.vm and args.image == "ghcr.io/ggml-org/llama.cpp:server-cuda":
        # VM instances boot a vastai/kvm image; llama.cpp runs INSIDE it via Docker.
        args.image = "docker.io/vastai/kvm:ubuntu_terminal"

    global MIN_DISK_GB
    if args.min_disk:
        MIN_DISK_GB = args.min_disk
    plan_dir = ROOT / args.plan_dir

    man = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))
    setup_name = "vm_setup.sh" if args.vm else "setup.sh"
    for f in ("run_all.sh", setup_name):
        p = plan_dir / f
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
    offers, why = find_offers(v, vm=args.vm)

    print(f"plan      : {man['totals']['models']} models, {man['totals']['missing_cells']} cells, "
          f"{man['totals']['download_gb']} GB")
    print(f"credit    : ${credit:.2f}")
    print(f"card gate : {REQUIRED_GPU} ONLY{', VM hosts only (Docker for the B8 oracle)' if args.vm else ''}, "
          f">={MIN_DISK_GB}GB disk, >={MIN_RAM_GB}GB RAM")
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

    # VM instances: NO onstart ssh-hygiene - the KVM image's sshd is vast-managed and
    # keyed from the ACCOUNT (keys cannot be changed on a running VM), and the login
    # user is not guaranteed to be root. The container-mode hygiene stays as-is.
    kw = dict(id=pick["id"], image=args.image, disk=MIN_DISK_GB, label=args.label,
              runtype="ssh")
    if not args.vm:
        kw["onstart_cmd"] = onstart(PUBKEY.read_text().strip())
    inst = v.create_instance(**kw)
    cid = inst.get("new_contract")
    print(f"created   : {cid}  (success={inst.get('success')})")
    (plan_dir / "INSTANCE").write_text(str(cid), encoding="utf-8")

    host = port = None
    for _ in range(80):
        d = v.show_instance(id=cid)
        if d.get("actual_status") == "running" and d.get("ssh_host"):
            host, port = "root@" + d["ssh_host"], d["ssh_port"]
            break
        time.sleep(15)
    # KVM images may log in as ubuntu rather than root - resolved at gate time below.
    ssh_users = ("root", "ubuntu") if args.vm else ("root",)
    if not host:
        print("box never came up - destroying")
        v.destroy_instance(id=cid)
        return 1
    (plan_dir / "ENDPOINT").write_text(f"{host} {port}", encoding="utf-8")

    # onstart installs sshd via apt, which can take 10-20 minutes on a slow host.
    # SILENCE IS "NOT READY YET", NOT "WRONG CARD" - conflating the two destroyed a
    # perfectly good box on the first attempt.
    gpu = ""
    for attempt in range(40):
        for u in ssh_users:
            cand = u + "@" + host.split("@", 1)[1]
            r = sshx(cand, port, "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                     timeout=45)
            if r.returncode == 0 and r.stdout.strip():
                gpu = r.stdout.strip()
                host = cand
                break
        if gpu:
            break
        if attempt and attempt % 8 == 0:
            print(f"  waiting for sshd ... ({attempt*15//60} min)", flush=True)
        time.sleep(15)

    if not gpu:
        print("box never answered over SSH; leaving it up for manual inspection")
        print(f"  instance {cid} at {host}:{port} - destroy with scripts/watch_run.py or the API")
        return 1
    print("on-box GPU:", gpu)
    if norm(REQUIRED_GPU) not in norm(gpu):
        print(f"WRONG CARD ({gpu}) - destroying immediately")
        v.destroy_instance(id=cid)
        return 3
    print(f"card gate PASSED: {gpu}  (ssh as {host.split('@')[0]})")
    (plan_dir / "ENDPOINT").write_text(f"{host} {port}", encoding="utf-8")
    if args.vm:
        # The entire reason for a VM: the B8 oracle shells out to Docker. Prove it
        # BEFORE shipping half a terabyte of models.
        r = sshx(host, port, "sudo -n docker info >/dev/null 2>&1 && echo DOCKER_OK || "
                             "(docker info >/dev/null 2>&1 && echo DOCKER_OK) || echo NO_DOCKER")
        if "DOCKER_OK" not in r.stdout:
            # KVM ubuntu images ship docker-ready but the daemon may need a start.
            sshx(host, port, "sudo -n systemctl start docker 2>/dev/null || "
                             "sudo -n service docker start 2>/dev/null || true")
            r = sshx(host, port, "sudo -n docker info >/dev/null 2>&1 && echo DOCKER_OK || echo NO_DOCKER")
        if "DOCKER_OK" not in r.stdout:
            print("NOT A WORKING VM: docker unavailable even after a daemon start - "
                  "destroying (a container box wearing a VM flag would just recreate "
                  "the 554 oracle-SETUPFAIL rows)")
            v.destroy_instance(id=cid)
            return 3
        print("VM gate PASSED: docker reachable")

    print("shipping repo + plan ...")
    # Verify each step. The tar used to run with check=False, so a failed archive
    # scp'd nothing and the run started against an empty box - which looks identical
    # to a model that would not load.
    tgz = plan_dir / "repo.tgz"
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

    B8R = "/opt/b8" if args.vm else "/root"
    SUDO = "sudo -n " if (args.vm and host.startswith("ubuntu@")) else ""
    sshx(host, port, f"{SUDO}mkdir -p {B8R} && {SUDO}chown -R $(whoami) {B8R}")
    if scp_to(host, port, tgz, f"{B8R}/repo.tgz").returncode != 0:
        print("scp of repo FAILED; instance left running for inspection")
        return 1
    r = sshx(host, port, f"cd {B8R} && tar xzf repo.tgz && test -f llmtest-v2/scripts/"
                         "bigmodel_gen.py && echo REPO_OK")
    if "REPO_OK" not in r.stdout:
        print("repo did not unpack on the box; instance left running:", r.stdout, r.stderr[:300])
        return 1

    if args.vm:
        # VM layout: the plan scripts travel INSIDE the repo (plan_vm/), and the runners
        # resolve everything under /opt/b8. NO canonical-shard seeding on this campaign:
        # every in-scope cell is re-run WHOLE, so a seeded shard would make the runner
        # skip exactly the rows being re-measured.
        r = sshx(host, port, "grep -l $'\r' " + B8R + "/llmtest-v2/plan_vm/*.sh "
                             "2>/dev/null || echo NO_CR")
        if "NO_CR" not in r.stdout:
            print("CRLF found in shipped plan_vm scripts - refusing to launch:", r.stdout)
            return 1
        print("  repo + plan_vm shipped, LF endings confirmed")
        guard = ROOT / "deploy" / "shutdown_guard.sh"
        if guard.exists():
            scp_to(host, port, guard, f"{B8R}/shutdown_guard.sh")
            sshx(host, port, f"chmod +x {B8R}/shutdown_guard.sh; "
                             f"tmux new-session -d -s shutguard 'B8_ROOT={B8R} "
                             f"bash {B8R}/shutdown_guard.sh'")
            print("  idle-cost guard armed")
        print("launching vm_setup -> run_all in tmux ...")
        sshx(host, port,
             f"tmux new-session -d -s work '{SUDO}bash {B8R}/llmtest-v2/plan_vm/vm_setup.sh "
             f"> {B8R}/setup_out.log 2>&1 && "
             f"bash {B8R}/llmtest-v2/plan_vm/run_all.sh > {B8R}/run_out.log 2>&1'")
    else:
        for f in ("setup.sh", "run_all.sh"):
            scp_to(host, port, plan_dir / f, f"/root/{f}")
        # setup.sh repoints registry local_path from this manifest (the p8 drivers resolve
        # the GGUF through local_path, which still holds the Windows authoring paths).
        scp_to(host, port, plan_dir / "manifest.json", "/root/plan_manifest.json")

        # SEED WHAT IS ALREADY COLLECTED (container-mode gap-closing only; the VM
        # campaign must NOT seed - see the vm branch above).
        sshx(host, port, "mkdir -p /root/out/games /root/out/security /root/out/tools")
        for sub, shard in (("games", "rows-games.jsonl"),
                           ("security", "rows-security.jsonl"),
                           ("tools", "rows-tools.jsonl")):
            local = ROOT / f"results_{sub}" / shard
            if local.exists():
                scp_to(host, port, local, f"/root/out/{sub}/{shard}")
                print(f"  seeded {sub}: {sum(1 for _ in local.open(encoding='utf-8'))} rows")
        r = sshx(host, port, "chmod +x /root/*.sh; "
                             "grep -l $'\r' /root/setup.sh /root/run_all.sh "
                             "2>/dev/null || echo NO_CR")
        if "NO_CR" not in r.stdout:
            print("CRLF found in shipped scripts - refusing to launch:", r.stdout)
            return 1
        print("  scripts shipped, LF endings confirmed")

        guard = ROOT / "deploy" / "shutdown_guard.sh"
        if guard.exists():
            scp_to(host, port, guard, "/root/shutdown_guard.sh")
            sshx(host, port, "chmod +x /root/shutdown_guard.sh; "
                             "tmux new-session -d -s shutguard 'bash /root/shutdown_guard.sh'")
            print("  idle-cost guard armed (powers off 15 min after completion)")

        # tmux, not `nohup ... &` - a backgrounded job over ssh dies with the session.
        print("launching setup -> run_all in tmux ...")
        sshx(host, port,
             "tmux new-session -d -s work 'bash /root/setup.sh > /root/setup_out.log 2>&1; "
             "bash /root/run_all.sh > /root/run_out.log 2>&1'")
    print(f"\nlaunched on {cid} ({host}:{port})")
    print("now run: python scripts/watch_run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
