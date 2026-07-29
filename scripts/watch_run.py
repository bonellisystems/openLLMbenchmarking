#!/usr/bin/env python3
"""Follow the gap-closing run: pull results per model, alarm on a stall, and DESTROY
the box the moment it finishes.

The pull happens on every poll and again immediately before teardown. A previous box
was left idle after its run completed, exhausted the balance overnight, could not be
restarted, and took 96 unpulled rows with it - so nothing here waits until the end.

    python scripts/watch_run.py                 # follow until done, then destroy
    python scripts/watch_run.py --no-destroy    # follow only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYFILE = Path.home() / ".config" / "vastai" / "vast_api_key"
PRIVKEY = "C:/Users/Michael/.ssh/vast_laguna"
SSH_COMMON = ["-i", PRIVKEY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
              "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
              "-o", "LogLevel=ERROR"]
DEST = ROOT / "results_gapclose"


def api():
    from vastai import VastAI
    return VastAI(api_key=KEYFILE.read_text().strip())


def wait_for_ssh(host, port, tries=60, delay=20):
    """onstart installs sshd via apt, which can take 10-20 minutes on a slow host.
    Silence here means NOT READY - it must never be mistaken for a failed check."""
    for _ in range(tries):
        r = sshx(host, port, "echo OK", timeout=30)
        if r.stdout.strip() == "OK":
            return True
        time.sleep(delay)
    return False


def endpoint():
    host, port = (ROOT / "plan" / "ENDPOINT").read_text(encoding="utf-8").split()
    return host, int(port)


def sshx(host, port, cmd, timeout=600):
    try:
        return subprocess.run(["ssh", *SSH_COMMON, "-p", str(port), host, cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 1, "", "timeout")


def pull(host, port):
    """Tar the whole output tree on the box and bring it back. scp needs -P."""
    DEST.mkdir(parents=True, exist_ok=True)
    r = sshx(host, port, "cd /root && tar czf out.tgz out run.log models_done failures 2>/dev/null; echo ok")
    if "ok" not in r.stdout:
        return False
    got = subprocess.run(["scp", *SSH_COMMON, "-P", str(port), f"{host}:/root/out.tgz",
                          str(DEST / "out.tgz")], capture_output=True, text=True)
    if got.returncode != 0:
        return False
    subprocess.run(["tar", "xzf", "out.tgz"], cwd=str(DEST), capture_output=True)
    (DEST / "out.tgz").unlink(missing_ok=True)
    return True


def rows_local():
    n = 0
    for p in DEST.rglob("*.jsonl"):
        try:
            n += sum(1 for _ in p.open(encoding="utf-8"))
        except OSError:
            pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--stall-polls", type=int, default=8)
    ap.add_argument("--floor", type=float, default=1.50)
    ap.add_argument("--no-destroy", action="store_true")
    args = ap.parse_args()

    cid = int((ROOT / "plan" / "INSTANCE").read_text(encoding="utf-8").strip())
    host, port = endpoint()
    v = api()
    log = DEST / "watch.log"
    DEST.mkdir(parents=True, exist_ok=True)

    last, stalls, miss, reason = -1, 0, 0, "TIMEOUT"
    for i in range(1, 400):
        alive = sshx(host, port, "echo OK").stdout.strip() == "OK"
        if not alive:
            miss += 1
            if miss >= 3:
                reason = "UNREACHABLE"
                break
            time.sleep(60)
            continue
        miss = 0

        stage = sshx(host, port,
                     "test -f /root/run_all_done && echo done || "
                     "(test -f /root/dl_done && echo running || "
                     "(test -f /root/setup_done && echo downloading || echo setup))").stdout.strip()
        done_models = sshx(host, port, "wc -l < /root/models_done 2>/dev/null || echo 0").stdout.strip()
        fails = sshx(host, port, "cat /root/failures 2>/dev/null | tr '\\n' ';'").stdout.strip()
        pull(host, port)
        rows = rows_local()
        try:
            bal = round(v.show_user().get("credit", 0), 2)
        except Exception:
            bal = None

        line = (f"{time.strftime('%H:%M:%S')} i={i} stage={stage} models_done={done_models} "
                f"rows={rows} bal={bal} stalls={stalls}" + (f" FAILURES[{fails}]" if fails else ""))
        print(line)
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        if stage == "done":
            reason = "DONE"
            break
        if rows == last and stage == "running":
            stalls += 1
            if stalls >= args.stall_polls:
                reason = f"STALLED at {rows} rows"
                break
        else:
            stalls = 0
        last = rows
        if bal is not None and bal < args.floor:
            reason = f"LOWCREDIT {bal}"
            break
        time.sleep(args.interval)

    print("exit reason:", reason)
    pull(host, port)
    print("final local rows:", rows_local())
    (DEST / "WATCH_DONE").write_text(reason, encoding="utf-8")
    if args.no_destroy:
        print(f"--no-destroy: instance {cid} left running")
        return 0
    try:
        print("destroying", cid, v.destroy_instance(id=cid))
    except Exception as e:
        print("destroy failed:", type(e).__name__, e, "- DESTROY MANUALLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
