#!/usr/bin/env python3
"""GO/NO-GO preflight for the VM hardware-consistency campaign. Spends nothing.

Michael's instruction, verbatim: "Make sure EVERYTHING is built in place before we spend
a dime and we're ready to hit the ground running from the moment the meter starts
running." This script IS that check. It must pass completely before `rent_and_run --vm
--go` is ever typed.

Checks:
  1. plan_vm artifacts exist, bash-parse clean, zero CR bytes.
  2. The plan covers EXACTLY the withdrawn GPU cells (manifest <-> superseded.yaml
     cross-check) - nothing silently dropped, nothing extra.
  3. config: suite-v2.2.0, b8.sandbox.enabled=true, deploy/blackwell files present.
  4. Live market: VM-capable RTX PRO 6000 offers, credit vs estimate.
  5. SSH key registered on the vast account (VM keys cannot be added after boot).
  6. END-TO-END SIMULATION: synthesize schema-valid v2.2.0 rows shaped exactly like the
     box's output tree, run the REAL merge, validate, rebuild the REAL dashboard, and
     assert the withdrawn cell flips back on. Every silent-failure bug this project has
     hit (append dedupe, merge KEY, rowselect wiring) lives on this path - so the path
     is executed, not reasoned about. Reverts itself; the tree must end clean.

    python scripts/preflight_vm.py            # all checks incl. simulation
    python scripts/preflight_vm.py --no-market  # offline checks only
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import yaml  # noqa: E402

PASS, FAIL = "  [OK]  ", "  [FAIL]"
failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL} {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------- 1. artifacts
def check_artifacts():
    print("\n== 1. plan artifacts ==")
    for f in ("vm_setup.sh", "run_all.sh", "manifest.json"):
        p = ROOT / "plan_vm" / f
        check(p.exists(), f"plan_vm/{f} exists")
    for f in ("vm_setup.sh", "run_all.sh"):
        p = ROOT / "plan_vm" / f
        if not p.exists():
            continue
        check(b"\r" not in p.read_bytes(), f"plan_vm/{f} has LF endings")
        # Git Bash mangles absolute Windows paths ("D:\..." -> "D:BUILT-TOOLS...");
        # a repo-relative POSIX path parses identically on both sides.
        r = subprocess.run(["bash", "-n", f"plan_vm/{f}"], capture_output=True,
                           text=True, cwd=str(ROOT))
        check(r.returncode == 0, f"plan_vm/{f} bash-parses", r.stderr.strip()[:80])
    for f in ("bootstrap.sh", "serve.sh", "smoke.sh", "Dockerfile.sandbox",
              "Dockerfile.prism", "pack_repo.sh"):
        check((ROOT / "deploy" / "blackwell" / f).exists(), f"deploy/blackwell/{f} exists")
    check((ROOT / "deploy" / "shutdown_guard.sh").exists(), "deploy/shutdown_guard.sh exists")


# ---------------------------------------------------------------- 2. coverage
def check_coverage():
    print("\n== 2. plan covers exactly the withdrawn GPU cells ==")
    man = json.loads((ROOT / "plan_vm" / "manifest.json").read_text(encoding="utf-8"))
    planned = {(m["id"], b) for m in man["models"] for b in m["batteries"]}

    sup = yaml.safe_load((ROOT / "config" / "superseded.yaml").read_text(encoding="utf-8"))
    want = set()
    for c in sup.get("cells", []):
        # 0.1 decision: the mxfp4 quant-arm's withdrawn B8 is REPLACED by re-running the
        # roster-canonical gemma quant; the arm itself is a later labelled ablation. So
        # the expectation folds onto the roster id, exactly as build_data's ARM_TO_ROSTER
        # folds the scores.
        mid = "gemma-4-26b-a4b" if c["model"] == "gemma-4-26b-a4b-mxfp4" else c["model"]
        want.add((mid, f"B{c['battery']}"))
    for c in sup.get("custom_rows", []):
        want.add((c["model"], f"B{c['battery']}"))
    # bonsai B10/B11 were never-run gaps (not superseded) - closable now via prism
    want |= {("bonsai-ternary-27b", "B10"), ("bonsai-ternary-27b", "B11")}

    missing = sorted(want - planned)
    extra = sorted(planned - want)
    check(not missing, "every withdrawn/target cell is in the plan",
          f"missing: {missing[:4]}" if missing else f"{len(want)} cells")
    check(not extra, "plan contains nothing outside the target set",
          f"extra: {extra[:4]}" if extra else "")
    check(("qwen3-235b", "B8") not in planned, "qwen3-235b stays excluded")
    ok_files = all(m["files"] for m in man["models"])
    check(ok_files, "every model's GGUF file list resolved from the HF API")


# ---------------------------------------------------------------- 3. config
def check_config():
    print("\n== 3. configuration ==")
    d = yaml.safe_load((ROOT / "config" / "suite.yaml").read_text(encoding="utf-8"))
    check(d["suite_version"] == "suite-v2.2.0", "suite_version is suite-v2.2.0",
          d["suite_version"])
    check(d["b8"]["sandbox"]["enabled"] is True,
          "b8.sandbox.enabled=true (the whole point of the VM)")
    run_all = (ROOT / "plan_vm" / "run_all.sh").read_text(encoding="utf-8")
    check("hardware-sku rtx-pro-6000-vm" in run_all, "rows will carry hardware_sku")
    check("suite-version suite-v2.2.0" in run_all, "custom rows will carry suite_version")
    check("prism-llama:1" in run_all, "bonsai routed to the prism image")
    check("B8-probe" in run_all, "infra-error models get a 1-task probe gate")
    check("sandbox.enabled=false" not in run_all,
          "nothing flips the sandbox off (the vast-container mistake)")


# ---------------------------------------------------------------- 4/5. market
def check_market(est_hours: float):
    print("\n== 4. live market + credit ==")
    from vastai import VastAI
    key = (Path.home() / ".config" / "vastai" / "vast_api_key").read_text().strip()
    v = VastAI(api_key=key)
    sys.path.insert(0, str(ROOT / "scripts"))
    from rent_and_run import find_offers
    offers, why = find_offers(v, vm=True)
    check(bool(offers), "VM-capable RTX PRO 6000 offer available",
          f"{len(offers)} offer(s); rejects: " + ", ".join(f"{k}={n}" for k, n in why.items() if n))
    credit = v.show_user().get("credit", 0)
    if offers:
        est = offers[0]["dph_total"] * est_hours
        check(est < credit, f"estimate ~{est_hours:.0f}h x ${offers[0]['dph_total']:.3f}/h "
                            f"= ${est:.2f} within credit ${credit:.2f}")
    print("\n== 5. account ==")
    ks = v.show_ssh_keys()
    rows = ks if isinstance(ks, list) else ks.get("ssh_keys", [])
    mine = Path("C:/Users/Michael/.ssh/vast_laguna.pub").read_text().split()[1]
    check(any((k.get("public_key") or "").split()[1:2] == [mine] for k in rows),
          "SSH key registered on the vast account (VMs key from the ACCOUNT at boot)")


# ---------------------------------------------------------------- 6. simulation
def simulate():
    """Run the REAL pipeline on synthetic v2.2.0 rows and prove a withdrawn cell
    comes back. Reverts every change; asserts the tree ends clean."""
    print("\n== 6. end-to-end result-loop simulation (merge -> validate -> dashboard) ==")
    from llmtest import schema

    sim = ROOT / "_sim_pull" / "out"
    if (ROOT / "_sim_pull").exists():
        shutil.rmtree(ROOT / "_sim_pull")
    (sim / "b8_gpt-oss-20b").mkdir(parents=True)
    (sim / "games").mkdir(parents=True)

    # --- synth a v2.2.0 B8 row from a real one (same shape, new version+id) ---
    tmpl = None
    for line in (ROOT / "results" / "rows-suite-v2.1.0.jsonl").open(encoding="utf-8"):
        if '"battery": 8' in line and '"gpt-oss-20b"' in line:
            r = json.loads(line)
            if r.get("battery") == 8 and r["model_id"] == "gpt-oss-20b":
                tmpl = r
                break
    assert tmpl, "no B8 template row found"
    row = copy.deepcopy(tmpl)
    row["suite_version"] = "suite-v2.2.0"
    row["session_id"] = "sim-preflight"
    row["det_checks"] = {"oracle": {"pass": True, "detail": "PASS", "stage": "behavior"}}
    row["metrics"]["completion"] = True
    row["metrics"]["terminal_status"] = "completed"
    row["row_id"] = schema.compute_row_id(
        suite_version="suite-v2.2.0", model_id=row["model_id"],
        quant_sha256=row["quant_sha256"], battery=8, task_id=row["task_id"],
        fixture_sha=row["fixture_sha"], condition=row["condition"],
        run_n=row["run_n"])
    (sim / "b8_gpt-oss-20b" / "rows-suite-v2.2.0.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    # --- synth a v2.2.0 B9 row (custom shard; KEY now includes suite_version) ---
    g = None
    for line in (ROOT / "results_games" / "rows-games.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["model_id"] == "gpt-oss-20b":
            g = r
            break
    assert g, "no B9 template row"
    g = copy.deepcopy(g)
    g["suite_version"] = "suite-v2.2.0"
    g["hardware_sku"] = "rtx-pro-6000-vm"
    (sim / "games" / "rows-games.jsonl").write_text(
        json.dumps(g, sort_keys=True) + "\n", encoding="utf-8")

    reverts = [ROOT / "results" / "rows-suite-v2.2.0.jsonl",
               ROOT / "results_b8_gpt-oss-20b" / "rows-suite-v2.2.0.jsonl"]
    try:
        # --- the REAL merge ---
        r = sh([sys.executable, str(ROOT / "scripts" / "merge_gapclose.py"),
                "--src", "_sim_pull/out"])
        merged = "suite B1-B7 : 1 read, 1 new" in r.stdout
        check(merged, "merge accepts the v2.2.0 suite row (no silent dedupe)",
              [l for l in r.stdout.splitlines() if "suite" in l][:1])
        check("B9 games    : 1 read, 1 new" in r.stdout,
              "merge accepts the v2.2.0 B9 row (KEY includes suite_version)")

        # --- validate ---
        r = sh([sys.executable, "-m", "llmtest", "validate"])
        check("0 errors" in r.stdout, "store validates with the sim rows in place")

        # --- the REAL dashboard build, into a scratch copy ---
        dash = ROOT / "dashboard"
        for f in ("data.json", "index.html"):
            shutil.copy2(dash / f, dash / (f + ".preflight-bak"))
        r = sh([sys.executable, str(dash / "build_data.py")])
        d = json.loads((dash / "data.json").read_text(encoding="utf-8"))
        b8 = d["matrix"]["gpt-oss-20b"]["B8"]
        check(b8.get("tested") is True and b8.get("n") == 1,
              "withdrawn B8 cell flips back on from the v2.2.0 row alone",
              f"tested={b8.get('tested')} n={b8.get('n')}")
        b9 = d["matrix"]["gpt-oss-20b"].get("B9", {})
        check(not b9.get("tested") and b9.get("partial"),
              "B9 with 1 of 24 rows stays honestly partial, not green",
              f"display={b9.get('display')}")
    finally:
        # --- revert everything the simulation touched ---
        for p in reverts:
            p.unlink(missing_ok=True)
        sh(["git", "-C", str(ROOT), "checkout", "--",
            "results_games/rows-games.jsonl"])
        dash = ROOT / "dashboard"
        for f in ("data.json", "index.html"):
            bak = dash / (f + ".preflight-bak")
            if bak.exists():
                shutil.move(str(bak), str(dash / f))
        shutil.rmtree(ROOT / "_sim_pull", ignore_errors=True)
    r = sh(["git", "-C", str(ROOT), "status", "--porcelain"])
    dirty = [l for l in r.stdout.splitlines() if l.strip()]
    check(not dirty, "working tree clean after simulation revert", str(dirty[:3]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-market", action="store_true",
                    help="skip the live vast.ai checks (offline)")
    ap.add_argument("--est-hours", type=float, default=33.0)
    args = ap.parse_args()

    check_artifacts()
    check_coverage()
    check_config()
    if not args.no_market:
        check_market(args.est_hours)
    simulate()

    print("\n" + "=" * 60)
    if failures:
        print(f"NO-GO: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GO: every preflight check passed. Launch with:")
    print("  python scripts/rent_and_run.py --vm --plan-dir plan_vm "
          "--label hw-consistency --est-hours 33 --check   # then --go")
    print("  python scripts/watch_run.py --plan-dir plan_vm --remote-root /opt/b8 "
          "--dest results_vm2200 --once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
