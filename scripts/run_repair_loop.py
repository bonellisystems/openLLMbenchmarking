#!/usr/bin/env python3
"""B9 self-correction: can the model FIX a broken game, and how many passes does it need?

Implements the loop protocol from TESTPLAN 5.6 for both tracks:

  planted-bug   a known-good game we authored, with a bug injected at a known
                difficulty (crash / logic / subtle). We know exactly what is wrong,
                so 'found it unaided' is meaningful.
  self-debug    the model's OWN broken one-shot build fed back to it. No planted
                answer key - the gate decides green.

Protocol (TESTPLAN 5.6), all deterministic - no LLM judging anywhere:
  * cap N=6 iterations, structured 2/2/2 hint escalation
      iters 1-2  H0  "something's wrong, find it"   (no symptom given)
      iters 3-4  H1  symptom described
      iters 5-6  H2  the actual console error / failing behaviour
  * thresholds: fixed-unaided <=2, fixed-with-symptom <=4, fixed-with-error <=6
  * early stops: gate goes green, OR two consecutive identical / no-op patches
    -> DNF-loop (reported per model)
  * the model's STATED DIAGNOSIS is logged every iteration, so detection rate and
    fix rate stay separable - a model can describe the bug correctly and still fail
    to fix it, and that is a different weakness
  * regressions logged per iteration (a check that was passing and now is not)

Fix iterations are edit tasks, so serve with n-gram spec-decode on: the plan notes
this is the cheapest decode in the suite (3-9x).

    python scripts/run_repair_loop.py --endpoint-url http://127.0.0.1:8080 \
        --model gpt-oss-20b --track planted --out results_repair
    python scripts/run_repair_loop.py --endpoint-url ... --track selfdebug \
        --from-games results_games
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.harness.game_oracle import det_checks_for, run_game_checks  # noqa: E402
from llmtest.harness.game_probes import probe_for  # noqa: E402

MAX_ITERS = 6
HINT_BANDS = [(1, 2, "H0"), (3, 4, "H1"), (5, 6, "H2")]

H0 = ("This game is broken: something about it does not work correctly when you run it. "
      "Find the problem and fix it.")
H1 = "This game is broken. Symptom: {symptom}\nFind the underlying cause and fix it."
H2 = ("This game is broken. Symptom: {symptom}\nThe browser reports: {error}\n"
      "Find the underlying cause and fix it.")
CONTRACT = ("Reply with the COMPLETE corrected HTML file and nothing else - no explanation "
            "outside the file, no markdown fence.\n"
            "Before the file, on a single line starting with 'DIAGNOSIS:', state in one "
            "sentence what you believe the bug is.")


def chat(url, prompt, *, max_tokens, temperature, timeout=1800):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_html(text):
    if not text:
        return ""
    m = re.search(r"```(?:html|HTML)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i = text.lower().find("<!doctype html")
    if i < 0:
        i = text.lower().find("<html")
    if i < 0:
        return ""
    j = text.lower().rfind("</html>")
    return text[i:j + 7] if j > i else text[i:]


def extract_diagnosis(text):
    m = re.search(r"DIAGNOSIS:\s*(.+)", text or "")
    return m.group(1).strip()[:400] if m else None


def band_for(i):
    for lo, hi, name in HINT_BANDS:
        if lo <= i <= hi:
            return name
    return "H2"


def symptom_from(checks, detail):
    """Describe the failure the way a user would, without naming the fix."""
    if not checks.get("loads", {}).get("pass"):
        return "the page throws an error as soon as it loads and nothing appears"
    if not checks.get("surface", {}).get("pass"):
        return "nothing is rendered - there is no visible game area"
    if not checks.get("paints", {}).get("pass"):
        return "the game area stays blank"
    if not checks.get("loop", {}).get("pass"):
        return ("the game draws once and then never moves - it does not advance on its own "
                "and does not react when keys are pressed")
    if not checks.get("input_safe", {}).get("pass"):
        return "pressing the controls throws an error and the game stops responding"
    if not checks.get("keys_wired", {}).get("pass"):
        return "the keyboard controls do nothing"
    for k, v in checks.items():
        if k.startswith("probe.") and not v.get("pass"):
            return {
                "probe.dies_at_wall": "the game ends at the wrong moment near the edge of the board",
                "probe.grows_on_eat": "eating does not update the score and the body together",
                "probe.render_in_sync": "what is drawn does not match where things actually are",
                "probe.advances": "the game does not advance",
                "probe.turns": "steering does not change direction",
            }.get(k, "one of the game rules behaves incorrectly")
    return "the game does not behave correctly"


def gate(path: Path, *, chrome, keys, probe_game: str | None):
    """Green = the generic gate passes AND, for a fixture build, every game-specific
    invariant holds. A subtle bug (off-by-one wall, score desync, renderer a tick
    behind) is invisible to the generic gate by construction, so without the probe a
    model could "fix" it by changing nothing."""
    res = run_game_checks(path, chrome_path=chrome, keys=keys)
    checks = det_checks_for(res)
    probe = probe_for(probe_game) if probe_game else None
    pr = None
    if probe is not None:
        pr = probe(path, chrome_path=chrome)
        for k, v in pr.checks.items():
            checks[f"probe.{k}"] = {"pass": bool(v)}
        checks["runs_clean"] = {"pass": res.runs_clean and pr.all_pass}
    return res, checks, (pr.to_dict() if pr else None)


def run_one(endpoint, model, name, html0, *, out: Path, chrome, keys, max_tokens,
            temperature, track, bug_kind=None, probe_game=None):
    """One repair loop over a single broken build. Returns the result row."""
    work = out / "iters"
    work.mkdir(parents=True, exist_ok=True)
    cur = html0
    prev_hashes = []
    iters = []
    green_at = None
    dnf = False

    base_p = work / f"{name}__iter0.html"
    base_p.write_text(cur, encoding="utf-8")
    base, base_checks, base_probe = gate(base_p, chrome=chrome, keys=keys, probe_game=probe_game)
    if base_checks.get("runs_clean", {}).get("pass"):
        return {"skipped": "starting build already passes the gate", "name": name}

    for i in range(1, MAX_ITERS + 1):
        band = band_for(i)
        sym = symptom_from(base_checks if not iters else iters[-1]["checks"], base.detail)
        err = (base.errors or [""])[0][:200] if not iters else (iters[-1]["error"] or "")
        if band == "H0":
            instr = H0
        elif band == "H1":
            instr = H1.format(symptom=sym)
        else:
            instr = H2.format(symptom=sym, error=err or "no console error; it simply does not run")
        prompt = f"{instr}\n\n{CONTRACT}\n\n--- FILE ---\n{cur}"

        t0 = time.time()
        try:
            resp = chat(endpoint, prompt, max_tokens=max_tokens, temperature=temperature)
            text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            gen_err = None
        except Exception as e:                                   # noqa: BLE001
            text, gen_err = "", f"{type(e).__name__}: {e}"
        secs = time.time() - t0

        new_html = extract_html(text)
        diagnosis = extract_diagnosis(text)
        h = hashlib.sha1((new_html or "").encode("utf-8", "replace")).hexdigest()
        noop = (not new_html) or (h in prev_hashes) or (new_html.strip() == cur.strip())
        prev_hashes.append(h)

        if new_html:
            p = work / f"{name}__iter{i}.html"
            p.write_text(new_html, encoding="utf-8")
            res, checks, probe_out = gate(p, chrome=chrome, keys=keys, probe_game=probe_game)
            if res.screenshot_b64:
                (work / f"{name}__iter{i}.png").write_bytes(base64.b64decode(res.screenshot_b64))
            clean = checks.get("runs_clean", {}).get("pass", False)
            first_err = (res.errors or [None])[0]
            # regression: a check that passed on the previous build now fails
            prev = iters[-1]["checks"] if iters else base_checks
            regressed = [k for k in checks
                         if prev.get(k, {}).get("pass") and not checks[k]["pass"]]
        else:
            checks, clean, first_err, regressed = {}, False, gen_err, []

        iters.append({"i": i, "hint": band, "diagnosis": diagnosis, "noop": noop,
                      "clean": clean, "checks": checks, "error": first_err,
                      "regressed": regressed, "seconds": round(secs, 1),
                      "html_bytes": len(new_html or "")})
        print(f"    iter{i} [{band}] {'GREEN' if clean else 'still broken'}"
              f"{' NO-OP' if noop else ''}{' REGRESSED:' + ','.join(regressed) if regressed else ''}"
              f"  {secs:.0f}s  dx={'yes' if diagnosis else 'no'}")

        if clean:
            green_at = i
            break
        if new_html:
            cur = new_html
        # two consecutive no-op / identical patches -> give up (DNF-loop)
        if len(iters) >= 2 and iters[-1]["noop"] and iters[-2]["noop"]:
            dnf = True
            break

    band_reached = iters[-1]["hint"] if iters else "H0"
    return {
        "battery": 9, "model_id": model, "task_id": f"b9repair.{name}",
        "run_n": 1, "condition": f"cond=B9repair;track={track};bug={bug_kind or 'self'}",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "det_checks": {
            "fixed": {"pass": green_at is not None},
            "fixed_unaided": {"pass": green_at is not None and green_at <= 2},
            "fixed_with_symptom": {"pass": green_at is not None and green_at <= 4},
            "fixed_with_error": {"pass": green_at is not None and green_at <= 6},
            "no_dnf_loop": {"pass": not dnf},
        },
        "metrics": {
            "track": track, "bug_kind": bug_kind, "green_at": green_at,
            "steps_to_green": green_at, "iterations": len(iters),
            "dnf_loop": dnf, "hint_band_reached": band_reached,
            "diagnosis_rate": round(sum(1 for x in iters if x["diagnosis"]) / max(1, len(iters)), 2),
            "regressions": sum(len(x["regressed"]) for x in iters),
            "total_seconds": round(sum(x["seconds"] for x in iters), 1),
        },
        "response_meta": {"iterations": iters},
        "artifacts": {"start": f"iters/{name}__iter0.html"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", choices=["planted", "selfdebug"], required=True)
    ap.add_argument("--out", default="results_repair")
    ap.add_argument("--from-games", default="results_games",
                    help="selfdebug: directory of one-shot builds to take failures from")
    ap.add_argument("--bugs-dir", default="suite/b9_games/planted",
                    help="planted: directory of pre-broken fixtures")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "rows-repair.jsonl"
    done = set()
    if shard.exists():
        for line in shard.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["model_id"], r["task_id"]))
            except Exception:
                continue

    jobs = []
    if args.track == "planted":
        bd = ROOT / args.bugs_dir
        for p in sorted(bd.glob("*.html")):
            kind = p.stem.split("__")[-1] if "__" in p.stem else "unknown"
            game = "snake" if "snake" in p.stem else None
            jobs.append((p.stem, p.read_text(encoding="utf-8"), kind, game))
    else:
        # the model's OWN failed one-shot builds
        gshard = Path(args.from_games) / "rows-games.jsonl"
        if not gshard.exists():
            print(f"no games shard at {gshard}")
            return 2
        for line in gshard.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model_id") != args.model:
                continue
            if r.get("metrics", {}).get("runs_clean"):
                continue
            hp = Path(args.from_games) / r["artifacts"]["html"]
            if hp.exists() and hp.stat().st_size > 200:
                # model-authored builds stay BLACK-BOX (amendment 18): no probe
                jobs.append((hp.stem, hp.read_text(encoding="utf-8"), "self", None))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{args.track}: {len(jobs)} broken build(s) for {args.model}")

    chrome = args.chrome if Path(args.chrome).exists() else None
    for name, html, kind, probe_game in jobs:
        if (args.model, f"b9repair.{name}") in done:
            print(f"  skip {name} (done)")
            continue
        print(f"  {name}  [{kind}]")
        row = run_one(args.endpoint_url, args.model, name, html, out=out, chrome=chrome,
                      keys=None, max_tokens=args.max_tokens, temperature=args.temperature,
                      track=args.track, bug_kind=kind, probe_game=probe_game)
        if row.get("skipped"):
            print(f"    skipped: {row['skipped']}")
            continue
        with shard.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        m = row["metrics"]
        print(f"    -> {'FIXED at iter ' + str(m['green_at']) if m['green_at'] else 'NOT FIXED'}"
              f" ({m['iterations']} iters, band {m['hint_band_reached']}"
              f"{', DNF-loop' if m['dnf_loop'] else ''}, {m['regressions']} regressions)")
    print("done ->", shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
