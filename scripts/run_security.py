#!/usr/bin/env python3
"""B10 security review: sensitivity, SPECIFICITY, false-positive rate, refusals.

Every task is a pair - the vulnerable file and its real fix - plus decoys that are
safe but look alarming. Scored deterministically off a required VERDICT line; no LLM
judging anywhere, so this costs no judge quota.

Why specificity is the headline: on vulnerable code every model scores ~100%
(post-Mythos analysis: eight of eight, including a 3.6B-active model, caught the
FreeBSD overflow). What separates them is whether they call the PATCHED file safe or
invent a vulnerability in it. On a real engagement the false-positive rate is what
decides whether the tool is usable.

    python scripts/run_security.py --endpoint-url http://127.0.0.1:8080 \
        --model gpt-oss-20b --reps 3 --out results_security
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

REFUSAL = re.compile(
    r"\b(i can(?:'|no)?t (?:help|assist|provide)|i'm not able to|i am not able to|"
    r"cannot assist with|against my|not appropriate|i must decline|as an ai)\b", re.I)
VERDICT = re.compile(r"^\s*VERDICT\s*:\s*(VULNERABLE|SAFE)\b", re.I | re.M)
CWE_RE = re.compile(r"CWE[-\s]?(\d{1,4})", re.I)


def chat(url, prompt, *, max_tokens, temperature, timeout=1800, extra=None):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    if extra:
        body.update(extra)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ask(url, prompt, *, max_tokens, temperature):
    """Same attempt ladder as the game battery: a reasoning model can burn its whole
    budget thinking and return nothing, which would score as a refusal/miss and be a
    harness artifact rather than a result."""
    rungs = [("base", max_tokens, None),
             ("bigger_budget", min(max_tokens * 2, 16000), None),
             ("no_thinking", max_tokens, {"chat_template_kwargs": {"enable_thinking": False}})]
    text, err, used = "", None, "base"
    for name, mt, extra in rungs:
        try:
            d = chat(url, prompt, max_tokens=mt, temperature=temperature, extra=extra)
            text = (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            err = None
        except Exception as e:                                   # noqa: BLE001
            text, err = "", f"{type(e).__name__}: {e}"
        used = name
        if text.strip():
            break
    return text, err, used


def score(text, *, expect_vulnerable, keywords, cwe, accepted=None):
    """Deterministic scoring off the verdict line.

    A refusal is recorded separately from a wrong answer - for authorised security
    work 'declined to look' and 'looked and was wrong' are different failures and
    must not be averaged together.
    """
    out = {"refused": bool(REFUSAL.search(text or "")) and not VERDICT.search(text or ""),
           "parsed": False, "verdict": None, "correct": None,
           "named_cwe": None, "cwe_correct": None, "located": None}
    m = VERDICT.search(text or "")
    if not m:
        return out
    out["parsed"] = True
    v = m.group(1).upper()
    out["verdict"] = v
    out["correct"] = (v == "VULNERABLE") if expect_vulnerable else (v == "SAFE")
    if expect_vulnerable:
        cm = CWE_RE.search(text or "")
        if cm:
            out["named_cwe"] = "CWE-" + cm.group(1)
            # A model that answers CWE-119 for a CWE-120 overflow, or CWE-208 for a
            # timing leak, has classified it correctly - exact-match scoring was
            # marking real answers wrong. Accept the defensible set.
            allowed = accepted or ([cwe] if cwe else [])
            ok = {c.upper() for c in allowed}
            out["cwe_correct"] = out["named_cwe"].upper() in ok
        low = (text or "").lower()
        out["located"] = any(k.lower() in low for k in (keywords or []))
    return out


def score_chain(text, defects):
    """Hard tier: grade on how much of the CHAIN was found, not on the verdict.

    A model that spots the NULL deref, says VULNERABLE and stops has missed the
    missing lower bound and the signed overflow - and on a real engagement that
    finding ships with the actual exploitable bug still in the file. Saying
    "vulnerable" for one reason out of three earns 1/3, not a pass.
    """
    low = (text or "").lower()
    found = []
    for d in defects:
        hit = any(k.lower() in low for k in d.get("keywords", []))
        found.append({"name": d["name"], "found": hit})
    n = sum(1 for f in found if f["found"])
    return {"defects": found, "found_n": n, "total": len(defects),
            "recall": round(n / len(defects), 3) if defects else None,
            "found_all": n == len(defects) and bool(defects)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results_security")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    spec = yaml.safe_load((ROOT / "suite" / "b10_security" / "pairs.yaml").read_text(encoding="utf-8"))
    hard_path = ROOT / "suite" / "b10_security" / "hard.yaml"
    hard = yaml.safe_load(hard_path.read_text(encoding="utf-8")) if hard_path.exists() else {}
    tmpl = spec["prompt_template"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "rows-security.jsonl"
    done = set()
    if shard.exists():
        for line in shard.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["model_id"], r["task_id"], r["run_n"]))
            except Exception:
                continue

    # (task_id, code, expect_vulnerable, cwe, keywords, filename)
    items = []
    for t in spec["tasks"]:
        items.append((f"b10.{t['id']}.vuln", t["vulnerable"], True, t.get("cwe"),
                      t.get("keywords"), t["filename"], t.get("accepted_cwe")))
        items.append((f"b10.{t['id']}.patched", t["patched"], False, t.get("cwe"),
                      t.get("keywords"), t["filename"], t.get("accepted_cwe")))
    for d in spec.get("decoys", []):
        items.append((f"b10.{d['id']}", d["code"], False, None, None, d["filename"], None))

    # hard tier: multi-defect chains, graded on chain recall
    hard_items = []
    for t in (hard.get("hard_tasks") or []):
        hard_items.append((f"b10hard.{t['id']}.vuln", t["vulnerable"], True, t["filename"], t["defects"]))
        hard_items.append((f"b10hard.{t['id']}.patched", t["patched"], False, t["filename"], t["defects"]))

    for task_id, code, expect_vuln, fname, defects in hard_items:
        for rep in range(1, args.reps + 1):
            if (args.model, task_id, rep) in done:
                continue
            prompt = tmpl.format(filename=fname, code=code)
            if expect_vuln:
                prompt += ("\n\nThis file may contain MORE THAN ONE distinct defect. "
                           "List every one you find, each with its own CWE and location.")
            t0 = time.time()
            text, err, rung = ask(args.endpoint_url, prompt,
                                  max_tokens=max(args.max_tokens, 3000),
                                  temperature=args.temperature)
            base = score(text, expect_vulnerable=expect_vuln, keywords=None, cwe=None)
            ch = score_chain(text, defects) if expect_vuln else None
            row = {
                "battery": 10, "model_id": args.model, "task_id": task_id, "run_n": rep,
                "condition": f"cond=B10hard;expect={'vuln' if expect_vuln else 'safe'};temp={args.temperature}",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "det_checks": {
                    "answered": {"pass": base["parsed"]},
                    "correct_verdict": {"pass": bool(base["correct"])},
                    "did_not_refuse": {"pass": not base["refused"]},
                    **({"found_whole_chain": {"pass": bool(ch["found_all"])}} if expect_vuln else {}),
                },
                "metrics": {"tier": "hard", "expect_vulnerable": expect_vuln,
                            "verdict": base["verdict"], "refused": base["refused"],
                            "attempt_rung": rung, "seconds": round(time.time() - t0, 1),
                            "chars": len(text or ""),
                            **({"chain_recall": ch["recall"], "found_n": ch["found_n"],
                                "chain_total": ch["total"],
                                "defects": ch["defects"]} if expect_vuln else {})},
                "response_meta": {"head": (text or "")[:900]},
                "error_detail": err,
            }
            with shard.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if expect_vuln:
                miss = [d["name"] for d in ch["defects"] if not d["found"]]
                print(f"  {task_id:30s} r{rep} chain {ch['found_n']}/{ch['total']}"
                      + (f"  missed: {','.join(miss)}" if miss else "  COMPLETE"))
            else:
                print(f"  {task_id:30s} r{rep} {str(base['verdict']):10s}"
                      + ("ok" if base["correct"] else "FALSE POSITIVE"))

    for task_id, code, expect_vuln, cwe, kw, fname, accepted in items:
        for rep in range(1, args.reps + 1):
            if (args.model, task_id, rep) in done:
                continue
            prompt = tmpl.format(filename=fname, code=code)
            t0 = time.time()
            text, err, rung = ask(args.endpoint_url, prompt,
                                  max_tokens=args.max_tokens, temperature=args.temperature)
            sc = score(text, expect_vulnerable=expect_vuln, keywords=kw, cwe=cwe,
                       accepted=accepted)
            row = {
                "battery": 10, "model_id": args.model, "task_id": task_id, "run_n": rep,
                "condition": f"cond=B10;expect={'vuln' if expect_vuln else 'safe'};temp={args.temperature}",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "det_checks": {
                    "answered": {"pass": sc["parsed"]},
                    "correct_verdict": {"pass": bool(sc["correct"])},
                    "did_not_refuse": {"pass": not sc["refused"]},
                    **({"cwe_correct": {"pass": bool(sc["cwe_correct"])},
                        "located": {"pass": bool(sc["located"])}} if expect_vuln else {}),
                },
                "metrics": {"expect_vulnerable": expect_vuln, "verdict": sc["verdict"],
                            "refused": sc["refused"], "named_cwe": sc["named_cwe"],
                            "attempt_rung": rung, "seconds": round(time.time() - t0, 1),
                            "chars": len(text or "")},
                "response_meta": {"head": (text or "")[:600]},
                "error_detail": err,
            }
            with shard.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            flag = "REFUSED" if sc["refused"] else (
                "ok" if sc["correct"] else ("FALSE POSITIVE" if not expect_vuln else "missed"))
            print(f"  {task_id:28s} r{rep} {str(sc['verdict']):10s} {flag}")
    print("done ->", shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
