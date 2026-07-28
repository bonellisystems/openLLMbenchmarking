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


def score(text, *, expect_vulnerable, keywords, cwe):
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
            out["cwe_correct"] = (out["named_cwe"].upper() == (cwe or "").upper())
        low = (text or "").lower()
        out["located"] = any(k.lower() in low for k in (keywords or []))
    return out


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
                      t.get("keywords"), t["filename"]))
        items.append((f"b10.{t['id']}.patched", t["patched"], False, t.get("cwe"),
                      t.get("keywords"), t["filename"]))
    for d in spec.get("decoys", []):
        items.append((f"b10.{d['id']}", d["code"], False, None, None, d["filename"]))

    for task_id, code, expect_vuln, cwe, kw, fname in items:
        for rep in range(1, args.reps + 1):
            if (args.model, task_id, rep) in done:
                continue
            prompt = tmpl.format(filename=fname, code=code)
            t0 = time.time()
            text, err, rung = ask(args.endpoint_url, prompt,
                                  max_tokens=args.max_tokens, temperature=args.temperature)
            sc = score(text, expect_vulnerable=expect_vuln, keywords=kw, cwe=cwe)
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
