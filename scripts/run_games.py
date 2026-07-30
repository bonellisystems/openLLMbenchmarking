#!/usr/bin/env python3
"""B9 game battery runner: ask a served model for each game, save the HTML, then
score it by DRIVING it in a real headless browser. No LLM judging anywhere.

    # against an already-running llama-server
    python scripts/run_games.py --endpoint-url http://127.0.0.1:8080 \
        --model gpt-oss-20b --reps 3 --out results_games

    python scripts/run_games.py --endpoint-url ... --model X --task snake --reps 1

Writes one row per (model, game, rep) into <out>/rows-games.jsonl plus the raw
.html and a .png screenshot per build, so every result stays inspectable and
playable in the explorer.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmtest.harness.game_oracle import det_checks_for, run_game_checks  # noqa: E402

import yaml  # noqa: E402


class Unreachable(Exception):
    """The endpoint did not answer in time, or at all.

    Kept distinct from "the model answered but emitted no HTML" because the two need
    OPPOSITE handling: the attempt ladder should retry the second and must NOT retry the
    first. A hung server retried three times costs three timeouts instead of one.
    """


def chat(url: str, prompt: str, *, max_tokens: int, temperature: float, timeout=600,
         extra: dict | None = None) -> dict:
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    if extra:
        body.update(extra)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (TimeoutError, socket.timeout) as e:
        raise Unreachable(f"no response in {timeout}s") from e
    except urllib.error.URLError as e:
        # URLError wraps a socket timeout as well as a refused connection.
        raise Unreachable(f"{type(e.reason).__name__}: {e.reason}") from e


def extract_html(text: str) -> str:
    """Models wrap output in fences or add commentary despite being told not to.
    Recovering the file is not scoring leniency - it separates 'ignored the output
    contract' (recorded) from 'could not build the game' (what we measure)."""
    if not text:
        return ""
    m = re.search(r"```(?:html|HTML)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i = text.lower().find("<!doctype html")
    if i < 0:
        i = text.lower().find("<html")
    if i < 0:
        return text if "<canvas" in text.lower() or "<script" in text.lower() else ""
    j = text.lower().rfind("</html>")
    return text[i:j + 7] if j > i else text[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results_games")
    ap.add_argument("--task", default=None, help="single game id")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--req-timeout", type=int, default=600,
                    help="seconds to wait for one generation. Was 1800 and unbounded by "
                         "any per-model budget: 24 games x 3 ladder rungs meant a stalled "
                         "server could hold a rented box for many hours. 600s is generous "
                         "for a single HTML file even on a slow dense model.")
    ap.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    ap.add_argument("--force", action="store_true", help="redo builds already on disk")
    args = ap.parse_args()

    spec = yaml.safe_load((ROOT / "suite" / "b9_games" / "games.yaml").read_text(encoding="utf-8"))
    tasks = [t for t in spec["tasks"] if (not args.task or t["id"] == args.task)]
    if not tasks:
        print(f"no such game: {args.task}")
        return 2

    out = Path(args.out)
    (out / "builds").mkdir(parents=True, exist_ok=True)
    shard = out / "rows-games.jsonl"
    done = set()
    if shard.exists() and not args.force:
        for line in shard.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["model_id"], r["task_id"], r["run_n"]))
            except Exception:
                continue

    chrome = args.chrome if Path(args.chrome).exists() else None
    for t in tasks:
        for rep in range(1, args.reps + 1):
            key = (args.model, f"b9.{t['id']}", rep)
            if key in done:
                print(f"  skip {t['id']} r{rep} (already built)")
                continue
            prompt = t["prompt"].strip() + "\n\n" + spec["output_contract"].strip()
            t0 = time.time()
            # Attempt ladder. A reasoning model can spend its entire budget on hidden
            # thinking and return an EMPTY answer - measured, not assumed: gemma
            # returned 0 visible characters at finish_reason=length, then a complete
            # file in 1,628 tokens with thinking off. Scoring that first attempt as
            # "cannot build a game" would be a harness artifact, so a run that emits
            # no file is retried with a larger budget and then with thinking
            # disabled. The rung that produced the result is recorded on the row.
            attempts = [
                ("base", args.max_tokens, None),
                ("bigger_budget", min(args.max_tokens * 2, 28000), None),
                ("no_thinking", args.max_tokens,
                 {"chat_template_kwargs": {"enable_thinking": False}}),
            ]
            text, usage, err, rung = "", {}, None, "base"
            for rung_name, mt, extra in attempts:
                try:
                    resp = chat(args.endpoint_url, prompt, max_tokens=mt,
                                temperature=args.temperature, extra=extra,
                                timeout=args.req_timeout)
                    text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    usage = resp.get("usage") or {}
                    err = None
                except Unreachable as e:
                    # DO NOT CLIMB THE LADDER ON A HANG. The ladder answers "the model
                    # spent its budget thinking and returned nothing"; it cannot fix a
                    # server that stopped responding, and retrying multiplies the wait.
                    # A 1800s timeout retried three times is 90 minutes for ONE game -
                    # measured: it silently burned 85 minutes of a rented box on
                    # llama-4-scout before the watcher noticed nothing was moving.
                    text, usage, err = "", {}, f"Unreachable: {e}"
                    rung = rung_name
                    break
                except Exception as e:                   # noqa: BLE001
                    text, usage, err = "", {}, f"{type(e).__name__}: {e}"
                rung = rung_name
                if extract_html(text):
                    break
            gen_s = time.time() - t0

            html = extract_html(text)
            stem = f"{args.model}__{t['id']}__r{rep}".replace("/", "-")
            hp = out / "builds" / f"{stem}.html"
            if html:
                hp.write_text(html, encoding="utf-8")

            if html:
                res = run_game_checks(hp, chrome_path=chrome, keys=t.get("keys"))
                checks = det_checks_for(res)
                if res.screenshot_b64:
                    (out / "builds" / f"{stem}.png").write_bytes(
                        base64.b64decode(res.screenshot_b64))
                detail = res.detail
                score, clean = res.score, res.runs_clean
            else:
                checks = {k: {"pass": False} for k in
                          ("loads", "surface", "paints", "loop", "keys_wired", "input_safe")}
                checks["runs_clean"] = {"pass": False,
                                        "detail": err or "model produced no HTML file"}
                detail, score, clean = {"no_html": True}, 0, False

            row = {
                "battery": 9, "model_id": args.model, "task_id": f"b9.{t['id']}",
                "run_n": rep, "condition": f"cond=B9;game={t['id']};temp={args.temperature}",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "det_checks": checks,
                "metrics": {"score": score, "runs_clean": clean,
                            "html_bytes": len(html), "gen_seconds": round(gen_s, 1),
                            "emitted_html": bool(html),
                            "followed_output_contract": bool(html) and "```" not in text[:200],
                            "completion_tokens": usage.get("completion_tokens"),
                            "attempt_rung": rung,
                            **{k: v for k, v in detail.items() if isinstance(v, (int, float, bool, str))}},
                "artifacts": {"html": f"builds/{stem}.html",
                              "screenshot": f"builds/{stem}.png"},
                "error_detail": err,
            }
            with shard.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {t['id']:10s} r{rep}  {score}/6 {'runs-clean' if clean else 'FAILS'}"
                  f"  {len(html):6d} bytes  {gen_s:5.0f}s"
                  f"  {(checks.get('runs_clean', {}).get('detail') or '')[:52]}")
    print("done ->", shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
