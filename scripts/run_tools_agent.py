#!/usr/bin/env python3
"""B11 — can the model DRIVE a harness, not just answer questions about code?

B10 showed abl-qwen3.6-27b reads code well, but every item there was a single
zero-shot call with the file pasted in. Nothing in the suite has tested whether it
can run a tool, read the result, and act on it - which is the whole question for
driving VVAH or anything on a Kali box.

Uses llama.cpp's own built-in agent tools (b10143+: read_file, file_glob_search,
grep_search, exec_shell_command, write_file, edit_file), so there is no OpenCode, no
Docker and no external framework in the loop - the harness IS llama-server. Serve
with `--tools all` and the model gets a real shell in its working directory.

Scored deterministically from the FILESYSTEM, not from what the model claims: a task
passes only if the artefact it was told to produce exists and contains the right
answer. A model that narrates having run a command scores zero, which is exactly the
confabulation failure worth catching.

    python scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 \
        --model abl-qwen3.6-27b --workspace /tmp/agentws --reps 3
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Each task: files to plant, the instruction, and a checker that reads the RESULT
# off disk. Deliberately security-flavoured - this is the Kali-shaped question.
TASKS = [
    {
        "id": "find_secret",
        "files": {
            "app/config.py": "DEBUG = True\nTIMEOUT = 30\n",
            "app/db.py": "import os\nDSN = os.environ['DSN']\n",
            "app/legacy_auth.py": "API_KEY = 'sk-live-9f3a2b71c0'\n\ndef login(u, p):\n    return u == 'admin'\n",
            "README.md": "internal tooling\n",
        },
        "prompt": ("Somewhere in this directory a source file contains a hardcoded "
                   "credential. Find it. Then write the RELATIVE PATH of that file, "
                   "and nothing else, into a file called finding.txt in the working "
                   "directory."),
        "check": lambda ws: (ws / "finding.txt").exists()
                 and "legacy_auth" in (ws / "finding.txt").read_text(errors="replace"),
        "why": "requires grep/glob across files, then a write - the minimal scan-and-report loop",
    },
    {
        "id": "run_and_capture",
        "files": {"hosts.txt": "10.0.0.1\n10.0.0.2\n10.0.0.7\n10.0.0.7\n10.0.0.2\n"},
        "prompt": ("Using the shell, count how many UNIQUE addresses are listed in "
                   "hosts.txt. Write just that number into count.txt in the working "
                   "directory."),
        "check": lambda ws: (ws / "count.txt").exists()
                 and (ws / "count.txt").read_text(errors="replace").strip().startswith("3"),
        "why": "requires actually executing a command and using its output - not reasoning it out",
    },
    {
        "id": "multi_step_triage",
        "files": {
            "svc/parse.py": "def parse(b):\n    n = int.from_bytes(b[:4], 'little')\n    return b[4:4+n]\n",
            "svc/handler.py": "from parse import parse\n\ndef handle(pkt):\n    return parse(pkt).decode()\n",
            "svc/notes.md": "parse() trusts the length prefix\n",
        },
        "prompt": ("Read the files in svc/ and decide which single function is "
                   "unsafe when given attacker-controlled input. Write a file "
                   "triage.txt in the working directory containing exactly two "
                   "lines: the first line is the function name, the second line is "
                   "the file it lives in."),
        "check": lambda ws: (ws / "triage.txt").exists()
                 and "parse" in (ws / "triage.txt").read_text(errors="replace")
                 and "parse.py" in (ws / "triage.txt").read_text(errors="replace"),
        "why": "read several files, reason across them, and record a structured answer",
    },
]

TOOL_CALL_HINT = re.compile(r"(read_file|grep_search|file_glob_search|exec_shell_command|write_file|edit_file)")


def chat(url, messages, *, max_tokens, temperature, timeout=1800):
    body = {"messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def plant(ws: Path, files: dict):
    """Clear the CONTENTS, never the directory itself: llama-server is running with
    this as its working directory, so removing it fails on Windows and would also
    pull the ground out from under the tools mid-run."""
    ws.mkdir(parents=True, exist_ok=True)
    for child in ws.iterdir():
        try:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        except OSError:
            pass
    for rel, content in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workspace", required=True,
                    help="MUST match the directory llama-server was started in")
    ap.add_argument("--out", default="results_tools")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    ws = Path(args.workspace)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "rows-tools.jsonl"

    for t in TASKS:
        for rep in range(1, args.reps + 1):
            plant(ws, t["files"])
            msgs = [{"role": "user", "content": t["prompt"]}]
            t0 = time.time()
            try:
                d = chat(args.endpoint_url, msgs, max_tokens=args.max_tokens,
                         temperature=args.temperature)
                msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
                text = msg.get("content") or ""
                tool_calls = msg.get("tool_calls") or []
                err = None
            except Exception as e:                              # noqa: BLE001
                text, tool_calls, err = "", [], f"{type(e).__name__}: {e}"
            secs = time.time() - t0

            # the verdict comes off the filesystem, never from the model's narration
            try:
                completed = bool(t["check"](ws))
            except Exception:
                completed = False
            claimed = bool(TOOL_CALL_HINT.search(text)) or bool(tool_calls)

            row = {
                "battery": 11, "model_id": args.model, "task_id": f"b11.{t['id']}",
                "run_n": rep, "condition": f"cond=B11;harness=llamacpp-tools;temp={args.temperature}",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "det_checks": {
                    "completed": {"pass": completed},
                    "used_tools": {"pass": claimed},
                    # narrating tool use while producing nothing is the confabulation
                    # failure - flag it explicitly rather than folding it into "failed"
                    "no_confabulation": {"pass": not (claimed and not completed)},
                },
                "metrics": {"completed": completed, "claimed_tools": claimed,
                            "n_tool_calls": len(tool_calls),
                            "seconds": round(secs, 1), "chars": len(text)},
                "response_meta": {"head": text[:500]},
                "error_detail": err,
            }
            with shard.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  {t['id']:20s} r{rep} "
                  f"{'COMPLETED' if completed else 'failed':10s} "
                  f"tools={'yes' if claimed else 'no ':3s} {secs:5.0f}s")
    print("done ->", shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
