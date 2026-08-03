#!/usr/bin/env python3
"""B11 — can the model DRIVE a tool loop, not just answer questions about code?

B10 showed the model reads code well, but every item there was one zero-shot call
with the file pasted in. This measures the thing that actually matters for driving
VVAH or anything on a Kali box: emit a tool call, read the result, decide what to do
next, and finish the job.

WHY THIS OWNS THE LOOP RATHER THAN USING llama.cpp's `--tools`
    Measured on b10156: with `--tools all` the model is never TOLD the tools exist.
    It either invents a name (`call: default_api:list_files` as plain text) or flatly
    answers "I do not have access to your local file system". Those built-ins are
    plumbed for the WebUI - the `--cors-origins localhost` note gives it away. The
    moment the client advertises the schema, the same model returns a clean
    structured call with finish_reason=tool_calls. So the harness advertises the
    schemas, executes the calls, and feeds results back - which is also the honest
    architecture: the scaffold owns fan-out, the model just reasons.

Scored from the FILESYSTEM, never from the model's narration: a task passes only if
the artefact it was told to produce exists with the right content. Describing a
command you never ran scores zero, and that confabulation is tracked separately.

    python scripts/run_tools_agent.py --endpoint-url http://127.0.0.1:8080 \
        --model abl-qwen3.6-27b --workspace /root/agentws --reps 3
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MAX_STEPS = 8

TOOLS = [
    {"type": "function", "function": {
        "name": "exec_shell_command",
        "description": "Run a shell command in the working directory and return stdout/stderr.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "the command"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file relative to the working directory.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write text to a file relative to the working directory, creating or overwriting it.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files under the working directory, recursively.",
        "parameters": {"type": "object", "properties": {}}}},
]

TASKS = [
    {
        "id": "find_secret",
        "files": {
            "app/config.py": "DEBUG = True\nTIMEOUT = 30\n",
            "app/db.py": "import os\nDSN = os.environ['DSN']\n",
            "app/legacy_auth.py": "API_KEY = 'sk-live-9f3a2b71c0'\n\ndef login(u, p):\n    return u == 'admin'\n",
            "README.md": "internal tooling\n",
        },
        "prompt": ("One source file in this working directory contains a hardcoded "
                   "credential. Find it, then write that file's relative path - and "
                   "nothing else - into a file named finding.txt."),
        "check": lambda ws: (ws / "finding.txt").exists()
                 and "legacy_auth" in (ws / "finding.txt").read_text(errors="replace"),
    },
    {
        "id": "run_and_capture",
        "files": {"hosts.txt": "10.0.0.1\n10.0.0.2\n10.0.0.7\n10.0.0.7\n10.0.0.2\n"},
        "prompt": ("Count how many UNIQUE addresses are listed in hosts.txt, using "
                   "the shell. Write just that number into count.txt."),
        "check": lambda ws: (ws / "count.txt").exists()
                 and (ws / "count.txt").read_text(errors="replace").strip().startswith("3"),
    },
    {
        "id": "multi_step_triage",
        "files": {
            "svc/parse.py": "def parse(b):\n    n = int.from_bytes(b[:4], 'little')\n    return b[4:4+n]\n",
            "svc/handler.py": "from parse import parse\n\ndef handle(pkt):\n    return parse(pkt).decode()\n",
            "svc/notes.md": "parse() trusts the length prefix\n",
        },
        "prompt": ("Read the files under svc/ and decide which single function is "
                   "unsafe on attacker-controlled input. Write triage.txt containing "
                   "exactly two lines: line 1 the function name, line 2 the file it "
                   "lives in."),
        "check": lambda ws: (ws / "triage.txt").exists()
                 and "parse" in (ws / "triage.txt").read_text(errors="replace")
                 and "parse.py" in (ws / "triage.txt").read_text(errors="replace"),
    },
    {
        "id": "chained_grep_fix",
        "files": {
            "svc/run.sh": "#!/bin/sh\ncurl -s http://internal/api?token=abc123 | tee /tmp/out\n",
            "svc/notes.md": "the token in run.sh must not be committed\n",
        },
        "prompt": ("A shell script under svc/ has a secret embedded in a URL. Replace "
                   "the secret value with the literal text REDACTED, in place, and "
                   "then write the name of the file you changed into changed.txt."),
        "check": lambda ws: (ws / "changed.txt").exists()
                 and "run.sh" in (ws / "changed.txt").read_text(errors="replace")
                 and "abc123" not in (ws / "svc" / "run.sh").read_text(errors="replace")
                 and "REDACTED" in (ws / "svc" / "run.sh").read_text(errors="replace"),
    },
]


def chat(url, messages, *, max_tokens, temperature, tools, timeout=1800, extra=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": False, "tools": tools}
    if extra:
        body.update(extra)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_tool(ws: Path, name: str, args: dict) -> str:
    """Execute one tool inside the workspace. Every path is resolved and confined -
    an agent with a shell will happily wander outside its scope if you let it, and
    that containment belongs in the harness, not in the prompt."""
    def safe(rel):
        p = (ws / str(rel)).resolve()
        if not str(p).startswith(str(ws.resolve())):
            raise ValueError("path escapes the workspace")
        return p
    try:
        if name == "exec_shell_command":
            cmd = str(args.get("command", ""))
            out = subprocess.run(cmd, shell=True, cwd=str(ws), capture_output=True,
                                 text=True, timeout=60)
            return ((out.stdout or "") + (out.stderr or ""))[:4000] or "(no output)"
        if name == "read_file":
            return safe(args.get("path", "")).read_text(errors="replace")[:6000]
        if name == "write_file":
            p = safe(args.get("path", ""))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {args.get('path')}"
        if name == "list_files":
            return "\n".join(sorted(str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()))[:4000]
        return f"unknown tool {name}"
    except Exception as e:                                        # noqa: BLE001
        return f"tool error: {type(e).__name__}: {e}"


def plant(ws: Path, files: dict):
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


def agent_loop(url, ws: Path, prompt: str, *, max_tokens, temperature):
    """Advertise -> call -> execute -> feed back, until the model stops or we hit the
    step cap. Returns (steps, tool_names, transcript_tail, error)."""
    msgs = [{"role": "system",
             "content": ("You are a security engineer working inside a scoped working "
                         "directory. Use the provided tools to inspect and modify files. "
                         "Take one action at a time and finish the task.")},
            {"role": "user", "content": prompt}]
    used, err = [], None
    for step in range(MAX_STEPS):
        try:
            d = chat(url, msgs, max_tokens=max_tokens, temperature=temperature, tools=TOOLS)
        except Exception as e:                                    # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            break
        m = (d.get("choices") or [{}])[0].get("message", {}) or {}
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     **({"tool_calls": calls} if calls else {})})
        if not calls:
            break
        for c in calls:
            fn = (c.get("function") or {})
            name = fn.get("name", "")
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except Exception:
                a = {}
            result = run_tool(ws, name, a)
            used.append(name)
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                         "name": name, "content": result[:4000]})
    tail = ""
    for msg in reversed(msgs):
        if msg.get("role") == "assistant" and msg.get("content"):
            tail = msg["content"][:400]
            break
    return len(used), used, tail, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite-version", default="suite-v2.2.0",
                    help="stamped into every row; rowselect uses it for latest-version-wins supersede")
    ap.add_argument("--hardware-sku", default="",
                    help="hardware SKU stamped into every row (e.g. rtx-pro-6000-vm). Legacy rows carry none, which is how 264 laptop B9 rows needed ledger archaeology to attribute - never again.")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", default="results_tools")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    ws = Path(args.workspace)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / "rows-tools.jsonl"

    for t in TASKS:
        for rep in range(1, args.reps + 1):
            plant(ws, t["files"])
            t0 = time.time()
            n_calls, used, tail, err = agent_loop(
                args.endpoint_url, ws, t["prompt"],
                max_tokens=args.max_tokens, temperature=args.temperature)
            secs = time.time() - t0
            try:
                completed = bool(t["check"](ws))
            except Exception:
                completed = False
            row = {
                "battery": 11, "suite_version": args.suite_version,
                "hardware_sku": args.hardware_sku, "model_id": args.model, "task_id": f"b11.{t['id']}",
                "run_n": rep,
                "condition": f"cond=B11;harness=client-loop;steps<={MAX_STEPS};temp={args.temperature}",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "det_checks": {
                    "completed": {"pass": completed},
                    "used_tools": {"pass": n_calls > 0},
                    # narrating tool use while producing nothing is the failure worth
                    # naming, so it is its own check rather than folded into "failed"
                    "no_confabulation": {"pass": not (n_calls == 0 and not completed)},
                },
                "metrics": {"completed": completed, "n_tool_calls": n_calls,
                            "tools_used": used, "seconds": round(secs, 1)},
                "response_meta": {"final": tail},
                "error_detail": err,
            }
            with shard.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print("  %-20s r%d %-10s calls=%-2d %s  %.0fs"
                  % (t["id"], rep, "COMPLETED" if completed else "failed", n_calls,
                     ",".join(dict.fromkeys(used))[:38], secs))
    print("done ->", shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
