#!/usr/bin/env python3
"""HANDOFF §1 probe. Stdlib only."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8888"
KEY = ""
MODEL = None


def call(path, payload=None, method="GET", timeout=120):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {KEY}"} if KEY else {}),
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:800]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def probe_models():
    st, body = call("/v1/models")
    print("GET /v1/models ->", st)
    if st == 200 and isinstance(body, dict):
        ids = [m.get("id") for m in body.get("data", [])]
        print("  models:", ids[:20])
        print("  raw0:", body.get("data", [{}])[:1])
        return ids
    print("  body:", str(body)[:400])
    return []


def probe_chat(model, **kw):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: PING OK"}],
        "max_tokens": 512,
        "temperature": 0,
        **kw,
    }
    t0 = time.time()
    st, body = call("/v1/chat/completions", payload, "POST")
    dt = time.time() - t0
    print(f"POST /v1/chat/completions -> {st} in {dt:.1f}s")
    if st != 200:
        print("  body:", str(body)[:400])
        return None
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message", {}) or {}
    content = msg.get("content") or ""
    print("  finish_reason :", ch.get("finish_reason"))
    print("  content       :", repr(content[:120]))
    print("  reasoning_content present:", bool(msg.get("reasoning_content")))
    print("  tool_calls    :", bool(msg.get("tool_calls")))
    print("  usage         :", body.get("usage"))
    print("  timings       :", body.get("timings"))
    return body


def probe_tools(model):
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo a string",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]
    st, body = call("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "Call echo with text=hi"}],
        "max_tokens": 256,
        "temperature": 0,
        "tools": tools,
    }, "POST")
    print("POST tools ->", st)
    if st != 200:
        print("  body:", str(body)[:400])
        return False
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message", {}) or {}
    print("  finish_reason:", ch.get("finish_reason"))
    print("  tool_calls:", msg.get("tool_calls"))
    print("  content:", repr((msg.get("content") or "")[:120]))
    return True


def main():
    ids = probe_models()
    model = ids[0] if ids else "glm-5.3-flash-dflash2"
    print("USING", model)
    probe_chat(model)
    print("--- thinking off ---")
    probe_chat(model, chat_template_kwargs={"enable_thinking": False})
    print("--- tools ---")
    probe_tools(model)


if __name__ == "__main__":
    main()
