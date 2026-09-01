"""Streaming OpenAI-compatible chat with per-call prefill/decode metrics."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def models(base: str, timeout: float = 5.0) -> list[dict]:
    url = base.rstrip("/")
    if url.endswith("/v1"):
        url = url
    else:
        url = url + "/v1"
    req = urllib.request.Request(url + "/models")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return list(data.get("data") or [])


def advertised_ctx(model_objs: list[dict], model: str) -> int | None:
    for m in model_objs:
        if m.get("id") != model and m.get("root") != model:
            continue
        for k in ("max_model_len", "context_length", "max_context_length"):
            if m.get(k):
                try:
                    return int(m[k])
                except (TypeError, ValueError):
                    pass
        meta = m.get("meta") or m.get("root_meta") or {}
        if isinstance(meta, dict) and meta.get("max_model_len"):
            try:
                return int(meta["max_model_len"])
            except (TypeError, ValueError):
                pass
    return None


def chat(
    url: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    tools: list | None = None,
    thinking: bool | None = None,
    timeout: int = 180,
    extra: dict | None = None,
) -> dict:
    """One completion. Stream when there are no tools so TTFT/decode are real."""
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if thinking is True:
        body["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
    elif thinking is False:
        body["chat_template_kwargs"] = {
            "enable_thinking": False,
            "thinking": False,
            "reasoning_effort": "low",
        }
    if tools:
        body["tools"] = tools
        body["stream"] = False
    else:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if extra:
        body.update(extra)

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = None
    usage = None
    err = None
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls = None
    finish = None
    raw_resp = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            if body.get("stream"):
                for raw in res:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    if line[6:] == "[DONE]":
                        break
                    try:
                        ch = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    usage = ch.get("usage") or usage
                    choice = (ch.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    msg = choice.get("message") or {}
                    finish = choice.get("finish_reason") or finish
                    c = delta.get("content") or msg.get("content") or ""
                    r = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or msg.get("reasoning_content")
                        or ""
                    )
                    if (c or r) and t_first is None:
                        t_first = time.perf_counter()
                    if c:
                        content.append(c)
                    if r:
                        reasoning.append(r)
                    if delta.get("tool_calls"):
                        tool_calls = delta.get("tool_calls")
            else:
                raw_resp = json.load(res)
                t_first = time.perf_counter()
                usage = raw_resp.get("usage") or {}
                choice = (raw_resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                finish = choice.get("finish_reason")
                content.append(msg.get("content") or "")
                reasoning.append(msg.get("reasoning_content") or "")
                tool_calls = msg.get("tool_calls")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        err = f"HTTPError {e.code}: {detail}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    t_end = time.perf_counter()
    usage = usage or {}
    text = "".join(content).strip()
    think = "".join(reasoning).strip()
    n = int(usage.get("completion_tokens") or 0)
    p = int(usage.get("prompt_tokens") or 0)
    rt = usage.get("reasoning_tokens") or usage.get("completion_tokens_details", {})
    if isinstance(rt, dict):
        rt = rt.get("reasoning_tokens")
    decode = None
    prefill = None
    if t_first and n > 1:
        decode = (n - 1) / (t_end - t_first)
    if t_first and p and (t_first - t0) > 0:
        prefill = p / (t_first - t0)
    elif not body.get("stream") and p and (t_end - t0) > 0 and n:
        # Non-stream: split e2e by token share (rough).
        decode = n / (t_end - t0)
    return {
        "ok": err is None and (bool(text) or bool(think) or bool(tool_calls) or n > 0),
        "error": err,
        "content": text,
        "reasoning": think[:6000],
        "tool_calls": tool_calls,
        "finish_reason": finish,
        "prompt_tokens": p,
        "completion_tokens": n,
        "reasoning_tokens": rt,
        "ttft_ms": round(1000 * (t_first - t0)) if t_first else None,
        "decode_tok_s": round(decode, 1) if decode else None,
        "prefill_tok_s": round(prefill, 1) if prefill else None,
        "e2e_s": round(t_end - t0, 2),
        "raw": raw_resp,
    }
