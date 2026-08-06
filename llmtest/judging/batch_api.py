"""Anthropic Message Batches transport for the Claude judge seat.

    NOT THE JUDGING PATH. EVALUATED 2026-08-06 AND DELIBERATELY NOT ADOPTED.

Michael's ruling: the judge seat stays on `claude -p`. This module is kept only so the
evidence behind that ruling survives, and so nobody re-opens the question without it.

MEASURED (scripts/judge_batch_control.py, 12 full-roster packets re-judged under the
same claude-opus-4-8 pin, $3.05):

    mean delta   -0.24 pts (API minus CLI)      exact match 44%, within 1pt 87%
    CAL-strong   API 8.56 vs CLI 8.78
    CAL-weak     API 0.78 vs CLI 1.22
    parse failures 3/12 (missing letter, no JSON, a 19-letter ranking for 18 letters)

The API path judges slightly STRICTER. That is small, but it is systematic, and the
existing 16-model scorecard was judged via the CLI - so adopting it would put a
delivery confound inside one scorecard, on top of the packet-size leniency already
being corrected for. Same principle as the hardware-consistency rule: one path, or the
comparison is not a comparison.

Cost was never the deciding factor; for the record it would have been ~$33 vs quota for
four models, and prompt caching does NOT apply here (see CACHE_CONTROL_ENABLED below).

Wiring this into config/judges.yaml requires re-judging the ENTIRE roster through it,
not just new models.

Why a second transport at all: the `claude` seat has always run through
`claude -p --model <pin> --output-format json` (see config/judges.yaml). That bills
Michael's subscription quota, spawns one Node process per packet, and - measured
2026-08-06 - charged ~37k cache-CREATION tokens on every call while never reading the
cache back. Judging is offline work with no interactivity, which is exactly the shape
the Batch API exists for.

DELIVERY IS NOT NEUTRAL, so this module never pretends it is. `claude -p` runs Claude
Code, with its own agent system prompt and tool surface wrapped around the packet; a
raw /v1/messages call sends the packet and nothing else. The model sees a different
context in each case even when the packet bytes are identical, so any judgment produced
here records `delivery: api-batch` alongside the model pin. A row that does not say how
it was judged is a row that cannot be retired later - the exact mistake the
suite-v2.2.0 hardware campaign spent a rental fixing.

The key is read from the repo-root .env as CLAUDE_API_KEY and is never logged, never
echoed, and never written into any artifact or shipped to a rented box.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API_ROOT = "https://api.anthropic.com"
API_VERSION = "2023-06-01"

# Cache reads cost 0.1x base and writes 1.25x, and the discounts stack with Batch -
# but a prefix only caches at all once it clears the model's minimum cacheable length.
# MEASURED on this repo's packets: the prefix shared by ALL packets is ~144 tokens, and
# the best case (grouping by business unit) is 789-957 tokens. Both sit under the
# 1024-token Opus-class floor, so caching is NOT expected to fire here and the Batch
# discount is the whole saving. Left unset deliberately rather than sprinkling
# cache_control that would only add cache-write premiums for no reads.
CACHE_CONTROL_ENABLED = False


def load_api_key(root: Path | str) -> str:
    """CLAUDE_API_KEY from the repo-root .env. Raises if absent - callers must not
    silently fall back to a CLI seat, because that would change the judging path
    without changing the provenance stamp."""
    p = Path(root)
    env = p / ".env" if (p / ".env").exists() else p.parent / ".env"
    if not env.exists():
        raise RuntimeError(f"no .env found at {env} - cannot judge via API")
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "CLAUDE_API_KEY":
            return v.strip().strip('"').strip("'")
    raise RuntimeError("CLAUDE_API_KEY not set in .env")


def _request(key: str, path: str, payload=None, method="GET", timeout=120):
    req = urllib.request.Request(
        API_ROOT + path,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"x-api-key": key, "anthropic-version": API_VERSION,
                 "content-type": "application/json"},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        # The key travels in a header, not the URL, so an error body is safe to
        # surface - but scrub anything key-shaped defensively before it reaches a log.
        raise RuntimeError(f"HTTP {e.code}: {_scrub(body)}") from None


def _scrub(text: str) -> str:
    import re
    return re.sub(r"sk-[A-Za-z0-9\-_]{8,}", "sk-REDACTED", text)


def build_request(custom_id: str, packet_text: str, model: str,
                  max_tokens: int = 4096) -> dict:
    """One batch request. The packet goes in as the sole user message, matching what
    the CLI seat delivers on stdin - the transport differs, the packet bytes do not."""
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            # NO temperature: claude-opus-4-8 rejects it outright ("`temperature` is
            # deprecated for this model" - all 12 control requests errored on it,
            # free of charge, 2026-08-06). Omitting it is also the parity-correct
            # choice: the CLI seat's invoke template never set temperature either.
            "messages": [{"role": "user", "content": packet_text}],
        },
    }


def submit(key: str, requests: list[dict]) -> str:
    """Create a batch, return its id. Limits are 100k requests / 256 MB per batch."""
    if not requests:
        raise ValueError("refusing to submit an empty batch")
    if len(requests) > 100_000:
        raise ValueError(f"{len(requests)} requests exceeds the 100k batch limit")
    out = _request(key, "/v1/messages/batches", {"requests": requests}, method="POST")
    return out["id"]


def poll(key: str, batch_id: str, interval: int = 20, max_wait: int = 24 * 3600,
         on_tick=None) -> dict:
    """Block until the batch ends. Most batches finish within an hour; a batch that has
    not completed in 24h expires, so max_wait mirrors the server-side deadline."""
    waited = 0
    while waited < max_wait:
        b = _request(key, f"/v1/messages/batches/{batch_id}")
        if on_tick:
            on_tick(b)
        if b.get("processing_status") == "ended":
            return b
        time.sleep(interval)
        waited += interval
    raise TimeoutError(f"batch {batch_id} still running after {max_wait}s")


def fetch_results(key: str, batch: dict) -> dict:
    """{custom_id: result_dict} from the batch's results .jsonl.

    Results stay retrievable for 29 days, so a lost local copy is recoverable from
    the batch id alone - worth knowing before anyone re-runs a judging pass.
    """
    url = batch.get("results_url")
    if not url:
        raise RuntimeError(f"batch {batch.get('id')} has no results_url")
    req = urllib.request.Request(
        url, headers={"x-api-key": key, "anthropic-version": API_VERSION})
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read().decode("utf-8")
    out = {}
    for line in body.splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["custom_id"]] = d
    return out


def reply_text(result: dict) -> tuple[str, str | None]:
    """(text, error) for one batch result. Batch reports per-request outcomes, so one
    bad packet never poisons the run - it comes back as its own error."""
    r = result.get("result") or {}
    kind = r.get("type")
    if kind != "succeeded":
        return "", f"batch result {kind}: {str(r.get('error'))[:200]}"
    msg = r.get("message") or {}
    parts = [c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text"]
    return "".join(parts), None


def usage_of(result: dict) -> dict:
    msg = ((result.get("result") or {}).get("message") or {})
    return msg.get("usage") or {}
