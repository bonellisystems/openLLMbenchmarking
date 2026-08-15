# Handoff: Benchmarking an Unknown LLM Endpoint

**For:** an agent (Grok/Build) testing an endpoint we have not characterised.
**From:** the LLMtest v2 project, which has measured 21 locally-served models across
11 batteries on RTX PRO 6000 hardware.
**Goal:** produce numbers for the new endpoint that are *comparable* to the baseline
table in §5, using the deterministic batteries only (no judge panel required).

You do not need our repository. Everything needed is described here. All code is
Python 3 standard library only.

---

## 0. Read this first: what makes a result trustworthy

Most of the ways this goes wrong are not "the model scored badly". They are **the
harness silently measuring something other than what it claims**. We lost days and
several hundred dollars to exactly that. §2 is the most valuable section in this
document; read it before writing any test code.

Three rules that govern everything below:

1. **Never infer a capability from config — measure it.** A setting you passed is not
   a setting that took effect.
2. **A missing measurement and a zero are different facts.** Never let "not run"
   collapse into "scored 0". Report them as distinct.
3. **Separate infrastructure failures from model failures.** A timeout, a 400, or a
   dropped connection is not the model being bad at the task. Excluded from the
   denominator, counted separately.

---

## 1. Phase 0 — Discover the endpoint before testing it

Do not skip this. The capability profile determines which batteries are even valid.

### 1.1 Probe script

```python
import json, time, urllib.request, urllib.error

BASE = "https://YOUR-ENDPOINT"      # no trailing slash
KEY  = "YOUR-KEY-OR-EMPTY"
MODEL = None                         # filled in by probe_models()

def call(path, payload=None, method="GET", timeout=120):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {KEY}"} if KEY else {})},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:500]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def probe_models():
    st, body = call("/v1/models")
    print("GET /v1/models ->", st)
    if st == 200 and isinstance(body, dict):
        ids = [m.get("id") for m in body.get("data", [])]
        print("  models:", ids[:20])
        return ids
    return []

def probe_chat(model, **kw):
    payload = {"model": model,
               "messages": [{"role": "user", "content": "Reply with exactly: PING OK"}],
               "max_tokens": 512, "temperature": 0, **kw}
    t0 = time.time()
    st, body = call("/v1/chat/completions", payload, "POST")
    dt = time.time() - t0
    print(f"POST /v1/chat/completions -> {st} in {dt:.1f}s")
    if st != 200:
        print("  body:", str(body)[:300]); return None
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message", {}) or {}
    print("  finish_reason :", ch.get("finish_reason"))
    print("  content       :", repr((msg.get("content") or "")[:80]))
    print("  reasoning_content present:", bool(msg.get("reasoning_content")))
    print("  usage         :", body.get("usage"))
    print("  timings       :", body.get("timings"))   # llama.cpp only
    return body
```

### 1.2 The capability profile to fill in

Record every one of these. Later sections branch on them.

| Question | How to answer it | Why it matters |
|---|---|---|
| OpenAI-compatible? | `/v1/chat/completions` returns `choices[0].message.content` | Everything below assumes this shape |
| Auth required? | try with and without the header | affects rate limits, error triage |
| Model id(s)? | `/v1/models`, else try the obvious name | some servers **require** `model`, some ignore it |
| Reports `usage`? | look for `prompt_tokens`/`completion_tokens` | needed for cost + throughput |
| Reports `timings`? | llama.cpp exposes `predicted_per_second` | best throughput source; else time it yourself |
| Emits `reasoning_content`? | see probe output | **critical** — see trap 2.1 |
| Supports `tools`? | send a request with a `tools` array | decides whether B2/B11 are valid |
| Supports streaming? | `"stream": true` | not required, but note it |
| Max context? | §1.3 | decides which B4 arms are valid |
| Concurrency limit? | §1.4 | decides B5 concurrency arms |

### 1.3 Find the real context limit — do not trust the docs

```python
def probe_context(model, lo=4096, hi=1_000_000):
    """Binary search the largest prompt the endpoint accepts.
    Filler is deliberately low-entropy; we are testing acceptance, not recall."""
    def fits(n):
        filler = ("word " * (n // 2))[:n * 4]
        st, _ = call("/v1/chat/completions",
                     {"model": model,
                      "messages": [{"role": "user", "content": filler + "\nSay OK."}],
                      "max_tokens": 16, "temperature": 0}, "POST", timeout=300)
        return st == 200
    if not fits(lo): return 0
    while lo < hi - 1024:
        mid = (lo + hi) // 2
        if fits(mid): lo = mid
        else: hi = mid
    return lo
```

**Trap:** an endpoint can advertise 128k and serve 8k per request, because the server
splits its context across parallel slots (`n_ctx_slot = n_ctx / n_slots`). We hit a
deterministic HTTP 400 on exactly one long-context task for this reason. The measured
number is the one that counts.

### 1.4 Concurrency

Fire N identical requests in parallel (N = 1, 2, 4, 8, 16). Record aggregate
tokens/sec and the error rate at each level. Stop when errors appear — that is the
endpoint's real ceiling, and it is a result worth reporting.

---

## 2. The traps — every one of these has bitten us

### 2.1 Reasoning models spend the whole budget thinking and return **nothing**

The single most expensive failure mode. A model that reasons internally emits its
thinking into a separate channel; if `max_tokens` runs out before it finishes, you get:

```
finish_reason: "length"
content: ""                       <- empty
reasoning_content: "....16000 chars...."
```

Measured on our roster: one model returned **empty content on 23 of 23 tasks** with a
16,000-token budget — it never stopped thinking. Another returned empty at 24 tokens
and a perfect answer at 512.

**Detection:** `content` empty AND `finish_reason == "length"`. Also: the SHA-256 of an
empty string is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` —
a cheap way to spot empty answers in bulk.

**Handling:** use a generous budget (we use 4k/8k/16k by task class). If content is
still empty at 16k, that is a **real finding about the model**, not a harness bug —
report it as such, and do not score those tasks as 0 quality.

Also try `chat_template_kwargs: {"enable_thinking": false}` — but verify the output is
sane. One model, with thinking disabled, emitted `"3333333333..."` for the entire
budget. Degenerate output is not a pass.

### 2.2 A declared setting is not an applied setting

Our suite recorded a speculative-decoding flag in every row's condition string for
**twenty models**, and the flag was never actually passed to the server. The tell: the
"on" and "off" arms returned identical throughput — 59.3 vs 59.5, 264.1 vs 264.3,
60.3 vs 60.3 — a speedup of exactly 1.00× across the whole roster. It survived for
months because someone had written a plausible explanation for it.

**Rule:** when you claim a configuration, prove it changed something measurable. If A/B
arms are within noise, assume the flag did not apply until proven otherwise.

### 2.3 Partial runs look identical to complete ones

A run that dies halfway leaves rows behind, and any code that sets `tested = True` on
"has at least one row" will show a green cell built from 5 of 24 measurements.

**Rule:** compute the expected count per cell from the task list, and assert it. Report
`n` beside every score, always.

### 2.4 Infra errors poison the denominator

Timeouts, 400s, connection resets, and rate-limit rejections are not task failures.
Count them separately. If more than ~10% of a cell is infra errors, the cell is not
reportable — fix the harness and re-run.

### 2.5 Temperature 0 is not determinism across machines

We re-ran identical prompts at temperature 0 on different GPUs and scores moved by up
to **13 points** (one battery went 87 → 100, another 97 → 90), because batching and
numerics shift borderline outputs. Do not compare a number from your endpoint to ours
and call a 5-point difference a ranking.

### 2.6 Narration is not execution

Models will happily *describe* running a command they never ran. Any tool-use test must
be scored from the **side effects** (files created, values returned), never from the
assistant's prose.

### 2.7 Small samples are mostly ties

Our per-cell `n` is 30 or fewer for most batteries. Differences under ~5 points are
noise. Report Wilson 95% intervals for pass-rates:

```python
import math
def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-m), 100*(c+m))
```

---

## 3. The batteries to run

Nine deterministic batteries. Each is scored by machine — no judge panel. For each,
run **3 repetitions per task at temperature 0** and report the mean plus `n`.

### B2 — Tool calling (%)
**Question:** can it emit a well-formed tool call at all?
**Method:** advertise 3–5 tool schemas in the `tools` array. Prompts that should
trigger: a single call, two independent calls (parallel), a chained call where the
second depends on the first's result, an error-recovery case where the first call
returns an error, and an abstention case where **no** tool applies.
**Score:** % of cases with a valid call — correct tool selected, arguments matching the
schema, JSON parseable.
**Note:** this is a *formation floor*, not agentic skill. Most models score ~100%. A
low score here means the endpoint's tool support is broken, which is a useful finding.
**Skip if** the endpoint has no `tools` support — record "not supported", not 0.

### B3 — Hallucination resistance (%)
**Question:** does it refuse/hedge instead of fabricating?
**Method:** unanswerable questions, false-premise questions ("why did X do Y" where Y
never happened), and trick questions. Mix in answerable controls so you can detect a
model that just refuses everything.
**Score:** % where it correctly declined or flagged the false premise, minus credit for
refusing the controls.
**Trap:** a small answer budget starves reasoning models and looks like fabrication.
Give it room (see 2.1).

### B4 — Long-context retrieval (%)
**Question:** can it find a needle in a large haystack?
**Method:** embed a distinctive fact at varying depths in filler text. Sweep context
lengths — 16k, 32k, 64k, 128k, 256k — **but only arms within the measured limit from
§1.3.** Include single-needle, multi-needle (2–3 facts), distractor (similar-looking
false facts), and multi-hop (fact A points to fact B) variants.
**Score:** % of needles correctly retrieved, reported **per context length**.
**Report the arms you could not run as "not run", never as 0.**

### B5 — Throughput (tokens/sec)
**Question:** how fast does it decode?
**Method:** prefer the server's own `timings.predicted_per_second`. Otherwise measure
wall-clock and divide by `usage.completion_tokens`. Two conditions: a short PEAK
generation, and a SUSTAINED one (~32k tokens of output) — they differ. Then the
concurrency ladder from §1.4.
**Report:** decode t/s and prefill t/s separately if available.
**Trap:** if you test any acceleration feature, verify it actually engaged (2.2).

### B6 — Coding (%)
**Question:** does the code work?
**Method:** 10 tasks — 5 from scratch (e.g. `is_prime`, a word-count CLI, a backup
shell script, a JS debounce, a SQL query), 5 planted-bug fixes where you give it
broken code and a failing test.
**Score:** % passing deterministic checks — run the code, run the tests. Do not grade
by reading it.

### B7 — Reproducibility (%)
**Question:** how much does the answer move when the harness moves?
**Method:** same prompts across a config matrix — system prompt present/minimal,
temperature 0 vs default, tool format native vs prompted.
**Score:** % of cells where deterministic signals agree with the baseline cell. This is
a *stability* measure; a low score means results are harness-sensitive and every other
number needs wider error bars.

### B9 — Program generation (%)
**Question:** does a one-shot generated program actually run?
**Method:** bare one-line prompts for self-contained browser programs (snake, tetris,
a simple 3D scene). Then **drive each build headlessly** — does it load without errors,
paint pixels, animate over time, respond to key events, survive a burst of input.
**Score:** % of builds passing those checks.
**Trap:** a generated infinite loop will wedge the browser and your test runner's own
timeouts may never fire, because the page stops servicing the automation protocol. Use
an out-of-band watchdog (e.g. SIGALRM) and score a hung build as a failure rather than
hanging the run. This cost us 59 minutes once.

### B10 — Security review (score)
**Question:** does it find real defects without inventing fake ones?
**Method:** vulnerable/patched **pairs** of the same code, plus safe-but-alarming
decoys. Sensitivity alone is useless — nearly every model flags the vulnerable version.
The discriminator is whether it stays quiet on the patched version and the decoys.
**Score:** usable-finding score = recall × specificity. Include a hard tier of
multi-defect chains graded on how much of the chain is found.
**Framing:** this is authorised defensive code review of code you supply.

### B11 — Tool loop (%)
**Question:** can it *drive* a loop — call, read the result, act on it?
**Method:** your client advertises schemas and owns execution. Give it a task needing
several dependent steps (list files → read one → write a modified version).
**Score:** from the **filesystem**, not the transcript (2.6).

### B8 — Agentic harness (optional, heavy)
Real coding agent in a disposable container against sealed tasks with a hidden
completion oracle. Requires a full agent harness plus Docker. Out of scope unless you
have both; if you attempt it, the oracle must be withheld from the model and
re-injected at scoring time, or it will be gamed.

---

## 4. What to record for every single result

One row per (task, repetition, condition):

```json
{
  "endpoint": "https://...",
  "model_id": "as reported by /v1/models",
  "battery": "B3",
  "task_id": "b3.false-premise-04",
  "run_n": 2,
  "condition": "temp=0;max_tokens=8000;tools=native",
  "status": "ok | infra_error | empty_output",
  "score": 1,
  "prompt_tokens": 812,
  "completion_tokens": 341,
  "finish_reason": "stop",
  "content_sha256": "…",
  "latency_s": 4.2,
  "error_detail": null,
  "ts": "2026-08-15T12:00:00Z"
}
```

`condition` matters: it is what makes two numbers comparable or not. Record what you
**actually sent and verified**, not what you intended.

---

## 5. Baseline — 21 models already measured

Same batteries, RTX PRO 6000, locally served. Use for calibration, not as a league
table to slot into: your endpoint runs on different hardware, so B5 especially is not
directly comparable.

Ranges across the roster:

| Battery | n models | min | median | max |
|---|---|---|---|---|
| B2 Tool calling | 21 | 33 | **100** | 100 |
| B3 Hallucination | 21 | 21 | **62** | 79 |
| B4 Long context | 21 | 19 | **68** | 100 |
| B5 Throughput t/s | 21 | 56 | **130** | 292 |
| B6 Coding | 21 | 73 | **97** | 100 |
| B7 Reproducibility | 21 | 62 | **91** | 100 |
| B9 Program gen | 20 | 4 | **58** | 83 |
| B10 Security | 20 | 6 | **50** | 83 |
| B11 Tool loop | 20 | 0 | **100** | 100 |

Selected models (full table available on request):

| model | B2 | B3 | B4 | B5 | B6 | B7 | B9 | B10 | B11 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.6-27b-dense | 100 | 72 | 40 | 59 | 100 | 80 | 58 | 21 | 100 |
| gemma-4-31b-dense | 100 | 77 | 68 | 57 | 93 | 97 | 79 | 42 | 100 |
| gpt-oss-120b | 81 | 38 | 100 | 177 | 73 | 86 | 62 | 67 | 92 |
| gpt-oss-20b | 90 | 33 | 95 | 244 | 83 | 83 | 58 | 69 | 100 |
| qwen3-235b | 100 | 79 | 100 | 58 | 100 | 94 | – | – | – |
| nemotron-3-nano-30b | 100 | 33 | 95 | 292 | 83 | 88 | 4 | 61 | 83 |
| llama-4-scout | 33 | 56 | 87 | 85 | 90 | 97 | 38 | 50 | 0 |

Reading the table: B2 near 100 is normal — it is a floor. B3 medians in the 60s mean
hallucination resistance is genuinely hard. B11 is bimodal — models either drive a loop
or cannot (note the 0). B10 spans 6 → 83, the widest spread and the most discriminating
battery we have.

---

## 6. What to hand back

1. **Capability profile** from §1 — including what the endpoint *cannot* do.
2. **Score table**: one row per battery with score, `n`, Wilson interval, and infra-error
   count. `not run` and `not supported` spelled out, never blank or zero.
3. **The conditions actually used** — model id, token budgets, temperature, tool format,
   measured context limit.
4. **Anomalies**, especially: empty outputs, degenerate outputs, arms that could not run,
   and any A/B that came back suspiciously identical (2.2).
5. **Raw rows** in the §4 shape, so results can be re-scored later without re-running.

A result you cannot explain the provenance of is a result that has to be thrown away
later. Recording the condition costs nothing now and is unrecoverable afterwards.
