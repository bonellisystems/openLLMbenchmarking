#!/usr/bin/env python3
"""Self-contained Spark llmtest progress HTML.

Reads ~/llmtest-spark/out (or --root). Never writes into llmtest-v2/results/.
Missing cells stay blank, never a zero.

Headline scores match the published dashboard:
  B3 = det_checks.correct (NOT AND-of-every-flag — fabricated.pass means the
  model fabricated, so treating it as a required pass forced a fake 0%).
  B9 hung generation (Unreachable / 0 HTML bytes) is a fail, not infra.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODELS = [
    ("glm-5.3-flash-dflash2", "GLM-5.3-Flash Beast DFlash2", "2× TP=2", "primary"),
    ("deepseek-v4-flash-0731", "DeepSeek-V4-Flash-0731", "2× TP=2", "primary"),
    ("qwen3.8-flash-next", "Qwen3.8-Flash-Next", "2× TP=2", "primary"),
    ("qwen3.8-27b-nvfp4", "Qwen3.8-27B NVFP4 dense", "1×", "explore"),
    ("minimax-m2.7-nvfp4", "MiniMax-M2.7 NVFP4", "?", "explore"),
    ("hy3-nvfp4-fp8", "Tencent Hy3 NVFP4-FP8", "?", "explore"),
    ("nemotron-3-super-120b-nvfp4", "Nemotron 3 Super 120B", "?", "explore"),
    ("gpt-oss-120b-unsloth", "gpt-oss-120b (unsloth)", "?", "explore"),
    ("laguna-s-2.1-nvfp4", "Laguna S 2.1 NVFP4", "?", "explore"),
]

# Display order is B1..B11. expected=None means "no fixed n / not a % cell".
BATTERIES = [
    {"id": 1, "label": "B1", "name": "Business", "expected": 360, "unit": "n",
     "scores": "Panel-judged 0–10 later. This cell is generations finished, not a quality score.",
     "blurb": "15 business units × 8 tasks × 3 reps. Spark run is generation-only; judging is off-box.",
     "roster": "PRO-6000 judged scores sat 5.0–7.6. Do not treat n/360 as a 0–10."},
    {"id": 2, "label": "B2", "name": "Tools", "expected": 30, "unit": "%",
     "scores": "% of rows where every applicable formation axis passed (schema, tool pick, parallel/chain/abstain).",
     "blurb": "Can it emit a well-formed tool call at all? Formation floor, not agentic skill.",
     "roster": "Most models ~100%. A low score usually means the endpoint's tools support is broken."},
    {"id": 3, "label": "B3", "name": "Hallucination", "expected": 39, "unit": "%",
     "scores": "Headline is CORRECT rate: hedge-and-not-trap on unanswerable probes, or the fact signal on closed-domain controls. fabricated.pass means the model asserted the trap — that is a fail, not a required check.",
     "blurb": "Unanswerable / false-premise / fabricated-artifact traps + answerable controls so an always-refuse strategy cannot game the battery.",
     "roster": "PRO-6000 correct-rate median 62% (min 21%, max 79%). Regex/keyword proxy — spot-check transcripts before ranking. Ambiguous ≠ fabricated."},
    {"id": 4, "label": "B4", "name": "Needle", "expected": None, "unit": "%",
     "scores": "% of needles retrieved, per context length that actually ran. Arms not planned are blank, never 0%.",
     "blurb": "Needle-in-a-haystack at 16k / 64k / 128k / 256k. This GLM boot only claimed 64k.",
     "roster": "On the live OpenAI endpoint the llama.cpp ctx/kv sweep planned 0 items — blank, not a zero."},
    {"id": 5, "label": "B5", "name": "Decode", "expected": None, "unit": "t/s",
     "scores": "Decode t/s from server timings. Spec-off vs spec-on must actually differ. 0.0 with tokens_out=0 is 'no timings', not 0 tok/s.",
     "blurb": "Throughput. Prefer the server's own predicted_per_second; otherwise completion_tokens / wall-clock.",
     "roster": "SGLang's OpenAI schema does not populate llama.cpp timings, so this attach recorded 0 tokens. Do not publish that as 0 t/s. Gauntlet numbers still stand."},
    {"id": 6, "label": "B6", "name": "Coding", "expected": 30, "unit": "%",
     "scores": "% of rows where every deterministic code check passed (fenced block, required constructs, compile on Python). Judged quality is not in this number.",
     "blurb": "5 from-scratch + 5 planted-bug fixes. Run the signals, do not grade by reading.",
     "roster": "PRO-6000 median 97% (min 73%)."},
    {"id": 7, "label": "B7", "name": "Repro", "expected": 80, "unit": "%",
     "scores": "% of non-baseline matrix cells whose content-signal pattern agrees with the baseline (≥0.8). Baseline cells are the reference, not in the %.",
     "blurb": "Same probes across system-prompt / temperature / tool-format / spec knobs. Stability, not capability.",
     "roster": "PRO-6000 median 91% (min 62%)."},
    {"id": 8, "label": "B8", "name": "OpenCode", "expected": None, "unit": "%",
     "scores": "% of sealed tasks the single-agent harness completed (hidden oracle). 1-task probe first; 23×5 only if eligible.",
     "blurb": "Real coding agent in a disposable container. Out of scope until the probe passes.",
     "roster": "Blank until the 1-task probe runs."},
    {"id": 9, "label": "B9", "name": "Games", "expected": 24, "unit": "%",
     "scores": "% of builds with runs_clean (load ∧ surface ∧ paint ∧ loop ∧ keys ∧ input-safe). A 600s generation timeout with 0 HTML bytes is a FAIL, not infra and not 'still running'.",
     "blurb": "One-shot browser programs (snake, tetris, arkanoid, flappy, doodle, asteroids, roguelike, flightsim) × 3 reps, then driven in headless Chrome.",
     "roster": "PRO-6000 runs_clean median 58% (min 4%, max 83%). A 0% here is an outlier and needs the fail taxonomy, not a shrug."},
    {"id": 10, "label": "B10", "name": "Security", "expected": 66, "unit": "%",
     "scores": "% correct_verdict on vulnerable/patched pairs + decoys. Drill-down splits sensitivity vs specificity — shouting VULNERABLE at everything is not a win.",
     "blurb": "Authorised defensive review of code we supply. Specificity is the discriminator.",
     "roster": "PRO-6000 usable-finding scores span 6–83."},
    {"id": 11, "label": "B11", "name": "Tool loop", "expected": 12, "unit": "%",
     "scores": "% of tasks whose FILESYSTEM shows the work happened (not the transcript).",
     "blurb": "Client advertises tools, executes them, feeds results back. Narrating an edit you never made fails.",
     "roster": "Bimodal on the roster: most 100%, a few 0%."},
]

BAT_BY_ID = {b["id"]: b for b in BATTERIES}
ART_ROOTS = [
    Path("/home/michaeldeblok/llmtest-spark/src/artifacts"),
    Path("/home/michaeldeblok/llmtest-spark/out/artifacts"),
]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def is_selftest(r: dict) -> bool:
    mid = r.get("model_id") or ""
    tid = r.get("task_id") or ""
    tags = r.get("tags") or []
    if mid in ("selftest", ""):
        return True
    if "selftest" in tags:
        return True
    if ".selftest." in tid or tid.endswith(".selftest"):
        return True
    return False


def battery_of(r: dict) -> int | None:
    b = r.get("battery")
    if b is not None:
        try:
            return int(b)
        except (TypeError, ValueError):
            pass
    tid = r.get("task_id") or ""
    m = re.match(r"b(\d+)", tid)
    return int(m.group(1)) if m else None


def _infra_msg(err: str) -> bool:
    low = (err or "").lower()
    return any(x in low for x in (
        "timeout", "timed out", "http", "conn", "reset", "unreachable", "httperror",
    )) or "HTTPError" in (err or "")


def _bool_checks(d: dict) -> list[bool]:
    return [bool(v["pass"]) for v in (d or {}).values()
            if isinstance(v, dict) and isinstance(v.get("pass"), bool)]


def failed_check_names(d: dict) -> list[str]:
    names = []
    for k, v in (d or {}).items():
        if isinstance(v, dict) and v.get("pass") is False:
            names.append(k)
    return names


def b3_reason(r: dict) -> str:
    m = r.get("metrics") or {}
    d = r.get("det_checks") or {}
    chars = int(m.get("chars") or 0)
    correct = bool(m["correct"] if "correct" in m else (d.get("correct") or {}).get("pass"))
    hedged = bool(m["hedged"] if "hedged" in m else (d.get("hedged") or {}).get("pass"))
    fab = bool(m["fabricated"] if "fabricated" in m else (d.get("fabricated") or {}).get("pass"))
    expect = m.get("expect")
    if correct:
        return "correct"
    if chars == 0:
        return "empty_output"
    if fab:
        return "fabricated"
    if hedged and expect == "answer":
        return "over_hedge_on_control"
    if hedged:
        return "hedged_but_trap_also_fired"
    return "ambiguous_regex_miss"


def b9_reason(r: dict) -> str:
    err = r.get("error_detail") or ""
    m = r.get("metrics") or {}
    d = r.get("det_checks") or {}
    if (d.get("runs_clean") or {}).get("pass") or m.get("runs_clean"):
        return "runs_clean"
    if m.get("no_html") or m.get("html_bytes") in (0, None) and not m.get("emitted_html"):
        if "Unreachable" in err or "timed out" in err.lower() or int(m.get("html_bytes") or 0) == 0:
            if "Unreachable" in err or int(m.get("gen_seconds") or 0) >= 590:
                return "gen_timeout_no_html"
            return "no_html"
    failed = failed_check_names(d)
    if failed:
        return "drive_fail:" + ",".join(x for x in failed if x != "runs_clean")
    return "fail"


def row_verdict(r: dict) -> tuple[bool | None, bool, str]:
    """(passed or None if unscored, is_infra, reason)."""
    err = (r.get("error_detail") or "") or ""
    status = r.get("status")
    b = battery_of(r)
    d = r.get("det_checks") or {}
    met = r.get("metrics") or {}

    if b == 9:
        reason = b9_reason(r)
        return (reason == "runs_clean"), False, reason

    if status and status not in ("ok", "pass"):
        if _infra_msg(err) or status == "error":
            return None, True, "infra:" + (err[:80] or status)
    if _infra_msg(err) and b != 9:
        return None, True, "infra:" + err[:80]

    if b == 1:
        content = ((r.get("response_meta") or {}).get("content")) or ""
        if not content and (r.get("response_meta") or {}).get("finish_reason") == "length":
            return None, False, "empty_thinking"
        return None, False, "generated"
    if b == 3:
        return (b3_reason(r) == "correct"), False, b3_reason(r)
    if b == 11:
        ok = bool((d.get("completed") or {}).get("pass") or met.get("completed"))
        return ok, False, "completed" if ok else "not_completed"
    if b == 10:
        ok = bool((d.get("correct_verdict") or {}).get("pass"))
        bucket = "other"
        if "decoy" in (r.get("task_id") or ""):
            bucket = "decoy"
        elif met.get("expect_vulnerable") is True:
            bucket = "sensitivity"
        elif met.get("expect_vulnerable") is False:
            bucket = "specificity"
        return ok, False, ("verdict_ok:" if ok else "verdict_wrong:") + bucket
    if b == 2:
        if met.get("det_pass") is not None:
            ok = bool(met["det_pass"])
        else:
            flags = _bool_checks(d)
            ok = all(flags) if flags else None
        return ok, False, "det_pass" if ok else "det_fail:" + ",".join(failed_check_names(d)[:6])
    if b in (4, 6):
        flags = _bool_checks(d)
        if not flags:
            return None, False, "no_checks"
        ok = all(flags)
        return ok, False, "det_pass" if ok else "det_fail:" + ",".join(failed_check_names(d)[:6])
    if b == 7:
        sab = d.get("signal_agreement_vs_baseline")
        if isinstance(sab, dict) and isinstance(sab.get("pass"), bool):
            ok = bool(sab["pass"])
            rate = sab.get("agreement_rate")
            return ok, False, f"agree:{rate}" if ok else f"disagree:{rate}"
        return None, False, "baseline_cell"
    if b == 5:
        return None, False, "throughput"
    if b == 8:
        m = met.get("completed")
        if m is None:
            m = (d.get("completed") or {}).get("pass")
        if m is None:
            return None, False, "unscored"
        return bool(m), False, "completed" if m else "not_completed"
    return None, False, "unscored"


def load_snippet(root: Path, r: dict, limit: int = 360) -> str:
    rel = ((r.get("artifacts") or {}).get("response") or {}).get("relpath") or ""
    if not rel:
        return ""
    candidates = [root / "artifacts" / rel, root.parent / "src" / "artifacts" / rel]
    candidates += [base / rel for base in ART_ROOTS]
    for p in candidates:
        if p.is_file():
            t = p.read_text(encoding="utf-8", errors="replace")
            if "ASSISTANT:" in t:
                t = t.split("ASSISTANT:", 1)[-1]
            t = " ".join(t.split())
            return t[:limit]
    return ""


def live_processes() -> dict:
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True, errors="replace")
    except Exception:
        return {}
    running: dict[str, int] = {}
    for bat in (1, 2, 3, 4, 5, 6, 7):
        if re.search(rf"llmtest run --battery {bat}\b", out):
            running["glm-5.3-flash-dflash2"] = bat
    if "run_games.py" in out:
        running["glm-5.3-flash-dflash2"] = 9
    if "run_security.py" in out:
        running["glm-5.3-flash-dflash2"] = 10
    if "run_tools_agent.py" in out:
        running["glm-5.3-flash-dflash2"] = 11
    if "run_b8_local.py" in out:
        running["glm-5.3-flash-dflash2"] = 8
    return running


def serving_model() -> str:
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8888/v1/models", timeout=2) as r:
            data = json.load(r)
        ids = [m.get("id") for m in data.get("data", [])]
        return ids[0] if ids else ""
    except Exception:
        return ""


def _empty_cell() -> dict:
    return {
        "n": 0, "pass": 0, "fail": 0, "infra": 0, "unscored": 0,
        "reasons": Counter(), "rows": [], "last_ts": "",
        "b3": Counter(), "b9": Counter(), "b10": Counter(),
        "b5_arms": [],
    }


def collect(root: Path) -> dict:
    rows: list[dict] = []
    rows += load_jsonl(root / "rows-suite-v2.3.0-spark.jsonl")
    rows += load_jsonl(root / "tools" / "rows-tools.jsonl")
    rows += load_jsonl(root / "security" / "rows-security.jsonl")
    rows += load_jsonl(root / "games" / "rows-games.jsonl")
    b8 = root / "b8_glm-5.3-flash-dflash2"
    if b8.exists():
        for p in b8.glob("rows-*.jsonl"):
            rows += load_jsonl(p)

    cells: dict[tuple[str, int], dict] = defaultdict(_empty_cell)
    for r in rows:
        if is_selftest(r):
            continue
        b = battery_of(r)
        mid = r.get("model_id") or ""
        if b is None or not mid:
            continue
        key = (mid, int(b))
        c = cells[key]
        passed, infra, reason = row_verdict(r)
        c["n"] += 1
        if infra:
            c["infra"] += 1
        elif passed is True:
            c["pass"] += 1
        elif passed is False:
            c["fail"] += 1
        else:
            c["unscored"] += 1
        c["reasons"][reason] += 1
        ts = r.get("ts") or ""
        if ts > c["last_ts"]:
            c["last_ts"] = ts

        met = r.get("metrics") or {}
        rec = {
            "task_id": r.get("task_id"),
            "run_n": r.get("run_n"),
            "passed": passed,
            "infra": infra,
            "reason": reason,
            "chars": met.get("chars") or met.get("code_chars") or met.get("html_bytes"),
            "seconds": met.get("gen_seconds"),
            "rung": met.get("attempt_rung"),
            "category": met.get("category") or met.get("expect") or met.get("track"),
            "html_bytes": met.get("html_bytes"),
            "completion_tokens": met.get("completion_tokens"),
            "condition": r.get("condition"),
        }
        if b == 3:
            rec["hedged"] = met.get("hedged")
            rec["fabricated"] = met.get("fabricated")
            rec["correct"] = met.get("correct")
            rec["expect"] = met.get("expect")
            c["b3"][reason] += 1
            if passed is not True:
                rec["snippet"] = load_snippet(root, r)
        elif b == 9:
            rec["game"] = (r.get("task_id") or "").split(".", 1)[-1]
            rec["loads"] = ((r.get("det_checks") or {}).get("loads") or {}).get("pass")
            rec["paints"] = ((r.get("det_checks") or {}).get("paints") or {}).get("pass")
            rec["error"] = (r.get("error_detail") or "")[:160]
            c["b9"][reason] += 1
        elif b == 10:
            c["b10"][reason.split(":", 1)[-1]] += 1
            rec["expect_vulnerable"] = met.get("expect_vulnerable")
        elif b == 5:
            rec["decode_tps"] = met.get("decode_tps")
            rec["pp_tps"] = met.get("pp_tps")
            rec["tokens_out"] = met.get("tokens_out")
            rec["ttft_ms"] = met.get("ttft_ms")
            rec["aggregate_tps"] = met.get("aggregate_tps")
            rec["streams_ok"] = met.get("streams_ok")
            c["b5_arms"].append(rec)
        elif passed is False and b in (2, 6, 7, 11):
            rec["failed_checks"] = failed_check_names(r.get("det_checks") or {})[:8]
        # Keep every row for drill-down; cap only snippets.
        c["rows"].append(rec)

    camp = ""
    p = root / "campaign.log"
    if p.exists():
        camp = p.read_text(encoding="utf-8", errors="replace")

    details = {}
    compact_cells = {}
    for (mid, b), c in cells.items():
        reasons = dict(c["reasons"])
        details[f"{mid}|{b}"] = {
            "n": c["n"], "pass": c["pass"], "fail": c["fail"],
            "infra": c["infra"], "unscored": c["unscored"],
            "reasons": reasons, "last_ts": c["last_ts"],
            "b3": dict(c["b3"]), "b9": dict(c["b9"]), "b10": dict(c["b10"]),
            "b5_arms": c["b5_arms"],
            "rows": c["rows"],
        }
        compact_cells[f"{mid}|{b}"] = {
            "n": c["n"], "pass": c["pass"], "fail": c["fail"],
            "infra": c["infra"], "unscored": c["unscored"], "last_ts": c["last_ts"],
        }

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "serving": serving_model(),
        "running": live_processes(),
        "campaign_tail": "\n".join(camp.strip().splitlines()[-16:]),
        "cells": compact_cells,
        "details": details,
        "batteries": BATTERIES,
        "total_rows": sum(v["n"] for v in compact_cells.values()),
    }


def _cell_state(st: dict, mid: str, bat: int) -> dict:
    spec = BAT_BY_ID[bat]
    exp = spec.get("expected")
    c = st["cells"].get(f"{mid}|{bat}")
    live = st["running"].get(mid) == bat
    n = c["n"] if c else 0
    if live and exp and n >= exp:
        live = False
    infra = c["infra"] if c else 0
    p = c["pass"] if c else 0
    f = c["fail"] if c else 0
    unscored = c["unscored"] if c else 0
    # Denominator is scored rows only. Infra and B7 baseline (unscored) are not zeros.
    eligible = p + f
    return {
        "spec": spec, "c": c, "live": live, "n": n, "infra": infra,
        "p": p, "f": f, "unscored": unscored, "eligible": eligible, "exp": exp,
    }


def cell_html(st: dict, mid: str, bat: int) -> str:
    s = _cell_state(st, mid, bat)
    spec, c, live, n = s["spec"], s["c"], s["live"], s["n"]
    if not c and not live:
        return (f'<td class="blank" data-mid="{mid}" data-bat="{bat}" '
                f'title="not run — click for metric">—</td>')
    p, f, infra, eligible, exp = s["p"], s["f"], s["infra"], s["eligible"], s["exp"]
    cls = "run"
    label = str(n)
    title = f"n={n} pass={p} fail={f} infra={infra} unscored={s['unscored']}"

    if spec["unit"] == "t/s":
        arms = (st["details"].get(f"{mid}|{bat}") or {}).get("b5_arms") or []
        toks = sum(int(a.get("tokens_out") or 0) for a in arms)
        tps = [a.get("decode_tps") for a in arms if a.get("decode_tps")]
        if live:
            cls, label = "live", f"▶ {n}"
        elif toks == 0 and (not tps or max(tps) == 0):
            cls, label = "infra", "no t/s"
            title += " — SGLang left llama.cpp timings empty; not a 0 tok/s result"
        else:
            cls, label = "mid", f"{max(tps):.0f} t/s"
    elif spec["unit"] == "n":
        if live:
            cls, label = "live", f"▶ {n}/{exp}" if exp else f"▶ {n}"
        elif exp:
            cls = "partial" if n < exp else "run"
            label = f"{n}/{exp}"
        else:
            label = f"n={n}"
    elif live:
        cls = "live"
        label = f"▶ {n}/{exp}" if exp else f"▶ {n}"
    elif exp and n < exp:
        cls, label = "partial", f"{n}/{exp}"
    elif spec["unit"] == "%" and eligible > 0 and (not exp or n >= exp or (p + f + infra + unscored) >= (exp or 0)):
        pct = 100.0 * p / eligible
        lo, hi = wilson(p, eligible)
        label = f"{pct:.0f}%"
        title += f" Wilson {100 * lo:.0f}–{100 * hi:.0f} scored={eligible}"
        if infra / max(n, 1) > 0.10:
            cls, label = "infra", "infra"
        else:
            cls = "good" if pct >= 80 else ("mid" if pct >= 50 else "bad")
        if bat == 3:
            d3 = (st["details"].get(f"{mid}|{bat}") or {}).get("b3") or {}
            title += (f" · correct={d3.get('correct', p)} fabricated={d3.get('fabricated', 0)} "
                      f"empty={d3.get('empty_output', 0)} ambiguous={d3.get('ambiguous_regex_miss', 0)}")
        if bat == 9:
            d9 = (st["details"].get(f"{mid}|{bat}") or {}).get("b9") or {}
            title += " · " + ", ".join(f"{k}={v}" for k, v in d9.items())
    elif n and not live:
        cls, label = "partial", f"n={n}"

    return (f'<td class="{cls}" data-mid="{mid}" data-bat="{bat}" '
            f'title="{title} — click to drill down">{label}</td>')


def render(st: dict) -> str:
    running = st["running"]
    body_rows = []
    for mid, name, topo, kind in MODELS:
        tds = "".join(cell_html(st, mid, b["id"]) for b in BATTERIES)
        mark = ""
        if st["serving"] == mid:
            mark = " ● serving"
        live = " live" if mid in running else ""
        body_rows.append(
            f'<tr class="{kind}{live}"><th>{name}'
            f'<div class="sub">{mid} · {topo}{mark}</div></th>{tds}</tr>'
        )
    heads = "".join(
        f'<th title="{b["name"]}: {b["blurb"]}">{b["label"]}<div class="sub">{b["name"]}</div></th>'
        for b in BATTERIES
    )
    metric_rows = "".join(
        f"<tr><th>{b['label']} {b['name']}</th><td>{b['unit']}</td>"
        f"<td>{b['scores']}</td><td>{b['roster']}</td></tr>"
        for b in BATTERIES
    )
    camp = (st.get("campaign_tail") or "").replace("&", "&amp;").replace("<", "&lt;")
    payload = json.dumps({
        "generated": st["generated"],
        "serving": st["serving"],
        "running": st["running"],
        "details": st["details"],
        "batteries": BATTERIES,
        "models": [{"id": m, "name": n} for m, n, *_ in MODELS],
    }, ensure_ascii=False)
    payload = payload.replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="refresh" content="20" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Spark llmtest — suite-v2.3.0-spark</title>
<style>
  :root {{
    --bg:#0b0d10; --card:#14181e; --text:#e8edf2; --muted:#93a0ab;
    --line:rgba(255,255,255,.08); --good:#34d399; --mid:#fbbf24;
    --bad:#f87171; --live:#c084fc; --partial:#5ea0ff; --infra:#fb923c;
    --accent:#7dd3fc;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg);
    color:var(--text); line-height:1.45;
  }}
  main {{ max-width:1480px; margin:0 auto; padding:28px 16px 80px; }}
  h1 {{ font-size:1.55rem; margin:0 0 6px; letter-spacing:-.03em; }}
  h2 {{ font-size:1.05rem; margin:28px 0 10px; }}
  .lead {{ color:var(--muted); margin:0 0 14px; }}
  table.score {{ width:100%; border-collapse:collapse; font-size:.82rem; background:var(--card);
    border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
  .score th, .score td {{ padding:8px 6px; border-bottom:1px solid var(--line); text-align:center; }}
  .score th:first-child, .score td:first-child {{ text-align:left; min-width:220px; }}
  .score thead th {{ color:var(--muted); font-weight:600; font-size:.72rem; }}
  .score td[data-mid] {{ cursor:pointer; }}
  .score td[data-mid]:hover {{ outline:1px solid var(--accent); }}
  .score td.on {{ box-shadow:inset 0 0 0 2px var(--accent); }}
  .sub {{ color:var(--muted); font-size:.7rem; font-weight:400; }}
  tr.explore th {{ opacity:.75; }}
  tr.live {{ background:rgba(192,132,252,.08); }}
  td.blank {{ color:#3d4650; }}
  td.live {{ color:var(--live); font-weight:700; }}
  td.partial {{ color:var(--partial); }}
  td.good {{ color:var(--good); font-weight:700; }}
  td.mid {{ color:var(--mid); }}
  td.bad {{ color:var(--bad); font-weight:700; }}
  td.infra {{ color:var(--infra); }}
  .legend {{ margin:12px 0 16px; color:var(--muted); font-size:.85rem; display:flex; gap:14px; flex-wrap:wrap; }}
  .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
  .sw.live {{ background:var(--live); }} .sw.partial {{ background:var(--partial); }}
  .sw.good {{ background:var(--good); }} .sw.mid {{ background:var(--mid); }}
  .sw.bad {{ background:var(--bad); }} .sw.infra {{ background:var(--infra); }}
  .sw.blank {{ background:#3d4650; }}
  pre {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:12px 14px; font-size:.8rem; overflow:auto; color:var(--muted); }}
  .note {{ color:var(--muted); font-size:.88rem; max-width:95ch; }}
  #drill {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:18px 18px 22px; margin:18px 0; min-height:8rem; }}
  #drill h3 {{ margin:0 0 6px; font-size:1.05rem; }}
  #drill .tiles {{ display:flex; gap:10px; flex-wrap:wrap; margin:12px 0; }}
  .tile {{ background:#0b0d10; border:1px solid var(--line); border-radius:8px; padding:10px 12px; min-width:110px; }}
  .tile .v {{ font-size:1.25rem; font-weight:700; }}
  .tile .k {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }}
  .tile.bad .v {{ color:var(--bad); }} .tile.good .v {{ color:var(--good); }}
  .tile.mid .v {{ color:var(--mid); }}
  table.fine {{ width:100%; border-collapse:collapse; font-size:.78rem; margin-top:10px; }}
  .fine th, .fine td {{ padding:5px 7px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
  .fine th {{ color:var(--muted); font-size:.7rem; text-transform:uppercase; }}
  .snip {{ color:#c9d4de; font-family:ui-monospace,Consolas,monospace; font-size:.75rem; }}
  .why {{ border-left:3px solid var(--bad); padding:8px 12px; margin:12px 0; background:rgba(248,113,113,.06); }}
  .why.ok {{ border-left-color:var(--good); background:rgba(52,211,153,.06); }}
  .why.warn {{ border-left-color:var(--mid); background:rgba(251,191,36,.06); }}
  #metrics td {{ font-size:.8rem; color:var(--muted); }}
  #metrics th {{ text-align:left; white-space:nowrap; padding-right:12px; }}
</style>
</head>
<body>
<main>
  <h1>2× DGX Spark llmtest — suite-v2.3.0-spark</h1>
  <p class="lead">hardware_sku <code>dgx-spark-gb10</code> · generated {st['generated']} UTC ·
  serving <strong>{st['serving'] or '(none)'}</strong> · {st['total_rows']} model rows · auto-refresh 20s.
  Click any cell to see what the number is and why it failed.</p>
  <p class="note">Blank is <em>not run</em>, never a zero. Infra (HTTP/connect) is excluded from
  the % denominator. Percentages only appear when a cell has finished its expected n.
  Wilson 95% is in the cell tooltip. Do not mix with the PRO-6000 scorecard.
  B3 headline is <strong>correct rate</strong> (hedge-and-not-trap / fact matched), not AND-of-every regex.
  B9 generation timeout with 0 HTML bytes is a <strong>model fail</strong>, not infra.</p>
  <div class="legend">
    <span class="sw live"></span> in progress
    <span class="sw partial"></span> partial
    <span class="sw good"></span> ≥80%
    <span class="sw mid"></span> 50–79%
    <span class="sw bad"></span> &lt;50%
    <span class="sw infra"></span> infra / no timings (not a model score)
    <span class="sw blank"></span> not run (not zero)
  </div>
  <table class="score">
    <thead><tr><th>Model</th>{heads}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>

  <div id="drill">
    <h3>Drill-down</h3>
    <p class="note">Click a score cell. Low cells open first so a 0% is never just a red number.</p>
  </div>

  <h2>What each battery actually scores</h2>
  <table class="fine" id="metrics">
    <thead><tr><th>Battery</th><th>Unit</th><th>Headline metric</th><th>How to read a low / zero</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>

  <h2>Campaign log</h2>
  <pre>{camp or '(no campaign.log yet)'}</pre>
  <p class="note">Primary three need the full pair (TP=2). Explore models wait until GLM → DeepSeek → Flash-Next finish.
  Results live in <code>~/llmtest-spark/out</code> on spark1, not <code>llmtest-v2/results/</code>.</p>
</main>
<script id="payload" type="application/json">{payload}</script>
<script>
(function(){{
  const DATA = JSON.parse(document.getElementById('payload').textContent);
  const drill = document.getElementById('drill');
  function esc(s){{
    return String(s==null?'':s).replace(/[&<>"'`]/g, c => ({{
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','`':'&#96;'
    }})[c]);
  }}
  function tiles(items){{
    return '<div class="tiles">' + items.map(([k,v,cls]) =>
      `<div class="tile ${{cls||''}}"><div class="v">${{esc(v)}}</div><div class="k">${{esc(k)}}</div></div>`
    ).join('') + '</div>';
  }}
  function reasonTable(reasons){{
    const keys = Object.keys(reasons||{{}}).sort((a,b)=>reasons[b]-reasons[a]);
    if (!keys.length) return '';
    return '<table class="fine"><thead><tr><th>Fail / bucket</th><th>n</th><th>What it means</th></tr></thead><tbody>' +
      keys.map(k => `<tr><td><code>${{esc(k)}}</code></td><td>${{reasons[k]}}</td><td>${{esc(explainReason(k))}}</td></tr>`).join('') +
      '</tbody></table>';
  }}
  function explainReason(k){{
    const map = {{
      correct: 'Matched the expected signal (hedge without asserting the trap, or the closed-domain fact).',
      fabricated: 'Asserted the planted false fact / fake CVE / fake package / fake endpoint and did not hedge.',
      empty_output: 'Zero characters of content. Often a reasoning model burning the token budget on hidden thinking.',
      ambiguous_regex_miss: 'Neither hedge nor trap/answer regex fired. First-pass proxy miss — read the snippet before calling it a hallucination.',
      hedged_but_trap_also_fired: 'Refused/hedged AND a trap keyword still matched (often quoting the fake citation while flagging it). Scored incorrect by the proxy.',
      over_hedge_on_control: 'Hedged on a knowable closed-domain control. That is over-caution, not fabrication.',
      gen_timeout_no_html: 'OpenAI chat call did not return in 600s (or the ladder ~1200s). 0 HTML bytes. Generation never finished — not a Chrome hang.',
      no_html: 'No HTML emitted.',
      runs_clean: 'Headless Chrome: loaded, painted, looped, keys wired, input-safe.',
      'drive_fail:surface,paints,loop,keys_wired': 'Page loaded but did not paint a surface / rAF loop / key handlers.',
      infra: 'HTTP/connect/timeout excluded from the model % denominator.',
      baseline_cell: 'B7 reference cell — not in the agreement %.',
      generated: 'B1 generation stored; quality is judged later.',
      throughput: 'B5 timing row. 0 tokens means the endpoint did not return llama.cpp timings.',
      det_pass: 'Every deterministic check on the row passed.',
      completed: 'Filesystem (B11) or oracle (B8) showed the work happened.',
      not_completed: 'Loop did not leave the required files / oracle failed.',
    }};
    if (map[k]) return map[k];
    if (k.startsWith('drive_fail:')) return 'HTML came back; headless Chrome failed the named drive checks.';
    if (k.startsWith('det_fail:')) return 'Named deterministic checks returned pass=false.';
    if (k.startsWith('verdict_')) return 'B10 correct_verdict on that bucket (sensitivity / specificity / decoy).';
    if (k.startsWith('agree:')) return 'B7 variant cell matched the baseline signal pattern.';
    if (k.startsWith('disagree:')) return 'B7 variant cell drifted from the baseline.';
    if (k.startsWith('infra:')) return 'Transport/HTTP — not a model quality zero.';
    return '';
  }}
  function rowTable(rows, bat){{
    if (!rows || !rows.length) return '<p class="note">No per-task rows stored for this cell.</p>';
    const failsFirst = rows.slice().sort((a,b)=> (a.passed===true)-(b.passed===true));
    const shown = failsFirst.slice(0, 80);
    const head = bat===9
      ? '<tr><th>Game</th><th>rep</th><th>result</th><th>bytes</th><th>sec</th><th>rung</th><th>tokens</th><th>detail</th></tr>'
      : bat===3
      ? '<tr><th>Task</th><th>rep</th><th>result</th><th>cat</th><th>chars</th><th>H/F/C</th><th>excerpt</th></tr>'
      : bat===5
      ? '<tr><th>Condition</th><th>decode t/s</th><th>tokens_out</th><th>TTFT ms</th><th>streams</th></tr>'
      : '<tr><th>Task</th><th>rep</th><th>result</th><th>reason</th><th>detail</th></tr>';
    const body = shown.map(r => {{
      const mark = r.infra ? 'infra' : (r.passed===true ? 'pass' : (r.passed===false ? 'FAIL' : '·'));
      if (bat===9) {{
        return `<tr><td>${{esc(r.game||r.task_id)}}</td><td>${{esc(r.run_n)}}</td><td>${{esc(mark)}}</td>
          <td>${{esc(r.html_bytes)}}</td><td>${{esc(r.seconds && r.seconds.toFixed ? r.seconds.toFixed(0) : r.seconds)}}</td>
          <td>${{esc(r.rung)}}</td><td>${{esc(r.completion_tokens)}}</td>
          <td>${{esc(r.error||r.reason)}}</td></tr>`;
      }}
      if (bat===3) {{
        const hfc = `${{r.hedged?'H':'-'}}/${{r.fabricated?'F':'-'}}/${{r.correct?'C':'-'}}`;
        return `<tr><td>${{esc(r.task_id)}}</td><td>${{esc(r.run_n)}}</td><td>${{esc(mark)}}</td>
          <td>${{esc(r.category)}}</td><td>${{esc(r.chars)}}</td><td>${{esc(hfc)}}</td>
          <td class="snip">${{esc(r.snippet||'')}}</td></tr>`;
      }}
      if (bat===5) {{
        return `<tr><td>${{esc(r.condition)}}</td><td>${{esc(r.decode_tps)}}</td><td>${{esc(r.tokens_out)}}</td>
          <td>${{esc(r.ttft_ms && r.ttft_ms.toFixed ? r.ttft_ms.toFixed(0) : r.ttft_ms)}}</td>
          <td>${{esc(r.streams_ok)}}</td></tr>`;
      }}
      return `<tr><td>${{esc(r.task_id)}}</td><td>${{esc(r.run_n)}}</td><td>${{esc(mark)}}</td>
        <td><code>${{esc(r.reason)}}</code></td>
        <td class="snip">${{esc((r.failed_checks||[]).join(', ') || r.condition || '')}}</td></tr>`;
    }}).join('');
    const more = rows.length>shown.length ? `<p class="note">${{rows.length-shown.length}} more rows not shown.</p>` : '';
    return `<table class="fine"><thead>${{head}}</thead><tbody>${{body}}</tbody></table>${{more}}`;
  }}
  function callout(bat, d, spec, pct){{
    if (bat===3) {{
      const fab = (d.b3||{{}}).fabricated || 0;
      const amb = (d.b3||{{}}).ambiguous_regex_miss || 0;
      const empty = (d.b3||{{}}).empty_output || 0;
      const trapped = (d.b3||{{}}).hedged_but_trap_also_fired || 0;
      if (pct===0) {{
        return `<div class="why"><strong>Why is this 0%?</strong> Headline is correct-rate, not “did it speak”.
          If fabricated=${{fab}} and the rest are empty/ambiguous, the model did not necessarily invent facts —
          the proxy never saw a hedge regex. Read the excerpts.</div>`;
      }}
      return `<div class="why ok"><strong>B3 is not a silent 0.</strong> Correct rate is the published number.
        Fabricated (asserted the trap with no hedge): <strong>${{fab}}</strong>.
        Empty output: <strong>${{empty}}</strong>.
        Ambiguous regex miss (often a real refusal the keyword list did not catch): <strong>${{amb}}</strong>.
        Hedged but trap keyword still matched: <strong>${{trapped}}</strong>.
        PRO-6000 median correct rate is 62% — this cell is comparable to that curve, not to 0.</div>`;
    }}
    if (bat===9) {{
      const to = (d.b9||{{}}).gen_timeout_no_html || 0;
      const drive = Object.keys(d.b9||{{}}).filter(k=>k.startsWith('drive_fail')).reduce((s,k)=>s+d.b9[k],0);
      return `<div class="why"><strong>Why is B9 ${{pct===0?'0%':(pct+'%')}}?</strong>
        ${{to}} / ${{d.n}} generations never returned HTML inside the 600s (ladder ~1200s) client timeout — 0 bytes, Unreachable.
        That is the model (or the thinking budget) not finishing a one-shot page, <em>not</em> headless Chrome wedging on an infinite loop.
        ${{drive}} page(s) did come back and then failed the drive checks (load/paint/loop/keys).
        Roster median runs_clean is 58%. A zero here is severe and this table is the answer to “how did it get 0%”.</div>`;
    }}
    if (bat===5) {{
      return `<div class="why warn"><strong>Do not publish 0 tok/s.</strong> ${{spec.roster}}</div>`;
    }}
    if (bat===4 && (!d || d.n===0)) {{
      return `<div class="why warn"><strong>Not run, not 0%.</strong> ${{spec.roster}}</div>`;
    }}
    if (pct===0) {{
      return `<div class="why"><strong>Finished at 0%.</strong> Every eligible row failed the headline check.
        The bucket table and per-task rows below are the explanation — there is no hidden infra exclusion here.</div>`;
    }}
    return '';
  }}
  function show(mid, bat){{
    document.querySelectorAll('.score td.on').forEach(el=>el.classList.remove('on'));
    const td = document.querySelector(`.score td[data-mid="${{CSS.escape(mid)}}"][data-bat="${{bat}}"]`);
    if (td) td.classList.add('on');
    const spec = (DATA.batteries||[]).find(b=>b.id===bat) || {{}};
    const d = (DATA.details||{{}})[mid+'|'+bat];
    const model = ((DATA.models||[]).find(m=>m.id===mid)||{{}}).name || mid;
    location.hash = encodeURIComponent(mid)+'|'+bat;
    if (!d) {{
      drill.innerHTML = `<h3>${{esc(model)}} · ${{esc(spec.label)}} ${{esc(spec.name)}}</h3>
        <p class="note">Not run. ${{esc(spec.blurb)}}</p>
        <p>${{esc(spec.scores)}}</p>
        <p class="note">${{esc(spec.roster)}}</p>
        ${{callout(bat, d, spec, null)}}`;
      return;
    }}
    const scored = d.pass + d.fail;
    let pct = null;
    if (spec.unit==='%' && scored>0) pct = Math.round(100*d.pass/scored);
    const lohi = (function(){{
      if (pct===null || scored<=0) return '';
      const z=1.96, p=d.pass/scored, den=1+z*z/scored;
      const c=(p+z*z/(2*scored))/den;
      const h=z*Math.sqrt(p*(1-p)/scored + z*z/(4*scored*scored))/den;
      return `Wilson 95% ${{Math.round(100*Math.max(0,c-h))}}–${{Math.round(100*Math.min(1,c+h))}}`;
    }})();
    const tileItems = [
      ['headline', spec.unit==='%' ? (pct===null? (d.n+' rows') : (pct+'%')) : (spec.unit==='t/s'?'no t/s': d.n+' gens'),
        pct===null?'': (pct>=80?'good':pct>=50?'mid':'bad')],
      ['n', d.n], ['pass', d.pass, 'good'], ['fail', d.fail, d.fail? 'bad':''],
      ['infra', d.infra, d.infra? 'mid':''], ['unscored', d.unscored],
    ];
    drill.innerHTML = `<h3>${{esc(model)}} · ${{esc(spec.label)}} ${{esc(spec.name)}}
        <span class="sub">${{esc(mid)}} · ${{esc(lohi)}}</span></h3>
      <p>${{esc(spec.blurb)}}</p>
      <p class="note"><strong>Headline:</strong> ${{esc(spec.scores)}}</p>
      ${{tiles(tileItems)}}
      ${{callout(bat, d, spec, pct)}}
      <h2>Buckets</h2>
      ${{reasonTable(d.reasons)}}
      <h2>Per-task rows (fails first)</h2>
      ${{rowTable(d.rows, bat)}}
      <p class="note">${{esc(spec.roster)}}</p>`;
    drill.scrollIntoView({{behavior:'smooth', block:'start'}});
  }}
  document.querySelectorAll('.score td[data-mid]').forEach(td => {{
    td.addEventListener('click', () => show(td.dataset.mid, parseInt(td.dataset.bat,10)));
  }});
  function openDefault(){{
    if (location.hash) {{
      const raw = decodeURIComponent(location.hash.slice(1));
      const i = raw.lastIndexOf('|');
      if (i>0) {{ show(raw.slice(0,i), parseInt(raw.slice(i+1),10)); return; }}
    }}
    // Prefer a finished <50% cell so a published 0% is never a mystery.
    const order = [];
    for (const m of (DATA.models||[])) {{
      for (const b of (DATA.batteries||[])) {{
        const d = (DATA.details||{{}})[m.id+'|'+b.id];
        if (!d || b.unit!=='%') continue;
        const elig = d.pass + d.fail;
        if (elig<=0) continue;
        const pct = 100*d.pass/elig;
        if (pct < 50) order.push([pct, m.id, b.id]);
      }}
    }}
    order.sort((a,b)=>a[0]-b[0]);
    if (order.length) show(order[0][1], order[0][2]);
  }}
  openDefault();
}})();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/michaeldeblok/llmtest-spark/out")
    ap.add_argument("--html", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    root = Path(args.root)
    st = collect(root)
    html = render(st)
    html_path = Path(args.html) if args.html else root / "progress.html"
    json_path = Path(args.json) if args.json else root / "progress.json"
    html_path.write_text(html, encoding="utf-8")
    slim = {k: v for k, v in st.items() if k != "details"}
    json_path.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    print("wrote", html_path, "rows", st["total_rows"], "serving", st["serving"],
          "running", st["running"], "cells", len(st["cells"]))


if __name__ == "__main__":
    main()
