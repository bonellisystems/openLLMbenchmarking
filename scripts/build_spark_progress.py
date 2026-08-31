#!/usr/bin/env python3
"""Write a self-contained progress HTML for the Spark llmtest campaign.

Reads ~/llmtest-spark/out (or --root). Never writes into llmtest-v2/results/.
Missing cells stay blank, never zero.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import defaultdict
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

BATTERIES = [
    (2, "B2 tools", 30, "formation floor"),
    (11, "B11 tool loop", 12, "filesystem"),
    (6, "B6 coding", 30, "run the code"),
    (3, "B3 halluc.", 39, "refuse vs fabricate"),
    (10, "B10 security", 66, "specificity"),
    (9, "B9 games", 24, "runs_clean"),
    (7, "B7 matrix", None, "config agreement"),
    (4, "B4 needle", 16, "16k+64k only on this GLM boot"),
    (5, "B5 decode", None, "spec on/off"),
    (1, "B1 business", 360, "gen only; judge later"),
    (8, "B8 OpenCode", 115, "23×5 after 1-task probe"),
]

EXPECTED = {b: n for b, _, n, _ in BATTERIES}


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


def row_pass(r: dict) -> tuple[bool | None, bool]:
    """Return (passed or None if unscored, is_infra)."""
    err = (r.get("error_detail") or "") or ""
    status = r.get("status")
    if status and status not in ("ok", "pass"):
        if any(x in err.lower() for x in ("timeout", "http", "conn", "reset", "400", "unreachable")):
            return None, True
        if status == "error":
            return None, True
    if "Unreachable" in err or "timed out" in err.lower() or "HTTPError" in err:
        # B9 hung gen: scored as fail (page didn't build), not missing
        if r.get("battery") == 9:
            return False, False
        return None, True
    d = r.get("det_checks") or {}
    met = r.get("metrics") or {}
    b = r.get("battery")
    if b == 9:
        return bool((d.get("runs_clean") or {}).get("pass") or met.get("runs_clean")), False
    if b == 11:
        return bool((d.get("completed") or {}).get("pass") or met.get("completed")), False
    if b == 10:
        return bool((d.get("correct_verdict") or {}).get("pass")), False
    if b in (2, 3, 6, 4, 7):
        flags = []
        for v in d.values():
            if isinstance(v, dict) and "pass" in v:
                flags.append(bool(v["pass"]))
        if flags:
            return all(flags), False
        if status == "ok":
            return True, False
        return None, False
    if b == 1:
        content = ((r.get("response_meta") or {}).get("content")) or ""
        if not content and (r.get("response_meta") or {}).get("finish_reason") == "length":
            return None, False  # empty thinking — not quality-0
        return None, False  # judged later
    if b == 5:
        return None, False
    if b == 8:
        m = met.get("completed") or (d.get("completed") or {}).get("pass")
        if m is None:
            return None, False
        return bool(m), False
    return None, False


def live_processes() -> dict:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "args"], text=True, errors="replace"
        )
    except Exception:
        return {}
    running = {}
    if re.search(r"llmtest run --battery 2", out):
        running["glm-5.3-flash-dflash2"] = 2
    if re.search(r"llmtest run --battery 3", out):
        running["glm-5.3-flash-dflash2"] = 3
    if re.search(r"llmtest run --battery 4", out):
        running["glm-5.3-flash-dflash2"] = 4
    if re.search(r"llmtest run --battery 5", out):
        running["glm-5.3-flash-dflash2"] = 5
    if re.search(r"llmtest run --battery 6", out):
        running["glm-5.3-flash-dflash2"] = 6
    if re.search(r"llmtest run --battery 7", out):
        running["glm-5.3-flash-dflash2"] = 7
    if re.search(r"llmtest run --battery 1", out):
        running["glm-5.3-flash-dflash2"] = 1
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


def collect(root: Path) -> dict:
    rows = []
    rows += load_jsonl(root / "rows-suite-v2.3.0-spark.jsonl")
    rows += load_jsonl(root / "tools" / "rows-tools.jsonl")
    rows += load_jsonl(root / "security" / "rows-security.jsonl")
    rows += load_jsonl(root / "games" / "rows-games.jsonl")
    for p in (root / "b8_glm-5.3-flash-dflash2").glob("rows-*.jsonl") if (root / "b8_glm-5.3-flash-dflash2").exists() else []:
        rows += load_jsonl(p)
    cells = defaultdict(lambda: {"n": 0, "pass": 0, "fail": 0, "infra": 0, "empty": 0, "last_ts": ""})
    for r in rows:
        mid = r.get("model_id") or ""
        if mid in ("selftest", ""):
            continue
        b = r.get("battery")
        if b is None:
            tid = r.get("task_id") or ""
            m = re.match(r"b(\d+)", tid)
            b = int(m.group(1)) if m else None
        if b is None:
            continue
        key = (mid, int(b))
        c = cells[key]
        passed, infra = row_pass(r)
        c["n"] += 1
        if infra:
            c["infra"] += 1
        elif passed is True:
            c["pass"] += 1
        elif passed is False:
            c["fail"] += 1
        ts = r.get("ts") or ""
        if ts > c["last_ts"]:
            c["last_ts"] = ts
    camp = ""
    p = root / "campaign.log"
    if p.exists():
        camp = p.read_text(encoding="utf-8", errors="replace")
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "serving": serving_model(),
        "running": live_processes(),
        "campaign_tail": "\n".join(camp.strip().splitlines()[-12:]),
        "cells": {f"{m}|{b}": v for (m, b), v in cells.items()},
        "total_rows": sum(v["n"] for v in cells.values()),
    }


def cell_html(st: dict, mid: str, bat: int, running: dict) -> str:
    exp = EXPECTED.get(bat)
    key = f"{mid}|{bat}"
    c = st["cells"].get(key)
    live = running.get(mid) == bat
    if not c and not live:
        return '<td class="blank" title="not run">—</td>'
    n = c["n"] if c else 0
    infra = c["infra"] if c else 0
    eligible = n - infra
    p = c["pass"] if c else 0
    f = c["fail"] if c else 0
    cls = "run"
    label = f"{n}"
    title = f"n={n} pass={p} fail={f} infra={infra}"
    if live:
        cls = "live"
        frac = f"{n}/{exp}" if exp else str(n)
        label = f"▶ {frac}"
    elif exp and n < exp:
        cls = "partial"
        label = f"{n}/{exp}"
    elif eligible > 0 and (p + f) == eligible:
        pct = 100.0 * p / eligible
        lo, hi = wilson(p, eligible)
        label = f"{pct:.0f}%"
        title += f" Wilson {100*lo:.0f}–{100*hi:.0f} n={eligible}"
        if infra / max(n, 1) > 0.10:
            cls = "infra"
            label = "infra"
        else:
            cls = "good" if pct >= 80 else ("mid" if pct >= 50 else "bad")
    elif n and not live:
        cls = "partial"
        label = f"n={n}"
    return f'<td class="{cls}" title="{title}">{label}</td>'


def render(st: dict) -> str:
    running = st["running"]
    rows = []
    for mid, name, topo, kind in MODELS:
        tds = "".join(cell_html(st, mid, b, running) for b, *_ in BATTERIES)
        mark = " ● serving" if st["serving"] and st["serving"] in mid or st["serving"] == mid else ""
        if st["serving"] == "glm-5.3-flash-dflash2" and mid == "glm-5.3-flash-dflash2":
            mark = " ● serving"
        elif st["serving"] and st["serving"] != mid:
            mark = ""
        live = " live" if mid in running else ""
        rows.append(
            f'<tr class="{kind}{live}"><th>{name}<div class="sub">{mid} · {topo}{mark}</div></th>{tds}</tr>'
        )
    heads = "".join(f"<th title=\"{tip}\">{lab}</th>" for _, lab, _, tip in BATTERIES)
    legend = """
    <span class="sw live"></span> in progress
    <span class="sw partial"></span> partial
    <span class="sw good"></span> ≥80%
    <span class="sw mid"></span> 50–79%
    <span class="sw bad"></span> &lt;50%
    <span class="sw infra"></span> infra (not a model score)
    <span class="sw blank"></span> not run (not zero)
    """
    camp = (st.get("campaign_tail") or "").replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="refresh" content="20" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Spark llmtest progress — suite-v2.3.0-spark</title>
<style>
  :root {{
    --bg:#0b0d10; --card:#14181e; --text:#e8edf2; --muted:#93a0ab;
    --line:rgba(255,255,255,.08); --good:#34d399; --mid:#fbbf24;
    --bad:#f87171; --live:#c084fc; --partial:#5ea0ff; --infra:#fb923c;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg);
    color:var(--text); line-height:1.4;
  }}
  main {{ max-width:1400px; margin:0 auto; padding:28px 16px 60px; }}
  h1 {{ font-size:1.6rem; margin:0 0 6px; letter-spacing:-.03em; }}
  .lead {{ color:var(--muted); margin:0 0 18px; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; background:var(--card);
    border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
  th, td {{ padding:8px 7px; border-bottom:1px solid var(--line); text-align:center; }}
  th:first-child, td:first-child {{ text-align:left; min-width:220px; }}
  thead th {{ color:var(--muted); font-weight:600; font-size:.75rem; }}
  .sub {{ color:var(--muted); font-size:.72rem; font-weight:400; }}
  tr.explore th {{ opacity:.75; }}
  tr.live {{ background:rgba(192,132,252,.08); }}
  td.blank {{ color:#3d4650; }}
  td.live {{ color:var(--live); font-weight:700; }}
  td.partial {{ color:var(--partial); }}
  td.good {{ color:var(--good); font-weight:700; }}
  td.mid {{ color:var(--mid); }}
  td.bad {{ color:var(--bad); }}
  td.infra {{ color:var(--infra); }}
  .legend {{ margin:14px 0; color:var(--muted); font-size:.85rem; display:flex; gap:14px; flex-wrap:wrap; }}
  .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
  .sw.live {{ background:var(--live); }} .sw.partial {{ background:var(--partial); }}
  .sw.good {{ background:var(--good); }} .sw.mid {{ background:var(--mid); }}
  .sw.bad {{ background:var(--bad); }} .sw.infra {{ background:var(--infra); }}
  .sw.blank {{ background:#3d4650; }}
  pre {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:12px 14px; font-size:.8rem; overflow:auto; color:var(--muted); }}
  .note {{ color:var(--muted); font-size:.88rem; max-width:90ch; }}
</style>
</head>
<body>
<main>
  <h1>2× DGX Spark llmtest — suite-v2.3.0-spark</h1>
  <p class="lead">hardware_sku <code>dgx-spark-gb10</code> · generated {st['generated']} UTC ·
  serving <strong>{st['serving'] or '(none)'}</strong> · {st['total_rows']} model rows · auto-refresh 20s</p>
  <p class="note">Blank is <em>not run</em>, never a zero. Infra (timeouts/HTTP) is excluded from
  the % denominator. Percentages only appear when a cell has finished its expected n.
  Wilson 95% is in the cell tooltip. Do not mix with the PRO-6000 scorecard.</p>
  <div class="legend">{legend}</div>
  <table>
    <thead><tr><th>Model</th>{heads}</tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <h2>Campaign log</h2>
  <pre>{camp or '(no campaign.log yet)'}</pre>
  <p class="note">Primary three need the full pair (TP=2). Explore models wait until GLM → DeepSeek → Flash-Next finish.
  Results live in <code>~/llmtest-spark/out</code> on spark1, not <code>llmtest-v2/results/</code>.</p>
</main>
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
    json_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
    print("wrote", html_path, "rows", st["total_rows"], "serving", st["serving"], "running", st["running"])


if __name__ == "__main__":
    main()
