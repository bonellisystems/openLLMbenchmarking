#!/usr/bin/env python3
"""Build morning.html from every hour-*/hour.json under --root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cell(b):
    if not b:
        return "—"
    if b.get("comply_rate") is not None:
        return f"{b['comply_rate']}% comply"
    if b.get("pct") is None:
        return "—"
    return f"{b['pct']:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/michaeldeblok/llmtest-spark/out")
    ap.add_argument("--html", default="")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for p in sorted(root.glob("hour-*/hour.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        bats = d.get("batteries") or {}
        c1 = (d.get("c1") or {}).get("decode_tok_s")
        cmax = d.get("cmax") or {}
        job = bats.get("job") or {}
        b9 = bats.get("b9") or {}
        times = ", ".join(
            f"{(x.get('id') or '').split('.')[-1]} {x.get('e2e_s')}s"
            for x in (b9.get("rows") or [])
        )
        rows.append({
            "model": d.get("model"),
            "wall": d.get("wall_s"),
            "c1": c1,
            "cmax": cmax.get("agg_tok_s"),
            "threads": cmax.get("threads"),
            "intel": cell(bats.get("intel")),
            "b1": cell(bats.get("b1")),
            "b2": cell(bats.get("b2")),
            "b3": cell(bats.get("b3")),
            "b6": cell(bats.get("b6")),
            "job": cell(job),
            "b9": cell(b9) + (f" ({times})" if times else ""),
            "b11": cell(bats.get("b11")),
            "b12": cell(bats.get("b12")),
            "path": str(p.parent / "hour.html"),
        })
    out = Path(args.html) if args.html else root / "morning.html"
    body = "".join(
        f"<tr><th>{r['model']}</th><td>{r['c1']}</td><td>{r['cmax']}@{r['threads']}</td>"
        f"<td>{r['intel']}</td><td>{r['b1']}</td><td>{r['b2']}</td><td>{r['b3']}</td>"
        f"<td>{r['b6']}</td><td>{r['job']}</td><td>{r['b9']}</td><td>{r['b11']}</td>"
        f"<td>{r['b12']}</td><td>{r['wall']}s</td></tr>"
        for r in rows
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Spark-hour overnight</title>
<style>
body{{background:#0b0d10;color:#e8edf2;font-family:Segoe UI,system-ui,sans-serif;margin:24px}}
table{{border-collapse:collapse;width:100%;background:#14181e}}
th,td{{border-bottom:1px solid #222;padding:8px;text-align:left;font-size:13px}}
</style></head><body>
<h1>Spark-hour overnight</h1>
<p>Ashby clone: Engineering - Internal AI Transformation @ ElevenLabs.
Nothing submitted off-box.</p>
<table><thead><tr>
<th>Model</th><th>C1</th><th>cMAX agg</th><th>Intel</th><th>B1</th><th>B2</th>
<th>B3</th><th>B6</th><th>Job</th><th>B9 time-to-HTML</th><th>B11</th><th>B12</th><th>wall</th>
</tr></thead><tbody>{body or '<tr><td>no hour.json yet</td></tr>'}</tbody></table>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    print("wrote", out, "models", len(rows))


if __name__ == "__main__":
    main()
