#!/usr/bin/env python3
import json
from pathlib import Path
root = Path("/home/michaeldeblok/llmtest-spark/out")
for p in sorted(root.rglob("*.jsonl")):
    lines = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    print("FILE", p, "n", len(lines))
    if not lines:
        continue
    r = json.loads(lines[0])
    print("  keys", sorted(r.keys())[:40])
    print("  sample", {k: r.get(k) for k in ("battery","model_id","task_id","status","run_n","hardware_sku") if k in r})
    if "det_checks" in r:
        print("  det_checks", list((r.get("det_checks") or {}).keys())[:12])
    if "metrics" in r:
        print("  metrics", r.get("metrics"))
    last = json.loads(lines[-1])
    print("  last task", last.get("task_id"), last.get("status") or last.get("pass") or last.get("result"))
print("B9LOG")
print((root/"b9.log").read_text()[-1500:] if (root/"b9.log").exists() else "none")
