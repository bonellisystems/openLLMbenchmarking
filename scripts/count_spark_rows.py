#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

p = Path("/home/michaeldeblok/llmtest-spark/out/rows-suite-v2.3.0-spark.jsonl")
if not p.exists():
    print("NO_ROWS")
    raise SystemExit(0)
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print("n", len(rows))
print("status", Counter(r.get("status") for r in rows))
print("models", Counter(r.get("model_id") for r in rows))
print("batteries", Counter(r.get("task_id", "").split(".")[0] for r in rows))
if rows:
    last = rows[-1]
    print("last", last.get("task_id"), last.get("status"), str(last.get("error_detail") or "")[:160])
    print("sku_sessions_check skip")
