#!/usr/bin/env python3
"""Wait until /v1/models lists the expected id (or any model if expected empty)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
expect = sys.argv[2] if len(sys.argv) > 2 else ""
timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 2700
url = base.rstrip("/") + "/v1/models"
deadline = time.time() + timeout
last = ""
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.load(r)
        ids = [m.get("id") for m in data.get("data") or []]
        last = ",".join(ids)
        if ids and (not expect or expect in ids):
            print("READY", last)
            raise SystemExit(0)
        print("waiting ids", last)
    except SystemExit:
        raise
    except Exception as e:
        print("waiting", type(e).__name__)
    time.sleep(10)
print("TIMEOUT", last)
raise SystemExit(2)
