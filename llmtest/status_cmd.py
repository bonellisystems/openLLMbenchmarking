"""llmtest status — done counts from resume keys (pending matrix arrives with plan() in Task 12)."""
from collections import Counter
from pathlib import Path

from llmtest.store import Store


def run_status(root: str | Path = ".") -> int:
    counts = Counter()
    for r in Store(Path(root).resolve() / "results").iter_rows():
        counts[(r["battery"], r["model_id"], r["status"])] += 1
    if not counts:
        print("status: no rows yet")
        return 0
    for (battery, model, status), n in sorted(counts.items()):
        print(f"B{battery} {model:24s} {status:8s} {n}")
    return 0
