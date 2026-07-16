"""Byte-deterministic tables (TESTPLAN 7.5): pure functions of rows, stable sorts, fixed float fmt, LF newlines."""
from pathlib import Path

from llmtest.store import Store


def render_serving_table(rows: list[dict]) -> str:
    keep = [r for r in rows
            if r.get("timing_authoritative") and r.get("status") == "ok"
            and "non-reportable" not in r.get("tags", [])]
    keep.sort(key=lambda r: (r["model_id"], r["condition"]))
    lines = ["# Serving (Battery 5) — timing_authoritative rows only", "",
             "| Model | Condition | decode t/s | PP t/s | TTFT ms |",
             "|---|---|---|---|---|"]
    for r in keep:
        m = r["response_meta"]
        lines.append(f"| {r['hf_repo']} | {r['condition']} | {m.get('decode_tps', 0):.1f} "
                     f"| {m.get('pp_tps', 0):.1f} | {m.get('ttft_ms', 0):.0f} |")
    return "\n".join(lines) + "\n"


def run_tables(root: str | Path = ".") -> int:
    root = Path(root).resolve()
    rows = list(Store(root / "results").iter_rows())
    out = root / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "serving.md").write_text(render_serving_table(rows),
                                    encoding="utf-8", newline="\n")
    print(f"tables: wrote serving.md from {len(rows)} rows")
    return 0
