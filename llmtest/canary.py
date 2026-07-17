"""Serving canary (TESTPLAN 7.5): Ornith ngram edit A/B vs reference band. Re-runnable health check."""
from pathlib import Path


def evaluate_canary(base_tps: float, ngram_tps: float, band: dict) -> tuple[bool, str]:
    ratio = ngram_tps / base_tps if base_tps else 0.0
    ok = band["min_speedup"] <= ratio <= band["max_speedup"]
    return ok, (f"canary: base={base_tps:.1f} t/s ngram={ngram_tps:.1f} t/s "
                f"speedup={ratio:.2f}x band=[{band['min_speedup']},{band['max_speedup']}] "
                f"{'PASS' if ok else 'FAIL'}")


def _edit_prompt(src: Path) -> str:
    return ("You are a code editor. Output the ENTIRE file again exactly as-is, "
            "inserting `<!-- MODIFIED -->` right after the opening <body> tag. "
            "Output only the full HTML. /no_think\n\n"
            + src.read_text(encoding="utf-8"))


def _decode_tps(handle, prompt: str) -> float:
    d = handle.chat([{"role": "user", "content": prompt}],
                    max_tokens=4096, temperature=0.0)
    return float(d.get("timings", {}).get("predicted_per_second", 0.0))


def run_canary(root: str | Path = ".") -> int:
    from llmtest.registry import load_config
    from llmtest.server import ServerManager
    from llmtest.store import Store
    root = Path(root).resolve()
    cfg = load_config(root)
    can = cfg.runtime_pins["canary"]
    prompt = _edit_prompt(Path(can["edit_source"]))
    mgr = ServerManager(cfg, Store(root / "results"))
    try:
        base = _decode_tps(mgr.request_endpoint(can["model_id"], ctx=10240,
                                                flags_overlay={"spec": "off"}), prompt)
        ngram = _decode_tps(mgr.request_endpoint(can["model_id"], ctx=10240), prompt)
    finally:
        mgr.teardown()
    ok, msg = evaluate_canary(base, ngram, can["reference"])
    print(msg)
    return 0 if ok else 1
