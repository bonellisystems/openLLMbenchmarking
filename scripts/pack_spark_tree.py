#!/usr/bin/env python3
"""Build a slim tarball of llmtest-v2 for the Sparks. Never includes results/."""
from __future__ import annotations

import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(r"D:\BUILT-TOOLS\LLMtesting\llmtest-v2-spark.tgz")
SKIP_DIR_NAMES = {
    "results", "artifacts", "_archive", "__pycache__", "llmtest.egg-info",
    "plan", "plan_vm", "plan_bonsai", "scratchpad",
}
SKIP_PREFIXES = ("results_",)


def keep(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(p in SKIP_DIR_NAMES for p in parts):
        return False
    if parts and any(parts[0].startswith(s) for s in SKIP_PREFIXES):
        return False
    if path.suffix in {".pyc", ".png"}:
        return False
    return True


def main() -> None:
    n = 0
    with tarfile.open(OUT, "w:gz") as tf:
        for p in ROOT.rglob("*"):
            if not p.is_file():
                continue
            if not keep(p):
                continue
            tf.add(p, arcname=str(Path("llmtest-v2") / p.relative_to(ROOT)))
            n += 1
    print("files", n, "out", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
