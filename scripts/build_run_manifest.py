#!/usr/bin/env python3
"""Build a verified manifest for closing every gap in the coverage matrix.

For each model that still has missing batteries this resolves the ACTUAL file list
from the Hugging Face API rather than trusting the registry's `quant_file` - sharded
repos keep their parts in a subdirectory and a wrong filename is the single most
common way one of these runs dies twenty minutes in.

Writes plan/manifest.json:
    {models: [{id, repo, files:[{path,size}], gb, batteries:[...], vram_ok}], totals}

    python scripts/build_run_manifest.py --out plan/manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT.parent / "llm-eval-dashboard"
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

# Batteries that cannot be filled by a plain generation run, with the reason, so the
# plan never silently promises something it cannot deliver.
NEEDS_EXTRA = {
    "B1": "generation only - a judging pass is required afterwards (judge quota, not GPU)",
    "B5": "timing-authoritative: only valid on the same card class as the frozen roster",
}

# Cells that already have rows but must be REGENERATED because the artifact changed
# underneath them - a changed quant means the existing rows measure a different file,
# and the coverage matrix would show the cell green while pooling two artifacts under
# one model identity.
#
# Currently empty. abl-opus-35b-a3b was briefly moved Q3_K -> Q4_K to respect the suite's
# 4-bit floor, which would have needed its B10 re-run; that was reverted because the
# Q4_K build (21.7GB) leaves no KV headroom on the 24GB laptop this model exists to run
# on. See its registry notes for the standing caveat.
REQUANT_RERUN: dict[str, list[str]] = {}


def hf_files(repo: str, want_hint: str = "") -> list[dict]:
    """Every .gguf in the repo, with sizes. Looks in the root and one level down,
    because sharded quants live in a per-quant subdirectory."""
    out: list[dict] = []
    seen = set()

    def walk(sub=""):
        url = f"https://huggingface.co/api/models/{repo}/tree/main"
        if sub:
            url += "/" + sub
        try:
            data = json.load(urllib.request.urlopen(url, timeout=30))
        except Exception:
            return
        for e in data:
            p = e.get("path", "")
            if e.get("type") == "directory" and not sub:
                walk(p)
            elif p.lower().endswith(".gguf") and p not in seen:
                seen.add(p)
                out.append({"path": p,
                            "size": (e.get("lfs") or {}).get("size") or e.get("size") or 0})

    walk()
    return out


def pick_files(files: list[dict], quant_file: str) -> list[dict]:
    """Select the shard set matching the registry's quant_file.

    A sharded model must be fetched WHOLE - taking only the named first shard is how
    llama.cpp ends up refusing to load ("wrong number of tensors")."""
    if not files:
        return []
    base = Path(quant_file).name
    exact = [f for f in files if Path(f["path"]).name == base]
    if exact:
        f = exact[0]
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", Path(f["path"]).name)
        if m:
            stem, _, total = m.groups()
            group = [x for x in files if Path(x["path"]).name.startswith(stem + "-")
                     and re.search(r"-of-%s\.gguf$" % total, Path(x["path"]).name)]
            return sorted(group, key=lambda x: x["path"])
        return [f]
    # registry filename not found upstream - fall back to the closest quant by name
    tag = re.search(r"(UD-)?(I?Q\d[_A-Z0-9]*|MXFP4[_A-Z]*|BF16|F16)", base, re.I)
    if tag:
        cand = [f for f in files if tag.group(0).lower() in f["path"].lower()]
        if cand:
            m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", Path(cand[0]["path"]).name)
            if m:
                stem, _, total = m.groups()
                return sorted([x for x in files if Path(x["path"]).name.startswith(stem + "-")],
                              key=lambda x: x["path"])
            return [cand[0]]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plan/manifest.json")
    ap.add_argument("--vram-gb", type=float, default=96.0, help="target card VRAM")
    args = ap.parse_args()

    reg = yaml.safe_load((ROOT / "config" / "registry.yaml").read_text(encoding="utf-8"))["models"]
    data = json.loads((DASH / "data.json").read_text(encoding="utf-8"))
    phases = [p["id"] for p in data["phases"]]
    matrix = data["matrix"]

    models = []
    for mid in data["models"]:
        missing = [p for p in phases if not matrix.get(mid, {}).get(p, {}).get("tested")]
        for p in REQUANT_RERUN.get(mid, []):
            if p not in missing and p in phases:
                missing.append(p)
        missing = [p for p in phases if p in missing]     # keep canonical order
        if not missing:
            continue
        entry = reg.get(mid) or {}
        repo = entry.get("hf_repo", "")
        qf = entry.get("quant_file", "")
        files = pick_files(hf_files(repo, qf), qf) if repo else []
        gb = sum(f["size"] for f in files) / 1e9
        reg_gb = entry.get("weights_gb")
        models.append({
            "id": mid,
            "repo": repo,
            "quant_file": qf,
            "files": files,
            "gb": round(gb, 1),
            "registry_gb": reg_gb,
            "resolved": bool(files),
            # a model needs offload if weights + KV headroom exceed the card
            "fits_card": (gb + 6) <= args.vram_gb if gb else None,
            "batteries": missing,
            "notes": {b: NEEDS_EXTRA[b] for b in missing if b in NEEDS_EXTRA},
        })

    unresolved = [m["id"] for m in models if not m["resolved"]]
    mismatch = [(m["id"], m["gb"], m["registry_gb"]) for m in models
                if m["resolved"] and m["registry_gb"]
                and abs(m["gb"] - m["registry_gb"]) > max(3.0, 0.15 * m["registry_gb"])]
    oversize = [m["id"] for m in models if m["fits_card"] is False]

    out = {
        "target_card": f"{args.vram_gb:.0f}GB (RTX PRO 6000 Blackwell only)",
        "models": models,
        "totals": {
            "models": len(models),
            "download_gb": round(sum(m["gb"] for m in models), 1),
            "missing_cells": sum(len(m["batteries"]) for m in models),
        },
        "warnings": {
            "unresolved_files": unresolved,
            "size_mismatch_vs_registry": mismatch,
            "needs_offload": oversize,
        },
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"models needing work : {out['totals']['models']}")
    print(f"missing cells       : {out['totals']['missing_cells']}")
    print(f"download            : {out['totals']['download_gb']} GB")
    print(f"unresolved files    : {unresolved or 'none'}")
    print(f"size mismatch       : {mismatch or 'none'}")
    print(f"needs offload       : {oversize or 'none'}")
    print(f"-> {p}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
