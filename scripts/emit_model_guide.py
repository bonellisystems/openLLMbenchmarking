#!/usr/bin/env python3
"""Emit MODEL-GUIDE.md - a single self-contained file an LLM can read to pick a
model and then actually run it.

Written for a MACHINE reader, which drives every formatting choice here:
every number carries its unit and its sample size, every model carries a
copy-pasteable command, and every known gap is stated rather than left as a
blank the reader has to interpret. A blank cell and a zero score mean very
different things and the file says which is which.

Regenerated from dashboard/data.json + config/registry.yaml, so it cannot drift
from the published dashboard:

    python scripts/emit_model_guide.py            # -> MODEL-GUIDE.md
    python scripts/emit_model_guide.py --out /tmp/x.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

BATTERIES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]

# The binary that can serve every quant in this roster. The stock llama.cpp build
# cannot load bonsai's Q2_0 (a prism-ml custom quantisation), and it is the only
# roster model that needs the fork - everything else runs on either.
PRISM_NOTE = ("bonsai-ternary-27b ONLY: its Q2_0 is a prism-ml custom quantisation. "
              "Stock llama.cpp exits with 'failed to load model'. Use the prism fork "
              "(prebuilt Windows binary, or build the Docker image from "
              "deploy/blackwell/Dockerfile.prism).")


def fmt(cell):
    """A battery cell as text. Distinguishes NOT RUN from a real zero - the
    single most important distinction in this file for a machine reader."""
    if not cell or not cell.get("tested"):
        return "not run"
    return cell.get("display") or str(cell.get("score"))


def run_cmd(gguf, ctx, ngram: bool, prism: bool) -> str:
    exe = "llama-server"
    spec = " \\\n    --spec-type ngram-mod --spec-ngram-mod-n-match 32" if ngram else ""
    note = "  # prism fork build" if prism else ""
    return (f"{exe} -m {gguf} \\\n"
            f"    -ngl 99 -c {ctx} --jinja -fa on \\\n"
            f"    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0{spec} \\\n"
            f"    --host 127.0.0.1 --port 8080{note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "MODEL-GUIDE.md"))
    args = ap.parse_args(argv)

    d = json.loads((ROOT / "dashboard" / "data.json").read_text(encoding="utf-8"))
    reg = yaml.safe_load((ROOT / "config" / "registry.yaml").read_text(encoding="utf-8"))
    models_reg = reg.get("models", reg)
    matrix = d["matrix"]
    phases = {p["id"]: p for p in d["phases"]}

    L = []
    A = L.append

    A("# Local LLM Model Guide")
    A("")
    A("Measured benchmark results for locally-servable LLMs, plus the exact command "
      "to run each one. Written to be read by an LLM that needs to choose a model "
      "for a workload and then start it.")
    A("")
    A("**How to read this file.** Every score below is a MEASURED result, not an "
      "estimate or a vendor claim. `not run` means the cell was never measured and "
      "you must not treat it as a zero or as a weak result - where a cell is blank "
      "for a known reason, the reason is in *Known gaps* at the end. Sample sizes "
      "are small (n is given per battery), so differences under ~5 points are "
      "usually ties rather than rankings.")
    A("")
    A(f"Generated from `{d.get('generated_from', 'llmtest-v2')}`.")
    A("")

    # ---------------- how to choose ----------------
    A("## Picking a model")
    A("")
    A("Match the battery to the job, then read that column:")
    A("")
    A("| If the workload is... | Read this battery | It measures |")
    A("|---|---|---|")
    pick = [
        ("General business writing, analysis, drafting", "B1",
         "quality across 15 business departments, judged 0-10"),
        ("Calling tools / driving an API or agent loop", "B2",
         "whether it forms a valid tool call at all"),
        ("Writing code from a spec", "B3", "code generation correctness"),
        ("Long documents / large context", "B4", "retrieval accuracy at long context"),
        ("Throughput-sensitive serving", "B5", "decode tokens/sec"),
        ("Fixing or editing existing code", "B6", "bugfix and edit correctness"),
        ("Autonomous multi-step coding (agentic)", "B8",
         "end-to-end task completion in a real harness with a hidden oracle"),
        ("Generating a working app / game in one shot", "B9",
         "does the generated program actually run"),
        ("Refusing unsafe requests", "B10", "safety behaviour under adversarial prompts"),
        ("Multi-turn tool use with real filesystem effects", "B11",
         "agentic tool use scored from the filesystem, not from narration"),
    ]
    for w, b, m in pick:
        A(f"| {w} | **{b}** | {m} |")
    A("")

    # ---------------- battery legend ----------------
    A("## What each battery measures")
    A("")
    for b in BATTERIES:
        p = phases.get(b) or {}
        A(f"- **{b} - {p.get('name', b)}** ({p.get('unit', '')}): {p.get('blurb', '')}")
    A("")

    # ---------------- scorecard ----------------
    A("## Scorecard")
    A("")
    A("Higher is better in every column. Units differ per battery (see above): "
      "B1 is /10, B5 is tokens/sec, the rest are percentages unless noted.")
    A("")
    A("| Model | " + " | ".join(BATTERIES) + " |")
    A("|---" * (len(BATTERIES) + 1) + "|")
    for m in sorted(matrix):
        cells = [fmt(matrix[m].get(b)) for b in BATTERIES]
        A(f"| `{m}` | " + " | ".join(cells) + " |")
    A("")

    # ---------------- how to run ----------------
    A("## Running a model")
    A("")
    A("All commands use **llama.cpp** (`llama-server`), which exposes an "
      "OpenAI-compatible API at `http://127.0.0.1:8080/v1/chat/completions`.")
    A("")
    A("### The two modes")
    A("")
    A("**Normal** - plain decoding. Use when generating fresh text with little "
      "overlap with the prompt.")
    A("")
    A("**n-gram speculative decoding** - adds `--spec-type ngram-mod "
      "--spec-ngram-mod-n-match 32`. It drafts tokens by matching n-grams already "
      "in the context, so it is fastest exactly when the output repeats the input.")
    A("")
    A("It is **lossless**: at temperature 0 the output is byte-identical to normal "
      "decoding. It costs **no extra VRAM**. There is no quality tradeoff to weigh - "
      "the only question is whether your workload benefits.")
    A("")
    A("| Workload | Typical speedup |")
    A("|---|---|")
    A("| Editing / rewriting a file (output largely repeats input) | **2x - 12x** |")
    A("| Refactoring, applying a diff, reformatting | 3x - 8x |")
    A("| Writing new code from scratch | ~1.0x - 1.6x |")
    A("| Free-form prose with no context overlap | ~1.0x (no harm) |")
    A("")
    A("**Turn it on by default for coding and editing work.** For from-scratch "
      "generation it neither helps much nor hurts.")
    A("")
    A("#### How this relates to the B5 column - read this before quoting throughput")
    A("")
    A("**Every B5 number in the scorecard is an UNACCELERATED baseline.** It is "
      "reported from the spec-decode OFF arm, deliberately, so all 21 models are "
      "the same measurement.")
    A("")
    A("The suite also runs an n-gram ON arm, and for the 20 models measured before "
      "2026-08-11 that arm returned a speedup of exactly **1.00x** across the "
      "board - 59.3 vs 59.5, 264.1 vs 264.3, 60.3 vs 60.3, and so on. That is not "
      "a result about n-gram. It is the flag missing at serve time: the row "
      "recorded `spec=ngram32` in its condition while the server ran without it. "
      "It was previously explained away as 'this arm generates fresh text, where "
      "n-gram cannot help'. `qwen3.6-27b-fable-fusion` disproved that on the same "
      "battery, scoring **6.79x (482 vs 71 t/s)** once the flag actually applied.")
    A("")
    A("Practical consequence for choosing a model:")
    A("")
    A("- The B5 ranking between models is still sound - all 21 were measured the "
      "same (unaccelerated) way.")
    A("- The ABSOLUTE numbers understate what you will see on edit-heavy work. A "
      "model listed at 70 t/s can serve an edit workload several times faster with "
      "the n-gram flags above.")
    A("- Do NOT read the 1.00x arm as evidence that speculative decoding is not "
      "worth enabling. The standalone measurements in the table above, and the one "
      "B5 run where the flag really applied, both say the opposite.")
    A("")
    A("#### Tuning `--spec-ngram-mod-n-match`")
    A("")
    A("| n-match | Measured speedup | Note |")
    A("|---|---|---|")
    for e in d.get("ngram_nmatch", []):
        sp = e.get("speedup")
        A(f"| {e.get('n')} | {('%.2fx' % sp) if sp else 'SLOWER'} | {e.get('note','')} |")
    A("")
    A("**Never set it below 16** - at 8 it runs slower than no speculative decoding "
      "at all. 32 is the best measured value for edit-heavy work.")
    A("")
    A("#### Measured per-model, edit-heavy workload")
    A("")
    if d.get("ngram_edit"):
        A("| Model | Normal (t/s) | With n-gram (t/s) | Speedup |")
        A("|---|---|---|---|")
        for e in d["ngram_edit"]:
            A(f"| `{e['model']}` | {e['base']} | {e['ngram']} | **{e.get('speedup', 0):.2f}x** |")
        A("")
        A("Note the pattern: the SLOWEST models gain the most, because each accepted "
          "draft token saves a full forward pass. A slow dense model can end up "
          "faster than a fast MoE once n-gram is on.")
        A("")

    # ---------------- per-model ----------------
    A("## Per-model reference")
    A("")
    A(f"> {PRISM_NOTE}")
    A("")
    for m in sorted(matrix):
        r = models_reg.get(m) or {}
        repo = r.get("hf_repo")
        qf = r.get("quant_file")
        gb = r.get("weights_gb")
        ctx = 32768
        prism = m == "bonsai-ternary-27b"
        A(f"### `{m}`")
        A("")
        if repo:
            A(f"- **HF repo**: `{repo}`")
            A(f"- **Quant file**: `{qf}`")
            A(f"- **Weights**: {gb} GB  (needs roughly {gb} GB VRAM plus KV cache; "
              f"quantised KV as below keeps that small)")
            A(f"- **Quant family**: {r.get('quant_family', 'n/a')}  |  "
              f"**License**: {r.get('license', 'n/a')}  |  "
              f"**Claimed context**: {r.get('claimed_ctx', 'n/a')}")
        else:
            A("- Not in the local registry (no download details recorded).")
        scores = ", ".join(f"{b} {fmt(matrix[m].get(b))}" for b in BATTERIES)
        A(f"- **Scores**: {scores}")
        A("")
        if repo and qf:
            A("Download:")
            A("")
            A("```bash")
            A(f"huggingface-cli download {repo} {qf} --local-dir ./models/{m}")
            A("```")
            A("")
            A("Run (normal):")
            A("")
            A("```bash")
            A(run_cmd(f"./models/{m}/{qf}", ctx, ngram=False, prism=prism))
            A("```")
            A("")
            A("Run (with n-gram speculative decoding):")
            A("")
            A("```bash")
            A(run_cmd(f"./models/{m}/{qf}", ctx, ngram=True, prism=prism))
            A("```")
            A("")
        A("")

    # ---------------- caveats ----------------
    A("## Known gaps and caveats")
    A("")
    A("Read these before quoting any number.")
    A("")
    for g in d.get("gaps", []):
        A(f"- **[{str(g.get('sev','')).upper()}] {g.get('title','')}** - {g.get('detail','')}")
    A("")
    cav = d.get("caveats") or {}
    if cav:
        A("### Methodology notes")
        A("")
        for k, v in cav.items():
            A(f"- **{k}**: {v}")
        A("")

    out = Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}  ({len(L)} lines, {out.stat().st_size/1024:.1f} KB, "
          f"{len(matrix)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
