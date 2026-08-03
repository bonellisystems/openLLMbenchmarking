#!/usr/bin/env python3
"""Generate explorer/ — a browsable record of every question, every answer, every
piece of code the models wrote, and the playable games, with a grading UI.

Why data lands in .js files rather than .json: Chrome blocks fetch() on file://,
so a JSON-fetching page would be dead when opened by double-click. Script tags are
not blocked, so each payload is emitted as `EXPL_LOAD("<model>", {...});` and pulled
in on demand by injecting a <script>. Works identically from file:// and Pages.

    python build_explorer.py            # regenerate explorer/
    python build_explorer.py --limit 2  # only 2 models, for a fast check
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Works in BOTH layouts: as a sibling checkout during local development
# (../llmtest-v2), and as a subdirectory of the published repo, where the suite lives in
# the parent. Resolved by looking for the results tree rather than by assuming a name, so
# neither layout needs a code change.
REPO = ROOT.parent / "llmtest-v2" if (ROOT.parent / "llmtest-v2" / "results").is_dir()     else ROOT.parent
# Ad-hoc playable game builds live in the WORKSPACE (D:\...\LLMtesting\michael), which
# sits beside the suite repo — one level up in the sibling layout, two in the published
# layout. On a fresh clone of the public repo neither exists, and then the games already
# COMMITTED under explorer/games/ are the only source; the emit step below must never
# destroy them (it did once: rmtree(OUT) wiped all 19 and an absent GAMES_SRC restored
# nothing).
_GAMES_CANDIDATES = (ROOT.parent / "michael", ROOT.parent.parent / "michael")
GAMES_SRC = next((p for p in _GAMES_CANDIDATES if p.is_dir()), _GAMES_CANDIDATES[0])
OUT = ROOT / "explorer"
DATA = OUT / "data"

MAX_ANSWER = 40_000       # per-answer cap so one runaway response can't bloat a bundle
MAX_CODE_FILE = 20_000
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def read_text(p: Path, cap=MAX_ANSWER):
    try:
        t = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    return t[:cap] + ("\n\n[... truncated ...]" if len(t) > cap else "")


def load_rows():
    """Every CURRENT row across all sources, selected through llmtest.rowselect so the
    explorer never shows superseded wrong-hardware runs beside their replacements.
    Withdrawn cells simply have no runs listed until their v2.2.0 re-run lands —
    exactly matching what the dashboard's matrix says about them."""
    import sys
    sys.path.insert(0, str(REPO))
    from llmtest.rowselect import (effective_custom_rows, effective_suite_rows,
                                   load_superseded, suite_cell_allowed, version_tuple)
    sup = load_superseded(REPO)

    suite = []
    for p in sorted((REPO / "results").glob("rows-suite-v2.*.jsonl")):
        if "shakedown" in p.name:
            continue
        for line in p.open(encoding="utf-8"):
            try:
                suite.append(json.loads(line))
            except Exception:
                continue
    # B8 dir rows join the suite pool BEFORE selection so per-cell max-version is
    # computed across both of B8's sources (same reasoning as build_data's take_b8).
    for p in sorted(REPO.glob("results_b8_*/*.jsonl")):
        for line in p.open(encoding="utf-8"):
            try:
                suite.append(json.loads(line))
            except Exception:
                continue
    cell_max = {}
    for r in suite:
        key = (r.get("model_id"), r.get("battery"))
        v = version_tuple(r.get("suite_version"))
        if v > cell_max.get(key, (-1,)):
            cell_max[key] = v
    # count_check off: this pool mixes shard + dirs copies of the same rows, so the
    # per-source n_expected bookkeeping lives with build_data; drift still fails the
    # build there before the explorer would ship anything stale.
    rows = [r for r in effective_suite_rows(suite, sup, count_check=False)
            if suite_cell_allowed(sup, r.get("model_id"), r.get("battery") or 0,
                                  cell_max.get((r.get("model_id"), r.get("battery"))))]

    # the newer batteries live in their own shards
    for extra, name in (("results_games/rows-games.jsonl", "rows-games.jsonl"),
                        ("results_security/rows-security.jsonl", "rows-security.jsonl"),
                        ("results_tools/rows-tools.jsonl", "rows-tools.jsonl")):
        p = REPO / extra
        if not p.exists():
            continue
        got = []
        for line in p.open(encoding="utf-8"):
            try:
                got.append(json.loads(line))
            except Exception:
                continue
        rows.extend(effective_custom_rows(got, sup, name, count_check=False))
    return rows


def load_prompts():
    """task_id -> prompt text, straight from the suite fixtures (rows don't carry
    the request; it's stripped for blinding)."""
    import sys
    sys.path.insert(0, str(REPO))
    cwd = os.getcwd()
    os.chdir(REPO)
    out = {}
    try:
        from llmtest.registry import load_config
        cfg = load_config(Path("."))
        try:
            from llmtest.batteries import b1_fixtures
            for unit in cfg.suite["b1"]["units_tier1"]:
                for t in b1_fixtures.load_unit_tasks(Path("."), unit):
                    out[f"b1.{t.id}"] = t.prompt
        except Exception as e:
            print("  ! b1 prompts:", e)
        for mod, pref in (("b2_fixtures", "b2"), ("b3_fixtures", "b3"),
                          ("b6_fixtures", "b6")):
            try:
                m = __import__(f"llmtest.batteries.{mod}", fromlist=["load_tasks"])
                for t in m.load_tasks(Path(".")):
                    out[f"{pref}.{t.id}"] = getattr(t, "prompt", "") or ""
            except Exception as e:
                print(f"  ! {pref} prompts:", e)
        try:
            from llmtest.batteries import b7_fixtures
            for t in b7_fixtures.load_probe_tasks(Path(".")):
                out[f"b7.{t.id}"] = getattr(t, "prompt", "") or ""
        except Exception as e:
            print("  ! b7 prompts:", e)
        try:
            from llmtest.batteries import b4_fixtures
            for t in b4_fixtures.load_longcontext_tasks(Path(".")):
                p = getattr(t, "prompt", "") or getattr(t, "question", "") or ""
                out[f"b4.{t.id}"] = (p[:2000] + "\n\n[haystack body omitted]") if p else \
                    "(needle-in-haystack: prompt is a generated haystack, omitted)"
        except Exception as e:
            print("  ! b4 prompts:", e)
        try:
            from llmtest.harness.tasks import load_b8_tasks
            for t in load_b8_tasks(Path(".")):
                out[f"b8.{t.id}"] = getattr(t, "prompt", "") or ""
        except Exception as e:
            print("  ! b8 prompts:", e)
    finally:
        os.chdir(cwd)
    return out


def b8_workspace_index():
    """Workspace dirs are named b8.<task>-run<N>-<8hex>, where the hex is a temp-dir
    id that appears nowhere in the rows - so a workspace cannot be linked to a row
    directly. The sweep ran one model at a time, though, so mtime falls inside
    exactly one model's run window. That attribution is INFERRED and labelled as
    such; the one ambiguous stretch (the original gpt-oss + gemma run, which
    interleaved) is left unattributed."""
    d = REPO / "artifacts" / "b8_workspaces"
    if not d.is_dir():
        return {}
    out = collections.defaultdict(list)
    for entry in os.scandir(d):
        if not entry.is_dir():
            continue
        m = re.match(r"(b8\.[a-z0-9\-]+)-run(\d+)-([0-9a-f]+)$", entry.name)
        if not m:
            continue
        files = []
        for r, dirs, fs in os.walk(entry.path):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for fn in fs:
                if fn.endswith((".pyc", ".pyo")) or fn.startswith(".opencode"):
                    continue
                fp = Path(r) / fn
                rel = str(fp.relative_to(entry.path)).replace("\\", "/")
                files.append({"path": rel, "text": read_text(fp, MAX_CODE_FILE)})
        if files:
            out[(m.group(1), int(m.group(2)))].append(
                {"ws": entry.name, "mtime": entry.stat().st_mtime,
                 "files": sorted(files, key=lambda f: f["path"])})
    return out


def model_windows(rows):
    win = {}
    for r in rows:
        if r.get("battery") != 8 or not r.get("ts"):
            continue
        try:
            t = datetime.datetime.fromisoformat(r["ts"].replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        w = win.setdefault(r["model_id"], [t, t])
        w[0] = min(w[0], t)
        w[1] = max(w[1], t)
    # any model whose window overlaps another's cannot be attributed by time
    amb = set()
    items = sorted(win.items(), key=lambda kv: kv[1][0])
    for i, (m1, w1) in enumerate(items):
        for m2, w2 in items[i + 1:]:
            if w1[1] > w2[0]:
                amb.add(m1)
                amb.add(m2)
    return win, amb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("loading rows ...")
    rows = load_rows()
    print(f"  {len(rows)} rows")
    print("loading prompts from suite fixtures ...")
    prompts = load_prompts()
    print(f"  {len(prompts)} prompts")
    print("indexing B8 workspaces (the code the agent left behind) ...")
    ws = b8_workspace_index()
    print(f"  {len(ws)} (task,run) groups")
    win, amb = model_windows(rows)

    art_roots = [REPO / "artifacts", REPO / "results" / "artifacts"]

    def answer_for(r):
        a = (r.get("artifacts") or {}).get("response")
        if not a or "relpath" not in a:
            return None
        for base in art_roots:
            p = base / a["relpath"]
            if p.exists():
                return read_text(p)
        return None

    by_model = collections.defaultdict(list)
    for r in rows:
        b = r.get("battery")
        if not b:
            continue
        mid = r["model_id"]
        tid = r.get("task_id", "")
        met = r.get("metrics") or {}
        det = r.get("det_checks") or {}
        checks = [v["pass"] for v in det.values() if isinstance(v, dict) and "pass" in v]
        item = {
            "b": b, "task": tid, "run": r.get("run_n"),
            "cond": r.get("condition", ""),
            "ok": (all(checks) if checks else None),
            "m": {k: met[k] for k in
                  ("completion", "steps", "terminal_status", "tokens_completion",
                   "tokens_prompt", "budget_exceeded", "correct", "fabricated", "hedged",
                   "decode_tps", "pp_tps", "needle_recall", "chars", "category",
                   "difficulty", "subagent_spawned") if k in met},
        }
        if b == 8:
            key = (tid, r.get("run_n"))
            cands = ws.get(key, [])
            w = win.get(mid)
            picked = None
            if w and mid not in amb:
                inw = [c for c in cands if w[0] - 300 <= c["mtime"] <= w[1] + 300]
                if len(inw) == 1:
                    picked = inw[0]
                elif inw:
                    picked = min(inw, key=lambda c: abs(c["mtime"] - (w[0] + w[1]) / 2))
            if picked:
                item["files"] = picked["files"]
                item["files_inferred"] = True
                item["ws"] = picked["ws"]
        else:
            txt = answer_for(r)
            if txt is not None:
                item["ans"] = txt
        by_model[mid].append(item)

    # ---- emit ----
    # Wipe ONLY the generated data dir. explorer/games/ holds committed artifacts that
    # can only be re-sourced on machines that have the workspace dir — an rmtree(OUT)
    # here once destroyed all 19 playable games on a checkout without it.
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    (OUT / "games").mkdir(parents=True, exist_ok=True)

    games = []
    if GAMES_SRC.is_dir():
        for p in sorted(GAMES_SRC.glob("*.html")):
            if p.name == "index.html":
                continue
            shutil.copy2(p, OUT / "games" / p.name)
            mt = re.match(r"(snake|tetris|arkanoid|flappy)[_-]?(.*)\.html$", p.name, re.I)
            games.append({"file": p.name,
                          "game": (mt.group(1).title() if mt else p.stem),
                          "model": (mt.group(2) or "?") if mt else "?",
                          "kb": round(p.stat().st_size / 1024, 1)})

    models = sorted(by_model)
    if args.limit:
        models = models[:args.limit]

    index = {"models": [], "prompts": prompts, "games": games,
             "batteries": {"9": "Game Builds", "10": "Security Review", "11": "Tool Loop",
                           "1": "Business Scorecard", "2": "Tool Calling",
                           "3": "Hallucination", "4": "Long Context", "5": "Serving",
                           "6": "Agentic Coding", "7": "Reproducibility",
                           "8": "Agentic Harness"},
             "ambiguous_attribution": sorted(amb)}
    for m in models:
        items = by_model[m]
        counts = collections.Counter(i["b"] for i in items)
        index["models"].append({
            "id": m, "slug": slug(m), "n": len(items),
            "counts": {str(k): v for k, v in sorted(counts.items())},
            "with_answer": sum(1 for i in items if i.get("ans")),
            "with_code": sum(1 for i in items if i.get("files")),
        })
        payload = json.dumps({"items": items}, ensure_ascii=True)
        (DATA / f"{slug(m)}.js").write_text(
            f'EXPL_LOAD({json.dumps(m)}, {payload});\n', encoding="utf-8")
        print(f"  {m}: {len(items)} items, "
              f"{sum(1 for i in items if i.get('ans'))} answers, "
              f"{sum(1 for i in items if i.get('files'))} code sets, "
              f"{(DATA / (slug(m) + '.js')).stat().st_size/1e6:.1f} MB")

    (DATA / "index.js").write_text(
        f"EXPL_INDEX = {json.dumps(index, ensure_ascii=True)};\n", encoding="utf-8")

    shell = ROOT / "explorer_shell.html"
    if shell.exists():
        shutil.copy2(shell, OUT / "index.html")
        print("copied explorer_shell.html -> explorer/index.html")
    else:
        print("WARNING: explorer_shell.html missing; explorer/index.html not written")

    total = sum(f.stat().st_size for f in DATA.glob("*.js"))
    print(f"\nexplorer/data: {total/1e6:.1f} MB across {len(list(DATA.glob('*.js')))} files")
    print(f"games copied: {len(games)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
