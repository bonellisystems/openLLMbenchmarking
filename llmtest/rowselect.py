"""Which rows are CURRENT — the one shared answer for every aggregator.

Michael's 2026-08-03 ruling: one consistent hardware metric (RTX PRO 6000 Blackwell),
and the store stays append-only. Superseded measurements are therefore never deleted;
they are out-versioned. Re-runs write a bumped ``suite_version`` (suite-v2.2.0) and
every consumer selects rows through this module:

1. **Latest version wins per (model_id, battery).** If a cell has rows at more than one
   suite version, only the highest version's rows are current. A cell that was never
   re-run keeps its old rows — bumping the suite version does not orphan untouched
   cells.
2. **Withdrawn cells read "not run" until replaced.** ``config/superseded.yaml`` lists
   cells whose only measurements violate the hardware requirement (laptop / RTX 5090
   32GB / oracle-SETUPFAIL). Such a cell is dropped from aggregation unless it has rows
   at or above ``replacement_version`` — so the dashboard honestly shows a gap in the
   interim instead of a number measured on the wrong machine.
3. **Legacy custom rows are matched by predicate.** B9/B10/B11 rows written before this
   module existed carry no ``suite_version``; the YAML's ``custom_rows`` rules select
   them by (shard, model, ts-window) with a load-bearing ``n_expected``: if a predicate
   stops matching exactly that many rows the store changed underneath us, and that is
   an error, never a silent reshape.

Consumers: dashboard/build_data.py, dashboard/build_explorer.py, llmtest/tables.py,
scripts/p8_report.py. Anything new that aggregates rows must import this module —
aggregating raw shards reintroduces the wrong-hardware numbers this exists to retire.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

_VER_RE = re.compile(r"(\d+)")


def version_tuple(suite_version) -> tuple:
    """'suite-v2.2.0' -> (2, 2, 0). Versionless/legacy -> (0,) so any real version
    beats it. Robust to prefix variants ('v2.1.0', 'suite-v2.1.0')."""
    if not suite_version:
        return (0,)
    nums = _VER_RE.findall(str(suite_version))
    return tuple(int(n) for n in nums) or (0,)


def load_superseded(root: Path | str) -> dict:
    """Parsed config/superseded.yaml, or an empty ruleset when absent (fresh clones of
    a hypothetical pre-policy checkout must not crash)."""
    p = Path(root) / "config" / "superseded.yaml"
    if not p.exists():
        return {"cells": [], "custom_rows": [], "replacement_version": None}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    d.setdefault("cells", [])
    d.setdefault("custom_rows", [])
    d.setdefault("replacement_version", None)
    return d


class SupersededDrift(RuntimeError):
    """A superseded predicate matched a different row count than the committed
    n_expected. The store changed under the policy file — refuse to aggregate."""


def _withdrawn_cells(superseded: dict) -> dict:
    """{(model, battery): rule} for the suite-cell rules."""
    return {(c["model"], int(c["battery"])): c for c in superseded.get("cells", [])}


def effective_suite_rows(rows, superseded: dict, count_check: bool = True) -> list:
    """Filter suite-store rows (anything with model_id/battery/suite_version) to the
    CURRENT set: latest version per cell, minus withdrawn cells not yet replaced.

    ``rows`` may be any iterable of dicts; returns a list. Shakedown rows are excluded
    defensively by version-string match (belt and suspenders — consumers already skip
    the shard by filename).
    """
    rows = [r for r in rows
            if "shakedown" not in str(r.get("suite_version", ""))]
    best: dict = {}
    for r in rows:
        key = (r.get("model_id"), r.get("battery"))
        v = version_tuple(r.get("suite_version"))
        if v > best.get(key, (-1,)):
            best[key] = v

    repl = version_tuple(superseded.get("replacement_version"))
    withdrawn = _withdrawn_cells(superseded)

    out, dropped = [], {}
    for r in rows:
        key = (r.get("model_id"), r.get("battery"))
        v = version_tuple(r.get("suite_version"))
        if v != best[key]:
            continue                      # out-versioned
        if key in withdrawn and best[key] < repl:
            dropped[key] = dropped.get(key, 0) + 1
            continue                      # withdrawn, not yet replaced
        out.append(r)

    if count_check:
        # n_expected asserts apply only to rules whose rows we actually saw AND whose
        # source includes the suite shard: dirs-sourced B8 rules are checked by the
        # B8-specific consumer, which sees both sources.
        for key, n in dropped.items():
            rule = withdrawn.get(key) or {}
            exp = rule.get("n_expected")
            src = rule.get("source", "shard")
            if exp is None or src == "dirs":
                continue
            if src == "shard" and n != exp:
                raise SupersededDrift(
                    f"superseded cell {key}: predicate dropped {n} rows, "
                    f"n_expected is {exp} — the store changed under "
                    f"config/superseded.yaml; investigate before aggregating")
    return out


def suite_cell_allowed(superseded: dict, model_id: str, battery: int,
                       cell_max_version: tuple | None = None,
                       row_version: tuple | None = None) -> bool:
    """Streaming-friendly variant for consumers that read a cell from MULTIPLE sources
    (build_data's take_b8 reads results_b8_<model>/ dirs AND the suite shard).
    ``cell_max_version`` is the highest suite version the caller has observed for the
    cell across all its sources ((0,) or None when every row is legacy); ``row_version``
    is THIS row's version.

    Two independent rules, both required:

    1. Latest version wins WITHIN a cell. Without this, a replaced cell re-admitted its
       superseded rows the moment the replacement arrived: after the 2026-08-05 campaign
       merged 115 fresh rows next to 69 old laptop rows, agents-a1-35b would have scored
       on all 184 - blending the two hardware generations this whole exercise exists to
       separate. The n_expected drift assert in build_data caught it.
    2. A withdrawn cell stays withdrawn until its replacement version actually exists.
    """
    if (row_version is not None and cell_max_version is not None
            and row_version < cell_max_version):
        return False
    rule = _withdrawn_cells(superseded).get((model_id, int(battery)))
    if rule is None:
        return True
    repl = version_tuple(superseded.get("replacement_version"))
    return (cell_max_version or (0,)) >= repl


def effective_custom_rows(rows, superseded: dict, shard_name: str,
                          count_check: bool = True) -> list:
    """Filter a custom shard (B9/B10/B11) to the CURRENT set.

    New-era rows carry suite_version + hardware_sku; latest version wins per cell
    exactly as for suite rows. Legacy rows (no suite_version) are dropped when a
    custom_rows predicate (shard, model, ts_before) matches them, or when the cell has
    any newer-versioned rows.
    """
    rules = [r for r in superseded.get("custom_rows", [])
             if str(r.get("shard", "")).endswith(shard_name)
             or shard_name in str(r.get("shard", ""))]
    rows = list(rows)

    best: dict = {}
    for r in rows:
        key = (r.get("model_id"), r.get("battery"))
        v = version_tuple(r.get("suite_version"))
        if v > best.get(key, (-1,)):
            best[key] = v

    matched = {id(r): False for r in rows}
    counts = {i: 0 for i in range(len(rules))}
    for i, rule in enumerate(rules):
        for r in rows:
            if r.get("model_id") != rule.get("model"):
                continue
            if r.get("suite_version"):
                continue                  # predicates only ever retire legacy rows
            if str(r.get("ts", "")) < str(rule.get("ts_before", "")):
                matched[id(r)] = True
                counts[i] += 1
    if count_check:
        for i, rule in enumerate(rules):
            exp = rule.get("n_expected")
            if exp is not None and counts[i] != exp:
                raise SupersededDrift(
                    f"custom rule {rule.get('model')}/B{rule.get('battery')} in "
                    f"{shard_name}: matched {counts[i]} rows, n_expected {exp} — "
                    f"the shard changed under config/superseded.yaml")

    out = []
    for r in rows:
        if matched[id(r)]:
            continue
        key = (r.get("model_id"), r.get("battery"))
        if version_tuple(r.get("suite_version")) != best[key]:
            continue
        out.append(r)
    return out
