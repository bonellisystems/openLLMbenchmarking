"""rowselect is the single gate between the append-only store and every published
number, so its selection rules are pinned here: latest version wins per cell, withdrawn
cells stay gaps until replaced, and a drifted predicate fails loudly."""
from __future__ import annotations

import pytest

from llmtest.rowselect import (SupersededDrift, effective_custom_rows,
                               effective_suite_rows, suite_cell_allowed,
                               version_tuple)

SUP = {
    "replacement_version": "suite-v2.2.0",
    "cells": [
        {"model": "m-laptop", "battery": 8, "n_expected": 2, "source": "shard",
         "reason": "laptop"},
        {"model": "m-dirs", "battery": 8, "n_expected": 3, "source": "dirs",
         "reason": "laptop sweep, rows live in results_b8_*/"},
    ],
    "custom_rows": [
        {"shard": "results_games/rows-games.jsonl", "model": "m-laptop", "battery": 9,
         "ts_before": "2026-07-29T22:11:00", "n_expected": 2, "reason": "laptop"},
    ],
}


def _r(model, battery, sv=None, ts="2026-07-20T00:00:00", **kw):
    d = {"model_id": model, "battery": battery, "ts": ts}
    if sv:
        d["suite_version"] = sv
    d.update(kw)
    return d


def test_version_tuple_orders_and_handles_legacy():
    assert version_tuple("suite-v2.2.0") > version_tuple("suite-v2.1.0")
    assert version_tuple("suite-v2.1.0") > version_tuple(None)
    assert version_tuple("v2.10.0") > version_tuple("v2.9.9")


def test_latest_version_wins_per_cell_and_untouched_cells_survive():
    rows = [_r("a", 2, "suite-v2.1.0"), _r("a", 2, "suite-v2.2.0"),
            _r("b", 3, "suite-v2.0.0")]
    out = effective_suite_rows(rows, SUP)
    assert [r["suite_version"] for r in out if r["model_id"] == "a"] == ["suite-v2.2.0"]
    # b/B3 was never re-run: its old rows remain current
    assert any(r["model_id"] == "b" for r in out)


def test_withdrawn_cell_is_a_gap_until_replaced_then_returns():
    old = [_r("m-laptop", 8, "suite-v2.1.0"), _r("m-laptop", 8, "suite-v2.1.0")]
    assert effective_suite_rows(old, SUP) == []          # withdrawn -> not run
    replaced = old + [_r("m-laptop", 8, "suite-v2.2.0")]
    out = effective_suite_rows(replaced, SUP)
    assert len(out) == 1 and out[0]["suite_version"] == "suite-v2.2.0"


def test_shard_count_drift_fails_loudly():
    rows = [_r("m-laptop", 8, "suite-v2.1.0")]           # n_expected says 2, we drop 1
    with pytest.raises(SupersededDrift):
        effective_suite_rows(rows, SUP)


def test_dirs_sourced_rule_is_not_count_checked_by_the_suite_path():
    # dirs-sourced cells are asserted by the B8 consumer that sees both sources;
    # the suite-shard path must not false-alarm on the copies it happens to see.
    rows = [_r("m-dirs", 8, "suite-v2.1.0")]
    assert effective_suite_rows(rows, SUP) == []


def test_suite_cell_allowed_streaming_variant():
    assert suite_cell_allowed(SUP, "m-dirs", 8, None) is False
    assert suite_cell_allowed(SUP, "m-dirs", 8, version_tuple("suite-v2.2.0")) is True
    assert suite_cell_allowed(SUP, "unlisted", 8, None) is True


def test_custom_rows_predicate_retires_legacy_and_new_rows_supersede():
    legacy_old = [_r("m-laptop", 9, ts="2026-07-25T10:00:00"),
                  _r("m-laptop", 9, ts="2026-07-26T10:00:00")]
    legacy_new = [_r("m-laptop", 9, ts="2026-07-30T10:00:00")]   # post-cutoff PRO-6000
    out = effective_custom_rows(legacy_old + legacy_new, SUP, "rows-games.jsonl")
    assert out == legacy_new
    # v2.2.0 rows for the cell evict remaining legacy rows entirely
    v22 = [_r("m-laptop", 9, "suite-v2.2.0", ts="2026-08-05T00:00:00")]
    out2 = effective_custom_rows(legacy_old + legacy_new + v22, SUP, "rows-games.jsonl")
    assert out2 == v22


def test_custom_count_drift_fails_loudly():
    rows = [_r("m-laptop", 9, ts="2026-07-25T10:00:00")]  # rule expects 2 matches
    with pytest.raises(SupersededDrift):
        effective_custom_rows(rows, SUP, "rows-games.jsonl")


def test_unlisted_shard_and_models_pass_through():
    rows = [_r("other", 10, ts="2026-07-20T00:00:00")]
    assert effective_custom_rows(rows, SUP, "rows-security.jsonl") == rows


def test_replaced_cell_does_not_readmit_its_superseded_rows():
    """The 2026-08-05 regression: once a withdrawn cell gained v2.2.0 rows, the
    cell-level gate opened and the OLD laptop rows were scored alongside the new ones -
    184 rows blending two hardware generations. A row older than its cell's max is
    dropped regardless of whether the cell was ever withdrawn."""
    new = version_tuple("suite-v2.2.0")
    old = version_tuple("suite-v2.1.0")
    # withdrawn cell, now replaced: new rows in, old rows out
    assert suite_cell_allowed(SUP, "m-dirs", 8, new, new) is True
    assert suite_cell_allowed(SUP, "m-dirs", 8, new, old) is False
    # cell never withdrawn but re-measured: latest still wins
    assert suite_cell_allowed(SUP, "unlisted", 8, new, new) is True
    assert suite_cell_allowed(SUP, "unlisted", 8, new, old) is False
    # withdrawn and NOT yet replaced: nothing is admitted
    assert suite_cell_allowed(SUP, "m-dirs", 8, old, old) is False
