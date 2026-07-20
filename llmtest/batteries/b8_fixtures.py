"""Battery 8 fixture loader -- thin wrapper over `llmtest.harness.tasks.
load_b8_tasks`, mirroring the `b6_fixtures`/`b7_fixtures` `load_*` shape so
`b8_harness.py` can `from llmtest.batteries.b8_fixtures import load_tasks`
just like B6 does for its own fixtures.

Deliberately thin: `llmtest.harness.tasks` (Task 3) already owns the full
manifest schema (setup_repo/oracle_files/protected_paths/budgets/oracle),
the fail-loud loader, `materialize_repo`, and `run_oracle`. This module adds
no new parsing or validation -- it exists purely so the battery layer has
its own, battery-scoped import path (consistent with every other
`bN_fixtures.py` in this package) without duplicating any of that logic.
Each `B8Task` already carries its own `fixture_sha` (sha256 of the raw
manifest bytes, computed by the loader) -- that's what rides in the
row-identity preimage (`compute_row_id`'s `fixture_sha` param), exactly like
every other battery's fixture_sha.
"""
from __future__ import annotations

from pathlib import Path

from llmtest.harness.tasks import B8Task, load_b8_tasks


def load_tasks(root: str | Path) -> list[B8Task]:
    """All B8 task manifests under `<root>/suite/b8_harness/`, sorted by id.
    Each returned `B8Task.fixture_sha` is what `b8_harness.py`'s `plan()`
    feeds into `schema.compute_row_id`. Returns `[]` if the fixture dir is
    absent (mirrors `load_b8_tasks`); raises `ValueError` on a malformed
    manifest (fail-loud, mirrors `b6_fixtures.load_tasks`)."""
    return load_b8_tasks(root)
