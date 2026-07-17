"""Append-only sharded results store (TESTPLAN 7.2). Write-time validation == CI validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from llmtest import schema


class SchemaError(ValueError):
    pass


class Store:
    def __init__(self, results_dir: Path | str):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _shard(self, suite_version: str) -> Path:
        return self.dir / f"rows-{suite_version}.jsonl"

    def existing_row_ids(self) -> set[str]:
        return {r["row_id"] for r in self.iter_rows()}

    def iter_rows(self) -> Iterator[dict]:
        for shard in sorted(self.dir.glob("rows-*.jsonl")):
            with shard.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def append(self, row: dict) -> bool:
        errs = schema.validate_row(row)
        if errs:
            raise SchemaError("; ".join(errs))
        if row["row_id"] in self.existing_row_ids():
            return False
        with self._shard(row["suite_version"]).open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return True

    def append_session(self, d: dict) -> None:
        with (self.dir / "sessions.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(d, sort_keys=True) + "\n")

    def iter_sessions(self) -> Iterator[dict]:
        p = self.dir / "sessions.jsonl"
        if not p.exists():
            return
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def _judgments_path(self) -> Path:
        return self.dir / "judgments.jsonl"

    def iter_judgments(self) -> Iterator[dict]:
        p = self._judgments_path()
        if not p.exists():
            return
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def existing_judgment_keys(self) -> set[tuple]:
        return {(j["packet_id"], j["judge_id"], j["letter"]) for j in self.iter_judgments()}

    def append_judgment(self, d: dict) -> bool:
        errs = schema.validate_judgment(d)
        if errs:
            raise SchemaError("; ".join(errs))
        key = (d["packet_id"], d["judge_id"], d["letter"])
        if key in self.existing_judgment_keys():
            return False
        with self._judgments_path().open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(d, sort_keys=True) + "\n")
        return True
