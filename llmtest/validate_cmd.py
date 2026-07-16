"""llmtest validate — shard + config integrity + mojibake lint. Same checks CI runs. Exit 0 = clean."""
from pathlib import Path

from llmtest import schema
from llmtest.registry import load_config
from llmtest.store import Store


def run_validate(root: Path | str = ".") -> int:
    root = Path(root).resolve()
    errors: list[str] = []
    cfg = load_config(root)
    order = cfg.suite["condition_order"]
    if len(order) != len(set(order)):
        errors.append("suite.yaml condition_order contains duplicates")
    for name, m in cfg.registry["models"].items():
        for k in ("hf_repo", "quant_file", "provenance", "license", "weights_gb"):
            if k not in m:
                errors.append(f"registry:{name} missing {k}")
    n = 0
    for row in Store(root / "results").iter_rows():
        n += 1
        errors += [f"row {row.get('row_id', '?')[:12]}: {e}"
                   for e in schema.validate_row(row)]
    scan_dirs = [root / "docs", root / "suite", root / "grading"]
    scan = [root / "TESTPLAN.md"] + [p for d in scan_dirs if d.exists()
                                     for p in d.rglob("*") if p.is_file()
                                     and p.suffix in {".md", ".yaml", ".txt", ".html"}]
    for p in scan:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-utf8 file: {p}")
            continue
        bad = sorted({c for c in text if ord(c) > 0x2E7F})   # mojibake/CJK lint (amendment 29)
        if bad:
            bad_hex = [f"U+{ord(c):04X}" for c in bad[:3]]
            errors.append(f"suspicious non-ASCII in {p.relative_to(root)}: {bad_hex}")
    for e in errors:
        print(f"VALIDATE-ERROR: {e}")
    print(f"validate: {n} rows checked, {len(errors)} errors")
    return 1 if errors else 0
