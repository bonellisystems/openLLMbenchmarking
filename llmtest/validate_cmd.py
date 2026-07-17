"""llmtest validate — shard + config integrity + mojibake lint. Same checks CI runs. Exit 0 = clean."""
from pathlib import Path

import yaml

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
    # Fixture linting for B1
    b1_config = cfg.suite.get("b1")
    if b1_config:
        b1_units = set(b1_config.get("units_tier1", []))
        difficulties = {"easy", "medium", "hard"}
        classes = {"short", "standard", "long"}
        valid_signal_types = {"contains", "regex", "numeric"}

        fixtures_dir = root / "suite" / "b1_business"
        if fixtures_dir.exists():
            for task_file in fixtures_dir.rglob("task-*.yaml"):
                try:
                    data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
                    rel_path = task_file.relative_to(root)

                    # Required keys
                    for key in ("id", "unit", "difficulty", "class", "prompt", "signals"):
                        if key not in data:
                            errors.append(f"fixture {rel_path} missing required key: {key}")

                    # Validate unit
                    if data.get("unit") not in b1_units:
                        errors.append(
                            f"fixture {rel_path} unit '{data.get('unit')}' not in b1.units_tier1"
                        )

                    # Validate difficulty
                    if data.get("difficulty") not in difficulties:
                        errors.append(
                            f"fixture {rel_path} difficulty '{data.get('difficulty')}' "
                            f"not in {difficulties}"
                        )

                    # Validate class
                    if data.get("class") not in classes:
                        errors.append(
                            f"fixture {rel_path} class '{data.get('class')}' not in {classes}"
                        )

                    # Validate id format
                    task_id = data.get("id", "")
                    unit = data.get("unit", "")
                    if unit and not task_id.startswith(f"{unit}-"):
                        errors.append(
                            f"fixture {rel_path} id '{task_id}' doesn't match pattern "
                            f"'{unit}-<NN>'"
                        )
                    import re
                    if not re.match(r"^[a-z_]+-\d{2}$", task_id):
                        errors.append(
                            f"fixture {rel_path} id '{task_id}' doesn't match "
                            f"pattern '<unit>-<NN>'"
                        )

                    # Validate signals
                    for sig_idx, sig in enumerate(data.get("signals", [])):
                        sig_type = sig.get("type")
                        if sig_type not in valid_signal_types:
                            errors.append(
                                f"fixture {rel_path} signal {sig_idx} has unknown type "
                                f"'{sig_type}' (must be one of {valid_signal_types})"
                            )
                except Exception as e:
                    rel_path = task_file.relative_to(root) if task_file.exists() else task_file
                    errors.append(f"fixture {rel_path} failed to parse: {e}")

    for e in errors:
        print(f"VALIDATE-ERROR: {e}")
    print(f"validate: {n} rows checked, {len(errors)} errors")
    return 1 if errors else 0
