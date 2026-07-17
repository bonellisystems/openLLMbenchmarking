"""llmtest validate — shard + config integrity + mojibake lint. Same checks CI runs. Exit 0 = clean."""
import re
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
        industries_vocab = set(b1_config.get("industries", []))
        difficulties = {"easy", "medium", "hard"}
        classes = {"short", "standard", "long"}
        valid_signal_types = {"contains", "regex", "numeric"}

        fixtures_dir = root / "suite" / "b1_business"
        if fixtures_dir.exists():
            # Collect fixtures by unit for distribution checks
            fixtures_by_unit = {}

            for task_file in fixtures_dir.rglob("task-*.yaml"):
                try:
                    data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
                    rel_path = task_file.relative_to(root)

                    # Required keys
                    for key in ("id", "unit", "difficulty", "class", "industry", "prompt", "signals"):
                        if key not in data:
                            errors.append(f"fixture {rel_path} missing required key: {key}")

                    # Validate unit
                    if data.get("unit") not in b1_units:
                        errors.append(
                            f"fixture {rel_path} unit '{data.get('unit')}' not in b1.units_tier1"
                        )

                    # Validate industry
                    if data.get("industry") not in industries_vocab:
                        errors.append(
                            f"fixture {rel_path} industry '{data.get('industry')}' not in b1.industries"
                        )

                    # Track fixtures by unit for distribution checks
                    unit = data.get("unit")
                    if unit not in fixtures_by_unit:
                        fixtures_by_unit[unit] = []
                    fixtures_by_unit[unit].append({
                        "file": rel_path,
                        "id": data.get("id"),
                        "industry": data.get("industry")
                    })

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

                        # Validate signal value exists and is the correct type
                        sig_value = sig.get("value")
                        if sig_value is None:
                            errors.append(
                                f"fixture {rel_path} signal {sig_idx} missing or null 'value' key"
                            )
                        elif sig_type == "regex":
                            # Try to compile regex
                            try:
                                re.compile(sig_value)
                            except re.error as e:
                                errors.append(
                                    f"fixture {rel_path} signal {sig_idx} regex value "
                                    f"'{sig_value}' failed to compile: {e}"
                                )
                            # Also check it's a string
                            if not isinstance(sig_value, str):
                                errors.append(
                                    f"fixture {rel_path} signal {sig_idx} regex value "
                                    f"must be string, got {type(sig_value).__name__}"
                                )
                        elif sig_type == "numeric":
                            # Check value is numeric
                            if not isinstance(sig_value, (int, float)) or isinstance(sig_value, bool):
                                errors.append(
                                    f"fixture {rel_path} signal {sig_idx} numeric value "
                                    f"must be int or float, got {type(sig_value).__name__}"
                                )
                            # Check tolerance if present
                            tolerance = sig.get("tolerance")
                            if tolerance is not None and not isinstance(tolerance, (int, float)):
                                errors.append(
                                    f"fixture {rel_path} signal {sig_idx} tolerance "
                                    f"must be numeric, got {type(tolerance).__name__}"
                                )
                        elif sig_type == "contains":
                            # Check value is string
                            if not isinstance(sig_value, str):
                                errors.append(
                                    f"fixture {rel_path} signal {sig_idx} contains value "
                                    f"must be string, got {type(sig_value).__name__}"
                                )
                except Exception as e:
                    rel_path = task_file.relative_to(root) if task_file.exists() else task_file
                    errors.append(f"fixture {rel_path} failed to parse: {e}")

            # Per-unit distribution check: ≥8 tasks → ≥5 distinct industries, ≤2 tasks per industry
            for unit, fixtures in fixtures_by_unit.items():
                if len(fixtures) >= 8:
                    # Check ≥5 distinct industries
                    distinct_industries = set(f["industry"] for f in fixtures)
                    if len(distinct_industries) < 5:
                        errors.append(
                            f"unit {unit}: {len(fixtures)} fixtures must span ≥5 distinct industries "
                            f"(found {len(distinct_industries)})"
                        )
                    # Check ≤2 tasks per industry
                    industry_counts = {}
                    for f in fixtures:
                        ind = f["industry"]
                        industry_counts[ind] = industry_counts.get(ind, 0) + 1
                    for ind, count in industry_counts.items():
                        if count > 2:
                            errors.append(
                                f"unit {unit}: industry '{ind}' appears {count} times "
                                f"(max 2 tasks per industry)"
                            )

    # Fixture linting for B3 (hallucination curve)
    b3_config = cfg.suite.get("b3")
    if b3_config:
        b3_categories = set(b3_config.get("categories", []))
        b3_industries = set(b3_config.get("industries", []))
        difficulties = {"easy", "medium", "hard"}
        classes = {"short", "standard", "long"}
        expects = {"hedge", "answer"}
        valid_signal_types = {"contains", "regex", "numeric"}

        b3_fixtures_dir = root / "suite" / "b3_hallucination"
        if b3_fixtures_dir.exists():
            for task_file in sorted(b3_fixtures_dir.glob("task-*.yaml")):
                try:
                    data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
                    rel_path = task_file.relative_to(root)

                    for key in ("id", "category", "difficulty", "class", "industry", "expect"):
                        if key not in data:
                            errors.append(f"fixture {rel_path} missing required key: {key}")

                    if data.get("category") not in b3_categories:
                        errors.append(
                            f"fixture {rel_path} category '{data.get('category')}' "
                            f"not in b3.categories"
                        )

                    if data.get("industry") not in b3_industries:
                        errors.append(
                            f"fixture {rel_path} industry '{data.get('industry')}' "
                            f"not in b3.industries"
                        )

                    if data.get("difficulty") not in difficulties:
                        errors.append(
                            f"fixture {rel_path} difficulty '{data.get('difficulty')}' "
                            f"not in {difficulties}"
                        )

                    if data.get("class") not in classes:
                        errors.append(
                            f"fixture {rel_path} class '{data.get('class')}' "
                            f"not in {classes}"
                        )

                    expect = data.get("expect")
                    if expect not in expects:
                        errors.append(
                            f"fixture {rel_path} expect '{expect}' not in {expects}"
                        )

                    # Exactly one of prompt/turns; turns needs >=2 entries.
                    has_prompt = "prompt" in data
                    has_turns = "turns" in data
                    if has_prompt == has_turns:
                        errors.append(
                            f"fixture {rel_path} must have exactly one of 'prompt' or 'turns'"
                        )
                    elif has_turns and (not isinstance(data["turns"], list) or len(data["turns"]) < 2):
                        errors.append(
                            f"fixture {rel_path} 'turns' must be a list of >=2 prompts"
                        )

                    # expect-conditional signal requirements.
                    if expect == "hedge" and not data.get("trap_signals"):
                        errors.append(
                            f"fixture {rel_path} expect=='hedge' requires >=1 trap_signals"
                        )
                    if expect == "answer" and not data.get("answer_signals"):
                        errors.append(
                            f"fixture {rel_path} expect=='answer' requires >=1 answer_signals"
                        )

                    # id format: hallucination-<NN>
                    task_id = data.get("id", "")
                    if not re.match(r"^hallucination-\d{2}$", task_id):
                        errors.append(
                            f"fixture {rel_path} id '{task_id}' doesn't match "
                            f"pattern 'hallucination-<NN>'"
                        )

                    # Validate every signal list present (trap/answer/hedge).
                    for sig_group in ("trap_signals", "answer_signals", "hedge_signals"):
                        for sig_idx, sig in enumerate(data.get(sig_group, [])):
                            sig_type = sig.get("type")
                            if sig_type not in valid_signal_types:
                                errors.append(
                                    f"fixture {rel_path} {sig_group} {sig_idx} has unknown "
                                    f"type '{sig_type}' (must be one of {valid_signal_types})"
                                )

                            sig_value = sig.get("value")
                            if sig_value is None:
                                errors.append(
                                    f"fixture {rel_path} {sig_group} {sig_idx} missing or "
                                    f"null 'value' key"
                                )
                            elif sig_type == "regex":
                                try:
                                    re.compile(sig_value)
                                except re.error as e:
                                    errors.append(
                                        f"fixture {rel_path} {sig_group} {sig_idx} regex "
                                        f"value '{sig_value}' failed to compile: {e}"
                                    )
                                if not isinstance(sig_value, str):
                                    errors.append(
                                        f"fixture {rel_path} {sig_group} {sig_idx} regex "
                                        f"value must be string, got {type(sig_value).__name__}"
                                    )
                            elif sig_type == "numeric":
                                if not isinstance(sig_value, (int, float)) or isinstance(sig_value, bool):
                                    errors.append(
                                        f"fixture {rel_path} {sig_group} {sig_idx} numeric "
                                        f"value must be int or float, got {type(sig_value).__name__}"
                                    )
                                tolerance = sig.get("tolerance")
                                if tolerance is not None and not isinstance(tolerance, (int, float)):
                                    errors.append(
                                        f"fixture {rel_path} {sig_group} {sig_idx} tolerance "
                                        f"must be numeric, got {type(tolerance).__name__}"
                                    )
                            elif sig_type == "contains":
                                if not isinstance(sig_value, str):
                                    errors.append(
                                        f"fixture {rel_path} {sig_group} {sig_idx} contains "
                                        f"value must be string, got {type(sig_value).__name__}"
                                    )
                except Exception as e:
                    rel_path = task_file.relative_to(root) if task_file.exists() else task_file
                    errors.append(f"fixture {rel_path} failed to parse: {e}")

    for e in errors:
        print(f"VALIDATE-ERROR: {e}")
    print(f"validate: {n} rows checked, {len(errors)} errors")
    return 1 if errors else 0
