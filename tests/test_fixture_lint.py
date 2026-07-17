"""Tests for fixture linting in validate_cmd."""
import shutil
from pathlib import Path
import pytest
import yaml
from llmtest.validate_cmd import run_validate

ROOT = Path(__file__).resolve().parents[1]


def test_fixture_lint_validates_signal_values(tmp_path, capsys):
    """Fixture lint must validate signal VALUES (not just types).

    Should catch:
    - missing 'value' key
    - regex that doesn't compile
    - numeric without int/float value
    - non-string contains value
    """
    # Set up a temp repo with config copied
    config_dir = tmp_path / "config"
    shutil.copytree(ROOT / "config", config_dir)

    # Create minimal suite.yaml with b1 config
    suite_yaml = tmp_path / "config" / "suite.yaml"
    suite_data = yaml.safe_load(suite_yaml.read_text(encoding="utf-8"))
    suite_data["b1"] = {"units_tier1": ["test_unit"]}
    suite_yaml.write_text(yaml.dump(suite_data), encoding="utf-8")

    # Create TESTPLAN.md (required by validate_cmd)
    (tmp_path / "TESTPLAN.md").write_text("# Test Plan\n", encoding="utf-8")

    # Create suite structure
    suite_dir = tmp_path / "suite" / "b1_business" / "test_unit"
    suite_dir.mkdir(parents=True)

    # Create a good fixture first to ensure b1_business dir exists
    good_fixture = suite_dir / "task-01.yaml"
    good_fixture.write_text("""\
id: test_unit-01
unit: test_unit
difficulty: easy
class: short
prompt: Test prompt
signals:
  - {type: contains, value: "test"}
""", encoding="utf-8")

    # Create fixture with missing value
    bad_fixture = suite_dir / "task-02.yaml"
    bad_fixture.write_text("""\
id: test_unit-02
unit: test_unit
difficulty: easy
class: short
prompt: Test prompt
signals:
  - {type: contains}
""", encoding="utf-8")

    # Create fixture with bad regex
    bad_regex = suite_dir / "task-03.yaml"
    bad_regex.write_text("""\
id: test_unit-03
unit: test_unit
difficulty: easy
class: short
prompt: Test prompt
signals:
  - {type: regex, value: "(unclosed"}
""", encoding="utf-8")

    # Create fixture with non-numeric numeric value
    bad_numeric = suite_dir / "task-04.yaml"
    bad_numeric.write_text("""\
id: test_unit-04
unit: test_unit
difficulty: easy
class: short
prompt: Test prompt
signals:
  - {type: numeric, value: "not a number"}
""", encoding="utf-8")

    # Create fixture with non-string contains value
    bad_contains = suite_dir / "task-05.yaml"
    bad_contains.write_text("""\
id: test_unit-05
unit: test_unit
difficulty: easy
class: short
prompt: Test prompt
signals:
  - {type: contains, value: 123}
""", encoding="utf-8")

    exit_code = run_validate(tmp_path)
    assert exit_code == 1
    captured = capsys.readouterr()
    output = captured.out

    # Should have errors for missing value, bad regex, bad numeric, bad contains
    assert "task-02.yaml" in output and "signal 0" in output
    assert "task-03.yaml" in output and "signal 0" in output and "compile" in output
    assert "task-04.yaml" in output and "signal 0" in output
    assert "task-05.yaml" in output and "signal 0" in output
