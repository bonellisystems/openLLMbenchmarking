import pytest
from pathlib import Path
from llmtest.batteries import b1_fixtures as f

ROOT = Path(__file__).resolve().parents[1]


def test_loader_reads_exemplar_and_hashes():
    tasks = f.load_unit_tasks(ROOT, "cybersecurity")
    assert tasks and tasks[0].id == "cybersecurity-01"
    assert len(tasks[0].fixture_sha) == 64


def test_check_signals_all_types():
    sig = [{"type": "contains", "value": "MFA"},
           {"type": "regex", "value": r"\bCVE-\d{4}-\d+\b"},
           {"type": "numeric", "value": 4200, "tolerance": 0.01}]
    out = f.check_signals("Enable MFA. CVE-2026-1234 applies. Cost: $4,200.", sig)
    assert all(v["pass"] for v in out.values())
    out2 = f.check_signals("nothing here", sig)
    assert not any(v["pass"] for v in out2.values())


def test_check_signals_keys_are_named():
    """Check that signal result keys are named (e.g. 'contains-0') not integer indices."""
    sig = [{"type": "contains", "value": "MFA"}]
    out = f.check_signals("MFA on", sig)
    assert "contains-0" in out
    assert 0 not in out


def test_check_signals_bad_regex_no_crash():
    """Bad regex at runtime returns error dict, not crash."""
    out = f.check_signals("text", [{"type": "regex", "value": "(unclosed"}])
    assert "regex-0" in out
    assert out["regex-0"]["pass"] is False
    assert "error" in out["regex-0"]


def test_loader_raises_on_malformed(tmp_path):
    """Malformed fixture should raise ValueError, not silently skip."""
    unit_dir = tmp_path / "suite" / "b1_business" / "cybersecurity"
    unit_dir.mkdir(parents=True)
    (unit_dir / "task-01.yaml").write_text("id: [broken", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fixture"):
        f.load_unit_tasks(tmp_path, "cybersecurity")


def test_loader_requires_industry_field(tmp_path):
    """Loader must require 'industry' field and fail loud if missing."""
    unit_dir = tmp_path / "suite" / "b1_business" / "test_unit"
    unit_dir.mkdir(parents=True)
    fixture = unit_dir / "task-01.yaml"
    fixture.write_text("""\
id: test_unit-01
unit: test_unit
difficulty: easy
class: short
prompt: Test prompt
signals: []
""", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed fixture|missing.*industry"):
        f.load_unit_tasks(tmp_path, "test_unit")


def test_loader_reads_industry_field(tmp_path):
    """Loader must read and return the industry field on Task objects."""
    unit_dir = tmp_path / "suite" / "b1_business" / "test_unit"
    unit_dir.mkdir(parents=True)
    fixture = unit_dir / "task-01.yaml"
    fixture.write_text("""\
id: test_unit-01
unit: test_unit
difficulty: easy
class: short
industry: financial_services
prompt: Test prompt
signals: []
""", encoding="utf-8")
    tasks = f.load_unit_tasks(tmp_path, "test_unit")
    assert len(tasks) == 1
    assert tasks[0].industry == "financial_services"
