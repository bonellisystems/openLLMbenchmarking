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
