import subprocess, sys

def test_cli_module_runs_and_reports_version():
    p = subprocess.run([sys.executable, "-m", "llmtest", "--version"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert p.stdout.strip().startswith("llmtest 2.")
