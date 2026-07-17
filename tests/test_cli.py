import subprocess, sys

from llmtest.cli import build_parser

def test_cli_module_runs_and_reports_version():
    p = subprocess.run([sys.executable, "-m", "llmtest", "--version"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert p.stdout.strip().startswith("llmtest 2.")


def test_judge_subcommand_parses_all_flags():
    args = build_parser().parse_args(
        ["judge", "--pending", "--judge", "claude", "--packets-only", "--fake",
         "--retry-errors"])
    assert args.command == "judge"
    assert args.pending is True
    assert args.judge == "claude"
    assert args.packets_only is True
    assert args.fake is True
    assert args.retry_errors is True


def test_judge_subcommand_defaults():
    args = build_parser().parse_args(["judge"])
    assert args.pending is False
    assert args.judge is None
    assert args.packets_only is False
    assert args.fake is False
    assert args.retry_errors is False


def test_status_judging_flag_parses():
    args = build_parser().parse_args(["status", "--judging"])
    assert args.command == "status"
    assert args.judging is True

    default_args = build_parser().parse_args(["status"])
    assert default_args.judging is False


def test_dispatch_routes_judge_to_run_judge(monkeypatch):
    from llmtest import dispatch

    called = {}

    def fake_run_judge(args):
        called["args"] = args
        return 0

    monkeypatch.setattr("llmtest.judge_cmd.run_judge", fake_run_judge)
    args = build_parser().parse_args(["judge", "--pending", "--fake"])
    assert dispatch.run(args) == 0
    assert called["args"] is args


def test_dispatch_routes_status_judging_flag_through(monkeypatch):
    from llmtest import dispatch

    called = {}

    def fake_run_status(*, judging=False):
        called["judging"] = judging
        return 0

    monkeypatch.setattr("llmtest.status_cmd.run_status", fake_run_status)
    args = build_parser().parse_args(["status", "--judging"])
    assert dispatch.run(args) == 0
    assert called["judging"] is True
