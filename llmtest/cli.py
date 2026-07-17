"""llmtest CLI — subcommands per TESTPLAN §7.5. P0-P2 wire: validate, status, run, tables."""
import argparse
import sys

from llmtest import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmtest")
    p.add_argument("--version", action="version", version=f"llmtest {__version__}")
    sub = p.add_subparsers(dest="command")
    v = sub.add_parser("validate", help="schema + fixture lint (== CI); --serving runs the canary")
    v.add_argument("--serving", action="store_true")
    st = sub.add_parser("status", help="done/pending matrix from resume keys")
    st.add_argument("--judging", action="store_true",
                     help="judging-pipeline view: cohorts/packets/(packet x judge) counts")
    r = sub.add_parser("run", help="execute battery work items")
    r.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    r.add_argument("--model", default=None)
    r.add_argument("--battery", type=int, default=None)
    r.add_argument("--task", dest="task_id", default=None)
    r.add_argument("--condition", default=None)
    r.add_argument("--force", action="store_true")
    r.add_argument("--keep-server", action="store_true")
    r.add_argument("--debug", action="store_true")
    sub.add_parser("tables", help="regenerate all tables (byte-deterministic)")
    j = sub.add_parser("judge", help="score needs_judging rows via the judge panel")
    j.add_argument("--pending", action="store_true",
                    help="process pending (packet x judge) work (currently the only mode)")
    j.add_argument("--judge", dest="judge", default=None,
                    help="restrict invocation to one judge id (default: full panel)")
    j.add_argument("--packets-only", action="store_true",
                    help="build packets only, skip invoking judges")
    j.add_argument("--fake", action="store_true",
                    help="use FakeJudgeAdapter instead of real judge CLIs")
    j.add_argument("--retry-errors", action="store_true",
                    help="treat pairs whose only rows are terminal '-' errors as pending again")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 2
    # Dispatch is wired per-task as commands land (Tasks 5, 7, 11, 13, 15).
    from llmtest import dispatch
    return dispatch.run(args)


def entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry()
