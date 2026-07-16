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
    sub.add_parser("status", help="done/pending matrix from resume keys")
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
