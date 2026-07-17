"""Command dispatch. Each command module registers here as tasks complete."""


def run(args) -> int:
    if args.command == "validate":
        if getattr(args, "serving", False):
            from llmtest.canary import run_canary
            return run_canary()
        from llmtest.validate_cmd import run_validate
        return run_validate()
    if args.command == "status":
        from llmtest.status_cmd import run_status
        return run_status(judging=getattr(args, "judging", False))
    if args.command == "run":
        from llmtest.run_cmd import run_run
        return run_run(args)
    if args.command == "tables":
        from llmtest.tables import run_tables
        return run_tables()
    if args.command == "judge":
        from llmtest.judge_cmd import run_judge
        return run_judge(args)
    raise SystemExit(f"unknown command {args.command}")
