# P3 backlog — deferred findings from the p0-p2 final review (2026-07-16)

Pre-P3 (first follow-up commits):
- Wire Battery.preflight() into run_cmd (TESTPLAN 7.4: battery refuses to execute on failure) — required before B6/B2/B4.
- Design --force re-measurement semantics: run_n bump or supersede tag (append() False under --force currently fails loudly by design).
- ServerManager reuse key must include timing_authoritative (authority-contradicting session reuse trap).
- Record sampling.max_tokens as actually sent (rows currently hardcode 0 — TESTPLAN 7.2 "as observed").
- compose_fork_flags must honor (or explicitly deprecate) runtime_pins.standard_flags; fix normalize_config ngramN!=32 spec_params gap in the same pass.
- Server logs: name by session_id (8080 log is overwritten per launch); close the log handle.
- Harden _power_state() detection + bench-night profile enforcement (shakedown ran "balanced"; TESTPLAN 11.10).

B5 next touch: render aggregate_tps column in serving.md; render stripped pp_tps/ttft_ms as "-" not 0; conc-table clarity.
Cosmetics/logged: run without --battery → argparse error not KeyError; --suite currently unused; _SPEC module-global caching; sweep_orphans SUCCESS-string localization; TESTPLAN 7.4 ABC sketch reconcile (fixtures(), plan signature); repo-scoped CLAUDE.md (TESTPLAN 7.1); tier/sku/kv hardcodes parametrized at T2/T3; unused imports (schema.field, server.os, test Path); thin branch coverage (validate_row, run_cmd flags); freeze script: empty-sha guard + allow_unicode + surgical per-model rewrite.
