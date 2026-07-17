# Battery 6 (agentic coding) — build report

Branch: `p4-b6-agenticcoding`. First-pass build per the explicit build contract
(not the full TESTPLAN §5.6 spec — see "Deltas vs TESTPLAN" below).

## Design

Mirrors B1's plugin shape exactly:

- `llmtest/batteries/b6_fixtures.py` — `Task` dataclass, `load_tasks()` loader
  (fail-loud on malformed fixtures, mirrors `b1_fixtures.load_unit_tasks`),
  `extract_code_block()`, `check_code_signals()`, `compile_check()`.
- `llmtest/batteries/b6_agenticcoding.py` — `@register class B6AgenticCoding(Battery)`,
  `id = 6`. `plan()`/`preflight()`/`execute()` follow `B1Business`'s structure:
  fixture_sha + prompt/signals ride in the `WorkItem.payload` from `plan()` so
  `execute()` never re-loads YAML; `force` bumps `run_n` scoped to
  `(model_id, task_id, condition)`, same bug-class B1 already fixed.
- `suite/b6_agenticcoding/task-01..10.yaml` — 10 fixtures, flat directory (no
  per-unit subdirs — B6 isn't organized by business unit).
- `config/suite.yaml` — added `b6:` block (`n_runs: 3`, `ctx: 32768`,
  `ctx_label: "32k"`, `max_tokens_by_track: {scratch: 6000, bugfix: 4000}`) and
  `"B6"` in `condition_vocab.cond`. Token sizing follows the same P3 Task 12
  reasoning-model-headroom lesson already documented next to `b1:`.
- `llmtest/batteries/__init__.py` — lazy-import branch for battery id 6.
- Condition: `runtime=fork;spec=ngram32;kv=q8;ctx=32k;cond=B6` — ngram spec-decode
  ON, per root CLAUDE.md ("edit/codegen is the n-gram-accelerated case").

## Task count: 10 (5 from-scratch + 5 planted-bug)

**Scratch** (function/CLI/query to spec, no code given): `scratch-01` python
(is_prime, easy), `scratch-02` python (CLI wordcount tool, medium), `scratch-03`
bash (timestamped backup.sh, easy), `scratch-04` js (debounce, easy), `scratch-05`
sql (30-day spend aggregate, medium).

**Bugfix** (buggy code + symptom, one-shot find-and-fix — no iterative loop, see
deltas below): `bugfix-01` python (missing colon → SyntaxError, easy/crash),
`bugfix-02` python (off-by-one `range(len(nums)-1)` drops last element, medium),
`bugfix-03` python (mutable-default-argument state leak, hard/subtle), `bugfix-04`
bash (`find -mtime $DAYS` vs `-mtime +$DAYS`, hard/silent), `bugfix-05` sql
(`WHERE o.customer_id = c.id` defeats the LEFT JOIN, medium).

Each bugfix task was traced by hand to its literal root-cause token; each carries
a `fix_signals` (positive regex/contains evidence the corrected pattern exists)
and a `regression_signals` (literal `absent` check that the exact buggy
line/token is gone — the "did they actually touch it or just echo it back"
no-op detector). Loader enforces both `buggy_code` and non-empty
`regression_signals` are present on every bugfix fixture — a bugfix task
without a discriminating regression signal fails fixture load, not silently
scores everything as "fixed."

Languages: python ×5, bash ×2, sql ×2, js ×1. Difficulty: easy ×4, medium ×4,
hard ×2.

## Scoring approach (deterministic, no execution)

1. `extract_code_block(text, language)` — regex over fenced ```` ``` ```` blocks;
   prefers a fence tagged with a known alias of the task's language, falls back
   to an untagged fence, then the first fence found; `None` if no fence exists.
2. `check_code_signals(code, signals, prefix)` — three signal types:
   `contains` (literal substring), `regex` (`re.search`), `absent` (literal
   substring must NOT be present — the regression/no-op check). Keys are
   namespaced (`required.*`, `fix.*`, `regression.*`) so the three signal lists
   never collide.
3. `compile_check(code)` — **Python tasks only**. Calls `compile(code, ...,
   "exec")`, which parses/byte-compiles to a code object and **never executes
   it** — no `exec()`/`eval()` call exists anywhere in `b6_fixtures.py` or
   `b6_agenticcoding.py`. Verified by two tests that monkeypatch
   `builtins.exec`/`builtins.eval` to raise `AssertionError` if called, then feed
   `compile_check()` — and the full `execute()` pipeline — code that would
   `sys.exit(1)` or `raise SystemExit` if it were ever actually run; both pass
   scoring without tripping the patched builtins.
4. `needs_judging=True` on every row: correctness (does `is_prime` actually work,
   does the SQL aggregate the right window) and completeness are judged axes —
   deterministic checks can only prove "the right shape/tokens are present,"
   not "the logic is right," without executing untrusted code.

## Deltas vs TESTPLAN §5.6 (flagged per the build contract)

TESTPLAN §5.6 specifies substantially more than this pass built, and the human
task instructions explicitly requested the narrower shape — noting this is a
deliberate scope cut, not an oversight:

- **No game roster / Playwright gate.** TESTPLAN wants Snake/Tetris/.../a 3D
  flight sim scored by a headless-Playwright load→motion→input→game-probe gate.
  This pass has no browser harness; scoring is pure static/compile-only. Tasks
  authored here are small (functions/CLI/queries), not full games.
- **No N=6 hint-escalation loop.** TESTPLAN's planted-bug track is an iterative
  H0→H1→H2 self-correction loop with early-stop/DNF-loop tracking. This pass is
  one-shot: symptom given directly, one attempt, static-scored. The `symptom`
  field on each bugfix fixture is written at TESTPLAN's H1 tier (states the
  observed wrong behavior, not the root cause) so it's a reasonable base to
  build the H0/H2 escalation and loop protocol on top of later, but that
  protocol does not exist yet.
- **No self-debug track** (model's own broken one-shot code fed back).
- **`preflight()` is a lightweight fixture-existence selftest**, not TESTPLAN's
  "known-good gate" (which requires the not-yet-built Playwright harness to
  pass on every known-good game before any bugged variant counts).
- **`llmtest validate` does not lint B6 fixtures.** B1's fixture lint block in
  `validate_cmd.py` only scans `suite/b1_business/`; I did not extend it to
  `suite/b6_agenticcoding/` since that file wasn't in the requested change list.
  Fixture validity is currently enforced only at load time (`load_tasks()`
  raises on malformed/incomplete fixtures) and by the loader tests in
  `tests/test_b6.py`, not by the `llmtest validate` CI-equivalent gate. Flagging
  this as a gap worth closing before B6 fixtures scale past 10 tasks.

## Known limitation: literal `absent` regression checks are reformat-fragile

`regression_signals` uses exact literal-substring matching against the known
buggy line (e.g. `"def summarize(nums)\n"`, `"range(len(nums) - 1)"`). A model
that fixes the bug but also reformats the surrounding line (e.g. reflows a
multi-line signature, changes whitespace) could produce a false-negative
regression check even though the bug is genuinely fixed. `fix_signals` (regex,
more tolerant of surrounding whitespace) is the primary positive evidence for
"root cause addressed"; `regression_signals` is a secondary no-op/DNF detector.
Both feed `det_checks` as evidence alongside the judge pass, not as a hard gate
— consistent with how B1 treats `det_checks` (informative, not blocking).

## Test count

35 new tests in `tests/test_b6.py`: fixture loader (6), code extraction (4),
signal checking (5), compile-safety (4, incl. the exec/eval-patch tests),
`plan()` (3), `execute()` (9, covering scratch-correct/wrong-symbol/
broken-syntax/no-fence and bugfix-correct-fix/no-op/non-python-no-compile/
sampling), `preflight()` (2), plus one end-to-end exec/eval-safety test through
the full `execute()` path.

Full suite: **211 passed** (176 pre-existing + 35 new), `python -m llmtest
validate` exits 0 (71 pre-existing rows checked, 0 errors — `results/` was not
touched).

## Ambiguities / things to review before GPU runs

1. **Scope cut vs TESTPLAN** (above) — is the static-only, no-Playwright,
   one-shot-only shape acceptable as "Battery 6" for the P8 baseline, or does
   this need a follow-up phase before baseline runs count it? The task
   instructions asked for exactly this narrower shape, but TESTPLAN §5.6 is the
   approved spec and this build does not satisfy it in full.
2. **`llmtest validate` doesn't lint B6 fixtures yet** (above) — worth a
   follow-up task to extend `validate_cmd.py`'s fixture-lint block the way it
   already does for B1.
3. **Token budgets are a guess.** `max_tokens_by_track: {scratch: 6000, bugfix:
   4000}` is sized by analogy to B1's reasoning-model-headroom lesson, not
   measured — first real GPU run should watch for `finish_reason=length` /
   `predicted_n==max_tokens` on empty-code rows the way B1's P3 Task 12 finding
   did, and bump if reasoning models burn the whole budget on hidden thinking
   before ever emitting the fenced block.
4. **10 tasks, not 8-12's upper end** — within the requested 8-12 range but on
   the lower side; easy to add more fixtures later since `load_tasks()` picks up
   any `task-*.yaml` file with no code changes required.
5. **JS/bash/sql get no static syntax check** (compile() is Python-only per the
   contract) — their scoring leans more heavily on signal checks + the judge
   than the Python tasks do. This is what was asked for, but it does mean a
   syntactically-broken bash/SQL answer that happens to contain the required
   substrings scores identically on det_checks to a syntactically valid one;
   the judge is the only backstop for gross syntax breakage in those languages.
