## Decision

Do both, in this order:

1. Invest locally in making B8 valid and building a genuinely discriminating task pool.
2. Then use Blackwell for model diversity and confirmatory measurement.

Do not spend more time repeating gpt-oss-20b on the current tasks, and do not top up the rental yet. A Blackwell run today would produce more precise-looking numbers from an invalid instrument. Conversely, one locally viable model can never establish a ranking, regardless of task count.

The current results should be labeled “B8 development/smoke results,” not leaderboard evidence. The latest 24/25 is five distinct tasks repeated five times—not 25 independent samples of coding ability.

## Ordered next steps

1. **Freeze the claim and scoring contract**

   Rank exact deployed systems:

   `model weights + quantization + server + chat template + adapter + OpenCode version + budgets`

   Primary metric: end-to-end oracle completion under a fixed resource envelope.

   Failure classification must be downstream and unable to affect the score or denominator:

   - `a`, `b`, `c`, and model-caused budget exhaustion count as failures.
   - Deterministically verified infrastructure failures are retried under a predeclared rule and excluded if unresolved.
   - A panel vote of `d` must never exclude a run from scoring.
   - Rank “B8/OpenCode systems,” not abstract model families.

   Start a new B8/scorer version after these changes. Do not mix current rows with the confirmatory run.

2. **Contain agent execution—hard blocker before any scaled run**

   Build a pinned Node/OpenCode runner image and use one disposable container or VM per attempt:

   - Non-root, read-only root filesystem, isolated process namespace.
   - Fresh OpenCode home/database per run.
   - Only an ephemeral task workspace writable.
   - No repository, result store, hidden oracle, Docker socket, credentials, or inherited host environment mounted.
   - No public network; permit only the inference endpoint through an isolated proxy.
   - Disable `webfetch` unless a task explicitly requires controlled network access.
   - PID, CPU, RAM, disk, output-size, and wall-time caps.
   - Kill the entire namespace before copying the result to a separate oracle container.

   Current code explicitly uses host execution and inherited environment, while sandbox execution is unimplemented: [opencode.py](/D:/BUILT-TOOLS/LLMtesting/llmtest-v2/llmtest/harness/opencode.py:303), [b8_harness.py](/D:/BUILT-TOOLS/LLMtesting/llmtest-v2/llmtest/batteries/b8_harness.py:304).

3. **Make completion scoring defensible**

   Before adding tasks, harden the evaluator:

   - Enforce token, step/tool-call, wall-time, and output limits externally. Today tokens and steps are provenance only: [suite.yaml](/D:/BUILT-TOOLS/LLMtesting/llmtest-v2/config/suite.yaml:243).
   - Give every task a reference solution, at least one alternate valid solution, and several plausible incorrect or shortcut patches.
   - Require deterministic repeatability in fresh containers.
   - Detect deletions, symlinks, special files, mode changes, test/config tampering, and all out-of-scope modifications.
   - Make oracle output machine-readable: stage, reason code, failing case, expected, actual, exit status. Avoid free-form stderr as the authoritative explanation.
   - Pin task, oracle, container, dependency, harness, prompt, tool schema, chat-template, model, and server hashes.
   - Add scripted pass/fail/timeout/tamper agents as harness conformance tests.

   The `.pyc` incident proves scorer validity is not theoretical: it changed the apparent result from 16/30 to 30/30.

4. **Fix the classifier, but demote it to diagnostics**

   The classifier bug invalidates the “why” labels, not completion—unless labels influence eligibility.

   Redesign it as:

   - Deterministic infrastructure validity first.
   - Deterministic parser/schema status second.
   - Panel only for genuinely ambiguous tool-misuse versus solution-logic cases.
   - Explicit `unknown/ambiguous`, with human review for every proposed harness bug.
   - Structured oracle rejection evidence included.
   - Typed, bounded, sanitized event summaries instead of verbatim tool input/output.

   Current rendering includes raw tool inputs and outputs and passes only the completion Boolean, not the oracle detail: [failure_class.py](/D:/BUILT-TOOLS/LLMtesting/llmtest-v2/llmtest/harness/failure_class.py:348), [classify_b8_local.py](/D:/BUILT-TOOLS/LLMtesting/llmtest-v2/scripts/classify_b8_local.py:213).

   Important nuance: an oracle assertion such as `90 -> B, expected A` proves the patch is wrong, but does not alone distinguish `b` from `c`; the ordered trace still determines whether bad tool use prevented a real solution attempt.

   Validate the classifier on a double-adjudicated gold corpus, including injected instructions. A reasonable publication gate is ≥0.90 macro-F1, ≥0.85 recall per class, and no successful label flips in the injection suite.

5. **Build breadth locally, not just “harder” microtasks**

   Create approximately 50–70 candidate development tasks and retain roughly 40–50 per frozen form. Difficulty should come from search, coupling, and invariants—not riddles.

   Cover at least:

   - Distant symptom/root-cause localization.
   - Cross-module contracts and backward compatibility.
   - Feature integration into an existing architecture.
   - Stateful/resource-lifecycle behavior.
   - Build, configuration, serialization, and migration failures.
   - Robustness, security, and performance edge cases.

   Use three or more ecosystems if the claim is general coding; otherwise explicitly call B8 Python-only. Limit shared repositories/task families because related tasks are statistically dependent.

   Keep the current tasks as regression and easy-anchor items, but give them little leaderboard weight. Use separate development and sealed confirmatory forms. Any task used to debug B8 is burned from the final holdout.

6. **Spend the existing Blackwell balance on a pilot—not a leaderboard**

   Once the local gates pass, run approximately:

   - 3–4 model configurations.
   - 8–12 development tasks.
   - 1–2 attempts per cell.
   - Randomized, paired task blocks.

   Include gpt-oss-20b as the cross-machine anchor, one weaker configuration, and at least one larger challenger. Qwen should initially be “not evaluated—adapter unqualified.” First pass canned protocol-conformance cases through its exact chat template and provider path. If that path is correct and Qwen still emits malformed calls, those become legitimate end-to-end compatibility failures.

   Do not combine 5090 and Blackwell leaderboard rows. Use cross-machine anchor runs only to detect backend confounding; conduct the official matrix in one environment.

   Go forward only if:

   - Zero containment or budget-control failures.
   - Infrastructure-invalid rate below roughly 2–5%.
   - At least several pilot tasks produce model-discordant outcomes.
   - Best-to-worst spread is around 15 points or greater.
   - Cost per valid attempt supports the powered full design.

   Use the existing balance with a hard spend cap. Top up only after measuring cost per valid attempt and estimating the confirmatory total plus approximately 25% rerun contingency.

7. **Run a sealed, multi-model confirmation**

   A strong design is:

   - 4–6 configurations on a frozen 48-task screening form.
   - Top contenders on a fresh 48-task confirmatory form.
   - A preselected 20–25% task subset receiving extra attempts to estimate stochasticity.

   Most compute should buy distinct tasks, not five repeats of the same task. Average repeats within each task first, then bootstrap over repositories/task families. Use paired simultaneous confidence intervals and a predeclared practical margin—10 percentage points is realistic for an initial coarse ranking.

   Publish tiers or a partial ordering. If intervals overlap the practical margin, say “unresolved”; do not force ranks 1–6.

8. **Add another harness only after OpenCode discriminates**

   More harnesses now multiply confounds. Later, run a small complete crossover—at least two harnesses, three representative models, and roughly 24 shared tasks—to measure model×harness interaction.

   If ordering reverses materially, publish harness-specific rankings. Never average “best harness per model,” which rewards unequal integration effort.

## Biggest threats to validity, in order

1. Uncontained execution, oracle leakage, and cross-run contamination.
2. False-positive/false-negative scorer behavior and mixing scorer versions.
3. Ceiling effects, only five independent hard tasks, and treating repetitions as task breadth.
4. Confounding model capability with chat-template/provider/OpenCode compatibility.
5. Unenforced budgets and inconsistent treatment of timeouts or infrastructure failures.
6. Benchmark contamination from repeatedly tuning against evaluated tasks.
7. Prompt-injected, evidence-starved failure classification.
8. Inadequate uncertainty analysis and forced total rankings.

The classifier mislabel is real, but it is not currently the largest threat to the ranking. The completion instrument, task population, and model–harness confound are. Fix those locally; use Blackwell only when the measurement system is frozen enough that additional models generate information rather than more debugging data.
EXIT=0
