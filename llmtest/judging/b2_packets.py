"""B2 axis-keyed judging packets (agentic-quality v2.1 design spec, Part 1).

`build_b2_axis_packets` mirrors `llmtest.judging.packets.build_cohort_packets`
in shape -- it groups needs_judging result rows, pairs each group with a
frozen calibration pair, blinds member identities behind per-judge letter
permutations, and writes a committed map (`results/packets/<packet_id>.map.json`)
plus one gitignored packet body per judge (`artifacts/packets/<packet_id>.<judge_id>.txt`)
built from `grading/judge_prompt.md`.

It differs from `build_cohort_packets` in ways specific to B2's judged axes
(5 = error-recovery, 8 = faithfulness-to-tool-results):

- **Cohort key is `(task_id, axis, run_n)`, not `(task_id, run_n)`** (spec
  1.1). Axis is folded into `packet_id`'s (and `base_seed`'s) hash preimage
  so axis-5 and axis-8 answers to the same scenario/run can never collide or
  blend into one packet, even if a future scenario carries both axes.
- **Rubric text and task prompt come from the fixture itself** (spec 1.2):
  `task.rubric["axis_N"]` fills the judge template's `anchors` slot, and the
  full tool interaction (`json.dumps({"tools": task.tools, "messages":
  task.messages})` -- tool contract + prompt + injected tool error/result +
  the model's recovery) fills `task_prompt`. There is no separate
  `grading/anchors/<unit>.md` file for B2 axes.
- **`fixture_sha` verification is per-row** (spec 1.2): a row whose
  `fixture_sha` doesn't match the loaded fixture's is excluded from that
  packet (never silently rebuilt against a mutated fixture) and reported in
  `skipped`; the rest of the cohort can still build if quorum allows.
- **Quorum, not "every cohort model"** (spec 1.6): a packet builds once
  `len(present_models) >= quorum`; models with no verified row are recorded
  as `missing_models` on the map rather than suppressing the whole packet.
- **CAL is a single frozen per-axis pair**, not per-task-with-fallback:
  `grading/calibration/b2/<dim.value>.yaml` (Task 2 output), read directly
  via `dimension.cal_ref` -- there is no global strong.md/weak.md tier for B2.
- **Blinding seed additionally folds in `packet_id`** (spec 1.7 hardening):
  `random.Random(sha256(base_seed + packet_id + judge_id))`, one step beyond
  `build_cohort_packets`'s `sha256(base_seed + judge_id)`.
- **The map records `dim`, `scenario`, `run_n`, `present_models`,
  `missing_models`, and per-model `fabrication_pass`** (bool, from the row's
  `det_checks["axis{N}_fabrication_guard"]["pass"]`) so Task 5's aggregation
  can apply the fabrication-guard hard cap without re-reading raw rows.

No evidence table is rendered per answer here (unlike `build_cohort_packets`'s
det-signal table) -- B2 doesn't have B1's free-text signal engine, and the
brief's reuse list omits `_format_evidence`; the answer body is the scrubbed,
answer-only rendering of the artifact (see `_render_answer`) -- never the raw
`/v1/chat/completions` envelope, which carries identity side channels
(`model`, `timings.predicted_per_second`, etc.) that must never reach a
blinded packet.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import yaml

from llmtest.batteries.b2_fixtures import _response_text, extract_tool_calls, load_tasks
from llmtest.judging.dimension import Dim, cal_ref, resolve_dims
from llmtest.judging.packets import _LETTERS, PacketRecord, _read_artifact_text, scrub


def _render_answer(raw_text: str) -> str:
    """Parse a B2 artifact's raw text into an answer-ONLY rendering.

    Real B2 artifacts (artifacts/b2/<sha>.json) are the whole
    `/v1/chat/completions` response envelope -- id, usage, timings (incl.
    `predicted_per_second`, a strong model-identity side channel), model
    path/name strings the scrub() regex won't catch, finish_reason, etc.
    Embedding that raw text verbatim as "the answer" would defeat the
    letter-permutation blinding. So: json.loads the artifact and keep ONLY
    the model's actual answer -- `choices[0].message.content` and
    `.tool_calls`, via the SAME parsers score_axes() uses
    (b2_fixtures._response_text / extract_tool_calls) -- rendered as
    readable text, never the envelope.

    Falls back to treating `raw_text` as the answer verbatim when it isn't
    valid JSON (or isn't a JSON object), so synthetic/plain-text artifacts
    (as written by tests, or by any future non-JSON transport) still work.
    """
    try:
        response = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return raw_text
    if not isinstance(response, dict):
        return raw_text

    content = _response_text(response)
    calls = extract_tool_calls(response)

    parts = []
    if content:
        parts.append(content)
    for call in calls:
        fn = call.get("function") or {}
        name = fn.get("name", "<unknown tool>")
        args = fn.get("arguments", "")
        parts.append(f"[tool call: {name}({args})]")
    return "\n\n".join(parts)


def _load_b2_calibration(root: Path, dim: Dim) -> tuple[str | None, str | None]:
    """Load the frozen (strong, weak) CAL pair for one B2 judged axis.

    Unlike B1's per-task-pair-with-global-fallback tiering, B2 CAL is a
    single frozen pair per axis at grading/calibration/b2/<dim.value>.yaml
    (Task 2) -- every scenario that carries this axis reuses the same
    author-pinned pair. Returns (None, None) when the file is missing or
    malformed (caller skips the whole axis-group).
    """
    cal_path = cal_ref(dim, root)
    if not cal_path.exists():
        return None, None
    try:
        data = yaml.safe_load(cal_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None, None
    if not isinstance(data, dict) or "strong" not in data or "weak" not in data:
        return None, None
    return str(data["strong"]), str(data["weak"])


def build_b2_axis_packets(
    rows: list[dict],
    *,
    root: Path,
    judge_ids: list[str],
    cohort_models: list[str],
    quorum: int,
    out_maps: Path,
    out_artifacts: Path,
    judge_prompt_path: Path,
) -> tuple[list[PacketRecord], list[dict]]:
    """Build blinded B2 axis-judging packets for every (task, axis, run)
    group that clears fixture_sha verification and quorum.

    Args:
        rows: B2 result rows (dicts, as read from Store). Only
            battery=2, needs_judging=True, status="ok" rows are considered;
            everything else is ignored outright (not reported as a skip,
            mirroring build_cohort_packets: `skipped` only covers rows/groups
            that were actual judging candidates).
        root: repo root -- resolves suite/b2_toolcalling fixtures
            (llmtest.batteries.b2_fixtures.load_tasks), answer artifacts
            (root/artifacts/... via packets._read_artifact_text), and the
            per-axis CAL yaml (grading/calibration/b2/<axis>.yaml).
        judge_ids: the panel's judge ids.
        cohort_models: the expected model roster. Rows from models outside
            this set are ignored; roster models with no verified row for a
            given (task, axis, run) show up as `missing_models` on the map.
        quorum: minimum number of present (fixture_sha-verified, cohort)
            models required for a packet to build (spec 1.6) -- may be lower
            than len(cohort_models); a shortfall is a skip, never a silently
            smaller anonymous cohort.
        out_maps: directory committed letter maps are written to.
        out_artifacts: directory packet bodies are written to (gitignored).
        judge_prompt_path: path to the judge_prompt.md template (shared
            with B1's build_cohort_packets).

    Returns:
        (packets, skipped) -- packets is a list of PacketRecord, one per
        (task_id, axis, run_n) group that cleared verification + quorum;
        skipped is a list of {battery, task_id, run_n, reason, ...} dicts
        (an `axis` key is present once axis resolution has happened; a bare
        `model` key marks a single-row exclusion such as a fixture_sha
        mismatch, distinct from a whole-group quorum/calibration skip).
    """
    root = Path(root)
    out_maps = Path(out_maps)
    out_artifacts = Path(out_artifacts)
    judge_prompt_path = Path(judge_prompt_path)

    out_maps.mkdir(parents=True, exist_ok=True)
    out_artifacts.mkdir(parents=True, exist_ok=True)

    tasks = {f"b2.{t.id}": t for t in load_tasks(root)}
    template = judge_prompt_path.read_text(encoding="utf-8")

    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        if r.get("battery") != 2 or not r.get("needs_judging") or r.get("status") != "ok":
            continue
        groups.setdefault((r["task_id"], r["run_n"]), []).append(r)

    packets: list[PacketRecord] = []
    skipped: list[dict] = []

    for (task_id, run_n) in sorted(groups):
        group_rows = groups[(task_id, run_n)]
        task = tasks.get(task_id)
        if task is None:
            skipped.append({"battery": 2, "task_id": task_id, "run_n": run_n,
                             "reason": f"no fixture loaded for {task_id!r}"})
            continue

        for dim in resolve_dims(2, task_id, task.axes):
            axis_num = int(dim.value.replace("axis", ""))

            def _skip(reason: str, **extra) -> None:
                skipped.append({"battery": 2, "task_id": task_id, "run_n": run_n,
                                 "axis": dim.value, "reason": reason, **extra})

            members: dict[str, tuple[str, str, dict]] = {}  # model_id -> (row_id, text, det_checks)
            for r in group_rows:
                model_id = r.get("model_id")
                if model_id not in cohort_models:
                    continue
                if r.get("fixture_sha") != task.fixture_sha:
                    _skip("fixture_sha mismatch: row does not match on-disk fixture",
                          model=model_id)
                    continue
                art = (r.get("artifacts") or {}).get("response")
                raw_text = _read_artifact_text(root, art) if art else None
                if raw_text is None:
                    _skip("missing artifact file", model=model_id)
                    continue
                text = _render_answer(raw_text)
                det = (r.get("det_checks") or {}).get(f"axis{axis_num}_fabrication_guard") or {}
                members[model_id] = (r.get("row_id"), text, det)

            present = sorted(members)
            missing = sorted(set(cohort_models) - set(present))
            if len(present) < quorum:
                _skip(f"quorum not met: {len(present)} present < {quorum} required",
                      present_models=present, missing_models=missing)
                continue

            rubric_text = (task.rubric or {}).get(f"axis_{axis_num}")
            if not rubric_text:
                _skip(f"fixture has no rubric.axis_{axis_num} text",
                      present_models=present, missing_models=missing)
                continue

            strong_text, weak_text = _load_b2_calibration(root, dim)
            if strong_text is None or weak_text is None:
                _skip(f"missing/malformed calibration: {cal_ref(dim, root)}",
                      present_models=present, missing_models=missing)
                continue

            # Spec 1.2: the body must show the fixture's tool contract
            # (task.tools) alongside task.messages, not messages alone --
            # the judge needs to see what tools the model actually had.
            interaction = json.dumps(
                {"tools": task.tools, "messages": task.messages}, ensure_ascii=False)

            member_row_ids = sorted(row_id for row_id, _t, _d in members.values())
            answer_shas = sorted(
                [hashlib.sha256(text.encode("utf-8")).hexdigest()
                 for _rid, text, _dc in members.values()]
                + [hashlib.sha256(strong_text.encode("utf-8")).hexdigest(),
                   hashlib.sha256(weak_text.encode("utf-8")).hexdigest()]
            )
            # Axis folded into base_seed AND packet_id (spec 1.1) -- the
            # single change that makes axis a first-class part of packet
            # identity, so axis-5/axis-8 packets for the same scenario/run
            # never collide even if member sets happened to coincide.
            base_seed = hashlib.sha256(
                (interaction + dim.value + "".join(answer_shas)).encode("utf-8")
            ).hexdigest()[:16]
            rubric_sha = hashlib.sha256((rubric_text + template).encode("utf-8")).hexdigest()
            packet_id = hashlib.sha256(
                ("|".join(member_row_ids) + rubric_sha + base_seed + dim.value).encode("utf-8")
            ).hexdigest()

            identities = list(member_row_ids) + ["CAL-strong", "CAL-weak"]
            letters = _LETTERS[:len(identities)]

            letters_by_judge = {}
            for judge_id in judge_ids:
                # Spec 1.7 hardening: fold packet_id into the blinding seed
                # (build_cohort_packets only folds in base_seed + judge_id).
                seed_hex = hashlib.sha256(
                    (base_seed + packet_id + judge_id).encode("utf-8")).hexdigest()
                rng = random.Random(seed_hex)
                shuffled = identities[:]
                rng.shuffle(shuffled)
                letters_by_judge[judge_id] = dict(zip(letters, shuffled))

            by_row_id = {row_id: text for _mid, (row_id, text, _dc) in members.items()}
            fabrication_pass = {
                model_id: (det.get("pass") if isinstance(det, dict) else None)
                for model_id, (_rid, _t, det) in members.items()
            }

            map_record = {
                "letters_by_judge": letters_by_judge,
                "base_seed": base_seed,
                "rubric_sha": rubric_sha,
                "task_id": task_id,
                "run_n": run_n,
                "dim": dim.value,
                "scenario": task.id,
                "present_models": present,
                "missing_models": missing,
                "fabrication_pass": fabrication_pass,
            }
            map_path = out_maps / f"{packet_id}.map.json"
            map_path.write_text(
                json.dumps(map_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            bodies: dict[str, Path] = {}
            for judge_id in judge_ids:
                letter_map = letters_by_judge[judge_id]
                parts = []
                for letter in letters:
                    identity = letter_map[letter]
                    if identity in ("CAL-strong", "CAL-weak"):
                        raw = strong_text if identity == "CAL-strong" else weak_text
                    else:
                        raw = by_row_id[identity]
                    parts.append(f"### Answer {letter}\n\n{scrub(raw)}\n")
                answers_block = "\n".join(parts)

                body = template.format(
                    anchors=rubric_text,
                    task_prompt=interaction,
                    letters=", ".join(letters),
                    answers_block=answers_block,
                )
                body_path = out_artifacts / f"{packet_id}.{judge_id}.txt"
                body_path.write_text(body, encoding="utf-8")
                bodies[judge_id] = body_path

            packets.append(PacketRecord(
                packet_id=packet_id, task_id=task_id, run_n=run_n, unit=dim.value,
                rubric_sha=rubric_sha, bodies=bodies, map_path=map_path,
            ))

    return packets, skipped
