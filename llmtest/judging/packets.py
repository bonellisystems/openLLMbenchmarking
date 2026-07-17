"""Judging packets -- cohort build, blinding, committed maps, scrubbing (TESTPLAN 6.1).

`build_cohort_packets` groups needs_judging result rows into per-(task, run)
cohorts, pairs each cohort with the two calibration answers, blinds the
member identities behind per-judge letter permutations, and writes:

- a committed map per packet: results/packets/<packet_id>.map.json
  (letters_by_judge, base_seed, rubric_sha, task_id, run_n, unit) -- this is
  the ONLY place the letter->identity mapping is recorded, so it must be
  committed to git (gitignore covers artifacts/packets/, not results/packets/).
- one packet body per judge: artifacts/packets/<packet_id>.<judge_id>.txt --
  gitignored, regenerable from the map + rows + grading/ content.

Content-hashed ids (packet_id, rubric_sha, base_seed) mean re-running the
builder over the same inputs is idempotent, and editing an anchor file or
judge_prompt.md automatically re-mints affected packets (new rubric_sha ->
new packet_id) without touching fixture_sha / row identity.

Challenger packet mode (TESTPLAN 6.2 -- single-answer spot checks against a
frozen reference set) is explicitly OUT OF SCOPE here; it is deferred to the
P7 intake plan. This module only builds full-cohort packets.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llmtest.batteries.b1_fixtures import check_signals, load_unit_tasks

_LETTERS = [chr(ord("A") + i) for i in range(26)]

# Best-effort scrub of self-identification phrasing and known model/vendor
# names. This is NOT a security boundary -- a determined model can still leak
# its identity through style; kin-delta (aggregate.py, Task 8) is the
# statistical backstop for that.
_SELF_ID_RE = re.compile(
    r"as an? (?:ai|assistant) (?:developed|created|made) by \w+",
    re.IGNORECASE,
)
_VENDOR_RE = re.compile(
    r"\b(?:gpt|openai|gemma|google|qwen|alibaba|claude|anthropic|"
    r"nemotron|nvidia|granite|ibm)\b",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    """Redact self-identification strings and model/vendor names with [model].

    Case-insensitive, best-effort. See module docstring re: kin-delta backstop.
    """
    text = _SELF_ID_RE.sub("[model]", text)
    text = _VENDOR_RE.sub("[model]", text)
    return text


@dataclass
class PacketRecord:
    packet_id: str
    task_id: str
    run_n: int
    unit: str
    rubric_sha: str
    bodies: dict = field(default_factory=dict)   # judge_id -> Path
    map_path: Path | None = None
    skipped_reason: str | None = None
    cal_fallback: bool = False   # True when the global strong.md/weak.md pair
                                  # was used because no per-task calibration
                                  # file (grading/calibration/<unit>/<task>.yaml)
                                  # was found.


def _unit_from_task_id(task_id: str) -> str:
    """'b1.cybersecurity-01' -> 'cybersecurity' (unit is everything before
    the trailing '-NN' task number)."""
    after_battery = task_id.split(".", 1)[1] if "." in task_id else task_id
    unit, sep, _num = after_battery.rpartition("-")
    return unit if sep else after_battery


def _task_suffix(task_id: str) -> str:
    """'b1.cybersecurity-01' -> 'cybersecurity-01' (matches Task.id from the loader)."""
    return task_id.split(".", 1)[1] if "." in task_id else task_id


def _answer_artifact(artifacts: dict) -> tuple[dict | None, list[str] | None]:
    """Pick the artifact entry that holds the model's answer text.

    Primary: the canonical TESTPLAN "response" key. Sole-entry fallback
    covers rows that write exactly one artifact under a different key. When
    there are MULTIPLE entries and none is "response", the answer is
    genuinely ambiguous -- we don't know which one is the model's answer --
    so this returns (None, sorted(keys)) instead of guessing, letting the
    caller report that distinctly from a plain missing-file skip.

    Returns (artifact_dict_or_None, ambiguous_keys_or_None).
    """
    if "response" in artifacts:
        return artifacts["response"], None
    if len(artifacts) == 1:
        return next(iter(artifacts.values())), None
    if len(artifacts) > 1:
        return None, sorted(artifacts)
    return None, None


def _read_artifact_text(root: Path, art: dict) -> str | None:
    if not art or "relpath" not in art:
        return None
    relpath = art["relpath"]
    # Battery artifact writers store relpath relative to artifacts/ (e.g.
    # "b1/<row_id>.txt", file lives at <root>/artifacts/b1/<row_id>.txt).
    # Accept a relpath that already includes the "artifacts/" prefix too,
    # defensively, since that convention isn't pinned down repo-wide yet.
    for candidate in (root / "artifacts" / relpath, root / relpath):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def _task_prompt(root: Path, unit: str, task_id: str) -> str | None:
    suffix = _task_suffix(task_id)
    for task in load_unit_tasks(root, unit):
        if task.id == suffix:
            return task.prompt
    return None


def _format_evidence(det_checks: dict) -> str:
    """Render the det-signal evidence block for one answer.

    Every letter in a packet (cohort answer or calibration reference) runs
    through this SAME formatter, so the evidence section is structurally
    identical for all of them -- only the pass/fail contents (which vary
    naturally) differ. This is deliberate: it must never be possible to
    pick out the calibration letters by evidence-block *shape* alone.
    """
    if not det_checks:
        return "Evidence: (no det-signal evidence)"
    lines = ["Evidence:", "", "| Check | Result |", "|---|---|"]
    for name in sorted(det_checks):
        result = det_checks[name]
        passed = result.get("pass") if isinstance(result, dict) else bool(result)
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    return "\n".join(lines)


def _resolve_calibration(
    calibration_dir: Path, unit: str, task_suffix: str,
) -> tuple[str | None, str | None, bool]:
    """Resolve the calibration (strong, weak) text pair for one task.

    Preference order (PLAN AMENDMENT, Task-5 review Finding 1c):
    1. Per-task pair at grading/calibration/<unit>/<task_suffix>.yaml,
       keys {strong, weak} (both required to count as present).
    2. Global fallback grading/calibration/strong.md + weak.md.

    Returns (strong_text, weak_text, used_fallback). When neither source is
    usable, returns (None, None, False) -- the caller skips the cohort.
    """
    per_task_path = calibration_dir / unit / f"{task_suffix}.yaml"
    if per_task_path.exists():
        try:
            data = yaml.safe_load(per_task_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
        if isinstance(data, dict) and "strong" in data and "weak" in data:
            return str(data["strong"]), str(data["weak"]), False

    strong_path = calibration_dir / "strong.md"
    weak_path = calibration_dir / "weak.md"
    if strong_path.exists() and weak_path.exists():
        return (strong_path.read_text(encoding="utf-8"),
                weak_path.read_text(encoding="utf-8"), True)

    return None, None, False


def build_cohort_packets(
    rows: list[dict],
    *,
    rubric_dir: Path,
    calibration_dir: Path,
    out_artifacts: Path,
    out_maps: Path,
    root: Path,
    judge_ids: list[str],
    cohort_models: list[str],
    judge_prompt_path: Path,
    signals_by_task: dict[str, list] | None = None,
) -> tuple[list[PacketRecord], list[dict]]:
    """Build blinded judging packets for every complete (battery, task, run) cohort.

    Args:
        rows: needs_judging result rows (dicts, as read from Store).
        rubric_dir: directory of per-unit anchor files (grading/anchors/<unit>.md).
        calibration_dir: directory holding calibration references -- either
            per-task grading/calibration/<unit>/<task>.yaml pairs (preferred)
            or the global strong.md / weak.md fallback.
        out_artifacts: directory packet bodies are written to (gitignored).
        out_maps: directory committed letter maps are written to.
        root: repo root -- used to resolve answer artifact files (root/artifacts/...)
            and to reload the task prompt via llmtest.batteries.b1_fixtures
            (root/suite/b1_business/<unit>/task-NN.yaml).
        judge_ids: the panel's judge ids, e.g. ["claude", "codex", "gemini"]
            (config/judges.yaml keys -- passed in by the caller for testability).
        cohort_models: the expected model set for a cohort to be "complete"
            (suite.yaml b1.cohort_models knob, wired by Task 12; default =
            full non-quant-arm roster is the caller's responsibility).
        judge_prompt_path: path to the judge_prompt.md template.
        signals_by_task: optional {task_id: [signal dicts]} map, used to run
            the calibration answers through the SAME check_signals() logic
            as cohort answers so every letter's evidence block is uniformly
            formatted (Task-5 review Finding 1 -- packets must never leak
            which letters are calibration references). When a task_id is
            absent from this map, ALL letters of that packet (cohort AND
            calibration) render an empty evidence table -- showing real
            det_checks for cohort answers only, while calibration answers
            go blank for lack of live signals, would itself be a structural
            tell, so the whole packet degrades uniformly instead.

    Returns:
        (packets, skipped) -- packets is a list of PacketRecord for every
        complete cohort with a resolvable anchor file, calibration pair, task
        prompt, and answer artifacts; skipped is a list of
        {battery, task_id, run_n, reason} dicts for every group that could
        not be turned into a packet.
    """
    signals_by_task = signals_by_task or {}
    root = Path(root)
    rubric_dir = Path(rubric_dir)
    calibration_dir = Path(calibration_dir)
    out_artifacts = Path(out_artifacts)
    out_maps = Path(out_maps)
    judge_prompt_path = Path(judge_prompt_path)

    out_artifacts.mkdir(parents=True, exist_ok=True)
    out_maps.mkdir(parents=True, exist_ok=True)

    judge_prompt_bytes = judge_prompt_path.read_bytes()
    judge_prompt_template = judge_prompt_bytes.decode("utf-8")

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if not row.get("needs_judging"):
            continue
        key = (row["battery"], row["task_id"], row["run_n"])
        groups.setdefault(key, []).append(row)

    packets: list[PacketRecord] = []
    skipped: list[dict] = []

    for (battery, task_id, run_n) in sorted(groups):
        group_rows = groups[(battery, task_id, run_n)]
        unit = _unit_from_task_id(task_id)

        def _skip(reason: str) -> None:
            skipped.append({"battery": battery, "task_id": task_id,
                             "run_n": run_n, "reason": reason})

        ok_by_model = {}
        for r in group_rows:
            if r.get("status") == "ok" and r.get("model_id") in cohort_models:
                ok_by_model[r["model_id"]] = r
        missing = sorted(set(cohort_models) - set(ok_by_model))
        if missing:
            _skip(f"incomplete cohort, missing models: {missing}")
            continue

        anchor_path = rubric_dir / f"{unit}.md"
        if not anchor_path.exists():
            _skip(f"missing anchor file: {anchor_path}")
            continue
        anchor_bytes = anchor_path.read_bytes()

        task_suffix = _task_suffix(task_id)
        strong_text, weak_text, cal_fallback = _resolve_calibration(
            calibration_dir, unit, task_suffix)
        if strong_text is None or weak_text is None:
            per_task_path = calibration_dir / unit / f"{task_suffix}.yaml"
            _skip(
                f"missing calibration: no per-task pair at {per_task_path} "
                f"and no global fallback in {calibration_dir}"
            )
            continue

        task_prompt = _task_prompt(root, unit, task_id)
        if task_prompt is None:
            _skip(f"could not load task prompt for {task_id} (unit={unit})")
            continue

        answers: dict[str, tuple[str, str, dict]] = {}  # model_id -> (row_id, text, det_checks)
        artifact_skip_reason = None
        for model_id, r in ok_by_model.items():
            art, ambiguous_keys = _answer_artifact(r.get("artifacts", {}) or {})
            if art is None:
                if ambiguous_keys is not None:
                    artifact_skip_reason = (
                        f"ambiguous artifact keys (no 'response'): {ambiguous_keys}")
                else:
                    artifact_skip_reason = f"missing artifact file for model {model_id}"
                break
            text = _read_artifact_text(root, art)
            if text is None:
                artifact_skip_reason = f"missing artifact file for model {model_id}"
                break
            answers[model_id] = (r["row_id"], text, r.get("det_checks", {}))
        if artifact_skip_reason:
            _skip(artifact_skip_reason)
            continue

        rubric_sha = hashlib.sha256(anchor_bytes + judge_prompt_bytes).hexdigest()

        member_row_ids = sorted(row_id for row_id, _t, _d in answers.values())
        answer_shas = sorted(
            [hashlib.sha256(text.encode("utf-8")).hexdigest()
             for _rid, text, _dc in answers.values()]
            + [hashlib.sha256(strong_text.encode("utf-8")).hexdigest(),
               hashlib.sha256(weak_text.encode("utf-8")).hexdigest()]
        )
        base_seed = hashlib.sha256(
            (task_prompt + "".join(answer_shas)).encode("utf-8")
        ).hexdigest()[:16]

        packet_id = hashlib.sha256(
            ("|".join(member_row_ids) + rubric_sha + base_seed).encode("utf-8")
        ).hexdigest()

        identities = list(member_row_ids) + ["CAL-strong", "CAL-weak"]
        letters = _LETTERS[:len(identities)]

        letters_by_judge = {}
        for judge_id in judge_ids:
            seed_hex = hashlib.sha256(
                (base_seed + judge_id).encode("utf-8")).hexdigest()
            rng = random.Random(seed_hex)
            shuffled = identities[:]
            rng.shuffle(shuffled)
            letters_by_judge[judge_id] = dict(zip(letters, shuffled))

        by_row_id = {row_id: (text, dc) for _mid, (row_id, text, dc) in answers.items()}

        map_record = {
            "letters_by_judge": letters_by_judge,
            "base_seed": base_seed,
            "rubric_sha": rubric_sha,
            "task_id": task_id,
            "run_n": run_n,
            "unit": unit,
            "cal_fallback": cal_fallback,
        }
        map_path = out_maps / f"{packet_id}.map.json"
        map_path.write_text(
            json.dumps(map_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Uniform evidence (Task-5 review Finding 1): every letter -- cohort
        # answer or calibration reference -- goes through _format_evidence()
        # with no special-cased marker text. When this task's signals aren't
        # available, the WHOLE packet degrades to empty evidence uniformly
        # rather than let CAL letters stand out as the only blank ones.
        task_signals = signals_by_task.get(task_id)
        if task_signals is None:
            evidence_by_identity = {identity: _format_evidence({}) for identity in identities}
        else:
            evidence_by_identity = {}
            for row_id, (_text, dc) in by_row_id.items():
                evidence_by_identity[row_id] = _format_evidence(dc)
            evidence_by_identity["CAL-strong"] = _format_evidence(
                check_signals(strong_text, task_signals))
            evidence_by_identity["CAL-weak"] = _format_evidence(
                check_signals(weak_text, task_signals))

        bodies: dict[str, Path] = {}
        for judge_id in judge_ids:
            letter_map = letters_by_judge[judge_id]
            parts = []
            for letter in letters:
                identity = letter_map[letter]
                if identity in ("CAL-strong", "CAL-weak"):
                    raw = strong_text if identity == "CAL-strong" else weak_text
                else:
                    raw, _dc = by_row_id[identity]
                evidence = evidence_by_identity[identity]
                parts.append(f"### Answer {letter}\n\n{evidence}\n\n{scrub(raw)}\n")
            answers_block = "\n".join(parts)

            body = judge_prompt_template.format(
                anchors=anchor_bytes.decode("utf-8"),
                task_prompt=task_prompt,
                letters=", ".join(letters),
                answers_block=answers_block,
            )
            body_path = out_artifacts / f"{packet_id}.{judge_id}.txt"
            body_path.write_text(body, encoding="utf-8")
            bodies[judge_id] = body_path

        packets.append(PacketRecord(
            packet_id=packet_id, task_id=task_id, run_n=run_n, unit=unit,
            rubric_sha=rubric_sha, bodies=bodies, map_path=map_path,
            cal_fallback=cal_fallback,
        ))

    return packets, skipped
