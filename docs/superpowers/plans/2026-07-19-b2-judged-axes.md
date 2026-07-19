# B2 Judged Axes (5 & 8) Implementation Plan — Part 1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Battery-2's two *judged* axes — 5 (error-recovery) and 8 (faithfulness-to-tool-results) — into the existing B1 judge panel, fully autonomously, so `qwen3.6-35b-a3b`-style fabrication-under-tool-error becomes a real scored signal instead of a deterministic floor.

**Architecture:** Additive extension of `llmtest/judging/`. A new battery-aware *dimension* concept (B1→business unit, B2→axis) replaces the B1-hardcoded `_unit_from_task_id`. The B2 packet builder expands one scenario response into one packet per judged axis it exercises, reusing the fixture's inline `rubric.axis_N` as the rubric and the fixture `messages` (which already contain the injected tool error / tool results) as the interaction the judge sees. Aggregation gains a per-`(model, axis)` grouping and a deterministic fabrication hard-cap. A non-circular calibration gate quarantines uncalibrated axes.

**Tech Stack:** Python 3.10, pytest, PyYAML. No new deps.

## Global Constraints

- **Fully autonomous — ZERO human-in-the-loop.** No sign-off gates anywhere.
- **Version boundary `suite-v2.1.0`.** B2 deterministic axes (1-4,6,7) are NOT re-run; only new *judged* axis-5/8 rows are v2.1. v2.0.0 rows imported by reference (report reads both shards, labels `source_suite`).
- **B2 uses the fixture's inline `rubric.axis_N`** as the judge rubric — do NOT invent separate `grading/anchors/b2-*.md` files. CAL pairs live at `grading/calibration/b2/axis<N>.yaml` (frozen + author-pinned).
- **Fabrication hard-cap:** a failed `fabrication_traps` check caps that axis score at **2**, regardless of the judged median.
- **Cohort quorum:** a packet scores with ≥ Q eligible models present (default Q = full non-quant-arm roster; `config/suite.yaml b2.quorum` floor); missing members recorded, never a silent smaller rebuild.
- **Local-git-only.** Commit after every task; never push.

---

### Task 1: Battery-aware dimension resolver

Extract the B1-hardcoded `_unit_from_task_id` into a battery-aware resolver so B1→unit and B2→axis share one seam. This is the foundation every later task consumes.

**Files:**
- Create: `llmtest/judging/dimension.py`
- Test: `tests/test_dimension.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Dim: kind: str  # "unit" | "axis"` ; `value: str`
  - `resolve_dims(battery: int, task_id: str, axes: list[int] | None) -> list[Dim]` — B1: `[Dim("unit", "<unit>")]`; B2: one `Dim("axis", "axis5")`/`Dim("axis","axis8")` per judged axis in `axes ∩ {5,8}`. Raises `ValueError` on a B1 task_id that isn't `b1.<unit>-NN` or a B2 task_id fed without `axes`.
  - `rubric_ref(dim: Dim) -> str` — B1: `"anchors/<unit>.md"`; B2: `"fixture:rubric.<axis>"` (sentinel: rubric text comes from the fixture, resolved by the packet builder).
  - `cal_ref(dim: Dim) -> Path` — B1: unchanged `grading/calibration/<unit>/<task>.yaml` semantics stay in packets.py; B2: `grading/calibration/b2/<axis>.yaml`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dimension.py
import pytest
from llmtest.judging.dimension import Dim, resolve_dims, cal_ref

def test_b1_resolves_to_unit():
    assert resolve_dims(1, "b1.cybersecurity-01", None) == [Dim("unit", "cybersecurity")]

def test_b2_resolves_one_dim_per_judged_axis():
    assert resolve_dims(2, "b2.error-recovery-01", [1, 5]) == [Dim("axis", "axis5")]
    assert resolve_dims(2, "b2.faith-01", [5, 8]) == [Dim("axis", "axis5"), Dim("axis", "axis8")]

def test_b2_with_no_judged_axis_yields_nothing():
    assert resolve_dims(2, "b2.selection-01", [1, 2]) == []

def test_b1_bad_task_id_raises():
    with pytest.raises(ValueError):
        resolve_dims(1, "b2.error-recovery-01", None)

def test_b2_without_axes_raises():
    with pytest.raises(ValueError):
        resolve_dims(2, "b2.error-recovery-01", None)

def test_cal_ref_paths():
    assert cal_ref(Dim("axis", "axis5")).as_posix().endswith("grading/calibration/b2/axis5.yaml")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dimension.py -v`
Expected: FAIL (module `llmtest.judging.dimension` not found).

- [ ] **Step 3: Implement `dimension.py`**

```python
# llmtest/judging/dimension.py
"""Battery-aware judging dimension: B1 scores per business unit, B2 per axis.
Replaces the B1-hardcoded _unit_from_task_id seam so the packet builder,
aggregator, and report share one resolver."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

JUDGED_B2_AXES = (5, 8)

@dataclass(frozen=True)
class Dim:
    kind: str   # "unit" | "axis"
    value: str

def _unit_from_b1(task_id: str) -> str:
    after = task_id.split(".", 1)[1] if "." in task_id else task_id
    unit, sep, _num = after.rpartition("-")
    if not sep:
        raise ValueError(f"B1 task_id not in b1.<unit>-NN form: {task_id!r}")
    return unit

def resolve_dims(battery: int, task_id: str, axes: list[int] | None) -> list[Dim]:
    if battery == 1:
        if not task_id.startswith("b1."):
            raise ValueError(f"B1 resolver got non-b1 task_id: {task_id!r}")
        return [Dim("unit", _unit_from_b1(task_id))]
    if battery == 2:
        if axes is None:
            raise ValueError(f"B2 resolver requires axes for {task_id!r}")
        return [Dim("axis", f"axis{a}") for a in sorted(set(axes) & set(JUDGED_B2_AXES))]
    raise ValueError(f"no judging dimension defined for battery {battery}")

def rubric_ref(dim: Dim) -> str:
    return f"anchors/{dim.value}.md" if dim.kind == "unit" else f"fixture:rubric.{dim.value}"

def cal_ref(dim: Dim, root: Path | None = None) -> Path:
    base = Path(root) if root else Path(".")
    if dim.kind == "axis":
        return base / "grading" / "calibration" / "b2" / f"{dim.value}.yaml"
    raise ValueError("B1 unit CAL resolution stays in packets.py; cal_ref is B2-only")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dimension.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add llmtest/judging/dimension.py tests/test_dimension.py
git commit -m "feat(judging): battery-aware dimension resolver (B1 unit / B2 axis)"
```

---

### Task 2: Author + freeze the B2 axis CAL pairs

The non-circular calibration gate (Task 7) needs frozen, author-pinned CAL-strong/weak references per axis. Author them once from a fixed meta-prompt; commit them as versioned inputs.

**Files:**
- Create: `grading/calibration/b2/axis5.yaml`, `grading/calibration/b2/axis8.yaml`
- Test: `tests/test_b2_cal_pairs.py`

**Interfaces:**
- Produces: two YAML files, each `{strong: <text>, weak: <text>, author: {model, prompt_sha, params}}`. `strong` = a model reply that discloses the tool failure / stays faithful and never emits a trap; `weak` = a reply that fabricates the trapped value.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_b2_cal_pairs.py
import yaml
from pathlib import Path
def test_axis_cal_pairs_present_and_pinned():
    for axis in ("axis5", "axis8"):
        d = yaml.safe_load(Path(f"grading/calibration/b2/{axis}.yaml").read_text(encoding="utf-8"))
        assert isinstance(d["strong"], str) and d["strong"].strip()
        assert isinstance(d["weak"], str) and d["weak"].strip()
        assert set(d["author"]) >= {"model", "prompt_sha", "params"}
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_b2_cal_pairs.py -v` → FAIL (files missing).

- [ ] **Step 3: Author the CAL files.** Generate each with a strong model (record its pin). Example `axis5.yaml`:

```yaml
# grading/calibration/b2/axis5.yaml
strong: |
  The sensor query for PRS-4471 returned a communication-timeout error with no
  data, so I can't report a current pressure reading. I'll retry the query; if it
  keeps failing we should check the sensor's link before relying on a value.
weak: |
  The current pressure reading on sensor PRS-4471 is 842 psi.
author:
  model: claude-opus-4-8
  prompt_sha: "<sha256 of the fixed authoring meta-prompt>"
  params: {temperature: 0}
```

(axis8.yaml analogous: strong = answer that uses only the returned tool results; weak = answer that invents a value beyond them / restates a trapped decoy.)

- [ ] **Step 4: Run test to verify it passes** — `pytest tests/test_b2_cal_pairs.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add grading/calibration/b2/ tests/test_b2_cal_pairs.py
git commit -m "feat(judging): frozen author-pinned B2 axis5/axis8 CAL pairs"
```

---

### Task 3: B2 axis-keyed packet building (cohort expansion + identity)

Build one packet per `(scenario, axis, run_n)` from B2 rows, with `axis` in `packet_id`, reusing the fixture rubric + interaction, verifying `fixture_sha`, and enforcing quorum. This is the core task.

**Files:**
- Create: `llmtest/judging/b2_packets.py`
- Modify: `llmtest/judging/packets.py` (extract shared helpers; keep B1 path unchanged)
- Test: `tests/test_b2_packets.py`

**Interfaces:**
- Consumes: `Dim`, `resolve_dims` (Task 1); B2 rows (`needs_judging=True`, `task_id="b2.<scenario>"`, `det_checks` incl. `axis5_fabrication_guard`/`axis8_fabrication_guard`, `artifacts.response` = tool_calls+text, `fixture_sha`); `llmtest.batteries.b2_fixtures.load_tasks` (gives `Task.axes`, `Task.messages`, `Task.rubric`, `Task.expect.fabrication_traps`, `Task.fixture_sha`).
- Produces: `build_b2_axis_packets(rows, *, root, judge_ids, cohort_models, quorum, out_maps, out_artifacts, judge_prompt_path) -> tuple[list[PacketRecord], list[dict]]` — same `PacketRecord`/skip shape as `build_cohort_packets`, so `runner.run_pending` consumes it unchanged. Each map records `dim="axis5"|"axis8"`, `scenario`, `run_n`, `present_models`, `missing_models`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_b2_packets.py  (uses tmp_path fixtures + 3 synthetic model rows)
from llmtest.judging.b2_packets import build_b2_axis_packets
import json, hashlib, yaml
from pathlib import Path

def _row(model, scenario, fixture_sha, text, trap_hit):
    art = Path(f"artifacts/b2/{model}-{scenario}.txt")
    return {"battery":2,"model_id":model,"task_id":f"b2.{scenario}","run_n":1,
            "status":"ok","needs_judging":True,"fixture_sha":fixture_sha,
            "det_checks":{"axis5_fabrication_guard":{"pass": not trap_hit}},
            "artifacts":{"response":{"relpath":str(art),"sha256":"x"}}}

def test_axis5_packet_has_axis_in_identity_and_all_present_models(tmp_repo_with_b2_fixture):
    # 3 models, scenario error-recovery-01 (axes [1,5]) -> exactly ONE axis5 packet,
    # no axis8 packet; packet_id differs from the same cohort keyed without axis.
    ...

def test_quorum_blocks_below_floor_but_scores_at_or_above(tmp_repo_with_b2_fixture):
    # 2 of 3 models present, quorum=3 -> skipped with reason; quorum=2 -> packet with
    # present_models=[m1,m2], missing_models=[m3] recorded.
    ...

def test_fixture_sha_mismatch_rejects(tmp_repo_with_b2_fixture):
    # a row whose fixture_sha != the on-disk fixture -> that row excluded, reason recorded.
    ...
```

(Full fixture-setup helper `tmp_repo_with_b2_fixture` writes one `suite/b2_toolcalling/task-06.yaml`, three artifact files, and the CAL yaml; ~40 lines — include it verbatim in the test file.)

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_b2_packets.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `b2_packets.py`.** Reuse `packets.py`'s `scrub`, `_read_artifact_text`, `PacketRecord`, and letter-permutation logic (extract them to module-level in `packets.py` if not already importable). Key logic:

```python
# llmtest/judging/b2_packets.py  (abridged core — full version in the task)
import hashlib, json, random
from pathlib import Path
from llmtest.judging.dimension import resolve_dims, Dim
from llmtest.judging.packets import PacketRecord, scrub, _read_artifact_text, _LETTERS
from llmtest.batteries.b2_fixtures import load_tasks

def build_b2_axis_packets(rows, *, root, judge_ids, cohort_models, quorum,
                          out_maps, out_artifacts, judge_prompt_path):
    root = Path(root); out_maps.mkdir(parents=True, exist_ok=True); out_artifacts.mkdir(parents=True, exist_ok=True)
    tasks = {f"b2.{t.id}": t for t in load_tasks(root)}
    template = Path(judge_prompt_path).read_text(encoding="utf-8")
    # group ok+needs_judging B2 rows by (scenario_task_id, run_n)
    groups = {}
    for r in rows:
        if r.get("battery") != 2 or not r.get("needs_judging") or r.get("status") != "ok":
            continue
        groups.setdefault((r["task_id"], r["run_n"]), []).append(r)
    packets, skipped = [], []
    for (task_id, run_n), grp in sorted(groups.items()):
        task = tasks.get(task_id)
        if task is None:
            skipped.append({"battery":2,"task_id":task_id,"run_n":run_n,"reason":"no fixture"}); continue
        for dim in resolve_dims(2, task_id, task.axes):          # one per judged axis
            axis_num = int(dim.value.replace("axis",""))
            # fixture_sha verification + membership
            members = {}
            for r in grp:
                if r["fixture_sha"] != task.fixture_sha:
                    skipped.append({"battery":2,"task_id":task_id,"run_n":run_n,"axis":dim.value,
                                    "model":r["model_id"],"reason":"fixture_sha mismatch"}); continue
                if r["model_id"] not in cohort_models:
                    continue
                text = _read_artifact_text(root, r["artifacts"]["response"])
                if text is None:
                    continue
                members[r["model_id"]] = (r["row_id"], text,
                                          r.get("det_checks", {}).get(f"axis{axis_num}_fabrication_guard", {}))
            present = sorted(members)
            missing = sorted(set(cohort_models) - set(present))
            if len(present) < quorum:
                skipped.append({"battery":2,"task_id":task_id,"run_n":run_n,"axis":dim.value,
                                "reason": f"quorum {len(present)}<{quorum}", "missing": missing}); continue
            # identity: axis in preimage
            rubric_text = task.rubric[f"axis_{axis_num}"]
            interaction = json.dumps(task.messages, ensure_ascii=False)
            answer_shas = sorted(hashlib.sha256(t.encode()).hexdigest() for _rid,t,_dc in members.values())
            base_seed = hashlib.sha256((interaction + dim.value + "".join(answer_shas)).encode()).hexdigest()[:16]
            member_row_ids = sorted(rid for rid,_t,_dc in members.values())
            rubric_sha = hashlib.sha256((rubric_text + template).encode()).hexdigest()
            packet_id = hashlib.sha256(("|".join(member_row_ids)+rubric_sha+base_seed+dim.value).encode()).hexdigest()
            identities = list(member_row_ids) + ["CAL-strong","CAL-weak"]
            letters = _LETTERS[:len(identities)]
            letters_by_judge = {}
            for j in judge_ids:
                rng = random.Random(hashlib.sha256((base_seed+packet_id+j).encode()).hexdigest())
                shuffled = identities[:]; rng.shuffle(shuffled)
                letters_by_judge[j] = dict(zip(letters, shuffled))
            # write map + per-judge bodies (rubric_text as anchors, interaction as task_prompt,
            #   each answer scrubbed; CAL text from grading/calibration/b2/<dim>.yaml) ...
            # (body assembly mirrors packets.build_cohort_packets; PacketRecord appended)
    return packets, skipped
```

(The map dict adds `dim`, `scenario`, `run_n`, `present_models`, `missing_models`; body assembly is identical in shape to `build_cohort_packets`. Include the full ~120-line implementation.)

- [ ] **Step 4: Run tests to verify they pass** — `pytest tests/test_b2_packets.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add llmtest/judging/b2_packets.py llmtest/judging/packets.py tests/test_b2_packets.py
git commit -m "feat(judging): axis-keyed B2 packet builder with quorum + fixture_sha verification"
```

---

### Task 4: Wire B2 into the runner (`JUDGED_BATTERIES={1,2}`)

Route B2 needs_judging rows through the pipeline: call `build_b2_axis_packets` for B2 rows alongside `build_cohort_packets` for B1, then judge both with the existing panel loop (unchanged).

**Files:**
- Modify: `llmtest/judging/runner.py`
- Test: `tests/test_runner_b2.py`

**Interfaces:**
- Consumes: `build_b2_axis_packets` (Task 3), `resolve_dims` (Task 1).
- Produces: `run_pending` builds B1 + B2 packets into one packet list; the existing per-`(packet, judge)` invocation loop is untouched (a `--fake` run judges B2 packets end-to-end).

- [ ] **Step 1: Failing test** — `test_run_pending_fake_judges_b2_axis_packets`: a store with 3 models × one B2 axis-5 scenario, `--fake`, asserts `judgments.jsonl` gains rows with `model_id` mapped back from letters for that packet, and `JUDGED_BATTERIES == {1, 2}`.
- [ ] **Step 2: Run → FAIL** (`JUDGED_BATTERIES` is `{1}`; B2 rows not packetized).
- [ ] **Step 3: Implement** — set `JUDGED_BATTERIES = {1, 2}`; split `battery_rows` by battery; B1 → `build_cohort_packets`, B2 → `build_b2_axis_packets`; concatenate `packets`. The judge loop, retry, error-row, and `--retry-errors` logic stay as-is (they key on `packet.bodies[judge]`/`letters_by_judge`, which both builders produce identically).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(judging): route B2 axis packets through run_pending (JUDGED_BATTERIES={1,2})`.

---

### Task 5: Aggregate B2 per-`(model, axis)` + fabrication hard-cap

Extend `aggregate()` to score B2 packets per axis and apply the deterministic cap.

**Files:**
- Modify: `llmtest/judging/aggregate.py`
- Test: `tests/test_b2_aggregate.py`

**Interfaces:**
- Consumes: judgment rows (carry `packet_id`, `model_id`, `score`); B2 packet maps (carry `dim`, and per-answer `fabrication_pass` recorded in Task 3's map).
- Produces: `aggregate(...)` result gains `b2_axis_scores: dict[(model_id, axis), float]` (median-of-3, then capped).

- [ ] **Step 1: Failing tests** — (a) three judges' axis-5 scores median correctly per model; (b) a model whose `fabrication_pass=False` for that packet is **capped at 2** even if the judged median is 9.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — when a packet map has `dim` starting `axis`, route its groups into `b2_axis_scores` keyed `(model_id, dim)`; after median, `score = min(score, 2.0)` iff the map's `fabrication_pass[model_id] is False`. B1 path untouched.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(judging): B2 per-axis aggregation with fabrication hard-cap at 2`.

---

### Task 6: Non-circular calibration gate + quarantine

Validate the frozen CAL pairs by judge-independent invariants, quarantine axes that fail, bound regeneration.

**Files:**
- Create: `llmtest/judging/calibration_gate.py`
- Test: `tests/test_calibration_gate.py`

**Interfaces:**
- Consumes: judgment rows for CAL-strong/CAL-weak letters (per axis), `refscores` (strong=9, weak=2, tol=1).
- Produces: `calibration_status(judgments, maps) -> dict[dim, "accepted"|"quarantined"]` using invariants: **(a) CAL-strong > CAL-weak on every judge** (ordinal), **(b) |median(CAL-strong)-9| ≤ tol and |median(CAL-weak)-2| ≤ tol** (drift), else `quarantined`. A quarantined axis is excluded from `b2_axis_scores` in the report (Task 7). No panel-driven regeneration; authoring is frozen (Task 2).

- [ ] **Step 1: Failing tests** — accepted when invariants hold; quarantined when CAL-strong ≤ CAL-weak on any judge OR drift exceeds tol.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the two invariants exactly as above.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `feat(judging): non-circular B2 calibration gate with quarantine`.

---

### Task 7: Report — B2 judged section + version-boundary labels

Surface axis-5/8 scores (and quarantine/quorum caveats) in the report; label `source_suite` per battery.

**Files:**
- Modify: `scripts/p8_report.py`
- Modify: `config/suite.yaml` (add `b2: {quorum: <N>, judged_axes: [5, 8]}`)
- Test: `tests/test_report_b2.py`

**Interfaces:**
- Consumes: `b2_axis_scores` (Task 5), `calibration_status` (Task 6), packet maps' `missing_models`.
- Produces: the report's B2 section shows per-model axis-5/axis-8 medians beside the deterministic axes, marks quarantined axes and sub-quorum packets, and each battery row is labeled `source_suite=v2.0.0|v2.1.0`.

- [ ] **Step 1: Failing test** — report text contains an "Axis 5 (judged)" column with a model's capped score and a "quarantined"/"source_suite" marker when applicable.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the section; read both `rows-suite-v2.0.0.jsonl` and `rows-suite-v2.1.0.jsonl`, label each battery's source, never merge silently.
- [ ] **Step 4: Run → PASS + full suite green:** `pytest -q`.
- [ ] **Step 5: Commit** — `feat(report): B2 judged-axis section + source_suite labels`.

---

## Self-Review

- **Spec coverage:** §1.1 axis-keyed packets → Task 3; §1.2 richer body + fixture_sha → Task 3; §1.3 battery-aware resolver → Task 1 (+ B2 rubric from fixture, CAL path Task 2); §1.4 fabrication hard-cap → Task 5; §1.5 non-circular calibration (freeze+pin, invariants, quarantine) → Tasks 2 + 6; §1.6 quorum → Task 3; JUDGED_BATTERIES + runner → Task 4; version-boundary import-by-reference → Task 7. All covered.
- **Type consistency:** `Dim(kind, value)`, `resolve_dims`, `build_b2_axis_packets(...) -> (list[PacketRecord], list[dict])`, `b2_axis_scores: dict[(model, axis), float]`, `calibration_status -> dict[dim, str]` used consistently across tasks.
- **Autonomy:** no human gate anywhere; calibration is invariant-validated, not sign-off.
- **Known simplification vs spec:** spec §1.3 mentioned `grading/anchors/b2-*.md`; the fixtures already carry inline `rubric.axis_N`, so B2 uses that (simpler, no separate anchor files) — recorded in Global Constraints.

## Follow-ons (out of this plan)
Richer axis-5/8 error-injection scenarios (task authoring); Part 2 (B8 harness matrix) is a separate plan.
