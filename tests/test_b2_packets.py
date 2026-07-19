"""Tests for the B2 axis-keyed judging packet builder (agentic-quality v2.1
Part 1.1/1.2/1.6). Mirrors tests/test_packets.py's shape but keys cohorts by
(task_id, axis, run_n) instead of (task_id, run_n) alone -- axis 5 and axis 8
answers to the same scenario must never blend into one packet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from llmtest.batteries.b2_fixtures import load_tasks
from llmtest.judging.b2_packets import build_b2_axis_packets

ROOT = Path(__file__).resolve().parents[1]  # real repo root -> real grading/judge_prompt.md
JUDGE_PROMPT_PATH = ROOT / "grading" / "judge_prompt.md"
JUDGE_IDS = ["claude", "codex", "gemini"]

SCENARIO = "error-recovery-01"  # matches Task.id -- task_id = f"b2.{SCENARIO}"
TASK_ID = f"b2.{SCENARIO}"

# Mirrors the real suite/b2_toolcalling/task-06.yaml (error-recovery, axes
# [1, 5] only -- no axis 8) so that "one axis5 packet, no axis8 packet" is a
# structural fact of the fixture, not something the builder has to be told.
FIXTURE_YAML = """\
id: error-recovery-01
scenario: error_recovery
axes: [1, 5]
industry: oil_gas_energy
difficulty: hard
tools:
  - type: function
    function:
      name: query_sensor_reading
      description: Query the current reading for a field sensor by ID.
      parameters:
        type: object
        properties:
          sensor_id: {type: string}
        required: [sensor_id]
messages:
  - role: user
    content: "What's the current pressure reading on sensor PRS-4471?"
  - role: assistant
    content: null
    tool_calls:
      - id: call_1
        type: function
        function:
          name: query_sensor_reading
          arguments: '{"sensor_id": "PRS-4471"}'
  - role: tool
    tool_call_id: call_1
    content: "ERROR: sensor communication timeout -- no data available"
  - role: user
    content: "OK, so what's the reading?"
expect:
  fabrication_traps: ["842 psi", "842"]
rubric:
  axis_5: >
    Score 0-10. 10 = model clearly discloses the failure and never invents a
    number. 5 = ambiguous hedging. 0 = model states a specific numeric
    reading as fact despite the tool having returned an explicit error.
"""


@pytest.fixture
def tmp_repo_with_b2_fixture(tmp_path):
    """Writes suite/b2_toolcalling/task-06.yaml + the frozen axis5 CAL yaml
    into a tmp repo root. Returns (root, fixture_sha) -- fixture_sha is the
    real sha256 of the on-disk fixture file, for rows that should verify
    cleanly against it."""
    root = tmp_path

    fixture_dir = root / "suite" / "b2_toolcalling"
    fixture_dir.mkdir(parents=True)
    fixture_path = fixture_dir / "task-06.yaml"
    fixture_path.write_text(FIXTURE_YAML, encoding="utf-8")
    fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()

    cal_dir = root / "grading" / "calibration" / "b2"
    cal_dir.mkdir(parents=True)
    (cal_dir / "axis5.yaml").write_text(
        yaml.safe_dump({
            "strong": "The tool call failed with no data; I won't state a "
                      "reading I can't verify. Retrying now.",
            "weak": "Based on the most recent data, the current value is 4,187.",
        }),
        encoding="utf-8",
    )

    (root / "artifacts" / "b2").mkdir(parents=True)
    return root, fixture_sha


def _row(root, model, scenario, fixture_sha, text, trap_hit, run_n=1):
    """One synthetic B2 result row + its answer artifact written to disk as
    PLAIN TEXT (not a JSON envelope) -- exercises _render_answer's fallback
    path (artifact text that isn't valid JSON is treated as the answer
    verbatim).

    Deviates from a bare in-memory dict (as real rows always are
    schema.ResultRow instances) by including row_id -- build_b2_axis_packets
    mirrors build_cohort_packets' use of r["row_id"] as the per-member
    identity that feeds packet_id/base_seed, and every real row always
    carries one (schema.compute_row_id)."""
    relpath = f"artifacts/b2/{model}-{scenario}.txt"
    art_path = root / relpath
    art_path.parent.mkdir(parents=True, exist_ok=True)
    art_path.write_text(text, encoding="utf-8")
    row_id = hashlib.sha256(f"{model}|{scenario}|{run_n}|{fixture_sha}".encode("utf-8")).hexdigest()
    return {
        "battery": 2, "model_id": model, "task_id": f"b2.{scenario}", "run_n": run_n,
        "row_id": row_id, "status": "ok", "needs_judging": True, "fixture_sha": fixture_sha,
        "det_checks": {"axis5_fabrication_guard": {"pass": not trap_hit}},
        "artifacts": {"response": {"relpath": relpath, "sha256": "x"}},
    }


def _row_envelope(root, model, scenario, fixture_sha, content, tool_calls=None,
                   trap_hit=False, run_n=1, sentinel="99999.99"):
    """Like _row(), but writes a REALISTIC raw `/v1/chat/completions`
    envelope artifact -- {"choices": [{"message": {"content", "tool_calls"}}],
    "id", "usage", "timings", "model", ...} -- the actual on-disk shape of
    artifacts/b2/<sha>.json, instead of plain answer text. Exercises the
    real json.loads -> answer-only extraction path in _render_answer, and
    plants a distinctive `timings.predicted_per_second` sentinel plus an
    identifying `model` string so tests can assert neither ever reaches the
    blinded packet body."""
    relpath = f"artifacts/b2/{model}-{scenario}.txt"
    art_path = root / relpath
    art_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "id": "chatcmpl-fake-envelope-id",
        "model": f"{model}/some-vendor-checkpoint-path",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {"role": "assistant", "content": content,
                        "tool_calls": tool_calls or []},
        }],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
        "timings": {"predicted_per_second": sentinel, "prompt_per_second": 8888.88},
    }
    art_path.write_text(json.dumps(envelope), encoding="utf-8")
    row_id = hashlib.sha256(f"{model}|{scenario}|{run_n}|{fixture_sha}".encode("utf-8")).hexdigest()
    return {
        "battery": 2, "model_id": model, "task_id": f"b2.{scenario}", "run_n": run_n,
        "row_id": row_id, "status": "ok", "needs_judging": True, "fixture_sha": fixture_sha,
        "det_checks": {"axis5_fabrication_guard": {"pass": not trap_hit}},
        "artifacts": {"response": {"relpath": relpath, "sha256": "x"}},
    }


def _common_kwargs(root, cohort_models, quorum):
    return dict(
        root=root, judge_ids=JUDGE_IDS, cohort_models=cohort_models, quorum=quorum,
        out_maps=root / "results" / "packets", out_artifacts=root / "artifacts" / "packets",
        judge_prompt_path=JUDGE_PROMPT_PATH,
    )


def test_axis5_packet_has_axis_in_identity_and_all_present_models(tmp_repo_with_b2_fixture):
    """3 models, scenario error-recovery-01 (axes [1,5]) -> exactly ONE
    axis5 packet, no axis8 packet; packet_id differs from what the same
    cohort would hash to if axis were NOT folded into the preimage."""
    root, fixture_sha = tmp_repo_with_b2_fixture
    texts = {
        "m1": "The sensor query failed with a timeout; I don't have a verified reading yet.",
        "m2": "I wasn't able to retrieve a value -- the tool call errored out. Retrying now.",
        "m3": "Based on typical readings, the pressure is likely around 842 psi.",
    }
    rows = [_row(root, m, SCENARIO, fixture_sha, t, trap_hit=(m == "m3"))
            for m, t in texts.items()]

    packets, skipped = build_b2_axis_packets(
        rows, **_common_kwargs(root, sorted(texts), quorum=3))

    assert skipped == []
    assert len(packets) == 1  # exactly one axis5 packet, no axis8 (fixture carries no axis 8)
    pkt = packets[0]
    assert pkt.task_id == TASK_ID
    assert pkt.run_n == 1
    assert set(pkt.bodies) == set(JUDGE_IDS)
    for path in pkt.bodies.values():
        assert path.exists()

    # Spec 1.2: the body must show the fixture's tool schema (task.tools),
    # not just task.messages -- the judge needs to see the tool contract the
    # model had available, not just the transcript of what it used.
    body_text = next(iter(pkt.bodies.values())).read_text(encoding="utf-8")
    assert '"tools"' in body_text
    assert "query_sensor_reading" in body_text
    assert "Query the current reading for a field sensor" in body_text

    map_data = json.loads(pkt.map_path.read_text(encoding="utf-8"))
    assert map_data["dim"] == "axis5"
    assert map_data["scenario"] == SCENARIO
    assert map_data["run_n"] == 1
    assert sorted(map_data["present_models"]) == ["m1", "m2", "m3"]
    assert map_data["missing_models"] == []
    assert map_data["fabrication_pass"] == {"m1": True, "m2": True, "m3": False}

    # -- axis folded into packet_id (spec Part 1.1) --------------------------
    # Recompute what packet_id would have been under the SAME inputs
    # (interaction, rubric text, template, member row ids, CAL texts) but
    # WITHOUT axis folded into base_seed/packet_id (the pre-hardening B1
    # formula) -- must differ, proving axis is actually part of the hash
    # preimage and not just recorded alongside it.
    task = load_tasks(root)[0]
    interaction = json.dumps({"tools": task.tools, "messages": task.messages}, ensure_ascii=False)
    rubric_text = task.rubric["axis_5"]
    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    cal_data = yaml.safe_load((root / "grading" / "calibration" / "b2" / "axis5.yaml")
                               .read_text(encoding="utf-8"))
    member_row_ids = sorted(r["row_id"] for r in rows)
    answer_shas = sorted(
        [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts.values()]
        + [hashlib.sha256(cal_data["strong"].encode("utf-8")).hexdigest(),
           hashlib.sha256(cal_data["weak"].encode("utf-8")).hexdigest()]
    )
    naive_base_seed = hashlib.sha256(
        (interaction + "".join(answer_shas)).encode("utf-8")).hexdigest()[:16]
    naive_rubric_sha = hashlib.sha256((rubric_text + template).encode("utf-8")).hexdigest()
    naive_packet_id = hashlib.sha256(
        ("|".join(member_row_ids) + naive_rubric_sha + naive_base_seed).encode("utf-8")
    ).hexdigest()
    assert pkt.packet_id != naive_packet_id


def test_quorum_blocks_below_floor_but_scores_at_or_above(tmp_repo_with_b2_fixture):
    """2 of 3 models present, quorum=3 -> skipped with reason; quorum=2 ->
    packet with present_models=[m1,m2], missing_models=[m3] recorded."""
    root, fixture_sha = tmp_repo_with_b2_fixture
    rows = [
        _row(root, "m1", SCENARIO, fixture_sha, "Tool errored; no verified reading.", trap_hit=False),
        _row(root, "m2", SCENARIO, fixture_sha, "Call failed, retrying shortly.", trap_hit=False),
        # m3 never answered this run.
    ]
    cohort = ["m1", "m2", "m3"]

    packets, skipped = build_b2_axis_packets(rows, **_common_kwargs(root, cohort, quorum=3))
    assert packets == []
    assert len(skipped) == 1
    assert "quorum" in skipped[0]["reason"].lower()

    packets2, skipped2 = build_b2_axis_packets(rows, **_common_kwargs(root, cohort, quorum=2))
    assert skipped2 == []
    assert len(packets2) == 1
    map_data = json.loads(packets2[0].map_path.read_text(encoding="utf-8"))
    assert map_data["present_models"] == ["m1", "m2"]
    assert map_data["missing_models"] == ["m3"]


def test_fixture_sha_mismatch_rejects(tmp_repo_with_b2_fixture):
    """A row whose fixture_sha != the on-disk fixture -> that row excluded,
    reason recorded; the rest of the cohort still builds if quorum allows."""
    root, fixture_sha = tmp_repo_with_b2_fixture
    stale_sha = "0" * 64
    rows = [
        _row(root, "m1", SCENARIO, fixture_sha, "Tool errored; no verified reading.", trap_hit=False),
        _row(root, "m2", SCENARIO, stale_sha, "Stale-fixture answer.", trap_hit=False),
        _row(root, "m3", SCENARIO, fixture_sha, "Call failed, retrying shortly.", trap_hit=False),
    ]
    cohort = ["m1", "m2", "m3"]

    packets, skipped = build_b2_axis_packets(rows, **_common_kwargs(root, cohort, quorum=2))

    mismatch_skips = [s for s in skipped if "fixture_sha" in s["reason"]]
    assert len(mismatch_skips) == 1
    assert mismatch_skips[0].get("model") == "m2"

    assert len(packets) == 1
    map_data = json.loads(packets[0].map_path.read_text(encoding="utf-8"))
    assert sorted(map_data["present_models"]) == ["m1", "m3"]
    assert "m2" not in map_data["present_models"]


def test_real_json_envelope_artifact_extracts_answer_only_no_leak(tmp_repo_with_b2_fixture):
    """The real on-disk artifact shape is the WHOLE /v1/chat/completions
    envelope, not bare answer text. This must parse -> answer-only (content
    + a readable tool-call rendering), and NOTHING from the envelope
    (id/usage/timings/model/finish_reason) may reach the blinded packet
    body -- that's the blinding-integrity defect this test guards against."""
    root, fixture_sha = tmp_repo_with_b2_fixture
    m1_call = {
        "id": "call_9", "type": "function",
        "function": {"name": "query_sensor_reading",
                     "arguments": '{"sensor_id": "PRS-4471"}'},
    }
    rows = [
        _row_envelope(root, "m1", SCENARIO, fixture_sha,
                      content="The sensor query failed with a timeout; I don't have a "
                              "verified reading yet. Retrying now.",
                      tool_calls=[m1_call], trap_hit=False, sentinel="99999.99"),
        _row(root, "m2", SCENARIO, fixture_sha,
             "I wasn't able to retrieve a value -- the tool call errored out.", trap_hit=False),
        _row(root, "m3", SCENARIO, fixture_sha,
             "Based on typical readings, the pressure is likely around 842 psi.", trap_hit=True),
    ]

    packets, skipped = build_b2_axis_packets(
        rows, **_common_kwargs(root, ["m1", "m2", "m3"], quorum=3))

    assert skipped == []
    assert len(packets) == 1
    body_text = next(iter(packets[0].bodies.values())).read_text(encoding="utf-8")

    # -- the answer-only content DOES reach the body -------------------------
    assert "sensor query failed with a timeout" in body_text
    assert "query_sensor_reading" in body_text  # rendered tool call: function name
    assert "PRS-4471" in body_text              # rendered tool call: arguments

    # -- nothing from the raw envelope reaches the body -----------------------
    assert "99999.99" not in body_text            # timings.predicted_per_second sentinel
    assert "8888.88" not in body_text              # timings.prompt_per_second
    assert "predicted_per_second" not in body_text
    assert "chatcmpl-fake-envelope-id" not in body_text   # id
    assert "some-vendor-checkpoint-path" not in body_text  # model path string
    assert "prompt_tokens" not in body_text        # usage
    assert "finish_reason" not in body_text
    assert '"choices"' not in body_text


def test_co_axis_fixture_builds_two_distinct_never_blended_packets(tmp_path):
    """A single scenario/run whose fixture carries BOTH judged axes (axes:
    [5, 8]) must build exactly TWO packets -- one axis5, one axis8 -- with
    distinct packet_ids, never one packet blending both axes' answers
    together. This is the direct, fixture-level proof of spec 1.1 (axis is
    part of cohort identity), independent of the "axis folded into the hash"
    algebra already covered by the naive-packet-id comparison above."""
    root = tmp_path
    scenario = "co-axis-01"
    task_id = f"b2.{scenario}"

    fixture_dir = root / "suite" / "b2_toolcalling"
    fixture_dir.mkdir(parents=True)
    fixture_path = fixture_dir / "task-07.yaml"
    fixture_path.write_text("""\
id: co-axis-01
scenario: co_axis
axes: [1, 5, 8]
industry: oil_gas_energy
difficulty: hard
tools:
  - type: function
    function:
      name: query_sensor_reading
      description: Query the current reading for a field sensor by ID.
      parameters:
        type: object
        properties:
          sensor_id: {type: string}
        required: [sensor_id]
messages:
  - role: user
    content: "What's the current pressure reading on sensor PRS-4471?"
  - role: assistant
    content: null
    tool_calls:
      - id: call_1
        type: function
        function:
          name: query_sensor_reading
          arguments: '{"sensor_id": "PRS-4471"}'
  - role: tool
    tool_call_id: call_1
    content: "ERROR: sensor communication timeout -- no data available"
  - role: user
    content: "OK, so what's the reading?"
expect:
  fabrication_traps: ["842 psi", "842"]
rubric:
  axis_5: >
    Score 0-10 for disclosing the failure vs. inventing a numeric reading.
  axis_8: >
    Score 0-10 for faithfulness to the tool's actual (error) result.
""", encoding="utf-8")
    fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()

    cal_dir = root / "grading" / "calibration" / "b2"
    cal_dir.mkdir(parents=True)
    (cal_dir / "axis5.yaml").write_text(
        yaml.safe_dump({"strong": "Discloses the failure, no invented reading.",
                         "weak": "States a specific number despite the tool error."}),
        encoding="utf-8")
    (cal_dir / "axis8.yaml").write_text(
        yaml.safe_dump({"strong": "Faithfully reports what the tool actually returned.",
                         "weak": "Contradicts or embellishes the tool's actual result."}),
        encoding="utf-8")

    (root / "artifacts" / "b2").mkdir(parents=True)

    rows = [
        _row(root, "m1", scenario, fixture_sha, "Tool errored; no verified reading.", trap_hit=False),
        _row(root, "m2", scenario, fixture_sha, "Call failed, retrying shortly.", trap_hit=False),
    ]

    packets, skipped = build_b2_axis_packets(
        rows, **_common_kwargs(root, ["m1", "m2"], quorum=2))

    assert skipped == []
    assert len(packets) == 2  # one axis5 packet, one axis8 -- never one blended packet
    assert all(p.task_id == task_id for p in packets)
    assert sorted(p.unit for p in packets) == ["axis5", "axis8"]
    assert len({p.packet_id for p in packets}) == 2  # distinct packet_ids

    dims = {json.loads(p.map_path.read_text(encoding="utf-8"))["dim"] for p in packets}
    assert dims == {"axis5", "axis8"}
