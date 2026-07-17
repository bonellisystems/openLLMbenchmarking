"""Battery 7 — harness/config sensitivity matrix (TESTPLAN 5.7).

INTERPRETATION NOTE (documented per build instructions -- B7 is the
least-specified battery; see .superpowers/sdd/b7-report.md for the full
writeup): TESTPLAN 5.7 describes B7 as a comparison across three REAL
external agentic-coding harnesses (Hermes-agent/WSL2, OpenCode, Claude Code
via a LiteLLM `/v1/messages` proxy) hitting an identical pinned
llama-server endpoint, and explicitly slates that axis for P6 ("WSL2 Hermes
+ OpenCode + LiteLLM-CC, pins recorded"). That axis needs environment work
(WSL2 install, OpenCode install, a running LiteLLM proxy) that doesn't
exist at this build point (P4) and is out of scope for this ticket.

This module instead implements a SENSITIVITY matrix over harness-ADJACENT
config knobs that already exist today against the same llama-server
backend: system-prompt variant, temperature (0 vs runtime default),
tool-call format (native `tools` API vs a prompted textual convention), and
n-gram speculative-decode on/off. It reuses the `cond=B7` condition slot
TESTPLAN reserves for the harness axis and is designed so the real
external-harness axis can be added later as one more matrix dimension
(e.g. `harness: [hermes, opencode, claudecode-litellm]`) without changing
the WorkItem/row shape -- see `_matrix_cells` below, which reads the whole
matrix generically from `suite.yaml` `b7.matrix.dimensions`.

Design: ONE-FACTOR-AT-A-TIME (OFAT) from a fixed "baseline" reference cell,
not a full factorial cross -- this directly matches TESTPLAN 5.7's Control
principle ("harness is the only variable") applied per-dimension, and keeps
the matrix small and each variant's effect individually attributable.
Scoring is deterministic throughout (content-signal checks reused from B1,
format compliance, tool-call compliance, and signal-agreement /
byte-identity vs the baseline cell) -- no judge calls in this build
(needs_judging=False on every row).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llmtest import schema
from llmtest.batteries import Battery, WorkItem, register
from llmtest.batteries.b1_fixtures import check_signals
from llmtest.batteries.b7_fixtures import load_probe_tasks

SYSTEM_PROMPTS = {
    "default": ("You are a precise assistant helping run an MSP's business "
                "operations. Answer completely and follow the requested "
                "format exactly."),
    "minimal": "Be concise.",
}

# signal_agreement_vs_baseline pass threshold — see suite.yaml b7.agreement_threshold
# (module default; execute() prefers the config value when present).
AGREEMENT_THRESHOLD = 0.8


def _dims_cfg(cfg) -> dict:
    return cfg.suite["b7"]["matrix"]["dimensions"]


def _baseline_dims(dims_cfg: dict) -> dict:
    return {k: v["baseline"] for k, v in dims_cfg.items()}


def _condition_for(dims: dict, order: list[str]) -> str:
    parts = {"runtime": "fork", "kv": "q8", "ctx": "8k", "cond": "B7", **dims}
    return schema.canonical_condition(parts, order)


def _matrix_cells(cfg, order: list[str]) -> list[tuple[str, str]]:
    """(cell_name, condition_string) pairs. Baseline is always first, then
    one OFAT variant per non-baseline value per dimension, read generically
    from suite.yaml b7.matrix.dimensions (adding a dimension or a 3rd value
    there grows the matrix with no code change)."""
    dims_cfg = _dims_cfg(cfg)
    baseline = _baseline_dims(dims_cfg)
    cells = [("baseline", _condition_for(baseline, order))]
    for dim, spec in dims_cfg.items():
        for val in spec["values"]:
            if val == spec["baseline"]:
                continue
            variant = dict(baseline)
            variant[dim] = val
            cells.append((f"{dim}-{val}", _condition_for(variant, order)))
    return cells


def _baseline_condition(cfg, order: list[str]) -> str:
    return _condition_for(_baseline_dims(_dims_cfg(cfg)), order)


def _check_json_format(text: str) -> bool:
    stripped = text.strip()
    candidates = [stripped]
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    for c in candidates:
        try:
            json.loads(c)
            return True
        except (ValueError, TypeError):
            continue
    return False


def _check_tool_call(response: dict, text: str, toolfmt: str, expected_name: str) -> dict:
    if toolfmt == "native":
        msg = response.get("choices", [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls") or []
        names = [tc.get("function", {}).get("name") for tc in tool_calls]
        return {"pass": expected_name in names, "detail": {"tool_calls_seen": names}}
    pattern = rf"TOOL_CALL:\s*{re.escape(expected_name)}\("
    return {"pass": bool(re.search(pattern, text)), "detail": {"toolfmt": "prompted"}}


def _signal_keys(det_checks: dict) -> set:
    return {k for k in det_checks if k.split("-")[0] in ("contains", "regex", "numeric")}


def _signal_agreement(this_checks: dict, baseline_checks: dict, threshold: float) -> dict:
    keys = _signal_keys(this_checks) & _signal_keys(baseline_checks)
    if not keys:
        return {"pass": True, "agreement_rate": 1.0, "n_compared": 0}
    matches = sum(1 for k in keys if this_checks[k].get("pass") == baseline_checks[k].get("pass"))
    rate = matches / len(keys)
    return {"pass": rate >= threshold, "agreement_rate": rate, "n_compared": len(keys)}


def _word_jaccard(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _prompted_tool_instructions(tool_schema: dict) -> str:
    fn = tool_schema["function"]
    name = fn["name"]
    props = fn.get("parameters", {}).get("properties", {})
    args = ", ".join(props.keys())
    return (f"You do not have native tool-calling in this harness. Instead, when you "
            f"need to call `{name}({args})`, emit a line of the exact form "
            f"`TOOL_CALL: {name}(<args>)` before your answer.")


def _find_row(store, **key) -> dict | None:
    row_id = schema.compute_row_id(**key)
    for row in store.iter_rows():
        if row["row_id"] == row_id:
            return row
    return None


def _read_artifact_text(root: Path, row: dict) -> str | None:
    info = row.get("artifacts", {}).get("response")
    if not info:
        return None
    p = root / "artifacts" / info["relpath"]
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


@register
class B7HarnessMatrix(Battery):
    id = 7

    def plan(self, cfg, store, model_filter=None, force=False) -> list[WorkItem]:
        """Generate WorkItems for the B7 harness/config sensitivity matrix.

        Covers all registry models WITHOUT role=quant-arm (same roster rule
        as B1), the fixed probe set, all matrix cells, and run_n in
        1..cfg.suite["b7"]["n_runs"].
        """
        order = cfg.suite["condition_order"]
        sv = cfg.suite["suite_version"]
        n_runs = cfg.suite["b7"]["n_runs"]
        probes = load_probe_tasks(cfg.root, cfg.suite["b7"].get(
            "probes_dir", "suite/b7_harnessmatrix/probes"))
        cells = [c for _, c in _matrix_cells(cfg, order)]

        items = []
        for model_id, m in sorted(cfg.registry["models"].items()):
            if model_filter and model_id != model_filter:
                continue
            if m.get("role") == "quant-arm":
                continue
            if str(m.get("local_path", "")).startswith("TO-"):
                continue

            for probe in probes:
                task_id = f"b7.{probe.id}"
                fixture_sha = probe.fixture_sha

                for condition in cells:
                    if force:
                        existing = [r["run_n"] for r in store.iter_rows()
                                   if r["task_id"] == task_id and r["model_id"] == model_id
                                   and r["condition"] == condition]
                        run_ns = [(max(existing) + 1) if existing else 1]
                    else:
                        run_ns = range(1, n_runs + 1)

                    for run_n in run_ns:
                        rid = schema.compute_row_id(
                            suite_version=sv, model_id=model_id,
                            quant_sha256=m["provenance"]["sha256"], battery=7,
                            task_id=task_id, fixture_sha=fixture_sha,
                            condition=condition, run_n=run_n)

                        items.append(WorkItem(
                            row_id=rid, model_id=model_id, battery=7,
                            task_id=task_id, condition=condition, run_n=run_n,
                            payload={
                                "model": m,
                                "fixture_sha": fixture_sha,
                                "suite_version": sv,
                                "prompt": probe.prompt,
                                "signals": probe.signals,
                                "expects_tool_call": probe.expects_tool_call,
                                "tool_schema": probe.tool_schema,
                                "expected_tool_name": probe.expected_tool_name,
                                "response_format": probe.response_format,
                            }))
        return items

    def execute(self, item: WorkItem, ctx) -> list[dict]:
        """Execute one (model, probe, matrix-cell, run_n) B7 WorkItem.

        Applies the cell's config to the request (system prompt, temperature,
        tool-call format, n-gram spec on/off), scores deterministically
        (content signals + format/tool-call compliance), and — for
        non-baseline cells — compares against the baseline cell's already-
        computed row (signal agreement; byte-identity when only ngram
        differs at temp=0, a direct check of the project's own "n-gram
        spec-decode is lossless at temp=0" claim).
        """
        cfg = ctx.cfg
        order = cfg.suite["condition_order"]
        model = item.payload["model"]
        parts = dict(p.split("=") for p in item.condition.split(";"))
        sysp = parts.get("sysp", "default")
        temp_key = parts.get("temp", "t0")
        toolfmt = parts.get("toolfmt", "native")
        spec = parts.get("spec", "ngram32")

        temperature = 0.0 if temp_key == "t0" else None
        flags_overlay = {"spec": "off"} if spec == "off" else None

        endpoint = ctx.server_manager().request_endpoint(
            item.model_id, ctx=cfg.suite["b7"].get("ctx", 8192), kv="q8_0",
            flags_overlay=flags_overlay, timing_authoritative=False)

        messages = []
        sys_text = SYSTEM_PROMPTS.get(sysp, SYSTEM_PROMPTS["default"])
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

        prompt = item.payload["prompt"]
        tools = None
        if item.payload["expects_tool_call"]:
            if toolfmt == "native":
                tools = [item.payload["tool_schema"]]
            else:
                prompt = _prompted_tool_instructions(item.payload["tool_schema"]) + "\n\n" + prompt
        messages.append({"role": "user", "content": prompt})

        max_tokens = cfg.suite["b7"].get("max_tokens", 900)
        response = endpoint.chat(messages, max_tokens=max_tokens,
                                 temperature=temperature, tools=tools)

        text = response["choices"][0]["message"].get("content") or ""

        det_checks = check_signals(text, item.payload["signals"])

        if item.payload["response_format"] == "json":
            det_checks["format_json"] = {"pass": _check_json_format(text)}

        if item.payload["expects_tool_call"]:
            det_checks["tool_call_compliance"] = _check_tool_call(
                response, text, toolfmt, item.payload["expected_tool_name"])

        metrics = {"chars": len(text)}

        baseline_condition = _baseline_condition(cfg, order)
        if item.condition != baseline_condition:
            baseline_row = _find_row(
                ctx.store, suite_version=item.payload["suite_version"],
                model_id=item.model_id, quant_sha256=model["provenance"]["sha256"],
                battery=7, task_id=item.task_id, fixture_sha=item.payload["fixture_sha"],
                condition=baseline_condition, run_n=item.run_n)
            if baseline_row is not None:
                threshold = cfg.suite["b7"].get("agreement_threshold", AGREEMENT_THRESHOLD)
                det_checks["signal_agreement_vs_baseline"] = _signal_agreement(
                    det_checks, baseline_row["det_checks"], threshold)
                root = ctx.root if hasattr(ctx, "root") else Path(".")
                baseline_text = _read_artifact_text(root, baseline_row)
                if baseline_text is not None:
                    metrics["length_ratio_vs_baseline"] = (
                        len(text) / len(baseline_text) if baseline_text else None)
                    metrics["word_jaccard_vs_baseline"] = _word_jaccard(text, baseline_text)
                    if spec == "off" and temp_key == "t0":
                        # Direct empirical test of the project's own claim
                        # (root CLAUDE.md): n-gram spec decode is lossless at
                        # temp=0 -- output should be byte-identical on vs off.
                        det_checks["byte_identical_vs_baseline"] = {"pass": text == baseline_text}

        artifacts_root = (ctx.root / "artifacts" / "b7") if hasattr(ctx, "root") else (Path("artifacts") / "b7")
        artifacts_root.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts_root / f"{item.row_id}.txt"
        artifact_path.write_text(text, encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        sampling = {"temp": temperature if temperature is not None else "runtime-default",
                    "max_tokens": max_tokens}

        row = schema.ResultRow.new(
            suite_version=item.payload["suite_version"], model_id=item.model_id,
            hf_repo=model.get("hf_repo", ""), quant_file=model.get("quant_file", ""),
            quant_sha256=model["provenance"]["sha256"], tier="T1", battery=7,
            task_id=item.task_id, fixture_sha=item.payload["fixture_sha"],
            condition=item.condition, run_n=item.run_n,
            session_id=endpoint.session_id, sampling=sampling,
            det_checks=det_checks, needs_judging=False, metrics=metrics,
            timing_authoritative=False,
            artifacts={"response": {"sha256": artifact_sha, "relpath": f"b7/{item.row_id}.txt"}},
            status="ok", tags=[])

        if response.get("timings"):
            row.response_meta.update({
                "predicted_n": response["timings"].get("predicted_n"),
                "predicted_per_second": response["timings"].get("predicted_per_second"),
            })

        return [row.to_dict()]
