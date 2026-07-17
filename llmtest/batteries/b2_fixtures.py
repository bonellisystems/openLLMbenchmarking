"""Battery 2 -- tool-calling fixture loader and axis scoring (TESTPLAN 5.2).

Fixture format: suite/b2_toolcalling/task-<NN>.yaml (see the sibling
_schema.md). Mirrors b1_fixtures.py's shape: a loader that fails loud on
malformed fixtures, plus pure checker functions the battery module calls
from execute(). See _schema.md for the full `expect:` block vocabulary.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_TASKS_DIRNAME = "b2_toolcalling"
_VALID_AXES = set(range(1, 9))          # TESTPLAN 5.2: axes 1..8
_JUDGED_AXES = {5, 8}
_JSON_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}
_PY_TYPES_BY_SCHEMA_TYPE = {
    "string": (str,), "number": (int, float), "integer": (int,),
    "boolean": (bool,), "array": (list,), "object": (dict,), "null": (type(None),),
}


@dataclass
class Task:
    """Fixture task representation."""
    id: str
    scenario: str
    axes: list[int]
    industry: str
    difficulty: str
    tools: list[dict]
    messages: list[dict]          # filler-expanded, ready to send verbatim
    expect: dict
    rubric: dict
    fixture_sha: str
    path: Path


# --- filler expansion (axis 7: long-context tasks) --------------------------

def _expand_filler(paragraph: str, target_tokens: int) -> str:
    """Repeat `paragraph` to approximately `target_tokens` (~4 chars/token,
    same heuristic as b5_serving.build_sustained_prompt) -- keeps long-context
    fixture FILES small while producing a genuinely long prompt at load time."""
    approx_chars = max(int(target_tokens) * 4, 1)
    body = (paragraph + "\n") * (approx_chars // max(len(paragraph), 1) + 1)
    return body[:approx_chars]


def _apply_filler(messages: list[dict], filler_spec: dict | None) -> list[dict]:
    if not filler_spec:
        return messages
    expanded = _expand_filler(filler_spec["unit_paragraph"], filler_spec["target_tokens"])
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str) and "{{FILLER}}" in content:
            m = dict(m)
            m["content"] = content.replace("{{FILLER}}", expanded)
        out.append(m)
    return out


# --- loader -------------------------------------------------------------

def load_tasks(root: Path) -> list[Task]:
    """Load all B2 task fixtures from suite/b2_toolcalling/task-*.yaml.

    Args:
        root: Repository root.

    Returns:
        List of Task objects, sorted by id. Empty list if the directory
        doesn't exist (mirrors b1_fixtures.load_unit_tasks).

    Raises:
        ValueError: If a fixture file is malformed and cannot be parsed.
    """
    tasks_dir = Path(root) / "suite" / _TASKS_DIRNAME
    if not tasks_dir.exists():
        return []

    tasks = []
    for task_file in sorted(tasks_dir.glob("task-*.yaml")):
        try:
            data = yaml.safe_load(task_file.read_text(encoding="utf-8"))
            fixture_sha = hashlib.sha256(task_file.read_bytes()).hexdigest()
            for key in ("id", "scenario", "axes", "industry", "tools", "messages", "expect"):
                if key not in data:
                    raise ValueError(f"missing required key: {key}")
            axes = data["axes"]
            if (not isinstance(axes, list) or not axes
                    or not set(axes) <= _VALID_AXES):
                raise ValueError(f"axes must be a non-empty list drawn from {sorted(_VALID_AXES)}, got {axes!r}")
            if not isinstance(data["tools"], list) or not data["tools"]:
                raise ValueError("tools must be a non-empty list")
            if not isinstance(data["messages"], list) or not data["messages"]:
                raise ValueError("messages must be a non-empty list")
            messages = _apply_filler(data["messages"], data.get("filler"))
            task = Task(
                id=data["id"],
                scenario=data["scenario"],
                axes=sorted(set(int(a) for a in axes)),
                industry=data["industry"],
                difficulty=data.get("difficulty", "medium"),
                tools=data["tools"],
                messages=messages,
                expect=data["expect"],
                rubric=data.get("rubric", {}),
                fixture_sha=fixture_sha,
                path=task_file,
            )
            tasks.append(task)
        except Exception as e:
            # Fail loud on malformed fixtures (mirrors b1_fixtures.load_unit_tasks).
            raise ValueError(f"malformed fixture {task_file}: {e}") from e

    return sorted(tasks, key=lambda t: t.id)


# --- tool schema / expect-block structural validation (preflight) -------

def validate_tool_schemas(tools: list[dict]) -> list[str]:
    """Structural validation that every tool def parses as a usable OpenAI
    `tools` entry. Returns a list of human-readable error strings (empty =
    clean). Used by B2ToolCalling.preflight() -- TESTPLAN 5.2: "preflight():
    all tool schemas parse."""
    errs: list[str] = []
    if not tools:
        return ["no tools defined"]
    seen_names: set[str] = set()
    for i, t in enumerate(tools):
        if not isinstance(t, dict) or t.get("type") != "function":
            errs.append(f"tool[{i}]: type must be 'function'")
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            errs.append(f"tool[{i}]: missing function block")
            continue
        name = fn.get("name")
        if not name or not isinstance(name, str):
            errs.append(f"tool[{i}]: missing/invalid function.name")
            name = f"<tool[{i}]>"
        elif name in seen_names:
            errs.append(f"tool[{i}] ({name}): duplicate tool name")
        else:
            seen_names.add(name)
        if not fn.get("description"):
            errs.append(f"tool[{i}] ({name}): missing description")
        params = fn.get("parameters")
        if not isinstance(params, dict):
            errs.append(f"tool[{i}] ({name}): missing parameters block")
            continue
        if params.get("type") != "object":
            errs.append(f"tool[{i}] ({name}): parameters.type must be 'object'")
        props = params.get("properties", {})
        if not isinstance(props, dict):
            errs.append(f"tool[{i}] ({name}): parameters.properties must be a dict")
            props = {}
        for pname, pschema in props.items():
            ptype = pschema.get("type") if isinstance(pschema, dict) else None
            if ptype not in _JSON_SCHEMA_TYPES:
                errs.append(f"tool[{i}] ({name}).{pname}: invalid/missing type {ptype!r}")
            if isinstance(pschema, dict) and "enum" in pschema and not isinstance(pschema["enum"], list):
                errs.append(f"tool[{i}] ({name}).{pname}: enum must be a list")
        required = params.get("required", [])
        if not isinstance(required, list):
            errs.append(f"tool[{i}] ({name}): required must be a list")
        else:
            unknown_required = [r for r in required if r not in props]
            if unknown_required:
                errs.append(f"tool[{i}] ({name}): required references unknown properties {unknown_required}")
    return errs


def validate_expect_block(task: Task) -> list[str]:
    """Cross-checks task.expect against task.tools (e.g. expect.tool_calls
    must name a tool the task actually declares). Returns error strings."""
    errs: list[str] = []
    tool_names = {t.get("function", {}).get("name") for t in task.tools
                  if isinstance(t, dict) and isinstance(t.get("function"), dict)}
    expect = task.expect or {}
    for c in expect.get("tool_calls", []):
        if c.get("name") not in tool_names:
            errs.append(f"expect.tool_calls references unknown tool {c.get('name')!r}")
    chain = expect.get("chain_check")
    if chain is not None:
        if "arg_path" not in chain or "expected_value" not in chain:
            errs.append("expect.chain_check requires arg_path and expected_value")
    if (5 in task.axes or 8 in task.axes) and not expect.get("fabrication_traps"):
        errs.append("axis 5/8 task should declare expect.fabrication_traps")
    return errs


# --- response parsing -----------------------------------------------------

def extract_tool_calls(response: dict) -> list[dict]:
    msg = (response.get("choices") or [{}])[0].get("message", {}) or {}
    return msg.get("tool_calls") or []


def _response_text(response: dict) -> str:
    msg = (response.get("choices") or [{}])[0].get("message", {}) or {}
    return msg.get("content") or ""


def parse_call_args(call: dict) -> tuple[bool, dict | None, str]:
    """Return (valid_json_object, parsed_args_or_None, raw_arguments_str)."""
    raw = (call.get("function") or {}).get("arguments", "")
    if raw in (None, ""):
        return True, {}, raw or ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False, None, raw
    if not isinstance(parsed, dict):
        return False, None, raw
    return True, parsed, raw


# --- axis 1: schema adherence ----------------------------------------------

def _check_arg_types(args: dict, properties: dict) -> list[str]:
    errs = []
    for pname, pval in args.items():
        pschema = properties.get(pname)
        if not isinstance(pschema, dict):
            continue                      # arg not declared -- caught elsewhere if desired
        ptype = pschema.get("type")
        py_types = _PY_TYPES_BY_SCHEMA_TYPE.get(ptype)
        if py_types is None:
            continue
        # bool is a subclass of int in Python; only accept bool for boolean-typed props.
        if isinstance(pval, bool) and ptype != "boolean":
            errs.append(f"{pname}: expected {ptype}, got boolean")
            continue
        if ptype in ("integer", "number") and isinstance(pval, bool):
            errs.append(f"{pname}: expected {ptype}, got boolean")
            continue
        if not isinstance(pval, py_types):
            errs.append(f"{pname}: expected {ptype}, got {type(pval).__name__}")
    return errs


def _check_enum_constraints(args: dict, properties: dict) -> list[str]:
    errs = []
    for pname, pval in args.items():
        pschema = properties.get(pname)
        if not isinstance(pschema, dict):
            continue
        enum = pschema.get("enum")
        if enum and pval not in enum:
            errs.append(f"{pname}: {pval!r} not in enum {enum}")
    return errs


def check_schema_adherence(calls: list[dict], tools_by_name: dict[str, dict]) -> tuple[bool, list[dict]]:
    """Axis 1. Every emitted call must be valid-JSON args, a known tool name,
    all required params present, arg types + enum constraints respected.
    Vacuously True (nothing to be invalid) when zero calls were emitted --
    other axes (2/3/4/6/7) judge WHETHER a call should have happened."""
    all_valid = True
    details = []
    for call in calls:
        name = (call.get("function") or {}).get("name")
        valid_json, args, _raw = parse_call_args(call)
        entry: dict[str, Any] = {"name": name, "valid_json": valid_json}
        if not valid_json:
            all_valid = False
            entry["error"] = "arguments not valid JSON object"
            details.append(entry)
            continue
        tool = tools_by_name.get(name)
        if tool is None:
            all_valid = False
            entry["error"] = "unknown tool name (hallucinated)"
            details.append(entry)
            continue
        params = (tool.get("function") or {}).get("parameters", {}) or {}
        required = params.get("required", []) or []
        properties = params.get("properties", {}) or {}
        missing = [p for p in required if p not in args]
        if missing:
            all_valid = False
            entry["missing_required"] = missing
        type_errors = _check_arg_types(args, properties)
        if type_errors:
            all_valid = False
            entry["type_errors"] = type_errors
        enum_errors = _check_enum_constraints(args, properties)
        if enum_errors:
            all_valid = False
            entry["enum_errors"] = enum_errors
        details.append(entry)
    return all_valid, details


# --- axis 2/3: call matching helpers ---------------------------------------

def _args_match(parsed: dict, expected: dict, mode: str = "subset") -> bool:
    if mode == "exact" and set(parsed.keys()) != set(expected.keys()):
        return False
    for k, v in expected.items():
        if k not in parsed:
            return False
        pv = parsed[k]
        if pv == v:
            continue
        if str(pv).strip() == str(v).strip():
            continue
        return False
    return True


def _call_matches_expected(call: dict, expected_call: dict) -> bool:
    fn = call.get("function") or {}
    if fn.get("name") != expected_call.get("name"):
        return False
    ok_json, parsed, _raw = parse_call_args(call)
    if not ok_json or parsed is None:
        return False
    return _args_match(parsed, expected_call.get("args", {}), expected_call.get("args_match", "subset"))


def check_tool_selection(calls: list[dict], expect: dict) -> dict | None:
    """Axis 2 (also reused for axis 7's base check). None if the task doesn't
    declare any expect.tool_calls (nothing to select among)."""
    expected_calls = expect.get("tool_calls", [])
    if not expected_calls:
        return None
    called_names = [(c.get("function") or {}).get("name") for c in calls]
    if not calls:
        return {"pass": False, "detail": "no tool call emitted", "called": [],
                "expected": sorted({c["name"] for c in expected_calls})}
    expected_names = {c["name"] for c in expected_calls}
    forbidden = set(expect.get("forbidden_tools", []))
    forbidden_hit = [n for n in called_names if n in forbidden]
    # Primary-call correctness: at least one emitted call must both target an
    # expected tool AND match that call's expected args (not just the name) --
    # a right-tool/wrong-args call is not "correct selection".
    right_call = any(any(_call_matches_expected(c, ec) for ec in expected_calls) for c in calls)
    return {"pass": right_call and not forbidden_hit,
            "called": called_names, "expected": sorted(expected_names),
            "forbidden_hit": forbidden_hit}


def check_parallel_calls(calls: list[dict], expect: dict) -> dict | None:
    """Axis 3. None unless expect.parallel_ok and >=2 expected calls."""
    expected_calls = expect.get("tool_calls", [])
    if not expect.get("parallel_ok") or len(expected_calls) < 2:
        return None
    if len(calls) < len(expected_calls):
        return {"pass": False, "n_calls": len(calls),
                "detail": f"expected {len(expected_calls)} parallel calls, got {len(calls)}"}
    remaining = list(calls)
    unmatched = []
    for ec in expected_calls:
        idx = next((i for i, c in enumerate(remaining) if _call_matches_expected(c, ec)), None)
        if idx is None:
            unmatched.append(ec.get("name"))
        else:
            remaining.pop(idx)
    return {"pass": not unmatched, "unmatched_expected": unmatched, "n_calls": len(calls)}


def check_chained_call(calls: list[dict], expect: dict) -> dict | None:
    """Axis 4. None unless expect.chain_check is set."""
    spec = expect.get("chain_check")
    if not spec:
        return None
    key_path = spec["arg_path"].split(".")
    target = spec["expected_value"]
    for call in calls:
        _ok, args, _raw = parse_call_args(call)
        if not args:
            continue
        val: Any = args
        for k in key_path:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                val = None
                break
        if val is not None and str(val) == str(target):
            return {"pass": True, "found_in": (call.get("function") or {}).get("name")}
    return {"pass": False, "detail": f"no call used {spec['arg_path']}={target!r}"}


def check_abstention(calls: list[dict], expect: dict) -> dict | None:
    """Axis 6. None unless expect.expect_no_call is set."""
    if not expect.get("expect_no_call"):
        return None
    if not calls:
        return {"pass": True, "detail": "no tool call emitted, as expected"}
    called = [(c.get("function") or {}).get("name") for c in calls]
    return {"pass": False, "detail": f"invented/used tool call(s) when none was appropriate: {called}"}


def check_fabrication_guard(text: str, calls: list[dict], expect: dict) -> dict | None:
    """Best-effort deterministic floor for the judged axes (5, 8): none of the
    planted `fabrication_traps` values may appear in the final answer text or
    in any emitted call's raw arguments. Passing is necessary, not sufficient
    -- real scoring for these two axes is judged (see module docstring)."""
    traps = expect.get("fabrication_traps")
    if not traps:
        return {"pass": None, "detail": "no fabrication_traps configured; judged axis only"}
    hit = [t for t in traps if t in text]
    for c in calls:
        _ok, _args, raw = parse_call_args(c)
        hit += [t for t in traps if raw and t in raw and t not in hit]
    return {"pass": not hit, "hit_traps": hit}


# --- top-level scorer -------------------------------------------------------

def score_axes(response: dict, task: Task) -> tuple[dict, bool, dict]:
    """Score one B2 response against `task`'s applicable axes.

    Returns (det_checks, needs_judging, metrics):
      - det_checks: {"axis1_schema_adherence": {...}, "axisN_...": {...}, ...}
        keyed only for axes the task actually exercises (axis 1 always runs).
      - needs_judging: True iff set(task.axes) & {5, 8}.
      - metrics: {"n_tool_calls", "axes_applicable", "det_pass"}.
    """
    calls = extract_tool_calls(response)
    text = _response_text(response)
    tools_by_name = {t.get("function", {}).get("name"): t for t in task.tools
                      if isinstance(t, dict) and t.get("function", {}).get("name")}
    expect = task.expect
    axes = set(task.axes)

    det_checks: dict[str, Any] = {}
    schema_ok, schema_details = check_schema_adherence(calls, tools_by_name)
    det_checks["axis1_schema_adherence"] = {"pass": schema_ok, "calls": schema_details}

    if 2 in axes:
        det_checks["axis2_tool_selection"] = check_tool_selection(calls, expect)
    if 3 in axes:
        det_checks["axis3_parallel_calls"] = check_parallel_calls(calls, expect)
    if 4 in axes:
        det_checks["axis4_chained_calls"] = check_chained_call(calls, expect)
    if 6 in axes:
        det_checks["axis6_abstention"] = check_abstention(calls, expect)
    if 7 in axes:
        det_checks["axis7_long_context_call"] = check_tool_selection(calls, expect)
    if 5 in axes:
        det_checks["axis5_fabrication_guard"] = check_fabrication_guard(text, calls, expect)
    if 8 in axes:
        det_checks["axis8_fabrication_guard"] = check_fabrication_guard(text, calls, expect)

    det_checks = {k: v for k, v in det_checks.items() if v is not None}
    needs_judging = bool(axes & _JUDGED_AXES)

    bool_results = [v.get("pass") for v in det_checks.values()
                     if isinstance(v, dict) and isinstance(v.get("pass"), bool)]
    metrics = {
        "n_tool_calls": len(calls),
        "axes_applicable": sorted(axes),
        "det_pass": all(bool_results) if bool_results else None,
    }
    return det_checks, needs_judging, metrics
