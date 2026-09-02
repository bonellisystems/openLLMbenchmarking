"""Job-application loop: tailor a resume to a JD, then drive a picky ATS form.

The form is a harness stand-in for a browser: required fields, LinkedIn URL
check, file upload, and inline validation errors on submit. No Playwright
required. Scores whether the model recovers from 'you forgot X' instead of
narrating that it applied.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from spark_hour.client import chat

JD = """\
Contract title: Senior Automation Engineer (6-month remote)
Rate: 1099
Must-haves:
- Python
- Browser automation (Playwright or equivalent) to drive real web UIs
- Filling application forms and recovering from client-side validation
  (missing required fields, files not uploaded, bad URLs)
- LLM tool calling / OpenAI-compatible endpoints
- Rewriting a resume and cover note to match EACH posting, not a generic dump
Nice: local GPU inference, DGX or similar, Docker, Linux.
Apply via the contractor portal form. Incomplete submits are rejected.
"""

BIO = """\
Alex Rivera, 11 years. Python services and internal tooling. Automated QA with
Selenium, then Playwright, including flaky form flows. Deployed local LLMs to
rewrite documents against a job description. Used to contractor boards where
the apply page throws 'LinkedIn required' and 'resume PDF missing' after
submit. Also: REST APIs, Docker, Linux, OpenAI-compatible local servers.
Not a full-time employee seeker; prefers 3–12 month contracts.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "read_listing",
        "description": "Read the job listing and the current application form state, including any validation errors from the last submit.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a text file in the workspace (use for resume.md / cover.txt).",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string"},
                           "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a workspace file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "fill_field",
        "description": "Set one form field. Names: full_name, email, linkedin, work_auth, cover_note.",
        "parameters": {"type": "object",
                       "properties": {
                           "name": {"type": "string"},
                           "value": {"type": "string"}},
                       "required": ["name", "value"]}}},
    {"type": "function", "function": {
        "name": "upload_file",
        "description": "Attach a workspace file as the resume upload.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "submit_form",
        "description": "Submit the application. Returns {ok, errors}. On errors, fix and submit again.",
        "parameters": {"type": "object", "properties": {}}}},
]


class Form:
    def __init__(self, ws: Path):
        self.ws = ws
        self.fields = {
            "full_name": "",
            "email": "",
            "linkedin": "",
            "work_auth": "",
            "cover_note": "",
        }
        self.resume_path = ""
        self.errors: list[str] = []
        self.submitted = False
        self.submit_attempts = 0
        self.error_rounds = 0

    def snapshot(self) -> str:
        return json.dumps({
            "listing": JD,
            "fields": self.fields,
            "resume_uploaded": bool(self.resume_path),
            "resume_path": self.resume_path,
            "last_errors": self.errors,
            "submitted": self.submitted,
        }, indent=2)

    def fill(self, name: str, value: str) -> str:
        if name not in self.fields:
            return f"unknown field {name}. valid: {sorted(self.fields)}"
        self.fields[name] = str(value or "").strip()
        return f"set {name}"

    def upload(self, path: str) -> str:
        p = (self.ws / path).resolve()
        if not str(p).startswith(str(self.ws.resolve())):
            return "path escapes workspace"
        if not p.is_file():
            return f"file not found: {path}"
        self.resume_path = path
        return f"uploaded {path} ({p.stat().st_size} bytes)"

    def submit(self) -> str:
        self.submit_attempts += 1
        err = []
        if len(self.fields["full_name"]) < 4:
            err.append("Full name is required.")
        if not re.search(r"^[^@]+@[^@]+\.[^@]+$", self.fields["email"]):
            err.append("A valid email is required.")
        if "linkedin.com/" not in self.fields["linkedin"].lower():
            err.append("LinkedIn URL is required (must contain linkedin.com/).")
        if not self.fields["work_auth"]:
            err.append("Work authorization is required.")
        if not self.resume_path:
            err.append("Resume file was not uploaded.")
        else:
            p = self.ws / self.resume_path
            body = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
            if len(body) < 400:
                err.append("Uploaded resume is too short / empty.")
            if "playwright" not in body.lower() and "selenium" not in body.lower():
                err.append("Resume does not mention browser automation (Playwright/Selenium).")
            if "python" not in body.lower():
                err.append("Resume does not mention Python.")
        self.errors = err
        if err:
            self.error_rounds += 1
            self.submitted = False
            return json.dumps({"ok": False, "errors": err})
        self.submitted = True
        return json.dumps({"ok": True, "errors": [], "confirmation": "APP-7741"})


def _exec_tool(form: Form, name: str, args: dict) -> str:
    ws = form.ws
    try:
        if name == "read_listing":
            return form.snapshot()
        if name == "write_file":
            p = (ws / str(args.get("path", ""))).resolve()
            if not str(p).startswith(str(ws.resolve())):
                return "path escapes workspace"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {args.get('path')} ({p.stat().st_size} bytes)"
        if name == "read_file":
            p = (ws / str(args.get("path", ""))).resolve()
            if not str(p).startswith(str(ws.resolve())):
                return "path escapes workspace"
            if not p.is_file():
                return "not found"
            return p.read_text(encoding="utf-8", errors="replace")[:8000]
        if name == "fill_field":
            return form.fill(str(args.get("name", "")), str(args.get("value", "")))
        if name == "upload_file":
            return form.upload(str(args.get("path", "")))
        if name == "submit_form":
            return form.submit()
        return f"unknown tool {name}"
    except Exception as e:
        return f"tool error: {type(e).__name__}: {e}"


def score_resume(text: str) -> dict:
    t = (text or "").lower()
    checks = {
        "long_enough": len(text or "") >= 600,
        "mentions_python": "python" in t,
        "mentions_browser": ("playwright" in t) or ("selenium" in t) or ("browser" in t),
        "mentions_title": "automation" in t,
        "has_experience": "experience" in t,
        "contract_not_fte": ("contract" in t) or ("1099" in t) or ("freelance" in t),
    }
    return {"pass": all(checks.values()), "checks": checks, "chars": len(text or "")}


def run_resume(chat_url: str, model: str, ws: Path, *, timeout: int = 180) -> dict:
    ws.mkdir(parents=True, exist_ok=True)
    prompt = (
        "You are rewriting a contractor resume for ONE posting. Use only the bio. "
        "Do not invent employers that are not implied. Output the resume as markdown.\n\n"
        f"JOB LISTING:\n{JD}\n\nCANDIDATE BIO:\n{BIO}\n"
    )
    t0 = time.perf_counter()
    resp = chat(chat_url, model, [{"role": "user", "content": prompt}],
                max_tokens=2048, thinking=True, timeout=timeout)
    text = (resp.get("content") or "") or (resp.get("reasoning") or "")
    (ws / "resume.md").write_text(text, encoding="utf-8")
    sc = score_resume(text)
    return {
        "battery": "job", "id": "job.resume_tailor",
        "passed": sc["pass"], "infra": bool(resp.get("error")),
        "chars": sc["chars"], "checks": sc["checks"],
        "e2e_s": round(time.perf_counter() - t0, 2),
        "decode_tok_s": resp.get("decode_tok_s"),
        "error": resp.get("error"),
        "content_chars": len(resp.get("content") or ""),
        "reasoning_chars": len(resp.get("reasoning") or ""),
    }


def run_form(chat_url: str, model: str, ws: Path, *, timeout: int = 180, max_steps: int = 14) -> dict:
    form = Form(ws)
    msgs = [
        {"role": "system", "content": (
            "You are applying to a contractor role through a web form. "
            "The listing and form are tools, not in the chat. "
            "Write a tailored resume to a file, fill every required field, upload the resume, "
            "and submit. If submit returns errors, fix those fields and submit again. "
            "Do not claim success unless submit_form returned ok true."
        )},
        {"role": "user", "content": (
            "Apply to the Senior Automation Engineer contract. "
            "Candidate bio is in the listing tool. Tailor the resume to the posting."
        )},
    ]
    t0 = time.perf_counter()
    used = []
    err = None
    for step in range(max_steps):
        try:
            resp = chat(chat_url, model, msgs, max_tokens=700, thinking=True,
                        timeout=timeout, tools=TOOLS)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        raw = resp.get("raw") or {}
        msg = ((raw.get("choices") or [{}])[0].get("message") or {})
        calls = msg.get("tool_calls") or resp.get("tool_calls") or []
        content = msg.get("content") or resp.get("content") or ""
        msgs.append({"role": "assistant", "content": content,
                     **({"tool_calls": calls} if calls else {})})
        if not calls:
            break
        for c in calls:
            fn = c.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _exec_tool(form, name, args)
            used.append(name)
            msgs.append({"role": "tool", "tool_call_id": c.get("id") or "",
                         "name": name, "content": result[:4000]})
        if form.submitted:
            break
    return {
        "battery": "job", "id": "job.form_recover",
        "passed": bool(form.submitted), "infra": bool(err),
        "steps": len(used), "tools": used,
        "submit_attempts": form.submit_attempts,
        "error_rounds": form.error_rounds,
        "errors_last": form.errors,
        "resume_path": form.resume_path,
        "e2e_s": round(time.perf_counter() - t0, 2),
        "error": err,
    }
