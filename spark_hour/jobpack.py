"""Ashby application clone: Engineering - Internal AI Transformation @ ElevenLabs.

Form layout matches the hosted Ashby apply page (system fields + this posting's
location/hybrid/visa questions). Nothing is submitted to ElevenLabs — the harness
is a local clone that returns the same class of inline validation errors.
Source: https://jobs.ashbyhq.com/elevenlabs/a3097257-a07a-4a7e-b9fe-b8555c1a0fa7
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from spark_hour.client import chat

POSTING_URL = "https://jobs.ashbyhq.com/elevenlabs/a3097257-a07a-4a7e-b9fe-b8555c1a0fa7"

JD = """\
Engineering - Internal AI Transformation @ ElevenLabs
Full-time · Remote · Hybrid 3 days/week in-office
Hubs: London, San Francisco, New York, Warsaw (also Dublin, Berlin, Amsterdam, …)
US compensation listed on the posting: $300,000–$420,000 + equity.

Apply at:
https://jobs.ashbyhq.com/elevenlabs/a3097257-a07a-4a7e-b9fe-b8555c1a0fa7

The role / impact
Internal AI Engineer, embedded across GTM, Operations, and Finance. Co-design and
ship agentic AI workflows that eliminate manual toil. Own automations from
discovery to production. Patterns become the blueprint other teams copy.

What you'll actually be working on
- Designing and iterating on AI agents and workflow orchestrations using
  ElevenAgents, Claude, and n8n.
- Integrating AI systems with the core business stack — Salesforce, Slack, Ashby.
- Building reusable automation services, patterns and shared templates.
- Owning experiments end-to-end and measuring impact.
- Evaluation and monitoring so AI-native workflows are reliable, auditable, safe.

What they're looking for
- You've designed, shipped, and iterated on AI agents or automated workflows
  that real people depend on (not just demos).
- You treat non-technical operators as your users.
- You move from messy problem to working software without a perfect spec.
- Production judgement: evals, failure modes, human-in-the-loop, when not to automate.
- Strong software engineering (Python) plus systems thinking.
- Bonus: GTM/RevOps/Finance tooling, Salesforce, data warehouses, ElevenLabs APIs.

How we work: high-velocity, no job titles, AI first, hybrid 3 days in a hub.
"""

BIO = """\
Alex Rivera, 11 years. Python services and internal tooling. Automated messy
operator workflows with n8n and custom agents (Claude + OpenAI-compatible local
LLMs). Integrated Slack and Salesforce for GTM/ops handoffs. Used Playwright
and form automation when a web UI was the only API. Comfortable shipping
without a perfect spec, adding evals, and putting a human in the loop when
the workflow can hurt someone. Contract and full-time mix; can work US timezones.
GitHub: github.com/example-arivera
LinkedIn: https://www.linkedin.com/in/example-arivera
Based near a major hub; open to 3 days/week in-office in New York.
US work authorized, no visa sponsorship required.
"""

HUBS = ("London", "San Francisco", "New York", "Warsaw", "Other")
HEAR = ("LinkedIn", "Ashby / careers page", "Referral", "X / Twitter", "Other")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_listing",
        "description": "Read the ElevenLabs Ashby job listing and the live application form, including any validation errors from the last submit.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a workspace file (resume.md, cover.txt, etc.).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a workspace file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "fill_field",
        "description": "Set one Ashby application field. Names: name, email, linkedin, phone, github, how_heard, hub, visa_sponsorship, additional.",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                       "required": ["name", "value"]}}},
    {"type": "function", "function": {
        "name": "upload_file",
        "description": "Attach a workspace file as Resume (Ashby required resume upload).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "submit_form",
        "description": "Click Apply. Returns {ok, errors} like Ashby inline validation. Fix errors and submit again.",
        "parameters": {"type": "object", "properties": {}}}},
]


class Form:
    """Ashby apply page clone for this posting. Not submitted off-box."""

    def __init__(self, ws: Path):
        self.ws = ws
        self.fields = {
            "name": "",
            "email": "",
            "linkedin": "",
            "phone": "",
            "github": "",
            "how_heard": "",
            "hub": "",
            "visa_sponsorship": "",
            "additional": "",
        }
        self.resume_path = ""
        self.errors: list[str] = []
        self.submitted = False
        self.submit_attempts = 0
        self.error_rounds = 0

    def snapshot(self) -> str:
        return (
            f"JOB POSTING\n{POSTING_URL}\n\n{JD}\n\n"
            "APPLICATION FORM (Ashby — same required fields as the live apply page)\n"
            "  Name *\n"
            "  Email *\n"
            "  Resume *  (file upload)\n"
            "  LinkedIn URL\n"
            "  Phone\n"
            "  GitHub / portfolio URL\n"
            f"  How did you hear about ElevenLabs? *  options: {', '.join(HEAR)}\n"
            f"  Which hub can you work from 3 days/week? *  options: {', '.join(HUBS)}\n"
            "  Will you require visa sponsorship? *  Yes / No\n"
            "  Additional information\n\n"
            "CURRENT STATE\n"
            + json.dumps({
                "fields": self.fields,
                "resume_uploaded": bool(self.resume_path),
                "resume_path": self.resume_path,
                "last_errors": self.errors,
                "submitted": self.submitted,
            }, indent=2)
        )

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
        name = self.fields["name"]
        if len(name.split()) < 2:
            err.append("Name is required (first and last).")
        if not re.search(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", self.fields["email"]):
            err.append("Email is required.")
        if not self.resume_path:
            err.append("Resume is required.")
        else:
            p = self.ws / self.resume_path
            body = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
            if len(body) < 400:
                err.append("Resume file is empty or too short.")
            low = body.lower()
            if "python" not in low:
                err.append("Resume does not mention Python.")
            if not any(w in low for w in ("agent", "workflow", "n8n", "automat")):
                err.append("Resume does not show shipped agents/workflows.")
        if self.fields["linkedin"] and "linkedin.com/" not in self.fields["linkedin"].lower():
            err.append("LinkedIn URL must contain linkedin.com/")
        if self.fields["how_heard"] not in HEAR:
            err.append("How did you hear about ElevenLabs? is required.")
        if self.fields["hub"] not in HUBS:
            err.append("Which hub can you work from 3 days/week? is required.")
        visa = self.fields["visa_sponsorship"].lower()
        if visa not in ("yes", "no"):
            err.append("Will you require visa sponsorship? is required (Yes/No).")
        self.errors = err
        if err:
            self.error_rounds += 1
            self.submitted = False
            return json.dumps({"ok": False, "errors": err})
        self.submitted = True
        return json.dumps({
            "ok": True,
            "errors": [],
            "confirmation": "Thank you for applying to ElevenLabs! We'll be in touch via email shortly.",
        })


def _exec_tool(form: Form, name: str, args: dict) -> str:
    ws = form.ws
    try:
        if name == "read_listing":
            return form.snapshot()
        if name == "write_file":
            p = (ws / str(args.get("path", ""))).resolve()
            if not str(p).startswith(str(self_ws(ws))):
                return "path escapes workspace"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {args.get('path')} ({p.stat().st_size} bytes)"
        if name == "read_file":
            p = (ws / str(args.get("path", ""))).resolve()
            if not str(p).startswith(str(self_ws(ws))):
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


def self_ws(ws: Path) -> str:
    return str(ws.resolve())


def score_resume(text: str) -> dict:
    t = (text or "").lower()
    checks = {
        "long_enough": len(text or "") >= 600,
        "mentions_python": "python" in t,
        "mentions_agents": any(w in t for w in ("agent", "agentic", "workflow", "n8n")),
        "mentions_stack": any(w in t for w in ("salesforce", "slack", "ashby", "claude", "eleven")),
        "has_experience": "experience" in t,
        "internal_ops": any(w in t for w in ("internal", "operator", "gtm", "ops", "finance", "automat")),
    }
    return {"pass": all(checks.values()), "checks": checks, "chars": len(text or "")}


def run_resume(chat_url: str, model: str, ws: Path, *, timeout: int = 180) -> dict:
    ws.mkdir(parents=True, exist_ok=True)
    prompt = (
        "Rewrite the candidate bio into a resume tailored to this ONE Ashby posting. "
        "Do not invent employers. Markdown only.\n\n"
        f"POSTING:\n{JD}\n\nBIO:\n{BIO}\n"
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


def run_form(chat_url: str, model: str, ws: Path, *, timeout: int = 180, max_steps: int = 16) -> dict:
    form = Form(ws)
    msgs = [
        {"role": "system", "content": (
            "You are applying on Ashby. The listing and form are tools. "
            "Write a tailored resume file, fill every required field, upload the resume, "
            "and submit. If Apply returns errors, fix those fields and submit again. "
            "Do not claim success unless submit_form returned ok true."
        )},
        {"role": "user", "content": (
            "Apply to Engineering - Internal AI Transformation at ElevenLabs on Ashby. "
            "Use the candidate bio from the listing tool. Tailor the resume to this posting."
        )},
    ]
    t0 = time.perf_counter()
    used = []
    err = None
    for _step in range(max_steps):
        try:
            resp = chat(chat_url, model, msgs, max_tokens=800, thinking=True,
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
        "posting": POSTING_URL,
    }
