"""ServerManager (TESTPLAN 7.3): all serving goes through here; provenance auto-attached.
Teardown by PID only. Orphan sweep before launch. fits() preflight. Fork implemented;
ollama sanctioned arm implemented (B5); vllm = remote-attach later phase."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llmtest.registry import Config, fits
from llmtest.schema import SessionRow, SCHEMA_VERSION
from llmtest.store import Store

_NGRAM_RE = re.compile(r"^ngram(\d+)$")

# Flags in runtime_pins.yaml's standard_flags that take a value (vs. bare boolean
# flags like --jinja). Needed to tokenize the pin string without guessing.
_VALUE_FLAGS = {"-ngl", "-c", "-np", "-ctk", "-ctv", "-fa",
                "--spec-type", "--spec-ngram-mod-n-match", "--cache-ram"}
# Flags always re-derived from the call's ctx/kv/parallel args, never taken from
# the pin verbatim (so a per-call override can never be silently ignored).
_OVERRIDE_FLAGS = {"-c", "-np", "-ctk", "-ctv"}


def _resolve_spec(spec: str) -> tuple[str, dict]:
    """Map a raw spec token ('off' | 'ngramN' | passthrough) to (spec_type,
    spec_params). TESTPLAN 2: never run ngram n-match < 16 -- enforced here so
    both compose_fork_flags() and normalize_config() get the guard for free,
    whether the spec came from a caller overlay or from the pin itself."""
    if spec == "off":
        return "none", {}
    m = _NGRAM_RE.match(spec)
    if m:
        n = int(m.group(1))
        if n < 16:
            raise ValueError("TESTPLAN 2: never run ngram n-match < 16")
        return "ngram-mod", {"n_match": n}
    return spec, {}


def _tokenize_flags(flags_str: str) -> list[tuple[str, str | None]]:
    """Split a llama.cpp-fork flag string into ordered (flag, value) pairs,
    value=None for bare boolean flags."""
    toks = flags_str.split()
    pairs: list[tuple[str, str | None]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _VALUE_FLAGS and i + 1 < len(toks):
            pairs.append((t, toks[i + 1]))
            i += 2
        else:
            pairs.append((t, None))
            i += 1
    return pairs


def _pinned_spec_string(cfg: Config) -> str:
    """The spec token implied by cfg.runtime_pins['standard_flags'] alone, in
    the same 'off' | 'ngramN' shape an overlay's "spec" key accepts -- so a
    caller that omits an overlay spec still gets pin-accurate provenance
    instead of a hardcoded default (the config can never silently lie)."""
    spec_type, n_match = None, None
    for flag, value in _tokenize_flags(cfg.runtime_pins["standard_flags"]):
        if flag == "--spec-type":
            spec_type = value
        elif flag == "--spec-ngram-mod-n-match":
            n_match = value
    if spec_type == "ngram-mod" and n_match is not None:
        return f"ngram{n_match}"
    if spec_type in (None, "none"):
        return "off"
    return spec_type


def normalize_config(*, runtime, ctx, kv, spec, parallel, flash_attn=True) -> dict:
    spec_type, spec_params = _resolve_spec(spec)
    return {"ctx": ctx, "kv_dtype": kv, "flash_attn": flash_attn,
            "spec_type": spec_type, "spec_params": spec_params, "parallel": parallel}


def compose_fork_flags(cfg: Config, *, ctx: int, parallel: int, kv: str,
                       overlay: dict | None) -> str:
    overlay = overlay or {}
    spec = overlay.get("spec") or _pinned_spec_string(cfg)
    spec_type, spec_params = _resolve_spec(spec)

    kept = []
    for flag, value in _tokenize_flags(cfg.runtime_pins["standard_flags"]):
        if flag in ("--spec-type", "--spec-ngram-mod-n-match") or flag in _OVERRIDE_FLAGS:
            continue                      # rebuilt below from call args / resolved spec
        kept.append(f"{flag} {value}" if value is not None else flag)

    parts = kept + [f"-c {ctx}"]
    if spec_type == "ngram-mod":
        parts += ["--spec-type ngram-mod",
                  f"--spec-ngram-mod-n-match {spec_params['n_match']}"]
    else:
        parts += [f"--spec-type {spec_type}"]
    if kv != "f16":
        parts += [f"-ctk {kv}", f"-ctv {kv}"]
    if parallel > 1:
        parts += [f"-np {parallel}"]
    return " ".join(parts)


def _free_port(start=8080) -> int:
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def _vram_free_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0]) / 1024


def _power_state() -> tuple[str, str]:
    """(power_mode, ac_state). power_mode from `powercfg /getactivescheme`'s
    active scheme name; ac_state from Win32_Battery.BatteryStatus via CIM
    (2=ac, 1=battery, no battery instance=ac, query error=unknown). Any
    failure anywhere in here falls back to ("unknown", "unknown") -- this must
    never raise and abort a launch."""
    try:
        scheme = subprocess.run(["powercfg", "/getactivescheme"],
                                capture_output=True, text=True, check=True).stdout
        low = scheme.lower()
        if "performance" in low or "ultimate" in low:
            mode = "performance"
        elif "balanced" in low:
            mode = "balanced"
        elif "power saver" in low:
            mode = "powersaver"
        else:
            mode = "unknown"

        ac = "unknown"
        try:
            ps = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Battery).BatteryStatus"],
                capture_output=True, text=True, check=True)
            out = ps.stdout.strip()
            if out == "2":
                ac = "ac"
            elif out == "1":
                ac = "battery"
            elif out == "":
                ac = "ac"           # no battery instance -> query ok, nothing to report
            else:
                ac = "unknown"
        except Exception:
            ac = "unknown"

        return mode, ac
    except Exception:
        return "unknown", "unknown"


@dataclass
class EndpointHandle:
    base_url: str
    session_id: str
    normalized_config: dict
    pid: int
    _mgr: "ServerManager"

    def chat(self, messages, *, max_tokens=512, temperature=0.0, tools=None,
             timeout=1200) -> dict:
        body = {"messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": False}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=timeout))


class ServerManager:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self._active: EndpointHandle | None = None
        self._active_key: tuple | None = None
        self._log_fh = None

    def sweep_orphans(self) -> list[str]:
        killed = []
        for name in ("llama-server.exe",):
            r = subprocess.run(["taskkill", "/F", "/IM", name],
                               capture_output=True, text=True)
            if "SUCCESS" in r.stdout:
                killed.append(name)
        time.sleep(2)
        return killed

    @staticmethod
    def _request_key(model_id: str, runtime: str, flags_overlay: dict | None,
                     parallel: int, ctx: int, kv: str,
                     timing_authoritative: bool) -> tuple:
        """Reuse key for config-match server reuse. Includes timing_authoritative:
        a timing-authoritative request must never silently reuse a warm endpoint
        that was launched under a non-authoritative (e.g. cold/contended) call,
        or vice versa."""
        return (model_id, runtime, json.dumps(flags_overlay, sort_keys=True),
               parallel, ctx, kv, timing_authoritative)

    def request_endpoint(self, model_id: str, runtime: str = "fork",
                         flags_overlay: dict | None = None, parallel: int = 1,
                         ctx: int = 8192, kv: str = "q8_0",
                         timing_authoritative: bool = False) -> EndpointHandle:
        key = self._request_key(model_id, runtime, flags_overlay, parallel,
                                ctx, kv, timing_authoritative)
        if self._active and self._active_key == key:
            return self._active                       # config-match reuse
        self.teardown()
        if runtime != "fork":
            raise NotImplementedError(f"runtime {runtime} lands in Task 12 (ollama) / later (vllm)")
        model = self.cfg.registry["models"][model_id]
        fit = fits(model, self.cfg.tiers, kv, tier="T1")
        if not fit.fits:
            raise RuntimeError(f"fits() preflight failed: {fit.detail}")
        if _vram_free_gb() < float(model["weights_gb"]) + 1.0:
            self.sweep_orphans()
            if _vram_free_gb() < float(model["weights_gb"]) + 1.0:
                raise RuntimeError("insufficient VRAM after orphan sweep")
        flags = compose_fork_flags(self.cfg, ctx=ctx, parallel=parallel, kv=kv,
                                   overlay=flags_overlay)
        port = _free_port()
        binary = self.cfg.runtime_pins["fork"]["binary"]
        invocation = (f'"{binary}" -m "{model["local_path"]}" {flags} '
                      f"--host 127.0.0.1 --port {port}")
        session_id = f"s-{uuid.uuid4().hex[:12]}"          # generated before Popen: log name = session
        log = Path("artifacts") / f"server-{session_id}.log"
        log.parent.mkdir(exist_ok=True)
        self._log_fh = log.open("w")
        proc = subprocess.Popen(invocation, stdout=self._log_fh,
                                stderr=subprocess.STDOUT, shell=True)
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(f"server died on launch; see {log}")
                time.sleep(1)
        else:
            raise RuntimeError("server health timeout")
        spec = (flags_overlay or {}).get("spec") or _pinned_spec_string(self.cfg)
        mode, ac = _power_state()
        self.store.append_session(SessionRow(
            schema_version=SCHEMA_VERSION, session_id=session_id,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            runtime="llamacpp-fork",
            runtime_build=self.cfg.runtime_pins["fork"]["build_id"],
            normalized_config=normalize_config(runtime="fork", ctx=ctx, kv=kv,
                                               spec=spec, parallel=parallel),
            raw_invocation=invocation, hardware_sku="rtx5090-laptop",
            measured_usable_vram_gb=self.cfg.tiers["tiers"]["T1"]["usable_gb"],
            tp_degree=1, topology=None, driver_env={},
            power_mode=mode, ac_state=ac,
            timing_authoritative=timing_authoritative).to_dict())
        self._active = EndpointHandle(base_url=base, session_id=session_id,
                                      normalized_config=normalize_config(
                                          runtime="fork", ctx=ctx, kv=kv,
                                          spec=spec, parallel=parallel),
                                      pid=proc.pid, _mgr=self)
        self._active_key = key
        return self._active

    def teardown(self) -> None:
        had_active = self._active is not None
        if had_active:
            subprocess.run(["taskkill", "/F", "/PID", str(self._active.pid), "/T"],
                           capture_output=True)
            self._active = None
            self._active_key = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        if not had_active:
            return
        deadline = time.time() + 30
        while time.time() < deadline and _vram_free_gb() < 5.0:
            time.sleep(1)
