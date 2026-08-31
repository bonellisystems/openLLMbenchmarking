"""ServerManager (TESTPLAN 7.3): all serving goes through here; provenance auto-attached.
Teardown by PID only. Orphan sweep before launch. fits() preflight. Fork implemented;
ollama sanctioned arm implemented (B5); vllm = remote-attach later phase."""
from __future__ import annotations

import json
import os
import re
import signal
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
    active scheme GUID (matched first) or fallback to scheme name;
    ac_state from Win32_Battery.BatteryStatus via CIM (2=ac, 1=battery,
    no battery instance=ac, query error=unknown). Any failure anywhere in
    here falls back to ("unknown", "unknown") -- this must never raise and
    abort a launch."""
    # Known Windows power scheme GUIDs (locale-independent)
    GUID_MAP = {
        "381b4222-f694-41f0-9685-ff5bb260df2e": "balanced",
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "performance",
        "e9a42b02-d5df-448d-aa66-ad42f0e084bd": "performance",  # Ultimate Performance
        "a1841308-3541-4fab-bc81-f71556f20b4a": "powersaver",
    }

    try:
        scheme = subprocess.run(["powercfg", "/getactivescheme"],
                                capture_output=True, text=True, check=True).stdout

        # Try GUID matching first (locale-independent)
        mode = "unknown"
        for guid, mode_name in GUID_MAP.items():
            if guid.lower() in scheme.lower():
                mode = mode_name
                break

        # Fall back to substring matching on scheme name if no GUID matched
        if mode == "unknown":
            low = scheme.lower()
            if "performance" in low or "ultimate" in low:
                mode = "performance"
            elif "balanced" in low:
                mode = "balanced"
            elif "power saver" in low:
                mode = "powersaver"

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
    model_name: str = ""

    def chat(self, messages, *, max_tokens=512, temperature=0.0, tools=None,
             timeout=1200) -> dict:
        body = {"messages": messages, "max_tokens": max_tokens,
                "stream": False}
        if self.model_name:
            body["model"] = self.model_name
        # temperature=None omits the key from body (runtime default); temperature=0.0 includes it
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools
        extra = os.environ.get("LLMTEST_CHAT_EXTRA")
        if extra:
            body.update(json.loads(extra))
        url = self.base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"
        req = urllib.request.Request(url + "/chat/completions",
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
        """Kill stray servers holding VRAM.

        Platform-aware because this class is no longer Windows-only: the whole
        canonical `llmtest run` path was unusable on a rented Linux box while
        this called `taskkill`, which does not exist there - it raised
        FileNotFoundError and took the launch down with it. That is why B4/B5/B7
        never ran through the VM path and needed bigmodel_gen, which cannot
        serve per-arm configs and so cannot drive them correctly.
        """
        killed = []
        if os.name == "nt":
            for name in ("llama-server.exe",):
                r = subprocess.run(["taskkill", "/F", "/IM", name],
                                   capture_output=True, text=True)
                if "SUCCESS" in r.stdout:
                    killed.append(name)
        else:
            for name in ("llama-server",):
                try:
                    r = subprocess.run(["pkill", "-f", name],
                                       capture_output=True, text=True)
                except FileNotFoundError:
                    continue
                if r.returncode == 0:
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
        runtime = os.environ.get("LLMTEST_RUNTIME", runtime)
        if runtime == "endpoint":
            return self._attach_live_endpoint(
                model_id, flags_overlay=flags_overlay, parallel=parallel,
                ctx=ctx, kv=kv, timing_authoritative=timing_authoritative,
                key=key)
        if runtime != "fork":
            raise NotImplementedError(f"runtime {runtime} lands in Task 12 (ollama) / later (vllm)")
        model = self.cfg.registry["models"][model_id]
        # Tier gates the VRAM-fit preflight and was pinned to T1 (the 24 GB
        # laptop) back when that was the only box this ran on. On the 96 GB
        # PRO 6000 that pin rejects models which fit with room to spare, so it
        # is overridable by env - the runner on a rented box exports
        # LLMTEST_TIER=T3. Default stays T1 so local behaviour is unchanged.
        tier = os.environ.get("LLMTEST_TIER", "T1")
        if tier not in self.cfg.tiers.get("tiers", self.cfg.tiers):
            raise RuntimeError(f"LLMTEST_TIER={tier!r} is not a known tier")
        fit = fits(model, self.cfg.tiers, kv, tier=tier)
        if not fit.fits:
            raise RuntimeError(f"fits() preflight failed: {fit.detail}")
        if _vram_free_gb() < float(model["weights_gb"]) + 1.0:
            self.sweep_orphans()
            if _vram_free_gb() < float(model["weights_gb"]) + 1.0:
                raise RuntimeError("insufficient VRAM after orphan sweep")
        flags = compose_fork_flags(self.cfg, ctx=ctx, parallel=parallel, kv=kv,
                                   overlay=flags_overlay)
        port = _free_port()
        # The pin in runtime_pins is a Windows path to the local prism build.
        # A rented Linux box has its own binary, so allow an override by env
        # rather than editing a pinned config that provenance depends on - the
        # pin still describes the local box, and the row's runtime field still
        # says "fork" either way.
        binary = os.environ.get("LLMTEST_FORK_BINARY") or \
            self.cfg.runtime_pins["fork"]["binary"]
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
            raw_invocation=invocation,
            hardware_sku=os.environ.get("LLMTEST_HARDWARE_SKU", "rtx5090-laptop"),
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

    def _attach_live_endpoint(self, model_id, *, flags_overlay, parallel, ctx,
                              kv, timing_authoritative, key) -> EndpointHandle:
        """Attach to an already-running OpenAI-compatible server (vLLM/SGLang).

        Required env:
          LLMTEST_ENDPOINT_URL   origin or .../v1  (e.g. http://127.0.0.1:8888)
        Optional:
          LLMTEST_SERVED_MODEL   wire model id (default: registry served_model_name or model_id)
          LLMTEST_HARDWARE_SKU   stamped on the session (default dgx-spark-gb10)
        Does not launch or kill the remote process.
        """
        raw = os.environ.get("LLMTEST_ENDPOINT_URL", "").strip()
        if not raw:
            raise RuntimeError("LLMTEST_RUNTIME=endpoint requires LLMTEST_ENDPOINT_URL")
        origin = raw.rstrip("/")
        if origin.endswith("/v1"):
            origin = origin[: -len("/v1")]
        model = self.cfg.registry["models"][model_id]
        served = (os.environ.get("LLMTEST_SERVED_MODEL")
                  or model.get("served_model_name")
                  or model_id)
        health_ok = False
        last_err = None
        for path in ("/health", "/v1/models"):
            try:
                urllib.request.urlopen(origin + path, timeout=5)
                health_ok = True
                break
            except Exception as e:
                last_err = e
        if not health_ok:
            raise RuntimeError(f"endpoint not healthy at {origin}: {last_err}")
        spec = (flags_overlay or {}).get("spec") or os.environ.get("LLMTEST_SPEC", "off")
        session_id = f"s-{uuid.uuid4().hex[:12]}"
        sku = os.environ.get("LLMTEST_HARDWARE_SKU", "dgx-spark-gb10")
        tier_name = os.environ.get("LLMTEST_TIER", "T_GB10")
        tiers = self.cfg.tiers.get("tiers", self.cfg.tiers)
        usable = float(tiers.get(tier_name, {}).get("usable_gb", 0) or 0)
        ncfg = normalize_config(runtime="endpoint", ctx=ctx, kv=kv,
                                spec=spec, parallel=parallel)
        self.store.append_session(SessionRow(
            schema_version=SCHEMA_VERSION, session_id=session_id,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            runtime="vllm",
            runtime_build=os.environ.get("LLMTEST_RUNTIME_BUILD", "endpoint-attach"),
            normalized_config=ncfg,
            raw_invocation=f"ATTACH {origin} model={served}",
            hardware_sku=sku,
            measured_usable_vram_gb=usable,
            tp_degree=int(os.environ.get("LLMTEST_TP", "2")),
            topology=os.environ.get("LLMTEST_TOPOLOGY", "2x-dgx-spark-gb10"),
            driver_env={},
            power_mode="unknown", ac_state="ac",
            timing_authoritative=timing_authoritative).to_dict())
        self._active = EndpointHandle(
            base_url=origin, session_id=session_id, normalized_config=ncfg,
            pid=0, _mgr=self, model_name=served)
        self._active_key = key
        return self._active

    def teardown(self) -> None:
        had_active = self._active is not None
        if had_active and self._active.pid == 0:
            self._active = None
            self._active_key = None
            return
        if had_active:
            pid = self._active.pid
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"],
                               capture_output=True)
            else:
                # The SECOND taskkill in this file, and the one that actually
                # bit: sweep_orphans was made platform-aware first and this was
                # missed, so B5 died on every single arm with
                # "FileNotFoundError: 'taskkill'" - after the server had already
                # launched and served. The launch path was fixed; the teardown
                # path was not, and B5 tears down between every arm.
                #
                # Popen ran with shell=True, so pid is the shell's: kill its
                # children first, then the shell, or the server outlives it and
                # the next launch finds the VRAM still held.
                subprocess.run(["pkill", "-TERM", "-P", str(pid)],
                               capture_output=True)
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
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
