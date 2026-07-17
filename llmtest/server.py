"""ServerManager (TESTPLAN 7.3): all serving goes through here; provenance auto-attached.
Teardown by PID only. Orphan sweep before launch. fits() preflight. Fork implemented;
ollama sanctioned arm implemented (B5); vllm = remote-attach later phase."""
from __future__ import annotations

import json
import os
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

_SPEC = {"ngram32": ("ngram-mod", {"n_match": 32}), "off": ("none", {})}


def normalize_config(*, runtime, ctx, kv, spec, parallel, flash_attn=True) -> dict:
    spec_type, spec_params = spec, {}
    if spec == "ngram32":
        spec_params = {"n_match": 32}
    return {"ctx": ctx, "kv_dtype": kv, "flash_attn": flash_attn,
            "spec_type": spec_type, "spec_params": spec_params, "parallel": parallel}


def compose_fork_flags(cfg: Config, *, ctx: int, parallel: int, kv: str,
                       overlay: dict | None) -> str:
    overlay = overlay or {}
    spec = overlay.get("spec", "ngram32")
    if spec.startswith("ngram") and spec not in _SPEC:
        n = int(spec.replace("ngram", "") or 0)
        if n < 16:
            raise ValueError("TESTPLAN 2: never run ngram n-match < 16")
        _SPEC[spec] = ("ngram-mod", {"n_match": n})
    spec_type, spec_params = _SPEC[spec]
    parts = ["-ngl 99", "--jinja", "-fa on", f"-c {ctx}", "--cache-ram 0"]
    if spec_type == "ngram-mod":
        parts += ["--spec-type ngram-mod",
                  f"--spec-ngram-mod-n-match {spec_params['n_match']}"]
    else:
        parts += ["--spec-type none"]
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
    try:
        scheme = subprocess.run(["powercfg", "/getactivescheme"],
                                capture_output=True, text=True).stdout
        mode = "performance" if "erformance" in scheme else "balanced"
    except Exception:
        mode = "unknown"
    return mode, "unknown"   # ac_state refinement lands with bench-night profile (TESTPLAN 11.10)


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

    def sweep_orphans(self) -> list[str]:
        killed = []
        for name in ("llama-server.exe",):
            r = subprocess.run(["taskkill", "/F", "/IM", name],
                               capture_output=True, text=True)
            if "SUCCESS" in r.stdout:
                killed.append(name)
        time.sleep(2)
        return killed

    def request_endpoint(self, model_id: str, runtime: str = "fork",
                         flags_overlay: dict | None = None, parallel: int = 1,
                         ctx: int = 8192, kv: str = "q8_0",
                         timing_authoritative: bool = False) -> EndpointHandle:
        key = (model_id, runtime, json.dumps(flags_overlay, sort_keys=True),
               parallel, ctx, kv)
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
        log = Path("artifacts") / f"server-{port}.log"
        log.parent.mkdir(exist_ok=True)
        proc = subprocess.Popen(invocation, stdout=log.open("w"),
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
        spec = (flags_overlay or {}).get("spec", "ngram32")
        session_id = f"s-{uuid.uuid4().hex[:12]}"
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
        if not self._active:
            return
        subprocess.run(["taskkill", "/F", "/PID", str(self._active.pid), "/T"],
                       capture_output=True)
        self._active = None
        self._active_key = None
        deadline = time.time() + 30
        while time.time() < deadline and _vram_free_gb() < 5.0:
            time.sleep(1)
