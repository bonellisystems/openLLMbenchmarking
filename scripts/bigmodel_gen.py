"""Single-model, single-GPU, multi-battery generation driver for a BIG model that
does NOT fit the framework's T1 tier (so the real ServerManager's fits(tier="T1")
preflight would reject it) and runs on LINUX (so ServerManager's taskkill teardown
is inapplicable). This is the same Handle/SM/Ctx injection the Verda p8_gen used for
the 4 big models — one server for the model, feed a fake ServerManager into the
CANONICAL Battery.execute() so every battery runs unmodified. Peer-correct battery
set for a 100B+ model is B1/B2/B3/B6 (B4 auto-empties via arm-selection, B5 is
box-dependent, B7 needs the fork's spec arm) — matches what the 235B ran.

Two serving modes:
  --serve --gguf PATH [--gpu N] [--bin PATH] [--ctx N]  -> launch llama-server here
  --endpoint-url http://127.0.0.1:8080                  -> use an already-running server
                                                          (used for the free LOCAL
                                                           mechanism-validation run)

  # local validate (server already up on 8080 with a small model):
  python bigmodel_gen.py --model gpt-oss-20b --batteries 3 --limit 2 \
      --endpoint-url http://127.0.0.1:8080 --results-dir results_bigval

  # on the box (A100 80GB, Laguna fits fully in VRAM):
  python bigmodel_gen.py --model laguna-s-2.1 --batteries 1,2,3,6 --serve \
      --gguf /root/models/laguna.gguf --bin /root/llama/llama-server \
      --ctx 32768 --results-dir /root/out
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_print_lock = threading.Lock()
def log(m):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Handle:
    """Mimics server.EndpointHandle: .chat() posts to the running server; the
    battery reads .session_id for provenance. session_id is fixed per driver run
    (the driver, not ServerManager, owns the one server's lifecycle)."""
    def __init__(self, base_url, session_id):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    def chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        body = {"messages": messages, "stream": False, "cache_prompt": False}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        for k in ("tools", "tool_choice", "top_p", "seed", "response_format", "stop", "system"):
            if kwargs.get(k) is not None:
                body[k] = kwargs[k]
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                     data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2400) as r:
            return json.loads(r.read().decode())


class SM:
    """Fake ServerManager: the driver already launched the one correct server, so
    request_endpoint() accepts the battery's ctx/kv/flags args (matching every
    battery's call signature) and returns the pre-launched Handle without
    re-composing flags or running the T1 fits()/taskkill path the real one would."""
    def __init__(self, h):
        self._h = h

    def request_endpoint(self, model_id=None, ctx=None, kv=None, flags_overlay=None,
                         timing_authoritative=False, parallel=None, runtime=None,
                         **_ignored):
        return self._h


class Ctx:
    """Battery execute() reads only .cfg, .root, .server_manager() (B1/B2/B3/B6);
    .store is carried for B7's baseline cross-reference (harmless for the others)."""
    def __init__(self, cfg, h, store, out_root):
        self.cfg = cfg
        self._sm = SM(h)
        self.root = out_root
        self.store = store
        self.server = None

    def server_manager(self):
        return self._sm


BASE_FLAGS = ["-ngl", "99", "--jinja", "-fa", "on", "--cache-ram", "0", "--no-webui",
              "--ctx-checkpoints", "0"]


def launch(binary, model_path, gpu, port, ctx_total, out_dir, extra_flags=(),
           parallel=1, kv="q8_0", libs=None):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if libs:
        env["LD_LIBRARY_PATH"] = libs + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [binary, "-m", model_path, "-c", str(ctx_total)] + BASE_FLAGS + list(extra_flags)
    if kv and kv != "f16":
        cmd += ["-ctk", kv, "-ctv", kv]
    if parallel > 1:
        cmd += ["--parallel", str(parallel)]
    cmd += ["--host", "127.0.0.1", "--port", str(port)]
    log("launch: " + " ".join(cmd))
    lf = open(Path(out_dir) / f"server.port{port}.log", "a")
    return subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)


def wait_health(port, timeout=1200):
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_battery(battery, cfg, store, store_lock, ctx, model, limit, parallel, force):
    done_ids = store.existing_row_ids()
    items = [i for i in battery.plan(cfg, store, model_filter=model, force=force)
             if force or i.row_id not in done_ids]
    if limit:
        items = items[:limit]
    if not items:
        log(f"B{battery.id} {model}: 0 pending")
        return 0, 0
    n_ok = [0]; n_err = [0]

    def run_item(it):
        try:
            rows = battery.execute(it, ctx)
            with store_lock:
                for row in rows:
                    store.append(row)
            n_ok[0] += 1
            if n_ok[0] % 10 == 0:
                log(f"B{battery.id} {model}: {n_ok[0]}/{len(items)} done")
        except Exception as e:
            n_err[0] += 1
            log(f"B{battery.id} {model} ERR {it.task_id} r{getattr(it,'run_n','?')} "
                f"cond={getattr(it,'condition','?')}: {e}")

    t0 = time.time()
    workers = max(1, min(parallel, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_item, items))
    log(f"B{battery.id} {model}: DONE {n_ok[0]} ok, {n_err[0]} err in "
        f"{time.time()-t0:.0f}s ({len(items)} items, workers={workers})")
    return n_ok[0], n_err[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="llmtest-v2 repo root (default: cwd)")
    ap.add_argument("--model", required=True, help="registry model id")
    ap.add_argument("--batteries", default="1,2,3,6")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    # serving
    ap.add_argument("--endpoint-url", default=None, help="use an already-running server")
    ap.add_argument("--serve", action="store_true", help="launch llama-server here")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--bin", dest="binary", default=None)
    ap.add_argument("--libs", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--kv", default="q8_0")
    args = ap.parse_args()

    repo = Path(args.repo or os.getcwd()).resolve()
    sys.path.insert(0, str(repo))
    from llmtest.registry import load_config
    from llmtest.store import Store
    from llmtest.batteries import get as get_battery

    cfg = load_config(repo)
    if args.model not in cfg.registry["models"]:
        log(f"FATAL: {args.model} not in registry"); return 2
    out = Path(args.results_dir); out.mkdir(parents=True, exist_ok=True)
    store = Store(out)
    lock = threading.Lock()
    batteries = [int(x) for x in args.batteries.split(",") if x.strip()]

    proc = None
    if args.serve:
        if not args.gguf or not args.binary:
            log("FATAL: --serve needs --gguf and --bin"); return 2
        if not os.path.exists(args.gguf):
            log(f"FATAL: gguf missing {args.gguf}"); return 2
        proc = launch(args.binary, args.gguf, args.gpu, args.port, args.ctx, out,
                      parallel=args.parallel, kv=args.kv, libs=args.libs)
        if not wait_health(args.port):
            log("FATAL: server health timeout"); proc.terminate(); return 1
        base_url = f"http://127.0.0.1:{args.port}"
        log(f"server healthy at {base_url}")
    elif args.endpoint_url:
        base_url = args.endpoint_url
    else:
        log("FATAL: need --serve or --endpoint-url"); return 2

    handle = Handle(base_url, f"bmg-{args.model}")
    ctx = Ctx(cfg, handle, store, out)

    try:
        summary = {}
        for bid in batteries:
            battery = get_battery(bid)
            ok, err = run_battery(battery, cfg, store, lock, ctx, args.model,
                                  args.limit, args.parallel, args.force)
            summary[bid] = (ok, err)
        log("SUMMARY: " + " ".join(f"B{b}={ok}ok/{err}err" for b, (ok, err) in summary.items()))
    finally:
        if proc is not None:
            proc.terminate()
            try: proc.wait(timeout=30)
            except Exception: proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
