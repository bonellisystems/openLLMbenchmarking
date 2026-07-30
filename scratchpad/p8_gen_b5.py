"""P8 parallel B5 (serving-speed/timing) generation on 2x RTX PRO 6000 --
TIMING-AUTHORITATIVE variant of p8_gen.py / p8_gen_serving.py.

Reuses p8_gen.py's proven skeleton (Handle/SM/Ctx injection into the CANONICAL
Battery.execute(), launch()/wait_health()/teardown(), one thread per GPU,
shared Store+lock, resumability via existing_row_ids, the log() helper) with
TWO deliberate departures from both p8_gen.py (B1/B2/B3/B6) and
p8_gen_serving.py (B4/B7):

  1. NO ThreadPoolExecutor across a model's WorkItems. B5 measures decode/
     prefill SPEED -- cross-model or cross-item concurrency on the same GPU
     would steal cycles/bandwidth from whichever probe is "in the timing
     window" and quietly corrupt every tps number this battery exists to
     produce. Every group's items run in a plain sequential for-loop
     (run_items() below), one item fully completing (store.append()'d)
     before the next starts. b5_serving.execute() is free to fire its OWN
     concurrent requests inside a single execute() call (the conc-ladder
     cells spin up `conc` threads against the one already-running server --
     see execute()'s `if conc > 1:` branch) -- that concurrency is b5's, not
     this driver's, and is exactly what the conc-ladder cells are measuring.
     Two GPUs running two different models' B5 items at the same time is
     fine and intentional: independent processes, independent VRAM, no
     shared timing window.

  2. Server -c is NOT multiplied by --parallel. p8_gen_serving.py's B7 groups
     use ctx_total = per_slot_ctx * grp_parallel (a deliberate B1-style
     "every concurrent slot gets the full per-request budget" convention).
     B5 does NOT follow that convention -- it follows the REAL ServerManager
     instead (llmtest/server.py::compose_fork_flags(), the authoritative
     production code path b5_serving.execute() is written against):
         parts = kept + [f"-c {ctx}"]                  # ctx used AS-IS
         ...
         if parallel > 1: parts += [f"-np {parallel}"]  # no ctx scaling
     i.e. llama-server splits the ONE -c budget across --parallel slots
     itself. b5_serving.py's own fixture sizing confirms this is intentional:
     the conc-ladder cells reuse the small conc_prompt/conc_max_tokens=400
     fixture (not peak_prompt/1000) at a constant ctx=8192, so even the top
     rung (conc=16 -> 8192/16=512 effective tokens/slot) is *meant* to just
     fit a ~24-token prompt + 400-token completion. This driver mirrors
     compose_fork_flags() exactly: -c <ctx-for-cond> (never scaled), -np
     <conc> appended only when conc>1. See group_b5()/gpu_worker() below.
     THIS IS THE BIGGEST UNVERIFIED-ON-HARDWARE ASSUMPTION IN THIS FILE --
     see the module-end risk note.

Handle/SM/Ctx are carried over unchanged from p8_gen.py's contract (as
already mirrored verbatim in p8_gen_serving.py): Handle.chat() posts to
/v1/chat/completions and returns the FULL parsed JSON response, which is
exactly what b5_serving.py needs -- it reads `d.get("timings", {})` for
predicted_per_second / prompt_per_second / prompt_ms / predicted_n /
prompt_n / draft_n / draft_n_accepted directly off that dict (see
peak_metrics()/concurrency_metrics() in llmtest/batteries/b5_serving.py).
Handle.chat() already sends "cache_prompt": false on every call (a
deliberate deviation from the production EndpointHandle.chat(), which sends
no such key) -- load-bearing for B5 in particular, since a cache hit on a
repeat/identical prompt would silently deflate pp_tps without erroring.
Ctx carries .cfg/.server_manager()/.root/.store, a strict superset of what
b5_serving.execute() actually touches (only ctx.cfg and
ctx.server_manager() -- it never reads ctx.store or ctx.root; .store/.root
are kept anyway for parity with the p8_gen.py contract and because they cost
nothing to carry).

  python3 p8_gen_b5.py --gpu0 m1,m2 --gpu1 m3,m4 --limit 0
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("LLMTEST_ROOT", "/root/llmtest-v2"))
OUT = Path(os.environ.get("LLMTEST_OUT", "/root/out"))
# See p8_gen_serving.py: BIN and LIBS must BOTH be overridable. B5 is
# timing-authoritative, so it may only run on a binary that actually supports
# --spec-type ngram-mod; setup.sh probes for that and records the answer in /root/caps.
BIN = os.environ.get("LLMTEST_BIN", "/root/prism-llama/build/bin/llama-server")
LIBS = os.environ.get("LLMTEST_LIBS", str(Path(BIN).parent))
sys.path.insert(0, str(ROOT))
from llmtest.batteries import get as get_battery        # noqa: E402
from llmtest.registry import load_config                # noqa: E402
from llmtest.store import Store                          # noqa: E402

_print_lock = threading.Lock()
def log(m):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# Handle/SM/Ctx injection -- identical contract to p8_gen.py's classes; the
# battery's execute() calls ctx.server_manager().request_endpoint(...) and
# gets back a Handle whose .chat()/.session_id it uses, never knowing the
# endpoint's ctx/kv/spec/parallel were actually pinned by the driver's group
# loop rather than freshly composed per-call by the real ServerManager.
# --------------------------------------------------------------------------

class Handle:
    def __init__(self, port, session_id):
        self.port = port; self.session_id = session_id

    def chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        import json
        body = {"messages": messages, "stream": False, "cache_prompt": False}
        if max_tokens is not None: body["max_tokens"] = max_tokens
        if temperature is not None: body["temperature"] = temperature
        for k in ("tools", "tool_choice", "top_p", "seed", "response_format", "stop", "system"):
            if kwargs.get(k) is not None:
                body[k] = kwargs[k]
        data = json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                     data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2400) as r:
            return json.loads(r.read().decode())


class SM:
    def __init__(self, h): self._h = h

    def request_endpoint(self, model_id, ctx=None, kv=None, flags_overlay=None,
                         timing_authoritative=False, parallel=None, runtime=None,
                         **_ignored):
        # b5_serving.execute() calls this once per item with the exact
        # ctx/kv/parallel/flags_overlay/timing_authoritative=True the driver's
        # group loop already launched a matching server for -- accepted here
        # (matching the call signature) but not re-applied; no relaunch.
        return self._h


class Ctx:
    def __init__(self, cfg, h, store):
        self.cfg = cfg; self._sm = SM(h); self.root = OUT; self.server = None
        self.store = store   # b5 never reads this; kept for p8_gen.py contract parity

    def server_manager(self): return self._sm


# --------------------------------------------------------------------------
# Server launch (unchanged shape from p8_gen.py/p8_gen_serving.py -- ctx is
# passed through AS-IS, never scaled here; the caller decides the value)
# --------------------------------------------------------------------------

BASE_FLAGS = ["-ngl", "99", "--jinja", "-fa", "on", "--cache-ram", "0", "--no-webui"]


def launch(model_path, gpu, port, ctx_total, extra_flags, parallel=1):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["LD_LIBRARY_PATH"] = LIBS
    cmd = [BIN, "-m", model_path, "-c", str(ctx_total)] + BASE_FLAGS + list(extra_flags)
    # Pass --parallel explicitly, including 1 (llama.cpp alias of -np; matches
    # server.compose_fork_flags()'s -np), so slot count never depends on a build default.
    # It matters most here: B5 is timing-authoritative and the slot count IS the
    # concurrency its throughput figure describes, so it must be stated, not inherited.
    cmd += ["--parallel", str(max(1, parallel))]
    cmd += ["--host", "127.0.0.1", "--port", str(port)]
    lf = open(OUT / f"server.gpu{gpu}.log", "a")
    return subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)


def wait_health(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200: return True
        except Exception:
            pass
        time.sleep(3)
    return False


def teardown(proc):
    proc.terminate()
    try: proc.wait(timeout=30)
    except Exception: proc.kill()
    time.sleep(3)


# --------------------------------------------------------------------------
# B5 grouping -- one server launch per (spec, cond, conc) cell. Under the
# default suite (b5_extra_runtimes unset/False in config/suite.yaml) this is
# 8 conditions/model = 8 groups of exactly 1 item each: {ngram32,off} x
# {PEAK,SUSTAINED32K} (conc=1) + ngram32 x PEAK x {2,4,8,16}. force=True runs
# (multiple run_n on the same condition) would put >1 item in a group; those
# still run sequentially inside that one server, never spanning a relaunch.
# --------------------------------------------------------------------------

_NGRAM_RE = re.compile(r"^ngram(\d+)$")
_COND_CTX = {"PEAK": 8192, "SUSTAINED32K": 36864}   # literal mirror of the
    # ctx_len = 36864 if parts["cond"] == "SUSTAINED32K" else 8192 line inside
    # b5_serving.execute() -- duplicated (not imported) because it's inline
    # logic, not an importable constant; keep in sync if b5_serving.py changes.
_KV_FLAGS = ["-ctk", "q8_0", "-ctv", "q8_0"]        # b5_serving.execute() hardcodes
    # kv="q8_0" on every request_endpoint() call -- never a group dimension.


def spec_flags(spec_token):
    """Mirrors server.compose_fork_flags()'s _resolve_spec(): 'off' -> spec
    disabled; 'ngramN' (N>=16, TESTPLAN 2 guard) -> ngram-mod at n-match=N."""
    if spec_token == "off":
        return ["--spec-type", "none"]
    m = _NGRAM_RE.match(spec_token)
    if not m:
        raise ValueError(f"B5: unrecognized spec token {spec_token!r} "
                         f"(expected 'off' or 'ngram<N>')")
    n = int(m.group(1))
    if n < 16:
        raise ValueError(f"B5: refusing ngram n-match={n} < 16 (TESTPLAN 2 guard)")
    return ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", str(n)]


def b5_condition_parts(item):
    try:
        return dict(p.split("=", 1) for p in item.condition.split(";"))
    except ValueError as e:
        raise ValueError(f"B5 item row_id={item.row_id} task_id={item.task_id}: "
                         f"cannot parse condition {item.condition!r}: {e}") from e


def group_b5(items):
    """key = (spec, cond, conc), parsed exactly the way b5_serving.execute()
    itself parses item.condition. Only runtime=fork items are supported by
    this driver (it launches the /root/prism-llama fork binary directly) --
    ollama/vllm b5 arms (b5_extra_runtimes=True) would need a different
    driver and raise loudly here rather than silently mis-launching."""
    groups = {}
    for it in items:
        parts = b5_condition_parts(it)
        runtime = parts.get("runtime", "fork")
        if runtime != "fork":
            raise NotImplementedError(
                f"B5 item row_id={it.row_id}: runtime={runtime!r} not supported by "
                f"this GPU-box driver (fork-only); b5_extra_runtimes arms need a "
                f"separate ollama/vllm driver")
        spec_token = parts.get("spec", "ngram32")
        try:
            cond_token = parts["cond"]
        except KeyError as e:
            raise KeyError(f"B5 item row_id={it.row_id}: condition {it.condition!r} "
                           f"has no 'cond' key") from e
        conc = int(parts.get("conc", 1))
        groups.setdefault((spec_token, cond_token, conc), []).append(it)
    return groups


# --------------------------------------------------------------------------
# Sequential item runner -- NO ThreadPoolExecutor across items (see module
# docstring point 1). Each item fully completes and is store.append()'d
# before the next one starts, so no two probes ever share the GPU's timing
# window on this driver's side.
# --------------------------------------------------------------------------

def run_items(gpu, tag, items, battery, cx, store, store_lock):
    if not items:
        return 0, 0
    n_ok = n_err = 0
    t0 = time.time()
    for it in items:
        try:
            rows = battery.execute(it, cx)
            with store_lock:
                for row in rows: store.append(row)
            n_ok += 1
        except Exception as e:
            n_err += 1
            log(f"GPU{gpu} {tag} ERR {it.task_id} r{it.run_n} cond={it.condition}: {e}")
    log(f"GPU{gpu} {tag}: DONE {n_ok} ok, {n_err} err in {time.time()-t0:.0f}s "
        f"({len(items)} item(s), sequential)")
    return n_ok, n_err


# --------------------------------------------------------------------------
# Per-GPU worker -- one model at a time; within a model, one server launch
# per serving-config group, groups run sequentially, teardown between each.
# --------------------------------------------------------------------------

# Per-model fatals recorded here so the process can exit non-zero. Threads append; the
# GIL makes list.append atomic enough for this.
FAILURES: list[str] = []


def gpu_worker(gpu, port, models, cfg, store, store_lock, limit, mpaths):
    battery = get_battery(5)
    done_ids = store.existing_row_ids()

    for model in models:
        mp = mpaths.get(model)
        if not mp or not os.path.exists(mp):
            log(f"GPU{gpu} SKIP {model}: file missing {mp}"); continue

        try:
            items = [i for i in battery.plan(cfg, store, model_filter=model, force=False)
                     if i.row_id not in done_ids]
            if limit:
                items = items[:limit]           # debug/smoke-test trunc, pre-grouping
                                                 # (may produce partial groups)
            if not items:
                log(f"GPU{gpu} {model}: 0 pending"); continue

            groups = group_b5(items)
            # Readability-only ordering (b5 rows have no cross-row dependency,
            # unlike B7's baseline lookups): PEAK before SUSTAINED32K, conc
            # ascending within PEAK, ngram32 before off.
            group_order = sorted(groups.keys(),
                                 key=lambda k: (k[1] != "PEAK", k[2], k[0] != "ngram32"))

            log(f"GPU{gpu} {model}: {len(items)} pending items across "
                f"{len(groups)} serving-config group(s): {group_order}")

            for gkey in group_order:
                spec_token, cond_token, conc = gkey
                group_items = groups[gkey]
                ctx_total = _COND_CTX[cond_token]   # NOT multiplied by conc -- see
                                                     # module docstring point 2.
                extra = spec_flags(spec_token) + _KV_FLAGS
                label = f"spec={spec_token} cond={cond_token} conc={conc}"

                log(f"GPU{gpu} {model} [{label}]: launching "
                    f"(ctx={ctx_total}, parallel={conc}, {len(group_items)} item(s))")
                p = launch(mp, gpu, port, ctx_total, extra, parallel=conc)
                try:
                    if not wait_health(port):
                        log(f"GPU{gpu} {model} [{label}]: HEALTH TIMEOUT -- skipping group")
                        continue
                    log(f"GPU{gpu} {model} [{label}]: healthy, running")
                    safe_label = re.sub(r"[^a-zA-Z0-9]+", "-", label)
                    handle = Handle(port, f"pB5-gpu{gpu}-{model}-{safe_label}")
                    cx = Ctx(cfg, handle, store)
                    run_items(gpu, f"{model} [{label}]", group_items, battery, cx,
                             store, store_lock)
                finally:
                    teardown(p)

            log(f"GPU{gpu} {model}: ALL GROUPS DONE")
        except Exception:
            log(f"GPU{gpu} {model}: FATAL error while planning/grouping/running -- "
               f"skipping remainder of this model, continuing roster")
            log(traceback.format_exc())
            # Continuing the roster is right, but the PROCESS must not still exit 0.
            # It did, and the runner recorded "B5 ok" for a step that wrote no rows at
            # all - the exact silent-success the per-battery exit codes exist to catch.
            FAILURES.append(f"{model}: {traceback.format_exc().splitlines()[-1]}")
            continue

    log(f"GPU{gpu}: ALL MODELS DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu0", required=True)
    ap.add_argument("--gpu1", required=True)
    ap.add_argument("--parallel", type=int, default=0,
                    help="Accepted for CLI parity with p8_gen.py/p8_gen_serving.py; "
                         "IGNORED. B5's concurrency ladder (conc=1/2/4/8/16) is fully "
                         "determined per-item by b5_serving.plan()'s condition set, "
                         "not by this flag -- items must run one-at-a-time per model "
                         "so each probe owns the GPU cleanly (see module docstring).")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug/smoke-test cap on a model's pending-item list, "
                         "applied BEFORE grouping (may produce partial groups).")
    args = ap.parse_args()

    if args.parallel:
        log(f"NOTE: --parallel={args.parallel} given but IGNORED by this B5 driver")

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT)
    store = Store(OUT)
    lock = threading.Lock()

    models_cfg = cfg.registry["models"]
    mpaths = {mid: m["local_path"] for mid, m in models_cfg.items()}

    g0 = [m for m in args.gpu0.split(",") if m]
    g1 = [m for m in args.gpu1.split(",") if m]
    log(f"Battery B5 | GPU0 models: {g0} | GPU1 models: {g1} | limit={args.limit or 'none'}")

    t0 = threading.Thread(target=gpu_worker,
                          args=(0, 8080, g0, cfg, store, lock, args.limit, mpaths))
    t1 = threading.Thread(target=gpu_worker,
                          args=(1, 8081, g1, cfg, store, lock, args.limit, mpaths))
    t0.start(); t1.start(); t0.join(); t1.join()
    log("ALL GPUS DONE")
    if FAILURES:
        log(f"EXITING NON-ZERO: {len(FAILURES)} model(s) failed: {FAILURES}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
