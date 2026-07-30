"""P8 parallel B4/B7 generation on 2x RTX PRO 6000 -- PER-SERVING-CONFIG variant of
p8_gen.py. Mirrors p8_gen.py's architecture exactly (Handle/SM/Ctx injection into
the CANONICAL Battery.execute(), shared Store+lock, ThreadPoolExecutor concurrency,
resumability via existing_row_ids, the log() helper) with ONE structural change:
p8_gen.py launches ONE server per MODEL with fixed flags; B4 and B7 need DIFFERENT
server flags for different subsets of a model's WorkItems, so this script groups
each model's pending items by a derived "serving-config key" and launches one
server per (model, group), running that group's items, then tearing down and
moving to the next group.

  B4 (long-context KV/ctx sweep): group key = (kv_short, ctx_tokens), taken
  directly from WorkItem.payload -- b4_longcontext.plan() already stashes the raw
  values there (payload["kv_short"], payload["ctx_tokens"]), so no condition-string
  parsing is needed. Server flags per group: -c <ctx_tokens> (exact, NOT multiplied
  by --parallel -- see VRAM note below) + -ctk/-ctv <dtype> when kv != f16 (KV_DTYPE
  imported directly from b4_longcontext.py so the short-label -> llama.cpp-dtype
  mapping can never drift out of sync with the battery). spec is fixed ngram32 for
  every B4 item (b4_longcontext.plan() hardcodes condition "spec": "ngram32"), so
  it's part of the constant base flags, not a group dimension.

  B7 (harness/config sensitivity matrix): group key = the "spec" field parsed out
  of WorkItem.condition (B7's payload does NOT carry the matrix cell's dims --
  only the condition string does; this driver parses it exactly the way
  b7_harnessmatrix.execute() itself does: dict(p.split("=") for p in
  item.condition.split(";"))). Of the 4 matrix dimensions (sysp, temp, toolfmt,
  spec), only "spec" (n-gram on/off) changes a server launch flag
  (--spec-type ngram-mod vs --spec-type none) -- sysp (system-prompt text),
  temp (the `temperature` kwarg), and toolfmt (native `tools=` vs a prompted
  textual convention) are all applied PER-REQUEST inside execute()/Handle.chat()
  against a single already-running server, so they are NOT grouping dimensions.
  kv and ctx are constant across every B7 cell (q8_0 / cfg.suite["b7"]["ctx"],
  usually 8192) -- also not grouping dimensions.

  B7 baseline ordering: b7_harnessmatrix.execute() looks up the baseline cell's
  ALREADY-STORED row (ctx.store.iter_rows()) to compute signal_agreement_vs_baseline
  / byte_identical_vs_baseline for every non-baseline cell. To guarantee that row
  exists before it's needed: (1) groups are run baseline-spec-value first, and (2)
  WITHIN the baseline-spec group, baseline-condition items run in their own
  ThreadPoolExecutor pass and are awaited BEFORE the group's remaining
  (sysp/temp/toolfmt-variant) items start their pass. The non-baseline "spec=off"
  group then finds its baseline row already committed by the earlier group. Ctx
  now carries `.store` (p8_gen.py's Ctx did not need this -- B1/B2/B3/B6 never
  read ctx.store) so _find_row() inside b7_harnessmatrix.execute() has something
  to read.

  python3 p8_gen_serving.py --battery 4 --gpu0 m1,m2 --gpu1 m3,m4 --parallel 4
  python3 p8_gen_serving.py --battery 7 --gpu0 m1,m2 --gpu1 m3,m4 --parallel 4
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("LLMTEST_ROOT", "/root/llmtest-v2"))
OUT = Path(os.environ.get("LLMTEST_OUT", "/root/out"))
# The P8 run built the prism fork at /root/prism-llama; the gap-closing run uses the
# official ggml container, where the binary and its shared libraries both live in /app.
# LIBS matters as much as BIN: a wrong LD_LIBRARY_PATH makes the server die on a missing
# .so, which the runner logs as "serve-fail" - indistinguishable from a model that will
# not load. Both are env-overridable so neither has to be edited per host.
BIN = os.environ.get("LLMTEST_BIN", "/root/prism-llama/build/bin/llama-server")
LIBS = os.environ.get("LLMTEST_LIBS", str(Path(BIN).parent))
sys.path.insert(0, str(ROOT))
from llmtest import schema                              # noqa: E402
from llmtest.batteries import get as get_battery        # noqa: E402
from llmtest.batteries.b4_longcontext import KV_DTYPE    # noqa: E402  ('f16'/'q8'/'q4' -> llama.cpp -ctk/-ctv dtype)
from llmtest.registry import load_config                # noqa: E402
from llmtest.store import Store                          # noqa: E402

_print_lock = threading.Lock()
def log(m):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------
# Handle/SM/Ctx injection -- identical contract to p8_gen.py's classes. The
# battery's execute() calls ctx.server_manager().request_endpoint(...) and
# gets back a Handle whose .chat()/.session_id it uses; it never knows the
# endpoint's ctx/kv/flags were actually pinned by the driver's group loop
# rather than freshly composed per-call the way the real ServerManager does.
# --------------------------------------------------------------------------

class Handle:
    def __init__(self, port, session_id):
        self.port = port; self.session_id = session_id

    def chat(self, messages, max_tokens=None, temperature=None, **kwargs):
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
        # The real ServerManager.request_endpoint() composes flags from these
        # args and (re)launches on mismatch (llmtest/server.py compose_fork_flags).
        # Here the driver's group loop has ALREADY launched the one server that's
        # correct for every item in the current group, so these args are accepted
        # (matching both b4_longcontext.execute()'s and b7_harnessmatrix.execute()'s
        # call signatures, including B7's flags_overlay kwarg) but not re-applied.
        return self._h


class Ctx:
    def __init__(self, cfg, h, store):
        self.cfg = cfg; self._sm = SM(h); self.root = OUT; self.server = None
        # b7_harnessmatrix.execute() reads ctx.store.iter_rows() (via _find_row)
        # to pull the baseline cell's already-committed row for signal_agreement
        # / byte_identical comparisons -- p8_gen.py's Ctx never needed this since
        # B1/B2/B3/B6 don't cross-reference other rows mid-battery.
        self.store = store

    def server_manager(self): return self._sm


# --------------------------------------------------------------------------
# Server launch
# --------------------------------------------------------------------------

BASE_FLAGS = ["-ngl", "99", "--jinja", "-fa", "on", "--cache-ram", "0", "--no-webui"]


def launch(model_path, gpu, port, ctx_total, extra_flags, parallel=1):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["LD_LIBRARY_PATH"] = LIBS
    cmd = [BIN, "-m", model_path, "-c", str(ctx_total)] + BASE_FLAGS + list(extra_flags)
    # Pass --parallel explicitly, including 1, so slot count never depends on a build
    # default that can change under us. Measured on build 10156: omitting it gives
    # n_slots=4, but with kv_unified='true' each slot still gets the FULL -c, so B4's
    # arms were served at their labelled contexts either way (n_ctx_slot was
    # 16384/65536/131072/262144 for the four arms). This is determinism, not a bugfix -
    # B7 is the case that actually depends on the count, since it sizes
    # ctx_total = per_slot_ctx * grp_parallel and runs with kv_unified='false'.
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
# B4 -- server flags per (kv_short, ctx_tokens) group. spec is constant
# (ngram32 on every B4 item), so it lives in the always-on base, not here.
# --------------------------------------------------------------------------

_B4_SPEC_FLAGS = ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "32"]


def b4_flags(kv_short):
    try:
        dtype = KV_DTYPE[kv_short]
    except KeyError as e:
        raise KeyError(f"B4: unknown kv short-label {kv_short!r} (KV_DTYPE={KV_DTYPE})") from e
    flags = list(_B4_SPEC_FLAGS)
    if dtype != "f16":
        flags += ["-ctk", dtype, "-ctv", dtype]
    return flags


def group_b4(items):
    """key = (kv_short, ctx_tokens) straight from payload -- these are the exact
    values b4_longcontext.execute() feeds to request_endpoint(ctx=, kv=), so
    grouping on them guarantees every item in a group wants the identical server."""
    groups = {}
    for it in items:
        try:
            key = (it.payload["kv_short"], it.payload["ctx_tokens"])
        except KeyError as e:
            raise KeyError(f"B4 item row_id={it.row_id} task_id={it.task_id}: "
                           f"payload missing expected key {e}") from e
        groups.setdefault(key, []).append(it)
    return groups


# --------------------------------------------------------------------------
# B7 -- server flags per "spec" value (the only matrix dim requiring a
# relaunch). sysp/temp/toolfmt are request-level and ride through unchanged
# on whichever server is currently up.
# --------------------------------------------------------------------------

_NGRAM_RE = re.compile(r"^ngram(\d+)$")


def b7_flags(spec_token):
    if spec_token == "off":
        return ["--spec-type", "none"]
    m = _NGRAM_RE.match(spec_token)
    if not m:
        raise ValueError(f"B7: unrecognized spec token {spec_token!r} "
                         f"(expected 'off' or 'ngram<N>')")
    return ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", m.group(1)]


def b7_condition_parts(item):
    try:
        return dict(p.split("=", 1) for p in item.condition.split(";"))
    except ValueError as e:
        raise ValueError(f"B7 item row_id={item.row_id} task_id={item.task_id}: "
                         f"cannot parse condition {item.condition!r}: {e}") from e


def group_b7(items):
    """key = parts['spec'] parsed from item.condition (payload has no per-cell
    dims for B7 -- only the condition string does). Mirrors the exact reparse
    b7_harnessmatrix.execute() itself does on item.condition."""
    groups = {}
    for it in items:
        parts = b7_condition_parts(it)
        if "spec" not in parts:
            raise KeyError(f"B7 item row_id={it.row_id}: condition {it.condition!r} "
                           f"has no 'spec' key -- condition_order/matrix dims changed?")
        groups.setdefault(parts["spec"], []).append(it)
    return groups


def b7_baseline_condition(cfg):
    """Reconstructs the exact string b7_harnessmatrix._baseline_condition() builds,
    so the phase-split below can tell baseline items apart from variant items in
    the same group. Duplicated (not imported) because it's a leading-underscore
    module-private helper; kept as a direct literal mirror of _condition_for()'s
    hardcoded base dict + suite.yaml's matrix.dimensions.*.baseline values so any
    drift between this file and b7_harnessmatrix.py is easy to spot on review."""
    order = cfg.suite["condition_order"]
    dims_cfg = cfg.suite["b7"]["matrix"]["dimensions"]
    baseline_dims = {k: v["baseline"] for k, v in dims_cfg.items()}
    parts = {"runtime": "fork", "kv": "q8", "ctx": "8k", "cond": "B7", **baseline_dims}
    return schema.canonical_condition(parts, order)


# --------------------------------------------------------------------------
# Shared item-running helper (both batteries funnel through this)
# --------------------------------------------------------------------------

def run_items(gpu, tag, items, battery, cx, store, store_lock, parallel):
    if not items:
        return 0, 0
    n_ok = [0]; n_err = [0]

    def run_item(it):
        try:
            rows = battery.execute(it, cx)
            with store_lock:
                for row in rows: store.append(row)
            n_ok[0] += 1
            if n_ok[0] % 10 == 0:
                log(f"GPU{gpu} {tag}: {n_ok[0]}/{len(items)} done")
        except Exception as e:
            n_err[0] += 1
            log(f"GPU{gpu} {tag} ERR {it.task_id} r{it.run_n} cond={it.condition}: {e}")

    t0 = time.time()
    workers = max(1, min(parallel, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_item, items))
    log(f"GPU{gpu} {tag}: DONE {n_ok[0]} ok, {n_err[0]} err in {time.time()-t0:.0f}s "
        f"({len(items)} items, workers={workers})")
    return n_ok[0], n_err[0]


# --------------------------------------------------------------------------
# Per-GPU worker
# --------------------------------------------------------------------------

# Per-model fatals recorded here so the process can exit non-zero.
FAILURES: list[str] = []


def gpu_worker(gpu, port, models, cfg, store, store_lock, parallel, limit, mpaths, battery_id):
    battery = get_battery(battery_id)
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
            if not items:
                log(f"GPU{gpu} {model}: 0 pending"); continue

            if battery_id == 4:
                groups = group_b4(items)
                m = cfg.registry["models"][model]
                claimed_ctx = m.get("claimed_ctx")
                # ctx tiers above claimed_ctx should already be pruned by
                # b4_longcontext.model_arms()/tiers_for_model() at plan() time --
                # this is a belt-and-suspenders guard per the ticket, not the
                # primary enforcement point.
                group_order = sorted(groups.keys(), key=lambda k: (k[1], k[0]))
            else:
                groups = group_b7(items)
                baseline_spec = cfg.suite["b7"]["matrix"]["dimensions"]["spec"]["baseline"]
                baseline_condition = b7_baseline_condition(cfg)
                group_order = sorted(groups.keys(), key=lambda k: (k != baseline_spec, k))

            log(f"GPU{gpu} {model}: {len(items)} pending items across "
                f"{len(groups)} serving-config group(s): {group_order}")

            for gkey in group_order:
                group_items = groups[gkey]

                if battery_id == 4:
                    kv_short, ctx_tokens = gkey
                    if claimed_ctx and ctx_tokens > claimed_ctx:
                        log(f"GPU{gpu} {model} kv={kv_short} ctx={ctx_tokens}: "
                            f"SKIP -- exceeds claimed_ctx={claimed_ctx} "
                            f"(should have been pruned in plan(); {len(group_items)} items dropped)")
                        continue
                    extra = b4_flags(kv_short)
                    ctx_total = ctx_tokens             # NOT multiplied by --parallel: ctx_tokens
                    # is the actual per-request document budget (16k..256k), sized by
                    # b4_longcontext.arm_fits_estimate() to just fit ONE slot on T1.
                    # -c splits across --parallel slots in llama.cpp, so multiplying
                    # would either shrink each slot's usable ctx below what the sweep
                    # is testing (silently corrupting the arm) or -- if instead scaled
                    # up to give every slot the full budget, p8_gen.py's B1 convention
                    # -- blow VRAM at the 128k/256k tiers (fit was only ever validated
                    # at a single slot). Groups run at effective parallel=1; --parallel
                    # is accepted on the CLI but does not apply to B4 server concurrency.
                    grp_parallel = 1
                    label = f"kv={kv_short} ctx={ctx_tokens}"
                else:
                    spec_token = gkey
                    extra = b7_flags(spec_token)
                    per_slot_ctx = cfg.suite["b7"].get("ctx", 8192)
                    grp_parallel = max(1, parallel)
                    ctx_total = per_slot_ctx * grp_parallel   # mirrors p8_gen.py's B1
                    # ctx_per_slot convention: B7's ctx is small/fixed (8192), so
                    # paying parallel x that cost for real concurrency is cheap and
                    # matches the proven pattern.
                    label = f"spec={spec_token}"

                log(f"GPU{gpu} {model} [{label}]: launching "
                    f"({len(group_items)} items, parallel={grp_parallel}, ctx={ctx_total})")
                p = launch(mp, gpu, port, ctx_total, extra, parallel=grp_parallel)
                try:
                    if not wait_health(port):
                        log(f"GPU{gpu} {model} [{label}]: HEALTH TIMEOUT -- skipping group")
                        continue
                    log(f"GPU{gpu} {model} [{label}]: healthy, generating")
                    safe_label = re.sub(r"[^a-zA-Z0-9]+", "-", label)
                    handle = Handle(port, f"pS-gpu{gpu}-{model}-b{battery_id}-{safe_label}")
                    cx = Ctx(cfg, handle, store)

                    if battery_id == 7:
                        # Baseline-condition items in THIS group must complete (and be
                        # appended to store) before this group's remaining variant
                        # items run, so b7_harnessmatrix.execute()'s _find_row()
                        # lookup for signal_agreement_vs_baseline succeeds instead of
                        # silently no-op'ing (it degrades gracefully to "skip the
                        # comparison" if the baseline row isn't there yet -- this
                        # phase split exists to avoid that, not to avoid a crash).
                        phase1 = [it for it in group_items if it.condition == baseline_condition]
                        phase2 = [it for it in group_items if it.condition != baseline_condition]
                        run_items(gpu, f"{model} [{label}] baseline-phase", phase1,
                                 battery, cx, store, store_lock, grp_parallel)
                        run_items(gpu, f"{model} [{label}] variant-phase", phase2,
                                 battery, cx, store, store_lock, grp_parallel)
                    else:
                        run_items(gpu, f"{model} [{label}]", group_items,
                                 battery, cx, store, store_lock, grp_parallel)
                finally:
                    teardown(p)

            log(f"GPU{gpu} {model}: ALL GROUPS DONE")
        except Exception:
            log(f"GPU{gpu} {model}: FATAL error while planning/grouping/running -- "
               f"skipping remainder of this model, continuing roster")
            log(traceback.format_exc())
            # Continue the roster, but do NOT let the process still exit 0: the runner
            # would record the battery as "ok" for a model that produced no rows.
            FAILURES.append(f"{model}: {traceback.format_exc().splitlines()[-1]}")
            continue

    log(f"GPU{gpu}: ALL MODELS DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", type=int, required=True, choices=[4, 7])
    ap.add_argument("--gpu0", required=True)
    ap.add_argument("--gpu1", required=True)
    ap.add_argument("--parallel", type=int, default=4,
                    help="B7: server slots per group (mirrors p8_gen ctx_per_slot). "
                         "B4: accepted but ignored -- B4 groups always run at "
                         "effective parallel=1 for VRAM safety (see gpu_worker).")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug/smoke-test cap on a model's pending-item list, "
                         "applied BEFORE grouping (may produce partial groups).")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT)
    store = Store(OUT)
    lock = threading.Lock()

    models_cfg = cfg.registry["models"]
    mpaths = {mid: m["local_path"] for mid, m in models_cfg.items()}

    g0 = [m for m in args.gpu0.split(",") if m]
    g1 = [m for m in args.gpu1.split(",") if m]
    log(f"Battery B{args.battery} | GPU0 models: {g0} | GPU1 models: {g1} | "
        f"parallel={args.parallel} limit={args.limit or 'none'}")

    t0 = threading.Thread(target=gpu_worker,
                          args=(0, 8080, g0, cfg, store, lock, args.parallel, args.limit,
                                mpaths, args.battery))
    t1 = threading.Thread(target=gpu_worker,
                          args=(1, 8081, g1, cfg, store, lock, args.parallel, args.limit,
                                mpaths, args.battery))
    t0.start(); t1.start(); t0.join(); t1.join()
    log("ALL GPUS DONE")
    if FAILURES:
        log(f"EXITING NON-ZERO: {len(FAILURES)} model(s) failed: {FAILURES}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
