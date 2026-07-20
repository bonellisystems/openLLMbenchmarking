"""Disposable Docker container sandbox (Task 2) -- the B8 security boundary
a real agent-harness run will (Phase 2, deferred) execute inside. No
harness adapter or battery here; this module only builds/tears down the
container and runs commands in it.

Security posture (spec Global Constraints, TESTPLAN B8 plan §Global
Constraints), each mapped to a concrete Docker mechanism:
  - **Read-only rootfs + rw workspace mount**: `--read-only` on the whole
    container plus the task workspace bind-mounted read-write at
    `/workspace` (with a `--tmpfs /tmp` for scratch space). Writes inside
    `/workspace` persist to the host dir; writes anywhere else fail.
  - **Process-tree kill + cleanup verification**: `__exit__` runs
    `docker rm -f`, which force-kills the container's whole cgroup (init
    process plus any backgrounded/reparented children -- Docker tracks the
    entire tree, not just PID 1), then verifies via `docker ps -a` that the
    container id is actually gone, raising if not.
  - **Egress restricted to the endpoint (deny-all by default)**: the
    container always runs with `--network none`. This is deliberately
    *stricter* than "endpoint-only" -- deny-all is a strict subset of
    endpoint-only egress, so it satisfies the spec today. `endpoint` is
    accepted and stored for provenance but NOT wired into a selective-egress
    policy yet: there is no live endpoint to test against in Phase 1 (real
    harness runs are deferred to Phase 2). Building an iptables-based
    selective-allow now would be untestable and is explicitly deferred --
    Phase 2 wires + live-tests it.
  - **No host-credential mounts, secret-free environment**: `__enter__`
    never mounts any host path other than the task workspace, and never
    passes host environment variables into the container.
  - **CPU/wall/token quotas**: `--cpus` bounds CPU; `run_in`'s `timeout`
    bounds wall-clock per command (via the in-container `timeout -s KILL`
    coreutil, with a Python-side subprocess timeout as a daemon-level
    backstop); `token_budget` is accepted/stored only -- token quotas are a
    caller-side (harness adapter) concern, not something the container
    itself can enforce.
  - **Anti-gaming hidden validation**: `hidden_validate(oracle, workspace)`
    copies the workspace (symlinks copied AS symlinks, never dereferenced
    host-side -- see its docstring) to a fresh temp dir and runs the
    oracle against it from a distinct, read-only mount (`/oracle`, never
    `/workspace`) in its own throwaway container, hardened the same as the
    main sandbox container (`--cap-drop ALL`, `--security-opt
    no-new-privileges`, `--cpus`/`--memory`) since it runs agent-produced
    code (Task 3's oracles compile/run the post-run workspace) --
    deliberately decoupled from any `B8Task` type (that lands in Task 3);
    the oracle here is a generic command list or callable.

Runs the pinned image by `image@digest` (not just the tag) for immutability.
Shells out to the `docker` CLI via `subprocess` rather than the docker-py
SDK, matching how `llmtest.server.ServerManager` already drives external
processes -- and avoiding a new pyproject dependency for a package that
happens to be installed on this machine but isn't declared.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

WORKSPACE_MOUNT = "/workspace"
ORACLE_MOUNT = "/oracle"

# Extra seconds of slack given to the Python-side subprocess.run() timeout
# beyond the in-container `timeout -s KILL <n>` wrapper, so the daemon-level
# backstop never fires before the in-container kill has had a chance to.
_TIMEOUT_SLACK_S = 15


def _default_pin(root: str | Path = ".") -> dict:
    """Read the `sandbox:` block from config/runtime_pins.yaml directly (not
    via `llmtest.registry.load_config`, so this module has no dependency on
    the rest of that config bundle)."""
    path = Path(root) / "config" / "runtime_pins.yaml"
    with path.open(encoding="utf-8") as f:
        pins = yaml.safe_load(f)
    return pins["sandbox"]


class Sandbox:
    """Disposable, isolated Docker container bound to one task workspace.

    Usage::

        with Sandbox(workspace=task_dir) as sbx:
            code, out, err = sbx.run_in(["bash", "-c", "..."], timeout=60)
            state = sbx.snapshot_workspace()
        # container is force-removed and verified gone here

    `hidden_validate` does not require an active (`__enter__`-ed) sandbox --
    it spins its own throwaway container per call -- so it can be invoked
    after the agent's sandbox has already exited, which is the intended B8
    flow (behavioral tests run AFTER the harness exits).
    """

    def __init__(self, workspace: str | Path, *, image: str | None = None,
                 digest: str | None = None, endpoint: tuple[str, int] | None = None,
                 cpus: float = 2.0, mem_limit: str = "2g",
                 token_budget: int | None = None, root: str | Path = "."):
        pin = _default_pin(root) if image is None or digest is None else {}
        self.workspace = Path(workspace)
        self.image = image or pin["image"]
        self.digest = digest or pin.get("digest")
        # Stored, not yet enforced -- see module docstring. Phase 2 wires a
        # selective-egress policy that opens exactly this host:port.
        self.endpoint = endpoint
        self.cpus = cpus
        self.mem_limit = mem_limit
        self.token_budget = token_budget
        self.container_id: str | None = None

    @property
    def _image_ref(self) -> str:
        return f"{self.image}@{self.digest}" if self.digest else self.image

    # -- lifecycle -----------------------------------------------------

    def __enter__(self) -> "Sandbox":
        self.workspace.mkdir(parents=True, exist_ok=True)
        argv = [
            "docker", "run", "-d",
            "--read-only",
            "--tmpfs", "/tmp",
            "-v", f"{self.workspace}:{WORKSPACE_MOUNT}:rw",
            "--network", "none",           # deny-all egress by default; see module docstring
            "--cpus", str(self.cpus),
            "--memory", self.mem_limit,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            # no -e / --env-file / extra -v: no host env, no host-credential mounts
            self._image_ref,
            "sleep", "infinity",           # keep-alive; real commands run via `run_in` (docker exec)
        ]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Sandbox: failed to start container: {r.stderr.strip()}")
        self.container_id = r.stdout.strip()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        cid, self.container_id = self.container_id, None
        if cid is None:
            return False
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, text=True)
        check = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", f"id={cid}"],
            capture_output=True, text=True)
        if check.stdout.strip():
            msg = f"Sandbox cleanup verification failed: container {cid} still present after docker rm -f"
            if exc_type is not None:
                # don't mask an in-flight exception with a cleanup error
                print(f"WARNING: {msg}")
            else:
                raise RuntimeError(msg)
        return False   # never suppress exceptions raised in the `with` block

    # -- running commands ------------------------------------------------

    def run_in(self, cmd: list[str], timeout: float | None = None) -> tuple[int, str, str]:
        """Run `cmd` (argv list) inside the live container via `docker exec`.

        `timeout` (seconds) is enforced two ways: primarily by wrapping the
        command with the in-container `timeout -s KILL <n>` coreutil (kills
        the exec'd process on the wall clock, not just this client), and as
        a `_TIMEOUT_SLACK_S`-second-later Python-side `subprocess.run`
        timeout backstop in case the docker daemon itself hangs.
        """
        if self.container_id is None:
            raise RuntimeError("run_in() called outside an active sandbox (no __enter__)")
        argv = ["docker", "exec", self.container_id]
        if timeout is not None:
            argv += ["timeout", "-s", "KILL", str(timeout)]
        argv += list(cmd)
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=(timeout + _TIMEOUT_SLACK_S) if timeout is not None else None)
        except subprocess.TimeoutExpired as e:
            return (124, e.stdout or "",
                    (e.stderr or "") + "\nrun_in: docker exec did not return within "
                    "the wall-clock timeout (daemon-level backstop fired)")
        return (r.returncode, r.stdout, r.stderr)

    # -- workspace state ---------------------------------------------------

    def snapshot_workspace(self) -> dict[str, bytes]:
        """Current workspace file state as {posix relative path: bytes}.
        Reads the host bind-mount directly (it mirrors the container's
        `/workspace` at all times) -- no container access needed.

        Symlinks are DELIBERATELY SKIPPED, not followed: this reads via the
        HOST process, so a workspace-planted symlink (e.g. `ln -s
        /etc/passwd leak`) resolving here would leak host file content, and
        a symlinked directory would let a walk traverse outside the
        workspace entirely. Uses `os.walk(followlinks=False)` -- not
        `Path.rglob`, which has no way in Python 3.10 to stop descending
        into a symlinked directory -- and additionally skips any symlinked
        *file* entry directly (a directory-prune alone doesn't cover a
        symlinked file sitting in an otherwise-real directory)."""
        result: dict[str, bytes] = {}
        base = self.workspace
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            # prune symlinked subdirectories before os.walk descends into them
            dirnames[:] = [d for d in dirnames
                           if not os.path.islink(os.path.join(dirpath, d))]
            for name in filenames:
                full = Path(dirpath) / name
                if full.is_symlink():
                    continue
                result[full.relative_to(base).as_posix()] = full.read_bytes()
        return result

    # -- hidden (anti-gaming) validation ------------------------------------

    def hidden_validate(self, oracle, workspace: str | Path) -> tuple[bool, str]:
        """Run `oracle` against a FRESH, READ-ONLY COPY of `workspace`,
        mounted OUTSIDE the agent's writable area -- so nothing the agent
        did to its own `/workspace` mount (or any process it left running)
        can influence the validation.

        Symlinks in `workspace` are copied AS SYMLINKS
        (`shutil.copytree(..., symlinks=True)`), never dereferenced by the
        HOST copy step. The accurate guarantee this gives: a workspace
        symlink (e.g. `ln -s /etc/passwd leak`, or one pointing at an
        arbitrary host path) cannot make the HOST process read host file
        content into the copy, and a dangling symlink cannot crash the
        copy. If the symlink resolves at all, it only does so INSIDE the
        isolated oracle container (`--network none`, copy mounted `:ro`,
        against the pinned image's own filesystem) -- e.g. `/etc/passwd`
        resolves to the *container's* passwd file, not the host's, and a
        symlink to a host-only absolute path simply fails to resolve
        (confined, not leaked).

        `oracle` is either:
          - a callable: invoked as `oracle(copy_path: Path) -> (bool, detail)`.
          - a command list (argv): run via `docker run --rm` in a brand-new
            throwaway container with the copy bind-mounted read-only at
            `/oracle` (never `/workspace`); exit code 0 => pass.

        Any failure setting up the copy (e.g. an unsupported file type, or
        any other host-side copy error) returns `(False, detail)` rather
        than raising -- consistent with the callable-oracle path below:
        an oracle failure is a validation result, not a crash.

        Deliberately decoupled from any `B8Task` type -- that type doesn't
        exist yet (Task 3). Task 3's oracle wiring is expected to build a
        command list or callable and call this method.
        """
        src = Path(workspace)
        try:
            with tempfile.TemporaryDirectory(prefix="llmtest-sbx-oracle-") as tmp:
                copy_root = Path(tmp) / "ws"
                shutil.copytree(src, copy_root, symlinks=True)

                if callable(oracle):
                    try:
                        ok, detail = oracle(copy_root)
                    except Exception as e:  # noqa: BLE001 - oracle failure is a validation result, not a crash
                        return False, f"oracle callable raised: {e!r}"
                    return bool(ok), str(detail)

                argv = [
                    "docker", "run", "--rm",
                    "--read-only", "--tmpfs", "/tmp",
                    "--network", "none",
                    "--cpus", str(self.cpus),
                    "--memory", self.mem_limit,
                    "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges",
                    "-v", f"{copy_root}:{ORACLE_MOUNT}:ro",
                    self._image_ref,
                ] + list(oracle)
                r = subprocess.run(argv, capture_output=True, text=True)
                detail = f"exit={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"
                return r.returncode == 0, detail
        except Exception as e:  # noqa: BLE001 - any pre-oracle setup failure (e.g. an
            # unsupported file type tripping copytree) is a validation
            # result, not a crash -- mirrors the callable-oracle handling above.
            return False, f"hidden_validate setup failed: {e!r}"
