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
  - **CPU/wall/process/token quotas**: `--cpus` bounds CPU; `--pids-limit`
    (both the main and oracle containers) bounds live processes so a fork
    bomb is capped by more than just `--memory` (a fork bomb can spin up
    many non-allocating processes without ever tripping a memory limit);
    `run_in`'s (and, since the post-review hardening below, `hidden_
    validate`'s) `timeout` bounds wall-clock per command (via the
    in-container `timeout -s KILL` coreutil, with a Python-side subprocess
    timeout as a daemon-level backstop); `token_budget` is accepted/stored
    only -- token quotas are a caller-side (harness adapter) concern, not
    something the container itself can enforce.
  - **Anti-gaming hidden validation, with its own wall-clock bound +
    cleanup guarantee**: `hidden_validate(oracle, workspace, timeout=...)`
    copies the workspace (symlinks copied AS symlinks, never dereferenced
    host-side -- see its docstring) to a fresh temp dir and runs the
    oracle against it from a distinct, read-only mount (`/oracle`, never
    `/workspace`) in its own throwaway, named container, hardened the same
    as the main sandbox container (`--cap-drop ALL`, `--security-opt
    no-new-privileges`, `--cpus`/`--memory`/`--pids-limit`) since it runs
    agent-produced code (Task 3's oracles compile/run the post-run
    workspace) -- deliberately decoupled from any `B8Task` type (that
    lands in Task 3); the oracle here is a generic command list or
    callable. `timeout` bounds the command-oracle path the same two ways
    `run_in` does (in-container `timeout -s KILL` + host-side backstop),
    closing a hang risk unique to this path: agent-produced code can
    contain a busy/infinite loop that `--cpus`/`--memory` alone would not
    stop. The oracle container is force-removed + verified gone in a
    `finally` regardless of outcome, since `docker run --rm` only
    auto-removes a container that exits on its own -- a host-side timeout
    kills only the local, attached `docker run` client, which would
    otherwise leave the container itself running, orphaned.

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
import uuid
from pathlib import Path

import yaml

WORKSPACE_MOUNT = "/workspace"
ORACLE_MOUNT = "/oracle"

# Extra seconds of slack given to the Python-side subprocess.run() timeout
# beyond the in-container `timeout -s KILL <n>` wrapper, so the daemon-level
# backstop never fires before the in-container kill has had a chance to.
_TIMEOUT_SLACK_S = 15

# Sane cap on live processes inside any sandbox container -- bounds a fork
# bomb by more than just --memory (a fork bomb can spin up many
# non-allocating processes without ever tripping a memory limit).
_PIDS_LIMIT = "512"


def _default_pin(root: str | Path = ".") -> dict:
    """Read the `sandbox:` block from config/runtime_pins.yaml directly (not
    via `llmtest.registry.load_config`, so this module has no dependency on
    the rest of that config bundle)."""
    path = Path(root) / "config" / "runtime_pins.yaml"
    with path.open(encoding="utf-8") as f:
        pins = yaml.safe_load(f)
    return pins["sandbox"]


def _walk_real_files(base: str | Path):
    """Yield `(relative_posix_path, full_Path)` for every REAL file under
    `base`, on the HOST filesystem, via `os.walk(followlinks=False)` --
    never descending into a symlinked subdirectory, never yielding a
    symlinked file. This is the ONE place this codebase decides how an
    agent-planted symlink is treated when a workspace is read or copied on
    the host: it is silently absent, never followed, never resolved here
    (it may still resolve, confined, inside an isolated Linux container
    that later mounts a copy of the result -- see `hidden_validate`).
    `Path.rglob` cannot be used for this: it has no way in Python 3.10 to
    stop descending into a symlinked directory.

    Shared traversal core for `Sandbox.snapshot_workspace` and
    `copy_real_files` (below) -- factored out so both treat symlinks in
    exactly the same way, rather than each hand-rolling its own walk.
    """
    base = Path(base)
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_symlink():
                continue
            yield full.relative_to(base).as_posix(), full


def copy_real_files(src: str | Path, dst: str | Path) -> Path:
    """Copy every REAL file under `src` into `dst` (creating `dst` and any
    needed subdirectories), via `_walk_real_files` -- i.e. a symlink-safe
    replacement for `shutil.copytree(src, dst, symlinks=True)` for callers
    that will subsequently write NEW files into specific paths of the
    copy.

    That distinction matters: `copytree(..., symlinks=True)` preserves an
    agent-planted symlink AS a symlink in the copy. That is safe for
    `hidden_validate`'s own internal copy (nothing ever writes back into
    it afterward; the copy is only read, either by a container mount or a
    read-only callable). It is NOT safe for a caller that then does
    `(dst / some_known_path).write_bytes(...)` for a set of known paths --
    if an agent planted a symlink at exactly one of those paths (e.g. a
    guessable hidden-oracle filename), that write follows the symlink to
    wherever it points, ON THE HOST, outside the copy entirely. Copying
    real files only means there is never a symlink in `dst` for such a
    write to collide with.
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for rel, full in _walk_real_files(src):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(full.read_bytes())
    return dst


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
            "--pids-limit", _PIDS_LIMIT,
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
        workspace entirely. Delegates to the module-level `_walk_real_files`
        (shared with `copy_real_files`) so this codebase has exactly ONE
        place that decides how an agent-planted symlink is treated when a
        workspace is read on the host."""
        return {rel: full.read_bytes() for rel, full in _walk_real_files(self.workspace)}

    # -- hidden (anti-gaming) validation ------------------------------------

    def hidden_validate(self, oracle, workspace: str | Path, *,
                         timeout: float | None = None) -> tuple[bool, str]:
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
        (confined, not leaked). (This copytree step never has anything
        written back into it afterward -- it is only ever read, via a
        `:ro` container mount or a read-only callable -- so, unlike Task
        3's re-injection copy in `llmtest.harness.tasks.run_oracle`, there
        is no write-through-symlink risk here; see `copy_real_files` above
        for the case where that risk IS live.)

        `oracle` is either:
          - a callable: invoked as `oracle(copy_path: Path) -> (bool, detail)`.
            `timeout` is NOT enforced for a callable oracle -- there is no
            subprocess to bound; a hanging callable is a caller bug, not
            something this method can watch from the outside.
          - a command list (argv): run via `docker run --rm` in a brand-new
            throwaway container with the copy bind-mounted read-only at
            `/oracle` (never `/workspace`); exit code 0 => pass.

        `timeout` (seconds), for the command-oracle path only, bounds the
        oracle's wall clock the same two ways `run_in` already does: the
        in-container `timeout -s KILL <n>` coreutil wraps the oracle
        command itself (so a busy/infinite loop inside agent-produced code
        gets killed regardless of CPU/memory use -- `--cpus` only
        throttles, and `--memory` never trips on a non-allocating spin
        loop), and a `_TIMEOUT_SLACK_S`-second-later Python-side
        `subprocess.run` timeout backstops a hung `docker run` itself.
        `timeout=None` (the default) disables both -- matching `run_in`'s
        own `timeout=None` opt-out and keeping every existing Task 2 caller
        of this method unaffected.

        The oracle container is given a random `--name` and is FORCE
        -REMOVED + verified gone in a `finally` block regardless of outcome
        (mirroring `__exit__`'s cleanup-verification discipline). This
        matters specifically for the timeout backstop: `docker run --rm`
        only auto-removes a container that exits ON ITS OWN: if the
        host-side `subprocess.run` timeout fires, only the LOCAL, ATTACHED
        `docker run` CLIENT process gets killed -- the container itself,
        managed independently by the Docker daemon, is never told to stop
        and would otherwise keep running, orphaned, forever.

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

                container_name = f"llmtest-b8-oracle-{uuid.uuid4().hex[:12]}"
                container_cmd = (["timeout", "-s", "KILL", str(int(timeout))]
                                  if timeout is not None else []) + list(oracle)
                argv = [
                    "docker", "run", "--rm", "--name", container_name,
                    "--read-only", "--tmpfs", "/tmp",
                    "--network", "none",
                    "--cpus", str(self.cpus),
                    "--memory", self.mem_limit,
                    "--pids-limit", _PIDS_LIMIT,
                    "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges",
                    "-v", f"{copy_root}:{ORACLE_MOUNT}:ro",
                    self._image_ref,
                ] + container_cmd
                host_timeout = (timeout + _TIMEOUT_SLACK_S) if timeout is not None else None
                timed_out = False
                r = None
                try:
                    r = subprocess.run(argv, capture_output=True, text=True, timeout=host_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    cleanup_status = self._force_remove_oracle_container(container_name)

                if timed_out:
                    return False, f"oracle timeout: exceeded wall-clock budget ({cleanup_status})"
                detail = f"exit={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"
                if r.returncode in (124, 137):
                    # 124: coreutils `timeout` itself killed the oracle command
                    # (its own exit code when the wrapped command was killed);
                    # 137 = 128+SIGKILL(9): the container's PID 1 (the `timeout`
                    # process) was killed with SIGKILL and that propagated as
                    # the container's own exit code. Either way, the
                    # in-container wall-clock bound fired.
                    return False, f"oracle timeout: {detail}"
                return r.returncode == 0, detail
        except Exception as e:  # noqa: BLE001 - any pre-oracle setup failure (e.g. an
            # unsupported file type tripping copytree) is a validation
            # result, not a crash -- mirrors the callable-oracle handling above.
            return False, f"hidden_validate setup failed: {e!r}"

    def _force_remove_oracle_container(self, name: str) -> str:
        """Force-remove the throwaway oracle container by NAME and verify
        it's actually gone -- mirrors `__exit__`'s force-remove + verify
        discipline, but by name rather than id: the oracle container is
        never run detached (no `-d`), so unlike `__exit__`'s
        `self.container_id` (captured from `docker run -d`'s stdout) there
        is no docker-assigned id available up front -- `--name` is what we
        control instead. Deliberately separate from `__exit__`'s own
        inline cleanup code (which already has its own tests) rather than
        a shared refactor, to keep this addition low-risk against Task 2's
        existing, stable behavior."""
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
        check = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", f"name={name}"],
            capture_output=True, text=True)
        if check.stdout.strip():
            return f"cleanup WARNING: oracle container {name} still present after docker rm -f"
        return "cleanup verified: oracle container removed"
