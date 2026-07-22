"""OpenCode headless harness adapter (Task 4, Part 2 Phase 2) -- the FIRST
live `HarnessAdapter`, driving the real `opencode` CLI headlessly against a
local OpenAI-compatible endpoint (llama-server) and mapping its own sqlite
trace store into the normalized `Trace` schema (Task 1). Sets the pattern
the LiteLLM (Task 5) and Hermes/WSL2 (Task 6) adapters follow.

RESOLVED DESIGN (Phase-0/0.2 spikes, see `docs/superpowers/notes/
b8-spike-serverprofile.md`, "Update 4 -- DEFINITIVE PASS" -- these are
confirmed facts, not choices re-made here):

- OpenCode only works headlessly with a config granting
  `permission: {edit, bash, webfetch: "allow"}` -- headless `opencode run`
  has no TTY, so without this the write/bash tools block on approval and
  the run hangs forever. `setup()` writes this config (plus a `local`
  `@ai-sdk/openai-compatible` provider pointed at `endpoint + "/v1"`) to
  `self._config_path`, a SIBLING of the workspace (never inside it), and
  `_launch` hands it to `opencode` via the `OPENCODE_CONFIG` env var --
  NOT as an `opencode.json` file in the cwd. (An earlier version of this
  adapter did write it into the workspace, matching what the Phase-0.2
  spike itself did; Task-4 review correctly flagged that as a live
  landmine -- `run_oracle`'s diff constraint, Task 3, would flag the
  leftover `opencode.json` as an out-of-bounds edit and FAIL an otherwise-
  correct run. Verified live: a run with zero `opencode.json` anywhere
  under the workspace, config supplied purely via `OPENCODE_CONFIG`,
  completed successfully end-to-end against the real endpoint.)
- `opencode run "<prompt>" -m local/<model>` is invoked with its own
  process's `stdout`/`stderr` NOT piped: OpenCode is verbose (INFO-level
  service logs on top of whatever `--print-logs` adds), and on Windows a
  piped, unread stdout deadlocks once the ~64 KB pipe buffer fills --
  `process.wait(timeout=...)` would then "time out" on a process that
  isn't actually hung, corrupting the very budget signal this adapter
  exists to measure. Output is redirected to a log file instead (mirrors
  `llmtest.server.ServerManager`'s own `Popen(..., stdout=logf)` pattern),
  written OUTSIDE the workspace (a sibling file, not under `/workspace`) so
  it never pollutes the agent-visible tree `run_oracle`'s diff constraint
  (Task 3) checks against.
- The native transcript is NOT read from `--format json` -- that mode
  buffers to stdout and yields NOTHING if the process is killed on a
  timeout, which is exactly the case this adapter most needs a trace for
  (a budget-exceeding hang is a real, expected B8 measurement, not a
  corner case -- gpt-oss-20b's own reasoning regularly runs long). Instead
  the trace is read from OpenCode's own **sqlite store**
  (`~/.local/share/opencode/opencode.db`, or an injected `db_path`), which
  is written incrementally as the run progresses and survives a kill:
    - `session(id, directory, time_created, ...)` -- the run's session is
      identified among rows with `time_created` at or after a timestamp
      recorded just before launch, preferring the one whose `directory`
      matches this run's workspace (exact, then case-insensitive, then a
      filesystem-ancestor check for a hypothesized git-worktree-root
      shape -- see `_find_session_id`'s docstring), but falling back to
      simply the NEWEST session in that window if none of those match --
      exactly one `opencode` subprocess is launched and synchronously
      awaited per `run()` call, so a directory-correlation miss must never
      by itself misclassify a genuinely-completed run as `infra-error`
      (Task-4 review finding #2).
    - `message(id, session_id, time_created, data)` -- `data` is a JSON
      blob: `role` ("user"/"assistant"), `tokens{input,output,reasoning}`,
      `finish` (finish reason; ABSENT, not e.g. `"error"`, on a
      provider-side failure -- confirmed against a live db with a real
      `ContextOverflowError` capture: the failing assistant message has
      `finish` unset and an `error{name,data}` field instead), `modelID`.
    - `part(id, message_id, session_id, time_created, data)` -- `data` is
      a JSON blob keyed by `type`: `"step-start"` is a turn boundary
      (`Trace.steps` = count of these, matching `Trace.from_events`'s own
      "turn"-event-count derivation); `"tool"` is
      `{callID, tool, state{status, input, output}}` (`tool == "task"` is
      OpenCode's subagent-delegation primitive); `"text"`/`"reasoning"`
      parts are assistant output, not separately represented in `Trace`
      (steps already come from `step-start`, and tool events are already
      separate) so they are read but intentionally not turned into events.

CONTAINMENT (Wave 2, B8 validity program -- the #1 threat per the
codex/kimi strategy review): the HOST-EXECUTION deferral below is now
resolved for the common case. When `sandbox_image` is set (threaded from
`suite.yaml`'s `b8.sandbox.image` via `llmtest.batteries.b8_harness.
_DEFAULT_HARNESS_FACTORIES`), `run()` drives `opencode` inside a
disposable, hardened `docker run` (`_launch_container`) instead of a host
subprocess -- no host env, no host-credential mounts, non-root, read-only
rootfs, resource-capped, endpoint reachable ONLY via `host.docker.internal`
(the served model's port on the HOST, not a container-local one),
`webfetch` DENIED in the containerized config. `sandbox_image=None` (the
default -- also what every unit test in `test_harness_opencode.py`
constructs) keeps the ORIGINAL host-subprocess path (`_launch`) byte-for-
byte unchanged; see that method's own docstring for why (Node CLI, no
Node in the (unrelated) `nvidia/cuda:...-base` Sandbox pin) and its still-
valid rationale for why Windows piping matters for THAT path specifically.

Two mounts only, both host-visible so the harness can read/verify what
happened without touching the container: the task workspace itself
(`-v <ws>:/workspace:rw`, so `run_oracle`'s post-run diff check sees
exactly what the containerized agent edited), and a FRESH per-run host
temp dir (`-v <tmp>:/home/node/.local/share/opencode:rw`, so the
container's own `opencode.db` lands somewhere `_read_trace` -- unchanged,
execution-environment-agnostic, it just reads a db path -- can read it
straight off the host filesystem after the container exits; "fresh per
run" also means no cross-run state ever leaks between two containerized
attempts, a hard requirement for B8 validity). `self.db_path`/
`self._config_path` are repointed into this directory in `setup()` for
container mode; the config itself (permission + `local` provider baseURL)
is otherwise generated by the SAME `_write_opencode_config()` host mode
uses, just with `webfetch: "deny"` instead of `"allow"` and delivered via
`OPENCODE_CONFIG` pointed at a container-visible path inside that same
mount (never `/workspace` -- would trip `run_oracle`'s out-of-bounds-edit
check exactly like the host-mode landmine this module already fixed once,
see `_write_opencode_config`'s own docstring).

Two LIVE findings from an actual container run against the served
gpt-oss-20b endpoint (`docker run --rm --user node --read-only ...
b8-sandbox:1 opencode run ...`), both now baked into `_launch_container`:
  1. `--tmpfs /home/node` ALONE is not enough -- Docker's own volume-mount
     setup (host-Docker-Desktop-side, effectively root) auto-creates the
     BIND mount's intermediate parent directories (`.local`, `.local/
     share`) as root:root 0755, regardless of the PARENT tmpfs's own
     uid/gid/mode option. opencode/bun then fails with `EACCES: permission
     denied, mkdir '/home/node/.local/state'` (a SIBLING dir it needs to
     create at runtime) because it can't write into the root-owned
     `.local`. Fix: mount `.local` ITSELF as a second, `node`-owned tmpfs
     (`--tmpfs /home/node/.local:uid=1000,gid=1000,mode=1777`), one level
     up from the bind target -- `.local` becomes writable even though
     `.local/share` (the deeper auto-created intermediate) stays
     root-owned, which is fine: nothing needs to write directly into
     `.local/share`, only into the bind-mounted `.local/share/opencode`
     (already rw via the bind mount itself) and siblings of `.local` like
     `.local/state`.
  2. `session.directory` inside the container-produced `opencode.db` is
     recorded as the literal CONTAINER cwd, `"/workspace"` -- never the
     host workspace path `_read_trace`/`_find_session_id` are handed.
     Directory correlation (buckets 1-3 of `_find_session_id`) therefore
     NEVER matches in container mode; every containerized run falls
     through to bucket 4 (newest session in the post-launch time window).
     This is provably safe, not a lucky accident: the opencode-home dir
     mounted at `/home/node/.local/share/opencode` is a FRESH `tempfile.
     mkdtemp()` per run (finding immediately above's mount), so the
     resulting `opencode.db` has ZERO pre-existing session rows -- the
     one session a single containerized `opencode run` writes IS,
     trivially, "the newest (and only) session in the window."
     `_find_session_id` needed no code change for this; it is documented
     here so a future reader doesn't mistake the directory-correlation
     miss for a bug.

TOKEN SOURCE (verified, per the family brief's "server-side llama-server
usage, NOT harness proxies" requirement): `message.data.tokens` IS the
`@ai-sdk/openai-compatible` provider's passthrough of llama-server's own
OpenAI-compatible `usage` object, not a value OpenCode recomputes locally.
Structural proof, not just plausibility: OpenCode's `tokens.cache.
{read,write}` fields map directly onto llama-server's own
`usage.prompt_tokens_details.cached_tokens` (confirmed live via a direct
non-streaming `/v1/chat/completions` request to the same endpoint -- the
raw response contains exactly that field). A client-side tokenizer has no
way to know how many prompt tokens hit the SERVER's own KV cache -- that
is purely a runtime fact about the server's own state, not a property of
the text -- so a populated, plausible `cache.read`/`cache.write` in
OpenCode's stored tokens could only have come from the server's response,
never from OpenCode computing token counts independently.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from llmtest.harness.base import HarnessAdapter
from llmtest.harness.tasks import materialize_repo
from llmtest.harness.trace import Trace, TraceEvent

DEFAULT_DB_PATH = Path(os.path.expanduser("~/.local/share/opencode/opencode.db"))
DEFAULT_MODEL = "gpt-oss-20b"
DEFAULT_WALL_CLOCK_S = 180.0

# Container-mode (Wave 2 CONTAINMENT) paths/constants -- see module
# docstring's CONTAINMENT section for the two live findings behind the
# extra `.local` tmpfs and the container-cwd session.directory shape.
_CONTAINER_HOME = "/home/node"
_CONTAINER_OPENCODE_HOME = f"{_CONTAINER_HOME}/.local/share/opencode"
_CONTAINER_WORKSPACE = "/workspace"
_CONTAINER_CONFIG_FILENAME = "opencode-config.json"

# Extra seconds of slack given to the Python-side `subprocess.run()`
# timeout on `docker run` beyond the in-container `timeout -s KILL <n>`
# wrapper -- mirrors `llmtest.harness.sandbox._TIMEOUT_SLACK_S`'s own
# two-layer discipline (in-container kill is primary; this is a backstop
# for a hung docker daemon/CLI, not the normal path). Slightly larger than
# Sandbox's 15s to give `docker run` (container create + mount setup, not
# just `docker exec` into an already-live container) a bit more headroom.
_CONTAINER_TIMEOUT_SLACK_S = 20

# Slack subtracted from the pre-launch timestamp before querying `session.
# time_created >= ?` -- guards against clock-rounding / millisecond-boundary
# skew between this process's `time.time()` and OpenCode's own timestamp,
# without being so generous it risks matching a stale prior session (the
# `directory` equality check is the real disambiguator; this is only slack
# on top of that).
_SESSION_TS_SLACK_MS = 2000

# Tool-part states this adapter treats as a genuine success. Anything else
# (an explicit "error", a "running" part that never resolved because the
# process was killed mid-tool-call, or a missing/malformed `state` entirely)
# is recorded as a failed tool_result rather than crashing the mapping.
_TOOL_SUCCESS_STATUSES = {"completed"}


class OpenCodeAdapter(HarnessAdapter):
    """Drives one model through the OpenCode CLI (headless) for one B8 task,
    on the HOST (see module docstring -- in-sandbox execution is deferred).
    """

    def __init__(self, *, model: str = DEFAULT_MODEL,
                 db_path: str | Path = DEFAULT_DB_PATH,
                 wall_clock_s: float | None = None,
                 max_steps: int | None = None,
                 opencode_bin: str = "opencode",
                 sandbox_image: str | None = None,
                 cpus: float = 2.0, mem_limit: str = "4g",
                 pids_limit: str = "512"):
        self.model = model
        self.db_path = Path(db_path)
        # None => fall back to the TASK's own budgets.wall_clock_s at run()
        # time (each B8 manifest declares its own), then DEFAULT_WALL_CLOCK_S
        # if the task has none either. An explicit value here (e.g. the live
        # smoke's ~180s) always wins.
        self.wall_clock_s = wall_clock_s
        # Native step-limit config (Wave 1a, B8 measurement-validity --
        # confirmed via Context7 against the real anomalyco/opencode
        # source): when set, threaded into `_write_opencode_config` as
        # `{"agent": {"build": {"steps": max_steps}}}` -- `opencode run`
        # with no `--agent` flag uses the built-in "build" agent, and
        # OpenCode's config layer lets you override a BUILT-IN agent's
        # fields by reusing its name as the config key (same mechanism a
        # custom agent uses). This is a SOFT, best-effort cap: on hitting
        # it, OpenCode's own agent loop is told to summarize its work and
        # stop, not hard-killed -- so `llmtest.batteries.b8_harness.
        # execute()`'s own post-hoc `trace.steps > budget_steps` check
        # remains the actual (and only, for a completion-token budget --
        # OpenCode has no native equivalent of that at all) backstop.
        # `None` (the default) omits the `agent` config key entirely,
        # byte-for-byte the pre-Wave-1a config shape.
        self.max_steps = max_steps
        self.opencode_bin = opencode_bin

        # Containment (Wave 2): `sandbox_image` set -> `run()` drives
        # opencode via `_launch_container` (disposable, hardened `docker
        # run`) instead of the host `_launch` subprocess -- see module
        # docstring's CONTAINMENT section. `None` (the default) is
        # byte-for-byte the pre-Wave-2 host path; `cpus`/`mem_limit`/
        # `pids_limit` mirror `llmtest.harness.sandbox.Sandbox`'s own
        # resource-cap conventions and are inert when `sandbox_image` is
        # `None`.
        self.sandbox_image = sandbox_image
        self.cpus = cpus
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit

        self.task = None
        self.endpoint: str | None = None
        self.workspace: Path | None = None
        self.process = None
        self._since_ts: int | None = None
        self._log_path: Path | None = None
        self._config_path: Path | None = None
        self._version: str | None = None
        # Container mode only (both None in host mode, for the lifetime of
        # this adapter):
        #   - `_opencode_home_dir`: the FRESH per-run host temp dir mounted
        #     at `/home/node/.local/share/opencode:rw` -- `self.db_path`/
        #     `self._config_path` are repointed here in `setup()`.
        #   - `_container_name`: set BEFORE `docker run` starts (so
        #     `teardown()` can force-remove it even if `_launch_container`
        #     itself never returns normally) and cleared to `None` once
        #     `_launch_container` has fully handled the outcome either way
        #     -- mirrors `_launch`'s own `self.process = None` reset
        #     discipline (post-review trivial fix #5) so a later
        #     `teardown()` is a true no-op, not a redundant `docker rm -f`
        #     against an already-gone container.
        self._opencode_home_dir: Path | None = None
        self._container_name: str | None = None

    # -- HarnessAdapter lifecycle -----------------------------------------

    def setup(self, task, endpoint, workspace) -> None:
        self.task = task
        self.endpoint = getattr(endpoint, "base_url", None) or str(endpoint)
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        materialize_repo(task, self.workspace)
        # Record BEFORE launch (with slack) so `_read_trace` can find this
        # run's session even if OpenCode's own clock is a hair behind ours.
        self._since_ts = int(time.time() * 1000) - _SESSION_TS_SLACK_MS
        token = uuid.uuid4().hex[:8]
        self._log_path = self.workspace.parent / f".opencode-{token}.log"
        if self.sandbox_image:
            # Containment (Wave 2): the endpoint's HOST-visible base_url
            # (e.g. "http://127.0.0.1:8080", or whatever a live
            # `EndpointHandle`/manual-endpoint object carries) is not
            # reachable from INSIDE the container -- rewritten here to
            # `host.docker.internal`, which `_launch_container`'s
            # `--add-host host.docker.internal:host-gateway` makes resolve
            # to the host's own gateway IP. Only `self.endpoint` (a plain
            # string on this adapter) is mutated -- the caller's own
            # endpoint object (and its `normalized_config`, read separately
            # by `llmtest.batteries.b8_harness.execute()` for row
            # provenance) is never touched.
            self.endpoint = _rewrite_endpoint_for_container(self.endpoint)
            # FRESH per run (codex requirement -- no cross-run state): a
            # brand-new host temp dir every `setup()` call, mounted at
            # `/home/node/.local/share/opencode:rw` by `_launch_container`.
            # `db_path`/`_config_path` are repointed HERE (not left at
            # whatever the constructor was given) so the container's own
            # `opencode.db` and the config this adapter writes both land
            # in the one directory the container actually sees -- see
            # module docstring's CONTAINMENT section, finding #2, for why
            # "fresh" also matters for session correlation, not just
            # cross-run hygiene.
            self._opencode_home_dir = Path(
                tempfile.mkdtemp(prefix="llmtest-b8-ochome-"))
            # LINUX containment fix: mkdtemp is root-owned mode 0700 (the run
            # process is root on the datacenter box), but the container runs
            # as --user node (uid 1000) and must mkdir bin/, write opencode.db,
            # etc. INSIDE this bind-mounted opencode-home. Without world-write
            # OpenCode/bun dies instantly (`EACCES: mkdir .../opencode/bin`).
            # On Docker Desktop the VM masked host ownership; native Linux
            # exposes it. 0777 on the dir is the minimal fix (files within
            # keep their own perms; the config this adapter writes stays 0644).
            os.chmod(self._opencode_home_dir, 0o777)
            self.db_path = self._opencode_home_dir / "opencode.db"
            self._config_path = self._opencode_home_dir / _CONTAINER_CONFIG_FILENAME
        else:
            self._config_path = self.workspace.parent / f".opencode-config-{token}.json"
        self._write_opencode_config()

    def run(self) -> Trace:
        prompt = self.task.prompt
        timeout = self.wall_clock_s
        if timeout is None:
            timeout = (self.task.budgets or {}).get("wall_clock_s") or DEFAULT_WALL_CLOCK_S

        if self.sandbox_image:
            returncode, timed_out, launch_error = self._launch_container(prompt, timeout)
        else:
            argv = [self.opencode_bin, "run", prompt, "-m", f"local/{self.model}"]
            returncode, timed_out, launch_error = self._launch(argv, cwd=self.workspace, timeout=timeout)

        if launch_error is not None:
            # subagent_spawned is "no", never "not_applicable" here: OpenCode
            # HAS a delegation primitive (the "task" tool) regardless of
            # whether this particular run ever got far enough to use it --
            # "not_applicable" is reserved for a harness that lacks the
            # capability entirely, which OpenCode never does.
            return Trace.from_events(
                [TraceEvent(kind="terminal", payload={"launch_error": launch_error})],
                terminal_status="infra-error", tokens_prompt=0, tokens_completion=0,
                subagent_spawned="no")

        (events, tokens_prompt, tokens_completion, subagent_spawned,
         last_finish, last_error, missing_usage) = self._read_trace(
            self.db_path, self.workspace, self._since_ts)

        if timed_out:
            terminal_status = "killed"
        elif returncode != 0:
            terminal_status = "infra-error"
        elif last_error is not None:
            # opencode exited 0, but the model/provider call itself failed
            # (e.g. a live-confirmed ContextOverflowError) -- finish is
            # absent in that case too, so this check is belt-and-suspenders
            # around the `last_finish is None` branch below.
            terminal_status = "infra-error"
        elif last_finish is not None:
            terminal_status = "completed"
        else:
            # Exited cleanly but no coherent finish reason was ever
            # recorded (e.g. no matching session at all) -- conservative:
            # this is an infra-level "couldn't confirm completion", not a
            # fabricated "completed".
            terminal_status = "infra-error"

        events = events + [TraceEvent(kind="terminal", payload={
            "returncode": returncode, "finish": last_finish,
            "error": last_error, "missing_usage": missing_usage,
        })]
        return Trace.from_events(events, terminal_status=terminal_status,
                                  tokens_prompt=tokens_prompt,
                                  tokens_completion=tokens_completion,
                                  subagent_spawned=subagent_spawned)

    def teardown(self) -> None:
        if self.process is not None:
            self._kill_process_tree(self.process)
            self.process = None
        if self._container_name is not None:
            # Belt-and-suspenders: `_launch_container` already force-
            # removes its own container on every outcome it fully handles
            # (see that method's docstring) and clears this to `None` --
            # a non-`None` value here means it was interrupted somehow
            # (e.g. an exception escaping `run()` before `_launch_container`
            # could finish), not the normal path.
            self._force_remove_container(self._container_name)
            self._container_name = None
        if self._opencode_home_dir is not None:
            # Disposable like the container itself -- the `Trace` this run
            # produced was already fully materialized in memory by `run()`
            # (which reads `self.db_path` before `teardown()` is ever
            # called, per the adapter lifecycle contract), so nothing
            # downstream still needs this directory on disk.
            shutil.rmtree(self._opencode_home_dir, ignore_errors=True)
            self._opencode_home_dir = None

    def version(self) -> str:
        if self._version is None:
            if self.sandbox_image:
                self._version = self._container_version()
            else:
                try:
                    argv = self._resolved_argv([self.opencode_bin, "--version"])
                    r = subprocess.run(argv, capture_output=True, text=True, timeout=15)
                    self._version = r.stdout.strip() or "unknown"
                except Exception:  # noqa: BLE001 - a version probe failure is not fatal
                    self._version = "unknown"
        return self._version

    def _container_version(self) -> str:
        """`version()`'s container-mode counterpart: probes the version of
        opencode actually baked into `self.sandbox_image` (the host may
        have a different -- or no -- `opencode` on PATH at all), via a
        throwaway `docker run --rm`. Same "never raises, 'unknown' on any
        failure" contract as the host probe -- a version-probe failure is
        not fatal and must never block a real run."""
        try:
            argv = ["docker", "run", "--rm", "--user", "node", self.sandbox_image,
                    self.opencode_bin, "--version"]
            r = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            return r.stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001 - a version probe failure is not fatal
            return "unknown"

    def _resolved_argv(self, argv: list[str]) -> list[str]:
        """Full argv to actually exec, with `argv[0]` resolved via
        `shutil.which()` (Windows PATHEXT shim resolution -- same fix
        already applied to judge CLIs in `llmtest.judging.adapters.
        BaseAdapter._resolve_argv0`: npm-global installs like `opencode`
        are `.cmd` shims, and `subprocess.Popen`/`.run(shell=False)` launch
        via `CreateProcess` directly, which does NOT apply `PATHEXT`
        resolution the way an interactive shell does -- a bare `"opencode"`
        raises `WinError 2` even though it's genuinely on PATH) and, if
        THAT resolves to a `.cmd`/`.bat` npm shim, additionally UNWRAPPED
        to invoke the underlying `node.exe <entry.js>` directly.

        The unwrap step is not defensive paranoia -- it fixes a real bug
        the live smoke surfaced: a B8 prompt is a multi-line YAML block
        scalar. Passed as one argv element with `shell=False`, a plain
        `.exe` target handles the embedded newlines fine (the target's own
        C-runtime argv parser honors Python's `list2cmdline` quoting). A
        `.cmd`/`.bat` target does NOT: Windows can only launch a batch
        file by implicitly running it through `cmd.exe /c`, and cmd.exe's
        own command-line tokenizer treats a raw newline as a statement
        separator regardless of quoting. Confirmed live: the first smoke
        attempt's own debug log recorded the actual received argv as
        `args=["run", "<prompt truncated at its first line>"]` -- the rest
        of the prompt AND the trailing `-m local/<model>` flag were both
        silently dropped, and OpenCode fell back to an unrelated default
        model. Reading the shim's own `%dp0%\\node_modules\\...\\bin\\
        <entry>` reference and invoking `node.exe` on it directly
        sidesteps cmd.exe (and its newline-as-separator parsing) entirely.
        """
        if not argv:
            return argv
        resolved = shutil.which(argv[0])
        if resolved is None:
            return argv
        unwrapped = _unwrap_npm_cmd_shim(resolved)
        if unwrapped is not None:
            return [*unwrapped, *argv[1:]]
        return [resolved, *argv[1:]]

    # -- config -------------------------------------------------------------

    def _write_opencode_config(self) -> None:
        """Write the `local`-provider config to `self._config_path` -- a
        SIBLING of the workspace (mirrors `_log_path`), NEVER inside
        `/workspace`. Delivered to the `opencode` process via the
        `OPENCODE_CONFIG` env var (`_launch`), not by placing
        `opencode.json` in the cwd.

        FIX (post-Task-4-review Important finding #1): the original version
        wrote `opencode.json` directly into the agent-visible workspace --
        confirmed live to be a real problem: after a run, `ls` on the
        workspace showed `opencode.json` sitting alongside the task's own
        files. `run_oracle`'s diff constraint (Task 3,
        `llmtest.harness.tasks.run_oracle`) treats any new file not in
        `allowed_diff_paths` as an out-of-bounds edit -> that would
        silently FAIL an otherwise-correct run. Verified the env-var path
        works live: a probe run with NO `opencode.json` anywhere under the
        workspace, config supplied purely via `OPENCODE_CONFIG`, completed
        successfully end-to-end against the real gpt-oss-20b endpoint.

        CONTAINMENT (Wave 2): `webfetch` is DENIED (not "allow") when
        `self.sandbox_image` is set -- codex review requirement: even
        inside the container, an agent has no legitimate reason to reach
        the open internet for a B8 task, and `--network` isolation alone
        (deferred, see `_launch_container`'s docstring) doesn't cover this
        -- OpenCode's own permission layer does. Host mode is unchanged
        (`"allow"`), matching the pre-Wave-2 config shape exactly."""
        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "permission": {"edit": "allow", "bash": "allow",
                          "webfetch": "deny" if self.sandbox_image else "allow"},
            "provider": {
                "local": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Local llama-server",
                    "options": {"baseURL": f"{self.endpoint}/v1", "apiKey": "sk-local"},
                    "models": {self.model: {"name": f"{self.model} (local)"}},
                }
            },
        }
        # Native step-limit override (Wave 1a -- see `self.max_steps`'s own
        # docstring in `__init__`): only added when a caller actually asked
        # for one (`b8.budgets.steps` via `_DEFAULT_HARNESS_FACTORIES` in
        # `llmtest.batteries.b8_harness`) -- absent here, the config shape
        # is byte-for-byte what it was before this key existed.
        if self.max_steps is not None:
            cfg["agent"] = {"build": {"steps": self.max_steps}}
        self._config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # -- seam: subprocess launch (host today; Sandbox.run_in later) --------

    def _launch(self, argv: list[str], cwd: Path, timeout: float):
        """Launch `argv` in `cwd`, waiting up to `timeout` seconds.

        Returns `(returncode, timed_out, launch_error)`:
          - clean exit: `(code, False, None)`
          - hang: `(None, True, None)` -- the process is force-killed
            (`_kill_process_tree`) before returning, so a caller never has
            to separately remember to clean up the timeout case.
          - launch itself failed (binary missing, permissions, etc. -- an
            `OSError` from `Popen` before any process exists):
            `(None, False, <str(exception)>)`.

        stdout/stderr are redirected to `self._log_path` (a sibling of the
        workspace, never inside it) rather than piped -- see the module
        docstring for why piping deadlocks a verbose headless OpenCode run
        on Windows.

        The config is delivered via `OPENCODE_CONFIG` (not a workspace
        file -- see `_write_opencode_config`), on top of the full parent
        environment (`opencode`/node need normal PATH/USERPROFILE/etc. to
        run at all; this is additive, not a replacement).

        `self.process` is reset to `None` once this method has fully
        handled the process either way (clean exit OR killed-on-timeout)
        -- post-review trivial fix #5: leaving the handle set after
        `_launch` itself already resolved the outcome meant a later
        `teardown()` would re-run `taskkill` against an already-finished
        PID instead of being a true no-op.
        """
        try:
            log_fh = open(self._log_path, "w", encoding="utf-8")
        except OSError as e:
            return None, False, f"could not open log file: {e!r}"
        try:
            resolved_argv = self._resolved_argv(argv)
            env = {**os.environ, "OPENCODE_CONFIG": str(self._config_path)}
            try:
                self.process = subprocess.Popen(resolved_argv, cwd=str(cwd), stdout=log_fh,
                                                 stderr=subprocess.STDOUT, env=env)
            except OSError as e:
                return None, False, str(e)

            try:
                self.process.wait(timeout=timeout)
                returncode = self.process.returncode
                self.process = None
                return returncode, False, None
            except subprocess.TimeoutExpired:
                self._kill_process_tree(self.process)
                self.process = None
                return None, True, None
        finally:
            log_fh.close()

    def _kill_process_tree(self, process) -> None:
        """Force-kill `process` and its whole descendant tree (a headless
        OpenCode run can itself spawn children, e.g. for a `bash` tool call
        -- killing only the top-level PID would leave those running).
        Windows `taskkill /T` mirrors `llmtest.server.ServerManager.
        teardown`'s existing convention for exactly this reason. Errors
        (e.g. the process already exited) are swallowed -- this is a
        best-effort cleanup, not something that should itself raise."""
        subprocess.run(["taskkill", "/F", "/PID", str(process.pid), "/T"],
                       capture_output=True)

    # -- seam: containerized launch (Wave 2 CONTAINMENT) --------------------

    def _launch_container(self, prompt: str, timeout: float):
        """Run opencode HEADLESSLY inside a disposable `self.sandbox_image`
        container instead of a host subprocess -- see module docstring's
        CONTAINMENT section for the full design + the two live findings
        behind the extra `.local` tmpfs and the container-cwd `session.
        directory` shape. Mirrors `llmtest.harness.sandbox.Sandbox`'s own
        hardening conventions (`--read-only`, `--cap-drop ALL`,
        `--security-opt no-new-privileges`, `--pids-limit`) and its
        kill-then-rm-f-BY-NAME cleanup discipline for a host-side timeout
        (`Sandbox.hidden_validate`'s docstring: `--rm` alone only
        auto-removes a container that exits ON ITS OWN -- a host-side
        `subprocess.run` timeout kills only the LOCAL, ATTACHED `docker
        run` CLIENT, leaving the container itself orphaned unless force-
        removed by name).

        Isolation actually achieved here, each mapped to a concrete flag:
          - non-root, disposable: `--user node` (the image's own uid-1000
            user, verified live), `--rm --name <fresh per-call name>`.
          - read-only rootfs + scratch: `--read-only --tmpfs /tmp`.
          - writable HOME: `--tmpfs {_CONTAINER_HOME}:uid=1000,gid=1000,
            mode=1777` PLUS a second `--tmpfs {_CONTAINER_HOME}/.local:...`
            -- NOT decoration, live-confirmed required (module docstring
            finding #1: without it, opencode/bun's own `mkdir ~/.local/
            state` fails `EACCES` because Docker auto-creates that bind
            mount's intermediate dirs as root:root).
          - agent workspace, read-write, persisting to the HOST dir so
            `run_oracle`'s post-run diff check sees exactly what the
            containerized agent edited: `-v <workspace>:{_CONTAINER_
            WORKSPACE}:rw` + `--workdir {_CONTAINER_WORKSPACE}`.
          - FRESH per-run opencode-home (`setup()` already pointed
            `self.db_path`/`self._config_path` at this same host dir):
            `-v <self._opencode_home_dir>:{_CONTAINER_OPENCODE_HOME}:rw`.
          - endpoint reachable without host networking: `--add-host
            host.docker.internal:host-gateway` (verified live against the
            served gpt-oss-20b endpoint) -- `self.endpoint`/the config's
            `baseURL` were already rewritten to `host.docker.internal` in
            `setup()`; this flag is what makes that hostname resolve.
          - resource caps, mirroring `Sandbox.__enter__`: `--cpus`,
            `--memory`, `--pids-limit`, `--cap-drop ALL`, `--security-opt
            no-new-privileges`.
          - NO host env, no host-credential mounts, no docker.sock: the
            ONLY `-e` flags are `OPENCODE_CONFIG` (pointed at the config
            THIS adapter wrote, inside the opencode-home mount -- never
            `os.environ`) and `HOME`; the ONLY `-v` flags are the two
            above.

        Egress restriction (spec Global Constraints): deliberately NOT
        `--network none` here -- unlike `Sandbox`, this container must
        reach the live endpoint. A strict "endpoint-port-only" egress
        policy (blocking the open internet while still allowing
        `host.docker.internal:<port>`) is HARD on Docker Desktop (no
        first-class per-container egress ACL short of a custom network
        plugin or iptables rules the daemon itself would need to own) and
        is NOT implemented here -- `webfetch: "deny"` in the opencode
        config (`_write_opencode_config`) is the containment actually
        achieved for "no open-internet access via the agent's own tools";
        raw network egress from an arbitrary `bash` call is NOT blocked at
        the Docker layer. Documented as a follow-up, not silently assumed
        solved (see the Wave 2 report).

        The in-container `timeout -s KILL <n>` wrapping the opencode
        command mirrors `Sandbox.run_in`/`hidden_validate`'s own two-layer
        timeout discipline: PRIMARY wall-clock bound (kills opencode dead
        from inside the container on the clock), with the Python-side
        `subprocess.run(..., timeout=timeout + _CONTAINER_TIMEOUT_SLACK_S)`
        as a backstop for a hung docker daemon/CLI itself.

        Returns the SAME `(returncode, timed_out, launch_error)` shape
        `_launch` does, so `run()`'s outcome-handling needs no branch on
        which launch path produced it:
          - in-container `timeout -s KILL` fired: `docker run` exits with
            the wrapped command's own kill exit code (124 or 137 -- same
            check `Sandbox.hidden_validate` already uses); `--rm` already
            auto-removed the container (it exited on its own).
            `(returncode, True, None)`.
          - the docker CLI/daemon itself hung (backstop fired):
            `subprocess.run` raises `TimeoutExpired` -- the container,
            NOT auto-removed by `--rm` in this case, is force `docker
            kill` + `docker rm -f`'d by name (`_force_remove_container`).
            `(None, True, None)`.
          - clean exit: `(returncode, False, None)`.
          - `docker` itself isn't runnable (missing binary, etc.):
            `(None, False, <str(exception)>)`.
        """
        self._container_name = f"llmtest-b8-opencode-{uuid.uuid4().hex[:12]}"
        container_cmd = [
            "timeout", "-s", "KILL", str(int(timeout)),
            self.opencode_bin, "run", prompt, "-m", f"local/{self.model}",
            "--print-logs", "--log-level", "INFO",
        ]
        argv = [
            "docker", "run", "--rm", "--name", self._container_name,
            "--user", "node",
            "--read-only",
            "--tmpfs", "/tmp",
            "--tmpfs", f"{_CONTAINER_HOME}:uid=1000,gid=1000,mode=1777",
            "--tmpfs", f"{_CONTAINER_HOME}/.local:uid=1000,gid=1000,mode=1777",
            "-v", f"{self.workspace.resolve()}:{_CONTAINER_WORKSPACE}:rw",
            "-v", f"{self._opencode_home_dir.resolve()}:{_CONTAINER_OPENCODE_HOME}:rw",
            "--add-host", "host.docker.internal:host-gateway",
            "--cpus", str(self.cpus),
            "--memory", self.mem_limit,
            "--pids-limit", self.pids_limit,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "-e", f"OPENCODE_CONFIG={_CONTAINER_OPENCODE_HOME}/{_CONTAINER_CONFIG_FILENAME}",
            "-e", f"HOME={_CONTAINER_HOME}",
            "--workdir", _CONTAINER_WORKSPACE,
            self.sandbox_image,
        ] + container_cmd

        host_timeout = timeout + _CONTAINER_TIMEOUT_SLACK_S
        try:
            # LIVE FINDING (Wave 2 smoke, real container run): `text=True`
            # alone decodes with the PLATFORM default (cp1252 on this
            # Windows box), which crashes a `subprocess.run` background
            # reader thread the moment the container's combined
            # stdout/stderr contains a byte sequence cp1252 can't
            # represent -- confirmed live: OpenCode's own log output
            # includes UTF-8 characters (e.g. a "checkmark" prefix on a
            # completed write) that reproduced exactly this crash.
            # `encoding="utf-8"` (the container is Linux/Node, always
            # UTF-8) plus `errors="replace"` as a hard backstop -- this
            # captured output is only ever written to a debug log
            # (`_write_container_log`), never parsed for control flow, so
            # "never crash" matters far more here than "never lossy."
            r = subprocess.run(argv, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=host_timeout)
        except subprocess.TimeoutExpired:
            self._force_remove_container(self._container_name)
            self._container_name = None
            return None, True, None
        except OSError as e:
            self._container_name = None
            return None, False, str(e)

        self._write_container_log(r)
        self._container_name = None
        if r.returncode in (124, 137):
            # coreutils `timeout`'s own kill exit code (124) or the
            # container's PID 1 dying to SIGKILL propagated as the
            # container's exit code (137 = 128+9) -- either way the
            # in-container wall-clock bound fired. Same convention
            # `Sandbox.hidden_validate` already uses for its oracle
            # container.
            return r.returncode, True, None
        return r.returncode, False, None

    def _write_container_log(self, r: subprocess.CompletedProcess) -> None:
        """Best-effort mirror of host mode's log-to-a-sibling-file
        behavior (`_launch`) -- `docker run`'s captured stdout/stderr
        (opencode's own `--print-logs --log-level INFO` output) written to
        `self._log_path` for debugging, never allowed to raise (a log-
        write failure must not turn a real run's result into a crash)."""
        try:
            with open(self._log_path, "w", encoding="utf-8") as f:
                f.write(r.stdout or "")
                f.write(r.stderr or "")
        except OSError:
            pass

    def _force_remove_container(self, name: str) -> None:
        """Force-remove a (possibly still-running) named container --
        mirrors `Sandbox._force_remove_oracle_container`'s kill-then-rm-f-
        by-name discipline (this adapter has no docker-assigned container
        id up front, unlike a `docker run -d` container: opencode runs
        ATTACHED, so `--name` is the only handle available). `docker kill`
        is tried first (fast SIGKILL to the container's init process while
        it may still be running under the daemon after only the local CLI
        died), then `docker rm -f` (removes it whether `kill` found it
        still running or it had already exited on its own) -- both
        best-effort, errors swallowed, mirroring `Sandbox.__exit__`'s own
        cleanup-is-not-fatal discipline. Called from BOTH the timeout
        branch of `_launch_container` (the normal case this exists for)
        and `teardown()` (belt-and-suspenders for an interrupted run)."""
        subprocess.run(["docker", "kill", name], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)

    # -- seam: sqlite trace mapping ------------------------------------------

    def _read_trace(self, db_path: Path, workspace: Path, since_ts: int):
        """Read OpenCode's sqlite store and map this run's session into
        `(events, tokens_prompt, tokens_completion, subagent_spawned,
        last_finish, last_error, missing_usage)`.

        Never raises on a malformed/partial trace (a killed run's db is
        expected to be partial) -- every row is defensively parsed; a row
        that can't be understood is skipped or recorded as a failure
        event, never a crash. `run()` decides `terminal_status` from the
        LAUNCH outcome plus `last_finish`/`last_error`, not from this
        method, so a missing session here (e.g. opencode crashed before
        ever writing one) still produces a well-formed, empty-ish result
        rather than raising -- `run()`'s own returncode/timeout handling is
        what actually explains why nothing was found.
        """
        events: list[TraceEvent] = []
        tokens_prompt = 0
        tokens_completion = 0
        subagent_spawned = "no"
        last_finish = None
        last_error = None
        missing_usage = False

        if not Path(db_path).exists():
            return events, tokens_prompt, tokens_completion, subagent_spawned, last_finish, last_error, missing_usage

        ws = str(Path(workspace).resolve())
        con = sqlite3.connect(str(db_path))
        try:
            session_id = self._find_session_id(con, ws, since_ts)
            if session_id is None:
                return events, tokens_prompt, tokens_completion, subagent_spawned, last_finish, last_error, missing_usage

            cur = con.cursor()
            cur.execute("SELECT time_created, data FROM message WHERE session_id = ? "
                        "ORDER BY time_created, id", (session_id,))
            for _tc, raw in cur.fetchall():
                data = _safe_json(raw)
                if data is None or data.get("role") != "assistant":
                    continue
                tokens = data.get("tokens")
                if isinstance(tokens, dict):
                    tokens_prompt += _as_int(tokens.get("input"))
                    tokens_completion += _as_int(tokens.get("output"))
                else:
                    missing_usage = True
                # LAST-assistant-message-scoped (per the brief: "completed
                # ... the LAST assistant message has a finish"), not "seen
                # anywhere across the whole session": unconditionally
                # overwritten on every assistant message, including back to
                # None/absent, so a mid-session error that the model then
                # recovered from (a later assistant message that finished
                # cleanly) can't leave a stale `last_error` behind that
                # would otherwise misclassify a genuinely-completed run.
                last_finish = data.get("finish")
                last_error = data.get("error")

            cur.execute("SELECT time_created, data FROM part WHERE session_id = ? "
                        "ORDER BY time_created, id", (session_id,))
            for _tc, raw in cur.fetchall():
                data = _safe_json(raw)
                if data is None:
                    continue
                ptype = data.get("type")
                if ptype == "step-start":
                    events.append(TraceEvent(kind="turn", payload={}))
                elif ptype == "tool":
                    call_event, result_event, is_task = _parse_tool_part(data)
                    events.append(call_event)
                    events.append(result_event)
                    if is_task:
                        subagent_spawned = "yes"
                        events.append(TraceEvent(kind="subagent_spawn",
                                                  payload={"callID": data.get("callID"),
                                                           "tool": data.get("tool")}))
                # text/reasoning/step-finish parts: assistant output, not
                # separately represented in Trace (see module docstring).
        finally:
            con.close()

        return events, tokens_prompt, tokens_completion, subagent_spawned, last_finish, last_error, missing_usage

    @staticmethod
    def _find_session_id(con: sqlite3.Connection, workspace: str, since_ts: int) -> str | None:
        """Correlate this run to ONE session, tried from most to least
        specific, among sessions created at/after `since_ts`:

          1. exact `directory` == `workspace` match.
          2. case-insensitive exact match (Windows drive-letter casing).
          3. `directory` is a filesystem ANCESTOR of `workspace`
             (`_is_ancestor_dir`) -- hardening for a hypothesized shape
             where OpenCode records a git-worktree ROOT instead of the
             literal cwd when the workspace sits inside a repo. NOTE: this
             was investigated live (a workspace nested one level inside a
             real `git init`-ed ancestor) and did NOT reproduce --
             `session.directory` was still the literal cwd; it was
             `project.worktree` (a DIFFERENT table) that held the git
             root. Kept anyway as defensive hardening for a shape not
             confirmed impossible in every OpenCode version/scenario.
          4. PRIMARY GUARANTEE -- if nothing above matched but at least one
             session exists in the time window, return the NEWEST one
             regardless of `directory`. `run()` launches and synchronously
             awaits exactly one `opencode` subprocess per call, so under
             that assumption the newest session created in the post-launch
             window IS this run's. This is what stops a genuinely-completed
             run from being misclassified `infra-error` purely because
             directory correlation happens to miss (Task-4 review finding
             #2) -- `directory` matching is now a preference for
             disambiguating among candidates, not a hard gate on whether a
             session is found at all.

        Only `rows` being completely EMPTY (no session at all in the time
        window -- e.g. opencode crashed before ever writing one) still
        yields `None`; `run()`'s own launch-outcome handling (nonzero
        exit / launch_error) is what explains that case, not this method.
        """
        cur = con.cursor()
        cur.execute("SELECT id, directory FROM session WHERE time_created >= ? "
                    "ORDER BY time_created DESC", (since_ts,))
        rows = cur.fetchall()
        if not rows:
            return None
        for sid, directory in rows:
            if directory == workspace:
                return sid
        norm_ws = os.path.normcase(os.path.normpath(workspace))
        for sid, directory in rows:
            if directory is not None and os.path.normcase(os.path.normpath(directory)) == norm_ws:
                return sid
        for sid, directory in rows:
            if directory is not None and _is_ancestor_dir(directory, workspace):
                return sid
        return rows[0][0]


def _rewrite_endpoint_for_container(base_url: str) -> str:
    """Rewrite a HOST-visible endpoint origin (e.g. `"http://127.0.0.1:
    8080"`, or `"http://127.0.0.1:9/"` -- an `EndpointHandle`/stub's
    `base_url` may or may not carry a trailing slash) to the same origin
    reachable from INSIDE a container via `host.docker.internal` --
    confirmed live (`docker run --add-host host.docker.internal:host-
    gateway ... curl http://host.docker.internal:8080/health` ->
    `{"status":"ok"}`, and a full real `opencode run` against it,
    end-to-end, via `_launch_container`).

    Only the HOSTNAME component is replaced -- scheme, port, path, query,
    and fragment all pass through unchanged, so a bare origin
    (`"http://127.0.0.1:8080"`) becomes `"http://host.docker.internal:
    8080"` and nothing else about the URL shape changes (no double
    slash, no dropped/duplicated port). Used from `OpenCodeAdapter.setup()`
    only when `sandbox_image` is set -- host mode never calls this, so a
    host-mode endpoint string is never touched."""
    parsed = urlsplit(base_url)
    netloc = f"host.docker.internal:{parsed.port}" if parsed.port else "host.docker.internal"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _is_ancestor_dir(candidate: str, workspace: str) -> bool:
    """True if `candidate` is `workspace` itself or a filesystem ANCESTOR
    of it (case-insensitive, Windows-path-safe comparison). Used by
    `OpenCodeAdapter._find_session_id` to accept a session whose recorded
    `directory` is a git-worktree root (or any other ancestor) rather than
    the literal workspace cwd."""
    cand = os.path.normcase(os.path.normpath(candidate))
    ws = os.path.normcase(os.path.normpath(workspace))
    return ws == cand or ws.startswith(cand + os.sep)


def _unwrap_npm_cmd_shim(cmd_path: str) -> list[str] | None:
    """If `cmd_path` is a standard npm-generated `.cmd`/`.bat` shim (the
    `@ECHO off ... "%dp0%\\node.exe" ... "%dp0%\\node_modules\\...\\bin\\
    <entry>" %*` shape `npm install -g` writes on Windows), return
    `[node_exe, entry_path]` so the caller can invoke the real entry point
    directly instead of through the shim -- see `OpenCodeAdapter.
    _resolved_argv`'s docstring for WHY (cmd.exe's command-line parsing
    breaks on a raw newline embedded in an argument, regardless of
    quoting, which a multi-line B8 prompt always contains).

    Returns `None` (never raises) for anything that doesn't match this
    exact shape -- a non-`.cmd`/`.bat` path, an unreadable file, or a
    `.cmd` file some OTHER tool (not an npm shim) installed is left alone
    and used via the normal resolved-path fallback instead of being
    force-unwrapped on a guess."""
    p = Path(cmd_path)
    if p.suffix.lower() not in (".cmd", ".bat"):
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(r'"%dp0%\\(node_modules\\[^"]+)"\s+%\*', text)
    if not m:
        return None
    dp0 = p.parent
    entry = dp0 / m.group(1)
    if not entry.exists():
        return None
    node_exe = dp0 / "node.exe"
    if not node_exe.exists():
        resolved_node = shutil.which("node")
        if resolved_node is None:
            return None
        node_exe = Path(resolved_node)
    return [str(node_exe), str(entry)]


def _safe_json(raw) -> dict | None:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _as_int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _parse_tool_part(data: dict) -> tuple[TraceEvent, TraceEvent, bool]:
    """Build the `tool_call`/`tool_result` event pair for one `type=="tool"`
    part, defensively -- a missing or malformed `state` (no `state` key at
    all, `state` not a dict, or `state.status` absent) is the
    malformed/failed-tool case this must record rather than crash on: it is
    reported as a failed tool_result with `status="error"`, not raised."""
    tool = data.get("tool")
    call_id = data.get("callID")
    state = data.get("state")
    if not isinstance(state, dict):
        state = {}
    tool_input = state.get("input")
    status = state.get("status")
    if status not in _TOOL_SUCCESS_STATUSES and status != "running":
        status = "error"
    call_event = TraceEvent(kind="tool_call", payload={
        "tool": tool, "callID": call_id, "input": tool_input,
    })
    result_event = TraceEvent(kind="tool_result", payload={
        "status": status, "output": state.get("output"),
    })
    return call_event, result_event, tool == "task"
