# Running the B8 2-model ranking on a Verda/DataCrunch Linux box

The B8 instrument (validity program W1–W5) is complete. The final *run* stalled
on the local Windows box: the llama-server's per-slot context-checkpoint cache
filled the KV under sustained OpenCode load until it crashed
(`GGML_ASSERT(logits != nullptr)`), giving escalating `infra-error`s; even after
the `--ctx-checkpoints 0` fix, ~29 % of runs still infra-errored from
Docker-Desktop/WSL2 container↔endpoint churn. This runbook moves the run to a
clean Linux datacenter box, where native Docker networking removes that churn
and the checkpoint fix removes the crash.

**This box only PRODUCES completion rows.** Classification + the final report run
**off-box** (they need the judge panel / agy creds), so the box needs no secrets.

---

## Cost (single small GPU is plenty — models are 20–26 B)

| box (`verda.py` instance_type) | $/hr | ~run cost (≈10 h) | notes |
|---|---|---|---|
| `1A100.40S.22V` (A100 40 GB) | **1.29** | **~$13** | cheapest that fits both models; FIN-02 |
| `1A100.22V` (A100 80 GB) | 1.79 | ~$18 | more headroom |
| `1RTXPRO6000.30V` (RTX PRO 6000, 96 GB) | 1.89 | ~$19 | the originally-requested box; **poll for availability** |
| `1H100.80S.32V` (H100 80 GB) | 3.25 | ~$32 | fastest, overkill |
| `1B200.30V` (B200, 180 GB) | 6.11 | ~$61 | Blackwell, big; overkill |

~10 h ≈ 2 models × 23 dev tasks × 5 replicates × ~2 min/run + setup. Without the
prism ngram speedup the decode is slower, so budget generously. Check balance:
`python verda.py balance`. **Provision only when steps 1–4 are ready to run
back-to-back** — idle GPU is wasted money.

---

## 0. Provision the box (`scratchpad/verda.py`)
```bash
python verda.py balance                    # confirm funds
python verda.py images                     # pick an Ubuntu 24.04 + CUDA image id
# RTX PRO 6000 is intermittently out of stock; poll:
python verda_probe.py                      # shows *RTXPRO6000* availability@location
```
Edit `verda.py`'s `DEPLOY` dict: set `instance_type` (table above), a `location_code`
where it's `AVAIL`, the CUDA `image`, and a bigger `os_volume` (≥ 200 GB for models),
then:
```bash
python verda.py deploy
python verda.py wait <instance-id>         # blocks until running + prints IP
```

## 1. Ship the repo (local Windows git-bash)
```bash
bash deploy/blackwell/pack_repo.sh         # -> tarball path in scratchpad
scp -i <key> <tarball> root@<ip>:/opt/b8/
ssh -i <key> root@<ip> 'mkdir -p /opt/b8 && tar xzf /opt/b8/llmtest-v2-b8.tgz -C /opt/b8'
```

## 2. Bootstrap (on the box)
```bash
cd /opt/b8/llmtest-v2 && sudo bash deploy/blackwell/bootstrap.sh
```
Installs Docker + NVIDIA toolkit, pulls `llama.cpp:server-cuda`, **verifies the
image has `--ctx-checkpoints`** (the fix depends on it), builds `b8-sandbox:1`,
and makes the venv. Re-runnable.

## 3. Fetch models (on the box)
```bash
sudo bash deploy/blackwell/fetch_models.sh     # ~27 GB; datacenter HF is fast
```
Downloads `openai/gpt-oss-20b` + `unsloth/gemma-4-26B-A4B-it-qat-GGUF` (both
native MXFP4) and symlinks stable names. **Probe HF throughput first** — a slow
host caps at ~4 MB/s (project memory); a datacenter box does ~200 MB/s.

## 4. Run the matrix (on the box)
```bash
bash deploy/blackwell/run_matrix.sh            # serves each model, runs task-by-task
```
Per-task health check restarts a dead server between tasks; `run_b8_local`'s 3×
infra-error retry bridges transients within a task. Prints a per-model
terminal-status + completion summary at the end. Long-running — use `tmux`/`nohup`.

## 5. Pull back + classify + report (OFF-BOX, local)
```bash
scp -i <key> -r root@<ip>:/opt/b8/llmtest-v2/results/rows-suite-v2.1.0.jsonl ./results/
scp -i <key> -r root@<ip>:/opt/b8/llmtest-v2/artifacts/b8_traces ./artifacts/
python scripts/classify_b8_local.py --suite suite-v2.1.0     # judge panel via agy
python scripts/p8_report.py                                  # regenerates REPORT
```
The report now **excludes `infra-error` runs from the completion denominator**
(codex eligibility rule) and surfaces the excluded count — see `_b8_group_stats`.

## 6. Teardown (STOP THE METER)
```bash
python verda.py delete <instance-id>       # verify status = deleted/404
python verda.py list                       # confirm empty
```

---

## Serving config (baked into `serve.sh`) — why it differs from the CLAUDE.md template
- `--ctx-checkpoints 0` — **the fix.** Disables the per-slot checkpoint cache that
  accumulated OpenCode's ~13 k-token prompts until the KV OOM-crashed.
- `--parallel 1` — single slot, no cross-request KV contention.
- `--network host` — llama.cpp binds the host's `0.0.0.0:8080`; OpenCode
  containers reach it at `host.docker.internal:8080` (host-gateway) unchanged —
  the OpenCodeAdapter's Linux path needs no edits.
- **No ngram spec-decode** — it's a prism-fork-only flag and only affects decode
  *speed*, never completion/oracle *outcome*, so the ranking is valid without it.
  To cut GPU-hours, build the prism fork on-box and add `--spec-type ngram-mod
  --spec-ngram-mod-n-match 32`.

## Eligibility rule (codex)
`infra-error` = harness/serving failure, **not** a model failure. It must never
count against a model: `run_b8_local` retries an infra-error cell up to 3× before
recording it, and the report computes k/N + Wilson over **eligible** runs only.
Clean Linux serving should make infra-errors rare-to-zero regardless.

## Pre-launch checklist (so the GPU meter never idles)
- [ ] `verda.py balance` covers the chosen box × ~10 h + margin
- [ ] tarball packed & SSH key ready (step 1)
- [ ] you can run steps 2→3→4 back-to-back once SSH is up
- [ ] plan to `verda.py delete` the instant step 4 finishes
