"""Rung 40 — render the unattended chain. NEVER a committed `.sh`.

Importable renderer, following rung 39's `_tools/chain.py`. What is committed is
this module — reproducible, reviewable, diffable. What it produces is pod scratch
and is deleted when done. A committed `.sh` is forbidden (CLAUDE.md,
CONSTITUTION §VIII.1, §IX.3) and a `.py` launcher is forbidden too, so the chain is
*rendered* and handed to `nohup bash`.

The chain runs ~16 h with nobody awake. Four things it must survive, each of which
has already cost this project something:

* **the pod outliving the work.** A `trap ... EXIT` fires on ANY exit — normal end,
  error, `kill`, crash. "Stops when it finishes" is not the same as "stops no
  matter what", and only the second is safe when legokna is asleep.
* **the disk quota.** `df` LIES on this volume (it reports the MooseFS cluster,
  not the ~670 GB quota). A run already died `Disk quota exceeded` MID-MERGE, after
  training. So: rung 38's merge is deleted BEFORE anything starts, and arm A's is
  deleted before arm B begins.
* **losing 14 h of work to a late failure.** Commit after EACH eval, never once at
  the end. 📌 **Commit only, no push** (legokna's call): the network volume outlives
  the pod, so the results are recoverable from S3 without starting anything. That
  removes git auth as a failure mode entirely — a push that fails at 3 a.m. leaves
  the results trapped, and there is nothing a retry can do about a bad credential.
* **a HUNG run.** The `trap` fires on exit; a training that stalls never exits, so
  nothing fires and the pod bills all night. This is the most likely expensive
  failure — likelier than SIGKILL. The watchdog watches the log's **mtime**: the
  27B logs a line every ~24 s, so 45 minutes of silence is unambiguous.

The watchdog is rendered as a SEPARATE script and launched with `setsid nohup`, so
it is not a child of the chain. A child would die with a process-group kill, which
is exactly the case it exists to cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# The branch this rung lives on. NOT main: rung 40 is unfinished work and the
# branch is deliberately unmerged.
BRANCH = "task/gen36-recipe-and-connector"


@dataclass
class ChainConfig:
    pod_id: str = ""                      # filled in when legokna launches the pod
    repo_root: str = "/workspace/repo_leo"
    exp_dir: str = "experiments/40-gen36-recipe-connector"
    env_python: str = "/workspace/envs/unsloth/bin/python"
    key_file: str = "/workspace/tmp/leo_runpod_key"
    script_path: str = "/workspace/tmp/leo_chain40.sh"
    log_path: str = "/workspace/tmp/leo_chain40.log"

    # 51.8 GB of rung 38's merged 27B — closed, scored, and regenerable from its
    # adapter. Deleting it takes the volume from ~136 GB free to ~188 GB.
    stale_merge: str = (
        "/workspace/repo_leo/experiments/38-gen36-ft-screen/runs/38_qwen36_27b_v1/merged"
    )

    # 🔴 HF_HOME is EMPTY on the pod and there are TWO caches with different
    # contents: the 27B is in /workspace/hf_cache (61 G) and NOT in
    # /workspace/.cache/huggingface (7.6 G, the judge). Unset, `from_pretrained`
    # re-downloads 52 GB — hours lost and the disk plan broken. Resolved by
    # LOOKING, per [[pod-hf-home-is-not-set]], not by hardcoding a guess.
    hf_home: str = "/workspace/hf_cache"
    model_cache_dir: str = "models--Qwen--Qwen3.6-27B"

    watchdog_path: str = "/workspace/tmp/leo_watchdog40.sh"
    watchdog_log: str = "/workspace/tmp/leo_watchdog40.log"

    run_arm_b: bool = True                # False = stop after arm A and its eval
    expect_unsloth: str = "2026.8.15"
    expect_unsloth_zoo: str = "2026.8.10"

    # A stalled 27B is silent; a live one writes a line every ~24 s.
    stale_seconds: int = 2700             # 45 min without log output => hung
    max_seconds: int = 16 * 3600          # hard wall clock, last resort
    stop_retries: int = 5                 # the stop call itself can fail on a flaky link


def render(cfg: ChainConfig) -> str:
    if not cfg.pod_id:
        raise AssertionError(
            "pod_id is empty — the chain would run and then fail to stop the pod, "
            "which is the one failure this whole design exists to prevent."
        )
    exp = f"{cfg.repo_root}/{cfg.exp_dir}"
    arm_b_block = _arm_b(cfg, exp) if cfg.run_arm_b else (
        '\necho "=== arm B skipped by config (run_arm_b=False) ==="\n'
    )
    return f"""#!/usr/bin/env bash
# RENDERED by _tools/chain40.py — do not edit here, and do not commit this file.
set -uo pipefail          # NOT -e: a failing arm must still reach its commit and the trap
echo "===== chain40 start $(date -u) ====="

# --- 0. the pod stops on ANY exit ------------------------------------------------
# Registered FIRST, before anything can fail. Reads the key at fire time and deletes
# the file, so the credential never outlives the run.
K=$(cat {cfg.key_file})
stop_pod() {{
  echo "===== stopping pod {cfg.pod_id} — $(date -u) ====="
  # Retried: the stop call itself can fail on a flaky link, and a single 000 would
  # leave the pod billing all night. Verified by reading the status back, because a
  # 200 alone proves nothing (measured 2026-08-13 on the test pod).
  for i in $(seq 1 {cfg.stop_retries}); do
    C=$(curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
        -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop)
    echo "stop attempt $i -> HTTP $C"
    sleep 5
    S=$(curl -s -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id} | grep -o '"desiredStatus":"[A-Z]*"')
    echo "  status now: $S"
    case "$S" in *EXITED*) echo "pod confirmed stopped"; break;; esac
    sleep 20
  done
  rm -f {cfg.key_file}
}}
trap stop_pod EXIT

# --- 0b. the watchdog: covers what the trap CANNOT ---------------------------------
# `trap` never fires on SIGKILL, and it never fires on a HUNG run because nothing
# exits. The watchdog is detached with setsid so a process-group kill cannot take it
# with the chain.
setsid nohup bash {cfg.watchdog_path} $$ > {cfg.watchdog_log} 2>&1 &
echo "watchdog detached (pid guard on $$) -> {cfg.watchdog_log}"

cd {exp} || exit 1

# --- 0c. HF_HOME, and PROOF the model is really in it ------------------------------
# Exported before anything imports transformers. The assert exists because the
# failure mode is silent and expensive: an unset HF_HOME does not error, it
# re-downloads 52 GB.
export HF_HOME={cfg.hf_home}
if [ ! -d "{cfg.hf_home}/hub/{cfg.model_cache_dir}" ]; then
  echo "FATAL: {cfg.model_cache_dir} is not in {cfg.hf_home}/hub — training would"
  echo "re-download 52 GB and blow the disk plan. Fix HF_HOME before relaunching."
  exit 1
fi
echo "HF_HOME={cfg.hf_home} — model present"

# --- 1. the versions the gates were measured against -----------------------------
# A mismatch means the merge gate and G4 do not transfer, so nothing below is
# interpretable. Fail here, before a 6 h arm.
{cfg.env_python} - <<'PY' || exit 1
import importlib.metadata as md, sys
want = {{"unsloth": "{cfg.expect_unsloth}", "unsloth_zoo": "{cfg.expect_unsloth_zoo}"}}
bad = {{p: md.version(p) for p in want if md.version(p) != want[p]}}
if bad:
    print("VERSION MISMATCH:", bad, "expected", want)
    sys.exit(1)
print("versions OK:", want)
PY

# --- 2. reclaim the quota BEFORE training ----------------------------------------
# `df` reports the MooseFS cluster and not the quota, so measure with du.
echo "--- volume before ---"; du -sx /workspace 2>/dev/null | tail -1
if [ -d "{cfg.stale_merge}" ]; then
  echo "removing rung 38's merged 27B (regenerable from its adapter)"
  rm -rf "{cfg.stale_merge}"
fi
echo "--- volume after ---";  du -sx /workspace 2>/dev/null | tail -1

# --- helpers ---------------------------------------------------------------------
# COMMIT ONLY, no push — legokna's call, and it is the right one. The network volume
# outlives the pod, so results are recoverable from S3 without starting anything.
# Pushing would add git auth as a failure mode at 3 a.m., and a retry cannot fix a
# bad credential. Nothing is lost: the commit and the artifacts are both on the volume.
commit_results() {{
  cd {cfg.repo_root} || return 1
  git add -A {cfg.exp_dir} 2>/dev/null
  git commit -q -m "$1" 2>/dev/null || echo "nothing to commit"
  git log --oneline -1
  cd {exp} || return 1
}}

run_nb() {{  # run_nb <notebook> <output-tag> [extra papermill args...]
  local nb="$1"; local tag="$2"; shift 2
  echo "===== $nb start $(date -u) ====="
  {cfg.env_python} -m papermill "$nb" "/workspace/tmp/leo_out_${{tag}}.ipynb" \\
    -p SMOKE False "$@" --log-output
  local rc=$?
  echo "===== $nb end rc=$rc $(date -u) ====="
  return $rc
}}

# --- 3. ARM A: lora_alpha 32 -> 16 -----------------------------------------------
run_nb 02_alpha_arm.ipynb arm_a
RC_A=$?
if [ $RC_A -ne 0 ]; then
  echo "ARM A FAILED (rc=$RC_A) — committing whatever it produced and stopping."
  commit_results "eval(40): arm A FAILED rc=$RC_A — committing the evidence"
  exit $RC_A
fi

run_nb 04_eval.ipynb eval_a -p MERGED_DIR "{cfg.repo_root}/{cfg.exp_dir}/runs/40_A_alpha_v1/merged" -p RUN_NAME 40_A_alpha_v1
commit_results "eval(40): arm A — lora_alpha 32->16, scored against rung 38 ep1"
{arm_b_block}
echo "===== chain40 end $(date -u) ====="
"""


def _arm_b(cfg: ChainConfig, exp: str) -> str:
    return f"""
# --- 4. reclaim arm A's merge before arm B ---------------------------------------
# Two 27B merges are ~104 GB and the volume has ~136 GB free. Sequential with a
# delete in between keeps the peak at ONE merge. Arm A is already scored and pushed
# at this point, so its merge is regenerable and expendable.
A_MERGED="{cfg.repo_root}/{cfg.exp_dir}/runs/40_A_alpha_v1/merged"
if [ -d "$A_MERGED" ]; then
  echo "removing arm A's merge (already scored and pushed)"
  rm -rf "$A_MERGED"
fi
du -sx /workspace 2>/dev/null | tail -1

# --- 5. ARM B: the connector trains, at 4e-5 -------------------------------------
run_nb 03_connector_arm.ipynb arm_b
RC_B=$?
if [ $RC_B -ne 0 ]; then
  echo "ARM B FAILED (rc=$RC_B) — committing the evidence."
  commit_results "eval(40): arm B FAILED rc=$RC_B — committing the evidence"
  exit $RC_B
fi

run_nb 04_eval.ipynb eval_b -p MERGED_DIR "{cfg.repo_root}/{cfg.exp_dir}/runs/40_B_connector_v1/merged" -p RUN_NAME 40_B_connector_v1
commit_results "eval(40): arm B — the connector trains at 4e-5, scored against rung 38 ep1"
"""


def render_watchdog(cfg: ChainConfig) -> str:
    """The watchdog. Detached from the chain, and it covers what the trap cannot.

    Three conditions, any of which stops the pod:

    * **the chain is gone** — `kill -0` on its PID fails. `trap ... EXIT` does NOT
      fire on SIGKILL, and the OOM killer uses SIGKILL. With a 27B that is not
      hypothetical.
    * **the log went stale** — a live run writes a line every ~24 s, so 45 minutes of
      silence means hung. The trap cannot help here: a hung process never exits, so
      nothing fires and the pod bills until morning. This is the likeliest expensive
      failure, and it is the reason this file exists.
    * **the wall clock** — last resort, if even the log hangs in some way we did not
      foresee.

    It stops the pod itself rather than signalling the chain: if the chain is hung or
    already dead, there is nobody left to ask.
    """
    if not cfg.pod_id:
        raise AssertionError("pod_id is empty — a watchdog that cannot stop the pod is decoration")
    return f"""#!/usr/bin/env bash
# RENDERED by _tools/chain40.py — do not edit here, and do not commit this file.
CHAIN_PID="${{1:-0}}"
START=$(date +%s)
K=$(cat {cfg.key_file})
echo "watchdog up — chain pid $CHAIN_PID, stale>{cfg.stale_seconds}s, wall>{cfg.max_seconds}s"

stop_now() {{
  echo "WATCHDOG STOPPING POD: $1 — $(date -u)"
  for i in $(seq 1 {cfg.stop_retries}); do
    C=$(curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
        -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop)
    echo "  stop attempt $i -> HTTP $C"
    sleep 5
    S=$(curl -s -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id} | grep -o '"desiredStatus":"[A-Z]*"')
    case "$S" in *EXITED*) echo "  confirmed stopped"; exit 0;; esac
    sleep 20
  done
  exit 0
}}

while true; do
  sleep 300

  # 1) the chain vanished without stopping the pod => SIGKILL / OOM
  if [ "$CHAIN_PID" != "0" ] && ! kill -0 "$CHAIN_PID" 2>/dev/null; then
    sleep 60   # grace: let a normal exit's own trap win the race
    S=$(curl -s -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id} | grep -o '"desiredStatus":"[A-Z]*"')
    case "$S" in *EXITED*) echo "chain ended and the pod is already stopping — done"; exit 0;; esac
    stop_now "chain pid $CHAIN_PID is gone and the pod is still up"
  fi

  # 2) the log went stale => hung
  if [ -f "{cfg.log_path}" ]; then
    AGE=$(( $(date +%s) - $(stat -c %Y "{cfg.log_path}") ))
    if [ "$AGE" -gt {cfg.stale_seconds} ]; then
      stop_now "log silent for ${{AGE}}s (a live 27B writes every ~24s)"
    fi
  fi

  # 3) wall clock
  if [ $(( $(date +%s) - START )) -gt {cfg.max_seconds} ]; then
    stop_now "wall clock exceeded"
  fi
done
"""


def write(cfg: ChainConfig, path: str | Path | None = None) -> str:
    """Render into pod scratch and return the path. RAISES if aimed at a checkout.

    A committed `.sh` is exactly what this module exists to avoid, and the cheapest
    way to violate that is a convenient default.
    """
    target = PurePosixPath(str(path) if path is not None else cfg.script_path)
    posix = target.as_posix()
    if cfg.repo_root.rstrip("/") in posix or "/repo" in posix:
        raise AssertionError(
            f"refusing to write the chain to {posix} — it is inside a repo checkout."
        )
    Path(posix).parent.mkdir(parents=True, exist_ok=True)
    Path(posix).write_text(render(cfg))
    Path(posix).chmod(0o755)
    # The watchdog goes beside it: a chain rendered without its watchdog would run
    # perfectly and leave the pod up on the two failures the trap cannot catch.
    wd = Path(cfg.watchdog_path)
    wd.parent.mkdir(parents=True, exist_ok=True)
    wd.write_text(render_watchdog(cfg))
    wd.chmod(0o755)
    return posix


def launch_line(cfg: ChainConfig) -> str:
    """The one line a human types. Printed, never executed by this module."""
    return (
        f"# 1) put the RunPod API key in {cfg.key_file} (the chain deletes it)\n"
        f"# 2) verify SSH works FIRST — it did not come up on a test pod on 2026-08-13\n"
        f"#    the chain launches {cfg.watchdog_path} itself; do not start it by hand\n"
        f"nohup bash {cfg.script_path} > {cfg.log_path} 2>&1 &\n"
        f"tail -f {cfg.log_path}"
    )
