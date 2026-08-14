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
* **losing 14 h of work to a late failure.** Commit and push after EACH eval, never
  once at the end.
* **the detached HEAD.** `git push origin <branch>` on the pod pushes the stale
  local ref — that cost rung 30 its push. Always `HEAD:<branch>`, with fetch+rebase
  retries because the volume is shared.
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

    run_arm_b: bool = True                # False = stop after arm A and its eval
    expect_unsloth: str = "2026.8.15"
    expect_unsloth_zoo: str = "2026.8.10"


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
  curl -s -o /dev/null -w "stop -> HTTP %{{http_code}}\\n" \\
    -X POST -H "Authorization: Bearer $K" \\
    https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop
  rm -f {cfg.key_file}
}}
trap stop_pod EXIT

cd {exp} || exit 1

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
# HEAD:{BRANCH} — the pod checkout is DETACHED, and a plain `git push origin
# <branch>` pushes the stale local ref. That cost rung 30 its push. Three tries with
# fetch+rebase because the volume is shared and teammates push in parallel.
push_results() {{
  cd {cfg.repo_root} || return 1
  git add -A {cfg.exp_dir} 2>/dev/null
  git commit -q -m "$1" 2>/dev/null || echo "nothing to commit"
  for i in 1 2 3; do
    git fetch -q origin && git rebase -q origin/{BRANCH} \\
      && git push -q origin HEAD:{BRANCH} && break
    echo "push retry $i"; sleep 30
  done
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
  push_results "eval(40): arm A FAILED rc=$RC_A — committing the evidence"
  exit $RC_A
fi

run_nb 04_eval.ipynb eval_a -p MERGED_DIR "{cfg.repo_root}/{cfg.exp_dir}/runs/40_A_alpha_v1/merged" -p RUN_NAME 40_A_alpha_v1
push_results "eval(40): arm A — lora_alpha 32->16, scored against rung 38 ep1"
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
  push_results "eval(40): arm B FAILED rc=$RC_B — committing the evidence"
  exit $RC_B
fi

run_nb 04_eval.ipynb eval_b -p MERGED_DIR "{cfg.repo_root}/{cfg.exp_dir}/runs/40_B_connector_v1/merged" -p RUN_NAME 40_B_connector_v1
push_results "eval(40): arm B — the connector trains at 4e-5, scored against rung 38 ep1"
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
    body = render(cfg)
    Path(posix).parent.mkdir(parents=True, exist_ok=True)
    Path(posix).write_text(body)
    Path(posix).chmod(0o755)
    return posix


def launch_line(cfg: ChainConfig) -> str:
    """The one line a human types. Printed, never executed by this module."""
    return (
        f"# 1) put the RunPod API key in {cfg.key_file} (the chain deletes it)\n"
        f"# 2) verify SSH works FIRST — it did not come up on a test pod on 2026-08-13\n"
        f"nohup bash {cfg.script_path} > {cfg.log_path} 2>&1 &\n"
        f"tail -f {cfg.log_path}"
    )
