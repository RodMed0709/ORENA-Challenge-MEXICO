"""Rung 39 — the self-closing serial chain, RENDERED rather than committed.

Folder-private glue. Importable; a notebook cell calls ``write(cfg, ...)`` and prints the launch
line for the human. Nothing here is a launcher and nothing here is run by hand.

## Why the chain is rendered text and not a file in the repo

Committing a ``.sh`` at the repo root — or anywhere — is forbidden (CLAUDE.md, CONSTITUTION
§VIII.1, §IX.3), and ``.py`` files are libraries, never launchers. So the chain is **not a committed
file**: it is *rendered text*. ``render(cfg) -> str`` is an importable, folder-private library
function; a notebook cell calls ``write(cfg, Path("/workspace/tmp/rung39_chain.sh"))`` and prints the
``nohup bash …`` line for the human to run. The rendered script lives in ``/workspace/tmp/`` — pod
scratch, outside the repo, deleted when the run is done (CONSTITUTION §IX.1-2).

What is committed is the **renderer**, which is reproducible, reviewable and diffable. The artifact
it produces is not, and does not need to be: it is fully determined by this file plus ``cfg``.

## 🔑 The one deliberate deviation from ``ktchain.sh``

``ktchain.sh`` ran ``python`` on a hand-written driver in ``/workspace/tmp``. That is a launcher,
which CONSTITUTION §VIII.1 forbids. The sanctioned headless path is **papermill executing the
notebook itself** (spec §5b) — a *command*, not a launcher file — and it writes an output notebook
that IS the provenance record, carrying ``metadata.papermill.parameters``, per-cell status, and a
non-zero exit plus ``exception: True`` on failure.

That non-zero exit is exactly the mechanism the blocking gate uses to stop the chain: the gate
notebook raises, papermill exits non-zero, and the chain publishes the gate result and stops the pod
without spending a single training GPU-hour.

## Every other ``ktchain.sh`` property is reproduced, and each was bought by a failure

* **WAIT for the GPU before touching it.** Rung 38's eval owns the card. The chain polls the eval's
  log mtime AND ``nvidia-smi``'s compute-apps list, waits a bounded time, and **aborts loudly rather
  than sharing the card**.
* **Commit only paths OUTSIDE ``runs/``.** ``runs/`` is gitignored; two earlier pushes were lost to
  committing into it. The chain adds only ``RESULTS_*.csv`` / ``RESULTS_*.json`` at the experiment
  root.
* **Push to ``HEAD:main``, never ``main``.** The pod checkout is detached; ``git push origin main``
  cost rung 30 its push.
* **Three push retries** with ``git fetch`` + ``git rebase`` between them — the volume is shared and
  another pod can land a commit mid-run.
* **The API key is read from a FILE that is then deleted**, never inlined. The rendered text contains
  only the path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

EXP_REL = "experiments/39-connector-lora"

#: Where the rendered script and the key file live. Pod scratch — outside the repo, deleted when
#: done (CONSTITUTION §IX.1-2, and the user rule that /workspace root stays clean).
POD_TMP = PurePosixPath("/workspace/tmp")


@dataclass
class ChainConfig:
    #: The pod-side checkout: `/workspace/repo_rodri` or `/workspace/repo_leo`.
    repo_root: str = "/workspace/repo_rodri"
    pod_id: str = "y6h32tbhwhgxxe"
    kernel: str = "infer"
    run_tag: str = "39_connector_v1"
    venv_activate: str = "/workspace/envs/infer/bin/activate"

    #: 🔴 The key is READ from this file and the file is deleted afterwards. It never appears in the
    #: rendered text, so the script is safe to `cat`, to log and to paste into a review.
    key_file: str = "/workspace/tmp/.rung39key"

    log: str = "/workspace/tmp/rung39_chain.log"

    #: Rung 38's eval owns the GPU. The chain blocks on this log's mtime before touching the card.
    wait_log: str = "/workspace/tmp/leo_eval38_full.log"
    max_wait_min: int = 240

    commit_message: str = "results(39): connector-LoRA gate and arm"

    @property
    def script_path(self) -> PurePosixPath:
        return POD_TMP / "rung39_chain.sh"

    @property
    def exp_dir(self) -> str:
        return f"{self.repo_root}/{EXP_REL}"


def render(cfg: ChainConfig | None = None) -> str:
    """The chain, as text. Pure — no IO — so it is reviewable and diffable before it ever runs."""
    cfg = cfg or ChainConfig()
    return f"""#!/bin/bash
# ===================================================================================
# Rung 39 - connector LoRA. RENDERED by {EXP_REL}/_tools/chain.py.
# POD SCRATCH, NOT A REPO FILE. Delete it when the run is done (CONSTITUTION IX.1-2).
# Regenerate with: import chain; chain.write(chain.ChainConfig(...))
# ===================================================================================
exec >> {cfg.log} 2>&1
echo "===== chain start $(date -u) ====="
source {cfg.venv_activate}
cd {cfg.exp_dir} || exit 3
git -C {cfg.repo_root} rev-parse --short HEAD

# --- 0. WAIT for the GPU: rung 38's eval owns it -----------------------------------
# Two independent signals, because either alone is fooled: a log can stall while the
# job lives, and nvidia-smi can be momentarily empty between phases. Bounded wait, and
# ABORT LOUDLY rather than share the card - two jobs on one L40S is an OOM, not a queue.
WAITED=0
while [ $WAITED -lt {cfg.max_wait_min} ]; do
  BUSY=0
  if [ -f {cfg.wait_log} ]; then
    M1=$(stat -c %Y {cfg.wait_log})
  else
    M1=0
  fi
  sleep 60
  WAITED=$((WAITED + 1))
  if [ -f {cfg.wait_log} ]; then
    M2=$(stat -c %Y {cfg.wait_log})
  else
    M2=0
  fi
  if [ "$M1" != "$M2" ]; then BUSY=1; fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then BUSY=1; fi
  if [ $BUSY -eq 0 ]; then break; fi
  echo "GPU busy (rung-38 eval) - waited $WAITED min"
done
if [ $WAITED -ge {cfg.max_wait_min} ]; then
  echo "ABORT: the GPU was still busy after {cfg.max_wait_min} min. Not sharing the card."
  exit 2
fi
echo "GPU free after $WAITED min"

# --- 1. GATE (BLOCKING) ------------------------------------------------------------
# The gate notebook RAISES on failure; papermill turns that into a non-zero exit and
# THAT is the mechanism. A failed gate is a publishable result, not an obstacle.
mkdir -p runs/{cfg.run_tag}
papermill 00_connector_gate.ipynb runs/{cfg.run_tag}/gate.ipynb -k {cfg.kernel} --log-output
GATE=$?
echo "gate rc=$GATE"

if [ $GATE -ne 0 ]; then
  echo "GATE FAILED -> no training. The connector is unreachable even when named explicitly."
  MSG="results(39): reachability gate FAILED - the connector is unreachable even when named explicitly"
else
  # --- 2. ARM: smoke, then full (build -> smoke -> full, CONSTITUTION VIII.6) -------
  papermill 01_connector_arm.ipynb runs/{cfg.run_tag}/arm_smoke.ipynb -p SMOKE True -k {cfg.kernel} --log-output
  SMOKE_RC=$?
  echo "arm smoke rc=$SMOKE_RC"
  if [ $SMOKE_RC -ne 0 ]; then
    echo "SMOKE FAILED -> no full run."
    MSG="results(39): gate PASSED, arm smoke failed - no full run"
  else
    papermill 01_connector_arm.ipynb runs/{cfg.run_tag}/arm_full.ipynb -p SMOKE False -k {cfg.kernel} --log-output
    ARM_RC=$?
    echo "arm full rc=$ARM_RC"
    MSG="{cfg.commit_message} (gate rc=0, arm rc=$ARM_RC)"
  fi
fi

# --- 3. commit ONLY paths outside the gitignored run dirs --------------------------
# Two earlier pushes were lost by staging generated output that git was ignoring, so the
# add list is an explicit allow-list of small artifacts at the experiment root.
cd {cfg.repo_root} || exit 3
git config user.name RodMed0709
git config user.email medrod2010@hotmail.com
for f in {EXP_REL}/RESULTS_*.csv {EXP_REL}/RESULTS_*.json; do
  if [ -f "$f" ]; then git add -f "$f"; fi
done
git commit -q -m "$MSG" || echo "nothing to commit"

# --- 4. push to HEAD:main - the pod checkout is DETACHED --------------------------
# `git push origin main` on a detached HEAD pushes the stale local ref and cost rung 30
# its push. Three retries with fetch+rebase, because the volume is shared.
for i in 1 2 3; do
  git fetch -q origin && git rebase -q origin/main && git push -q origin HEAD:main && break
  echo "push retry $i"
  sleep 30
done
git log --oneline -1

# --- 5. stop the pod; the key is read from a file and the file is deleted ----------
K=$(cat {cfg.key_file})
curl -s -X POST -H "Authorization: Bearer $K" https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop
rm -f {cfg.key_file}
echo "===== chain end $(date -u) ====="
"""


def write(cfg: ChainConfig | None = None, path: str | Path | None = None) -> str:
    """Render the chain into pod scratch and return the path. NEVER into the repo.

    RAISES if asked to write anywhere under a repo checkout: a committed ``.sh`` is exactly what this
    module exists to avoid, and the cheapest way to violate that is a convenient default.
    """
    cfg = cfg or ChainConfig()
    target = Path(str(path)) if path is not None else Path(str(cfg.script_path))
    posix = target.as_posix()
    if cfg.repo_root.rstrip("/") in posix or "/repo" in posix:
        raise AssertionError(
            f"refusing to write the chain to {posix} — it is inside a repo checkout. The chain is "
            f"pod scratch and lives under {POD_TMP} (CONSTITUTION §VIII.1, §IX.3)."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(cfg), encoding="utf-8")
    return str(target)


def launch_command(cfg: ChainConfig | None = None, path: str | Path | None = None) -> str:
    """The line the HUMAN runs. The notebook prints it; the notebook does not shell out."""
    cfg = cfg or ChainConfig()
    target = str(path) if path is not None else str(cfg.script_path)
    return (
        f"# 1) put the RunPod API key in {cfg.key_file} (it is deleted by the chain)\n"
        f"# 2) nohup bash {target} > /dev/null 2>&1 &\n"
        f"# 3) tail -f {cfg.log}"
    )


__all__ = ["ChainConfig", "EXP_REL", "POD_TMP", "launch_command", "render", "write"]
