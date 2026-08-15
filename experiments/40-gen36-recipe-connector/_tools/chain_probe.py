"""Rung 40 — render the thinking-probe chain. NEVER a committed `.sh`.

Two GPU arms in SERIES, because the volume cannot hold two 52 GB merges: on 2026-08-15
it had ~43 GB free with a teammate writing frames to the same quota.

    1. conn4e5 + thinking   — its merge already exists, so nothing is written first
    2. delete conn4e5's merge                       (+52 GB)
    3. rebuild alpha16's merge from its 383 MB adapter
    4. alpha16 + thinking
    5. delete alpha16's merge

Both adapters are kept throughout; either merge regenerates in ~10 min, which is the
trade `remerge.py` encodes. Peak disk stays at ONE merge, exactly as the arm chain does.

🔴 The trap and the watchdog are IMPORTED from `chain40`, not re-written. That machinery
is the only reason a $2.09/h pod cannot be left billing, and the fastest way to lose it
is a second copy that drifts. `chain40._shutdown_block` and `chain40.render_watchdog`
carry the retries, the status read-back and the setsid detachment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from chain40 import ChainConfig, _shutdown_block, render_watchdog  # noqa: F401


@dataclass
class ProbeChainConfig(ChainConfig):
    """A ChainConfig whose *body* is the probe. Inherits every safety field verbatim."""

    script_path: str = "/workspace/tmp/leo_chain_probe.sh"
    log_path: str = "/workspace/tmp/leo_chain_probe.log"
    watchdog_path: str = "/workspace/tmp/leo_watchdog_probe.sh"
    watchdog_log: str = "/workspace/tmp/leo_watchdog_probe.log"

    n_subset: int = 625
    max_new_tokens: int = 512
    probe_seed: int = 42
    # A reasoning pass is 3-5x slower per question than a bare answer, and this runs two
    # arms plus a re-merge. 6 h is generous on purpose: the wall clock is a backstop, not
    # a schedule, and a probe that trips it has gone wrong in a way worth stopping.
    max_seconds: int = 6 * 3600


def render(cfg: ProbeChainConfig) -> str:
    if cfg.stop_pod_on_exit and not cfg.pod_id:
        raise AssertionError("pod_id is empty — the chain could not stop the pod")
    exp = f"{cfg.repo_root}/{cfg.exp_dir}"
    runs = f"{exp}/runs"
    B = f"{runs}/40_B_connector_v1"
    A = f"{runs}/40_A_alpha_v1"

    return f"""#!/usr/bin/env bash
# RENDERED by _tools/chain_probe.py — do not edit here, and do not commit this file.
set -uo pipefail
echo "===== thinking probe start $(date -u) ====="

{_shutdown_block(cfg)}

cd {exp} || exit 1
export HF_HOME={cfg.hf_home}

run_probe() {{  # run_probe <tag> <merged-dir> <run-name>
  echo "===== probe $3 start $(date -u) ====="
  {cfg.env_python} -m papermill 05_thinking_probe.ipynb "/workspace/tmp/leo_out_$1.ipynb" \\
    -p MERGED_DIR "$2" -p RUN_NAME "$3" -p WORK_DIR "{runs}" -p HF_HOME "{cfg.hf_home}" \\
    -p N_SUBSET {cfg.n_subset} -p SEED {cfg.probe_seed} \\
    -p MAX_NEW_TOKENS {cfg.max_new_tokens} --log-output
  local rc=$?
  echo "===== probe $3 end rc=$rc $(date -u) ====="
  return $rc
}}

echo "--- volume before ---"; du -sx /workspace 2>/dev/null | tail -1

# --- 1. conn4e5 + thinking (its merge already exists) ----------------------------
if [ ! -d "{B}/merged" ]; then
  echo "FATAL: {B}/merged is gone — rebuild it from its adapter before running this."
  exit 1
fi
run_probe probe_conn "{B}/merged" 40_B_conn4e5_think
RC1=$?
[ $RC1 -ne 0 ] && echo "conn4e5 thinking arm FAILED (rc=$RC1) — its partial output is on the volume."

# --- 2. reclaim before the rebuild ----------------------------------------------
# Unconditional, and BEFORE step 3 rather than after step 1's success: with ~43 GB free
# a 52 GB write cannot start, so this is the step that makes step 3 possible at all.
echo "removing conn4e5's merge (adapter kept — regenerable in ~10 min)"
rm -rf "{B}/merged"
du -sx /workspace 2>/dev/null | tail -1

# --- 3. rebuild alpha16's merge from its 383 MB adapter --------------------------
{cfg.env_python} - <<'PY' || exit 1
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, "{exp}/_tools")
from remerge import RemergeConfig, remerge
print(remerge(RemergeConfig(adapter_dir="{A}/ckpt/checkpoint-901",
                            out_dir="{A}/merged", hf_home="{cfg.hf_home}")))
PY

# --- 4. alpha16 + thinking -------------------------------------------------------
run_probe probe_alpha "{A}/merged" 40_A_alpha16_think
RC2=$?
[ $RC2 -ne 0 ] && echo "alpha16 thinking arm FAILED (rc=$RC2)"

# --- 5. leave the volume as we found it ------------------------------------------
echo "removing alpha16's merge (adapter kept)"
rm -rf "{A}/merged"
echo "--- volume after ---"; du -sx /workspace 2>/dev/null | tail -1

echo "===== thinking probe end rc1=$RC1 rc2=$RC2 $(date -u) ====="
"""


def write(cfg: ProbeChainConfig, path: str | Path | None = None) -> str:
    """Render into pod scratch. RAISES if aimed at a checkout — a committed `.sh` is
    exactly what this module exists to avoid."""
    target = PurePosixPath(str(path) if path is not None else cfg.script_path)
    posix = target.as_posix()
    if cfg.repo_root.rstrip("/") in posix or "/repo" in posix:
        raise AssertionError(f"refusing to write the chain to {posix} — inside a checkout.")
    Path(posix).parent.mkdir(parents=True, exist_ok=True)
    Path(posix).write_text(render(cfg))
    Path(posix).chmod(0o755)
    wd = Path(cfg.watchdog_path)
    wd.write_text(render_watchdog(cfg))
    wd.chmod(0o755)
    return posix
