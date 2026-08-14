"""Render an unattended pod chain as SHELL TEXT, and print a launch line for a human.

Promoted to `src/frame/` because `_tools/` is folder-private
(`EXPERIMENT_REPO_STRUCTURE_SPEC.md` §9) and two experiments now need this.

Every clause below is a scar. The comment on each says which one, so nobody
"simplifies" it back into the failure it prevents.

NOT a launcher: this module RENDERS text and returns a path. A notebook cell calls
`render()` and prints `launch_line`; a human runs that line. Nothing here shells out.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChainConfig:
    pod_id: str                              # NO DEFAULT. See render().
    stages: list[tuple[str, str]] = field(default_factory=list)  # (label, shell)
    workdir: Path = Path("/workspace/repo")
    log_path: Path = Path("/workspace/tmp/chain.log")
    key_file: Path = Path("/workspace/tmp/.rp_key")
    max_seconds: int = 20 * 3600             # WALL. Must exceed the measured budget.
    stop_retries: int = 5
    grace_seconds: int = 60
    env: dict[str, str] = field(default_factory=dict)
    commit_globs: list[str] = field(default_factory=list)   # explicit allow-list
    git_user: str = "orena-pod"
    git_email: str = "orena-pod@localhost"
    min_free_gb: float = 25.0
    require_versions: dict[str, str] = field(default_factory=dict)
    out: Path = Path("/workspace/tmp/chain.sh")
    watchdog: Path = Path("/workspace/tmp/watchdog.sh")


def _stop_pod_fn(cfg: ChainConfig) -> str:
    # L2: a 200 alone proves nothing — read desiredStatus back and only break on EXITED.
    return f"""
stop_pod() {{
  rc=$?
  echo "[trap] chain exiting rc=$rc — stopping pod {cfg.pod_id}"
  for i in $(seq 1 {cfg.stop_retries}); do
    curl -s -X POST "https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop" \\
         -H "Authorization: Bearer $RP_KEY" -o /dev/null
    sleep 5
    st=$(curl -s "https://rest.runpod.io/v1/pods/{cfg.pod_id}" \\
         -H "Authorization: Bearer $RP_KEY" | grep -o '"desiredStatus":"[A-Z]*"' | head -1)
    echo "[trap] attempt $i -> $st"
    case "$st" in *EXITED*) echo "[trap] pod stopped."; break;; esac
  done
  # NOTE: the key file is deliberately NOT deleted here. After a failed stop a human
  # needs it to retry by hand; deleting it removes the only credential that would help.
}}
"""


def _watchdog(cfg: ChainConfig) -> str:
    # L3 detached, L4 liveness+grace, L5 hard wall.
    # L6: NO stale-log detector. MooseFS merges/du/17.5GB shard writes are silent for
    # tens of minutes; a silence detector kills a healthy run and covers nothing that
    # L4+L5 do not already cover.
    return f"""#!/usr/bin/env bash
set -uo pipefail
CHAIN_PID="$1"
RP_KEY="$2"          # passed as an argument: the chain's trap may delete the file
START=$(date +%s)
while true; do
  sleep 60
  if ! kill -0 "$CHAIN_PID" 2>/dev/null; then
    # L4: trap ... EXIT never fires on SIGKILL, and the OOM killer uses SIGKILL.
    # Grace period lets a NORMAL exit's own trap win the race.
    sleep {cfg.grace_seconds}
    st=$(curl -s "https://rest.runpod.io/v1/pods/{cfg.pod_id}" \\
         -H "Authorization: Bearer $RP_KEY" | grep -o '"desiredStatus":"[A-Z]*"' | head -1)
    case "$st" in *EXITED*) exit 0;; esac
    echo "[wd] chain gone and pod alive — stopping"
    for i in $(seq 1 {cfg.stop_retries}); do
      curl -s -X POST "https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop" \\
           -H "Authorization: Bearer $RP_KEY" -o /dev/null; sleep 5
    done
    exit 0
  fi
  NOW=$(date +%s)
  if [ $((NOW-START)) -gt {cfg.max_seconds} ]; then
    echo "[wd] wall clock {cfg.max_seconds}s exceeded — stopping"
    for i in $(seq 1 {cfg.stop_retries}); do
      curl -s -X POST "https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop" \\
           -H "Authorization: Bearer $RP_KEY" -o /dev/null; sleep 5
    done
    exit 0
  fi
done
"""


def render(cfg: ChainConfig) -> dict:
    """Render chain + watchdog to disk. Returns paths and the launch line."""
    # L1: an empty pod_id is the one failure this whole design exists to prevent, and a
    # library DEFAULT is how it happens (rung 39 shipped a dead pod's id as a default).
    if not cfg.pod_id or not cfg.pod_id.strip():
        raise ValueError("pod_id is empty — refusing to render a chain that cannot stop its pod")
    if len(cfg.pod_id) < 8:
        raise ValueError(f"pod_id {cfg.pod_id!r} looks wrong — refusing")
    if not cfg.stages:
        raise ValueError("no stages")
    # never render into a repo checkout: a rendered chain is a temporary, not an artifact
    # (as_posix, because on Windows str(Path) uses backslashes and a substring test on
    #  "/repo/" silently never matches — caught by this module's own negative control)
    for p in (cfg.out, cfg.watchdog):
        parts = Path(p).parts
        if ".git" in parts or "repo" in parts or "experiments" in parts \
                or "experiments_segment" in parts:
            raise ValueError(f"refusing to render into a repo checkout: {p}")

    env = "\n".join(f'export {k}={shlex.quote(v)}' for k, v in cfg.env.items())
    vers = "\n".join(
        f'python -c "import {m}; assert {m}.__version__==\'{v}\', '
        f"f'{m} '+{m}.__version__+' != {v}'\" || {{ echo '[preflight] {m} version wrong'; exit 12; }}"
        for m, v in cfg.require_versions.items()
    )
    globs = " ".join(shlex.quote(g) for g in cfg.commit_globs) or "''"

    body = []
    for i, (label, sh) in enumerate(cfg.stages, 1):
        body.append(f"""
echo "=== stage {i}/{len(cfg.stages)}: {label} ==="
{sh}
STAGE_RC=$?
echo "[stage {i}] rc=$STAGE_RC"
# P6: with `set +e` a non-zero exit does NOT stop the chain by itself. The explicit
# test is the mechanism; without it a raising gate is only a log line.
if [ $STAGE_RC -ne 0 ]; then echo "[stage {i}] FAILED — stopping chain"; exit $STAGE_RC; fi
# L8: commit after EACH stage. A single commit at the end loses everything on a crash.
cd {cfg.workdir} && git add -- {globs} 2>/dev/null
git -C {cfg.workdir} commit -m "results(segment): stage {i} — {label}" 2>&1 | tail -2
""")

    chain = f"""#!/usr/bin/env bash
# L7: -u and pipefail, but NEVER -e — a failing arm must still reach its commit and trap.
set -uo pipefail
exec >> {cfg.log_path} 2>&1     # the script writes its OWN log; never rely on the caller
echo "=== chain start $(date -u +%FT%TZ) pod={cfg.pod_id} ==="

# L9 + trap-order fix: read the key and ASSERT IT IS NON-EMPTY *before* registering the
# trap. With `set +e` a missing file silently leaves RP_KEY="" and then NEITHER the trap
# NOR the watchdog can stop the pod — it bills until a human notices.
RP_KEY="$(cat {cfg.key_file} 2>/dev/null || true)"
if [ -z "$RP_KEY" ]; then echo "[fatal] no API key at {cfg.key_file}"; exit 10; fi
{_stop_pod_fn(cfg)}
trap stop_pod EXIT

# L3: detached watchdog, NOT a child — a process-group kill is exactly the case it covers.
# The key goes as an ARGUMENT so the watchdog survives any later deletion of the file.
setsid nohup bash {cfg.watchdog} $$ "$RP_KEY" >> {cfg.log_path}.wd 2>&1 &
echo "[init] watchdog detached, wall={cfg.max_seconds}s"

{env}

# ── preflight ────────────────────────────────────────────────────────
# P1: df LIES on this volume (it reports the whole MooseFS cluster, not our quota).
FREE_GB=$(python - <<'PYEOF'
import subprocess
used=int(subprocess.run(["du","-sx","/workspace"],capture_output=True,text=True).stdout.split()[0])
print(round(640 - used/1048576, 1))
PYEOF
)
echo "[preflight] free_gb=$FREE_GB (vs the ~640GB empirical wall, not df)"
python -c "import sys; sys.exit(0 if float('$FREE_GB') >= {cfg.min_free_gb} else 1)" \\
  || {{ echo "[preflight] only $FREE_GB GB free, need {cfg.min_free_gb}"; exit 11; }}

# P2: an unset HF_HOME does not error, it re-downloads 52 GB.
[ -n "${{HF_HOME:-}}" ] || {{ echo "[preflight] HF_HOME unset"; exit 13; }}
[ -d "$HF_HOME" ] || {{ echo "[preflight] HF_HOME $HF_HOME missing"; exit 13; }}

# P3: the documented installer has produced a CPU-only torch twice.
{vers}
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \\
print('[preflight] gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability())" \\
  || exit 14

# P5: two jobs on one GPU is an OOM, not a queue.
NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
echo "[preflight] compute apps already on this GPU: $NPROC"
[ "$NPROC" -eq 0 ] || {{ echo "[preflight] GPU busy — refusing to share"; exit 15; }}

# L8: without this, `git commit` fails on a fresh pod and `|| true` swallows it,
# leaving the results as untracked files nobody ever sees.
git -C {cfg.workdir} config user.name  {shlex.quote(cfg.git_user)}
git -C {cfg.workdir} config user.email {shlex.quote(cfg.git_email)}
echo "[init] HEAD $(git -C {cfg.workdir} rev-parse --short HEAD) on $(git -C {cfg.workdir} rev-parse --abbrev-ref HEAD)"
{chr(10).join(body)}
echo "=== chain done $(date -u +%FT%TZ) ==="
"""

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(chain, encoding="utf-8")
    cfg.watchdog.write_text(_watchdog(cfg), encoding="utf-8")
    return {
        "chain": str(cfg.out),
        "watchdog": str(cfg.watchdog),
        # printed for a human to run; this module never shells out
        "launch_line": f"setsid nohup bash {cfg.out} >> {cfg.log_path} 2>&1 & disown; "
                       f"echo launched $!",
    }
