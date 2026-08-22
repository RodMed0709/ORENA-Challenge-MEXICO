"""Rung 19b — wait for training to end, then run the epoch sweep. Folder-private glue.

This is the ONE thing in this rung that is a launcher, and it is one on purpose: the
sweep has to start at ~03:35 on a box nobody will be watching. It runs `papermill` over
`01_eval_epochs_full6252.ipynb`, which is where the config and the gates live — this file
decides only WHEN, never WHAT.

    nohup <env>/bin/python chain_eval.py > /dev/null 2>&1 &

## What "training finished" means here, and why three signals

`--save_strategy epoch` writes `checkpoint-6295` and the trainer then appends its
end-of-run trailer. Neither alone is enough:

* **the trailer only** — `train_runtime` in `logging.jsonl` — says the trainer returned,
  but the adapter write may still be in flight;
* **the checkpoint only** says a file exists, and `safetensors` will happily load a
  truncated file it can parse (rung 47's scar, `assert_checkpoint_complete`);
* **the process only** could mean it died at step 6,000 with rc≠0, and a chain that runs
  the eval on a crashed run scores four epochs and calls the fifth missing.

So: trailer AND non-empty adapter AND the `swift/cli/sft.py` process gone. Then a grace
pause before the first read, because the last two are checked from a different process.

🔴 **It never kills anything and never touches the training.** If the wait times out it
writes why and exits — a chain that "helpfully" proceeds on a half-finished run turns a
recoverable delay into a bad number that looks fine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORK = Path("/mnt/storage/uaq_user/rung19b")
RUN = "19b_merged_ep5_v1"
REPO = Path("/mnt/storage/uaq_user/repo_leo")
NB = REPO / "experiments" / "19b-external-recognition" / "01_eval_epochs_full6252.ipynb"
PY = Path("/mnt/storage/uaq_user/envs/orena-train/bin/python")

FINAL_STEP = 6295
LOG = WORK / "code" / "chain_eval.log"
STATUS = WORK / "code" / "chain_eval_status.json"
NB_OUT = WORK / "runs" / RUN / "nb"

POLL_S = 120
MAX_WAIT_H = 12          # ~5.4 h of training left at launch; 12 h is slack, not a guess
GRACE_S = 180            # let the final adapter write settle before anything reads it

# The smoke runs the two epochs the rung is actually read on, so it exercises BOTH the
# cross-arm pairing (ep4 has a rung-47 control) and the adjacent-epoch curve. One epoch
# would leave the curve path untested until the full run, which is when it costs hours.
SMOKE_EPOCHS = [4, 5]
FULL_EPOCHS = [1, 2, 3, 4, 5]


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(**kw) -> None:
    kw["updated"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    STATUS.write_text(json.dumps(kw, indent=1) + "\n", encoding="utf-8")


def ckpt_root() -> Path:
    hits = sorted((WORK / "runs" / RUN / "ckpt").glob("v0-*"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one v0-* under ckpt/, found {len(hits)}")
    return hits[0]


def training_state(root: Path) -> dict:
    """The three signals, read fresh. Never cached — that is the whole point."""
    adapter = root / f"checkpoint-{FINAL_STEP}" / "adapter_model.safetensors"
    logfile = root / "logging.jsonl"

    trailer, last_step = False, None
    if logfile.exists():
        for ln in logfile.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue                      # a line mid-write is not a finding
            if "train_runtime" in r:
                trailer = True
            s = r.get("global_step/max_steps")
            if isinstance(s, str) and "/" in s:
                last_step = int(s.split("/")[0])

    alive = subprocess.run(
        ["pgrep", "-f", f"swift/cli/sft.py.*{RUN}"], capture_output=True, text=True
    ).returncode == 0

    return {
        "trailer": trailer,
        "adapter_bytes": adapter.stat().st_size if adapter.exists() else 0,
        "process_alive": alive,
        "last_step": last_step,
    }


def wait_for_training(root: Path) -> dict:
    deadline = time.time() + MAX_WAIT_H * 3600
    log(f"waiting for training to finish (max {MAX_WAIT_H} h) — {root}")
    while time.time() < deadline:
        st = training_state(root)
        done = st["trailer"] and st["adapter_bytes"] > 0 and not st["process_alive"]
        write_status(phase="waiting", **st)
        if done:
            log(f"training finished — step {st['last_step']}, adapter "
                f"{st['adapter_bytes'] / 1e6:.0f} MB")
            log(f"grace pause {GRACE_S}s before reading the last checkpoint")
            time.sleep(GRACE_S)
            return st
        # A dead process with no trailer is a CRASH, not a finish. Say so and stop:
        # the eval would silently score whatever epochs happened to land.
        if not st["process_alive"] and not st["trailer"]:
            write_status(phase="ABORTED", reason="process gone without a trailer", **st)
            raise SystemExit(
                f"training process is gone but wrote no end-of-run trailer (last step "
                f"{st['last_step']} of {FINAL_STEP}). This is a crash, not a finish — "
                "the sweep is NOT run. Read the log and decide by hand."
            )
        time.sleep(POLL_S)
    write_status(phase="ABORTED", reason=f"timed out after {MAX_WAIT_H} h")
    raise SystemExit(f"training did not finish within {MAX_WAIT_H} h — sweep not run")


def pick_gpu() -> str:
    """GPU 0 is this rung's own and frees itself. Fall back to 1 if someone took it.

    Not clever on purpose: an eval that lands on a busy card OOMs an hour in, and this
    box is shared ([[prepare-local-ask-before-upload]]).
    """
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    used = {}
    for ln in q.stdout.strip().splitlines():
        i, m = (x.strip() for x in ln.split(","))
        used[i] = int(m)
    for i in ("0", "1"):
        if used.get(i, 10**9) < 2048:
            log(f"GPU {i} is free ({used.get(i)} MiB used) — using it")
            return i
    raise SystemExit(f"no free GPU: {used}. The sweep is not run.")


def papermill(tag: str, params: dict, gpu: str) -> None:
    NB_OUT.mkdir(parents=True, exist_ok=True)
    out = NB_OUT / f"{tag}_{datetime.now():%Y%m%d-%H%M%S}.ipynb"
    # 🔴 One `-y` payload, not a pile of `-p`. `-p` passes every value as a STRING, so
    # `-p SMOKE False` injects the string "False", which is TRUTHY — the full sweep would
    # run in smoke mode and write a 40-row table as if it were the 6,252. JSON is a YAML
    # subset, so `json.dumps` gives correct types for bools, ints and lists alike.
    yml = "\n".join(f"{k}: {json.dumps(v)}" for k, v in params.items())
    cmd = [str(PY), "-m", "papermill", str(NB), str(out),
           "--log-output", "--no-progress-bar", "-y", yml]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               HF_HOME="/mnt/storage/uaq_user/hf_cache",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log(f"[{tag}] {' '.join(cmd)}")
    write_status(phase=tag, notebook=str(out), gpu=gpu)
    t0 = time.perf_counter()
    with open(LOG, "a", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    mins = (time.perf_counter() - t0) / 60
    if p.returncode != 0:
        write_status(phase="FAILED", stage=tag, notebook=str(out), rc=p.returncode)
        raise SystemExit(f"[{tag}] papermill rc={p.returncode} after {mins:.0f} min "
                         f"— see {out} for the cell that raised. Full sweep NOT run.")
    log(f"[{tag}] OK in {mins:.0f} min -> {out}")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("rung 19b chain: wait -> smoke -> full sweep on the 6,252")
    root = ckpt_root()
    wait_for_training(root)
    gpu = pick_gpu()

    # 🔴 Smoke first, always. It is ~10 min against ~3 h, and every gate in the notebook
    # RAISES — a corpus-legality or judge failure found at 04:00 costs nothing, the same
    # failure found at 07:00 costs the night (CONSTITUTION: build -> smoke -> full).
    papermill("smoke", {"SMOKE": True, "EPOCHS": SMOKE_EPOCHS, "N_BOOT": 200}, gpu)
    papermill("full", {"SMOKE": False, "EPOCHS": FULL_EPOCHS, "N_BOOT": 4000}, gpu)

    write_status(phase="DONE", epochs=FULL_EPOCHS, gpu=gpu)
    log("chain complete — RESULTS_*.csv written under "
        f"{REPO / 'experiments' / '19b-external-recognition'}")


if __name__ == "__main__":
    main()
