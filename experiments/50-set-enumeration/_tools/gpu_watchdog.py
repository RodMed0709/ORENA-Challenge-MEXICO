"""Free the UNAM card the moment our run stops needing it — nobody else will reclaim it.

`context/UNAM_SERVER.md` §4: *"Free your card the moment the run ends — kill your own tmux
session rather than leaving an idle engine resident; there is no scheduler to reclaim it."*
A finished `swift sft` exits and releases the GPU on its own, but the **tmux session survives**,
which is what makes `tmux ls` read BUSY to a teammate long after the card is idle. And a run that
*hangs* holds ~19 GiB indefinitely with nobody watching.

🔴 **It kills BY RECORDED PID AND SESSION NAME ONLY.** It never scans for "whatever is holding a
GPU" — `UNAM_SERVER.md` §2 exists because a working 40 GB job was killed that way on 2026-08-17.
If the pid it was given is gone or belongs to someone else, it does nothing and says so.

Two triggers, both conservative:
  * the watched process has exited  -> tidy the tmux session, regenerate ESTADO.md
  * no new training step for `stall_minutes` -> the run is wedged: kill it and free the card,
    recording that it was a STALL and not a completion, so nobody scores a truncated run
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# Two shapes carry progress and BOTH must parse, because reading neither would leave the
# stall detector frozen at `None` and fire a false STALL on a perfectly healthy run — which is
# how this watchdog nearly killed arm A at 45 minutes. tqdm is the one that is always present.
STEP_RE = re.compile(r"global_step/max_steps'?:\s*'?(\d+)/(\d+)")
TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[")
STORAGE = Path("/mnt/storage/uaq_user")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # someone else's process — not ours to touch
        return False
    return True


def _last_step(log: Path) -> tuple[int, int] | None:
    try:
        tail = log.read_text(errors="ignore")[-200_000:]
    except OSError:
        return None
    best = None
    for rx in (STEP_RE, TQDM_RE):
        for m in rx.finditer(tail):
            if best is None or m.end() > best[0]:
                best = (m.end(), int(m.group(1)), int(m.group(2)))
    return (best[1], best[2]) if best else None


def _free(session: str, pid: int, reason: str, report: Path) -> None:
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(30):
                if not _alive(pid):
                    break
                time.sleep(2)
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as exc:
            print(f"could not signal {pid}: {exc}", flush=True)
    subprocess.run(["tmux", "kill-session", "-t", session], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(STORAGE / "estado.py")], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report.write_text(json.dumps({"reason": reason, "session": session, "pid": pid,
                                  "freed_at_epoch_seconds": int(time.time())}, indent=1))
    print(f"card released: {reason}", flush=True)


def _mtime(log: Path) -> float:
    try:
        return log.stat().st_mtime
    except OSError:
        return 0.0


def watch(session: str, pid: int, log: Path, report: Path, *,
          stall_minutes: int = 90, poll_seconds: int = 60) -> None:
    """Free the card on exit, or on a genuine stall. Biased HARD against a false positive.

    🔴 Liveness is *step changed* OR *log file was written*, never the step alone. `_train` reads
    the child's stdout line by line and tqdm updates in place with `\\r`, emitting `\\n` only
    occasionally — so a perfectly healthy run can show the same step for many minutes. Killing a
    20-hour job on that would cost far more than a card held an extra hour, which is why the
    default window is 90 minutes and not 45.
    """
    print(f"watchdog on session={session} pid={pid} log={log} stall={stall_minutes}m", flush=True)
    last, last_mt, last_change = _last_step(log), _mtime(log), time.time()
    while True:
        time.sleep(poll_seconds)
        if not _alive(pid):
            # give the process a moment to flush its final lines before we read them
            time.sleep(10)
            done = _last_step(log)
            finished = bool(done and done[0] >= done[1])
            _free(session, pid, "training exited (complete)" if finished
                  else f"training exited EARLY at step {done}", report)
            return
        now, mt = _last_step(log), _mtime(log)
        if now != last or mt != last_mt:
            last, last_mt, last_change = now, mt, time.time()
            continue
        if (time.time() - last_change) / 60 >= stall_minutes:
            _free(session, pid, f"STALLED at step {last} for {stall_minutes}m with no log write — "
                                "the card was held by a wedged run, NOT a finished one", report)
            return


if __name__ == "__main__":
    watch(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
          stall_minutes=int(sys.argv[5]) if len(sys.argv) > 5 else 45)
