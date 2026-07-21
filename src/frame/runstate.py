"""Observable run state — everything a watcher needs, on disk, never only on a screen.

Why this exists
---------------
A pod run outlives the terminal that started it. The operator leaves; the power may drop;
the SSH session dies. Anything knowable only from scrollback is lost, and "how far along is
it?" then gets answered with a guess. This module makes the answer a file read.

Four properties, each one load-bearing:

1. **Atomic.** `STATE.json` is written to a temp file, fsynced, then `os.replace`d. A reader
   polling at any instant sees either the old state or the new one — never a half-written
   file. Without this a watcher eventually parses a truncated JSON and concludes the run
   crashed.

2. **Durable.** Every write is fsynced before the rename. A power cut loses at most the
   update in flight, not the file. This is the difference between "the log said so" and
   "the disk says so".

3. **Append-only history.** `events.jsonl` records what happened and when, one flushed line
   per event. `STATE.json` answers *where are we now*; the history answers *how did we get
   here* after the fact, which is what a post-mortem needs and what scrollback cannot give.

4. 🔴 **Progress is MEASURED, not estimated.** `step`/`total_steps` come from the trainer's
   own counter, and the ETA is computed from the observed rate over a trailing window. When
   the underlying numbers are not available the fields are `null` — this module never
   invents a percentage. A confident wrong ETA is worse than an honest absent one.

The tqdm problem, and why it is not incidental
----------------------------------------------
`for line in proc.stdout` splits on ``\\n``. Progress bars emit ``\\r`` and no newline, so a
naive line loop shows **nothing** until the bar finishes — which is exactly the "the log only
updates every so often" complaint this module exists to kill. `iter_fragments` reads raw
bytes and splits on both, so a bar redraw is an event like any other.

Terminal contract for an automated watcher
------------------------------------------
`status` is one of ``pending | running | done | failed | aborted``. A watcher may shut the
machine down when ``finished`` is true, and **only** then. `safe_to_shutdown` is written
explicitly rather than inferred, so the decision to power off is never a guess about
whether some later stage was still coming.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

TERMINAL = ("done", "failed", "aborted")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    """Write, fsync, rename. A reader never observes a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # fsync the directory too, or the rename itself can be lost on power failure
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


# HF/swift tqdm: " 12%|#1        | 120/1000 [02:11<16:04,  1.10s/it]"
_TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+)")
# swift/transformers dict lines: {'loss': 1.23, 'epoch': 0.42, ...}
_KV_RE = re.compile(r"'(loss|eval_loss|epoch|learning_rate|grad_norm)':\s*"
                    r"(-?[\d.]+(?:[eE][-+]?\d+)?|nan|-?inf)")


def iter_fragments(stream) -> Iterator[str]:
    """Yield output fragments split on BOTH ``\\n`` and ``\\r``.

    A plain line loop hides progress bars entirely (they carriage-return in place), which is
    the single most common reason a long run looks frozen when it is fine.
    """
    buf = ""
    while True:
        chunk = stream.read(1)
        if not chunk:
            break
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", "replace")
        if chunk in ("\n", "\r"):
            if buf.strip():
                yield buf
            buf = ""
        else:
            buf += chunk
    if buf.strip():
        yield buf


def parse_progress(fragment: str) -> dict[str, Any]:
    """Pull measured numbers out of one output fragment. Absent → absent, never guessed."""
    out: dict[str, Any] = {}
    m = _TQDM_RE.search(fragment)
    if m:
        out["step"], out["total_steps"] = int(m.group(1)), int(m.group(2))
        out["elapsed_str"], out["trainer_eta_str"] = m.group(3), m.group(4)
    for k, v in _KV_RE.findall(fragment):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


@dataclass
class RunState:
    """The single file a watcher polls, plus the history it can reconstruct from."""

    root: Path
    pipeline: str
    stages: list[str]
    _stage: str | None = None
    _status: str = "pending"
    _data: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.time)
    _rate: deque = field(default_factory=lambda: deque(maxlen=64))
    _last_flush: float = 0.0
    _shutdown_ok: bool | None = None

    @property
    def state_path(self) -> Path:
        return self.root / "STATE.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    # ── history ──────────────────────────────────────────────────────────
    def event(self, kind: str, **payload: Any) -> None:
        """Append one durable line. Flushed and fsynced — the point is surviving a crash."""
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _utc(), "kind": kind, "stage": self._stage, **payload}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ── current state ────────────────────────────────────────────────────
    def _eta(self) -> tuple[float | None, str | None]:
        """Seconds remaining from the OBSERVED step rate, or (None, None).

        Deliberately independent of the trainer's own ETA, which resets between stages. A
        trailing window rather than a lifetime average, so a slow warm-up stops poisoning
        the estimate an hour later.
        """
        step, total = self._data.get("step"), self._data.get("total_steps")
        if not step or not total or len(self._rate) < 4:
            return None, None
        (t_old, s_old), (t_new, s_new) = self._rate[0], self._rate[-1]
        if s_new <= s_old or t_new <= t_old:
            return None, None
        rate = (s_new - s_old) / (t_new - t_old)  # steps per second
        remaining = (total - step) / rate
        eta = datetime.fromtimestamp(time.time() + remaining, timezone.utc)
        return round(remaining, 1), eta.isoformat(timespec="seconds")

    def flush(self, force: bool = False, min_interval: float = 2.0) -> None:
        """Rewrite STATE.json. Throttled so a fast progress bar does not thrash the disk."""
        now = time.time()
        if not force and (now - self._last_flush) < min_interval:
            return
        self._last_flush = now
        step, total = self._data.get("step"), self._data.get("total_steps")
        eta_s, eta_at = self._eta()
        finished = self._status in TERMINAL
        payload = {
            "pipeline": self.pipeline,
            "status": self._status,
            "finished": finished,
            # 🔴 written explicitly, never inferred by the watcher
            "safe_to_shutdown": (finished if self._shutdown_ok is None
                                 else self._shutdown_ok),
            "stage": self._stage,
            "stage_index": (self.stages.index(self._stage) + 1) if self._stage in self.stages else None,
            "stage_total": len(self.stages),
            "stages": self.stages,
            "step": step,
            "total_steps": total,
            "pct": round(100 * step / total, 2) if (step and total) else None,
            "eta_seconds": eta_s,
            "eta_utc": eta_at,
            "elapsed_seconds": round(now - self._t0, 1),
            "updated_utc": _utc(),
            "heartbeat_epoch": round(now, 3),
            **{k: v for k, v in self._data.items()
               if k in ("loss", "eval_loss", "epoch", "learning_rate", "note", "error")},
        }
        _atomic_write(self.state_path, json.dumps(payload, indent=1))

    # ── transitions ──────────────────────────────────────────────────────
    def start_stage(self, stage: str, **note: Any) -> None:
        self._stage, self._status = stage, "running"
        self._data = {}
        self._rate.clear()
        self.event("stage_start", **note)
        self.flush(force=True)

    def mark_stage(self, stage: str) -> None:
        """Record a stage as the current position WITHOUT emitting a start event.

        Used when resuming past an already-stamped stage: a fully-resumed run must still
        report where it got to, or the final state reads `stage: null` and a post-mortem
        cannot tell a completed pipeline from one that never began.
        """
        self._stage = stage

    def observe(self, fragment: str) -> dict[str, Any]:
        """Feed one output fragment; updates measured progress and the rate window.

        🔴 Flushes on EVERY fragment, not only on ones that parse. A checkpoint save or a
        kernel compile can print nothing parseable for minutes; if the heartbeat only moved
        on parsed lines, `is_stale` would call that healthy stretch a crash. Any output at
        all is evidence of life, and that is what the heartbeat must mean.
        """
        got = parse_progress(fragment)
        if got:
            self._data.update(got)
            if "step" in got:
                self._rate.append((time.time(), got["step"]))
        self.flush()
        return got

    def heartbeat(self, **note: Any) -> None:
        """Prove liveness from a stage that produces no output of its own.

        Frame extraction and map building are silent for long stretches; without this they
        are indistinguishable from a hang.
        """
        self._data.update(note)
        self.flush(force=True)

    def finish_stage(self, **note: Any) -> None:
        self.event("stage_done", **note)
        self.flush(force=True)

    def finish(self, status: str = "done", safe_to_shutdown: bool | None = None,
               **note: Any) -> None:
        """Close the run. `safe_to_shutdown` defaults to "any terminal state".

        🔴 Pass it explicitly to withhold permission after a crash: `finished` still becomes
        true so a watcher never waits forever, but powering the machine off is a separate
        decision from having stopped. They are only the same when the run's artifacts are
        already persisted off the machine.
        """
        if status not in TERMINAL:
            raise ValueError(f"{status!r} is not terminal — a watcher would never shut down")
        self._status = status
        self._shutdown_ok = bool(status in TERMINAL if safe_to_shutdown is None
                                 else safe_to_shutdown)
        self._data.update(note)
        self.event("pipeline_" + status, **note)
        self.flush(force=True)
        # a marker file, so a watcher can decide with a stat() and no JSON parsing at all
        _atomic_write(self.root / status.upper(), _utc() + "\n")


def read(root: Path | str) -> dict[str, Any]:
    """Read the current state. Safe against a concurrent write by construction."""
    return json.loads((Path(root) / "STATE.json").read_text(encoding="utf-8"))


def is_stale(root: Path | str, max_age_s: float = 300) -> bool:
    """Has the heartbeat stopped? Distinguishes 'crashed' from 'slow' — a watcher needs both.

    A finished run is never stale: its heartbeat legitimately stops.
    """
    st = read(root)
    if st.get("finished"):
        return False
    return (time.time() - st.get("heartbeat_epoch", 0)) > max_age_s


def summary(root: Path | str) -> str:
    """One line, everything a human or a watcher needs. Reads the file, guesses nothing."""
    try:
        s = read(root)
    except FileNotFoundError:
        return f"{root}: no STATE.json — the pipeline has not started"
    except json.JSONDecodeError:
        return f"{root}: STATE.json unreadable (should be impossible — writes are atomic)"
    bits = [f"[{s['status'].upper()}]",
            f"{s.get('stage')} ({s.get('stage_index')}/{s.get('stage_total')})"]
    if s.get("pct") is not None:
        bits.append(f"{s['pct']}% {s['step']}/{s['total_steps']}")
    if s.get("eta_seconds") is not None:
        bits.append(f"ETA {s['eta_seconds'] / 60:.1f} min (measured)")
    else:
        bits.append("ETA unavailable")  # never invented
    for k in ("loss", "eval_loss", "epoch"):
        if s.get(k) is not None:
            bits.append(f"{k}={s[k]}")
    age = time.time() - s.get("heartbeat_epoch", 0)
    bits.append(f"heartbeat {age:.0f}s ago" + (" ⚠️ STALE" if is_stale(root) else ""))
    if s["finished"]:
        bits.append("shutdown-ok" if s["safe_to_shutdown"] else "🔴 do NOT shut down")
    if s.get("error"):
        bits.append(f"error={s['error']}")
    return " | ".join(bits)


if __name__ == "__main__":  # `python -m frame.runstate <root>` — the operator's one-liner
    import sys as _sys

    print(summary(_sys.argv[1] if len(_sys.argv) > 1 else "."))
