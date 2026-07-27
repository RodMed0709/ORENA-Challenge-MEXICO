"""Probe 16a — can the model emit ZERO at all, and did our SFT take that away?

## Why this probe exists

Our training data contains **no negative example in any numeric or class format**:
0 of 2,495 per-class count golds are 0, 0 of 2,628 total-instance golds are 0, and 0 of
8,969 `fo_class` golds are `none`. Only `binary` carries negatives, and those are
co-occurrence questions about class PAIRS.

The obvious remedy — mint zero-count questions from the per-frame inventory — is expensive
and carries ~8.4% one-directional label noise. Before paying that, one measurement decides
whether it is even the right tool.

HoloCount (arXiv:2607.06420, Table 3, "Null-Target Prompting": 250 samples asking about a
concept absent from the image) reports **Qwen3-VL-8B at 96.4%** — ahead of Gemini-3-Flash
(62.4%) and Gemini-3.1-Pro (55.2%). Its verbatim finding: *"larger models become reluctant
to output zero, even when no target exists."*

So the capability plausibly ships in our backbone. If our LoRA lost it, the disease is our
own SFT's answer prior and the cure is regularisation — NOT 37k synthetic rows that also
push the count bias further down (we already sit at -0.66, and 8.4% of minted zeros would
be wrong *in the direction of zero*).

**The probe:** ask the base model and the fine-tuned checkpoint the same absent-class and
present-class questions on the same real frames, and compare.

## What is and is not measured

🔴 The ABSENT label is derived from the per-frame gold inventory under a **closure
assumption** — that the `fo_class` gold names every class present. Measured non-circularly
against the 1,008 `no` co-occurrence binaries: **0.00% false positives** (the list never
names an absent class) but **8.4% under-naming** (n=333; a second cross-check on
"are all objects the same class?" agrees at 10.7%). So roughly 1 in 12 ABSENT labels is
wrong, always in the same direction.

Therefore this module reports TWO things and they must not be conflated:

- **`emit_rate`** — how often the model emits `0` / `no` at all. **Noise-free**: it is a
  property of the model's output, not of the label. This is the primary reading and the
  one comparable to HoloCount.
- **`acc`** — agreement with the closure-derived label. Carries the ~8.4% noise. Read it
  for direction, never as a point estimate.

The PRESENT arm is the control that stops "always answer 0" from scoring well, per the
two-axis discipline (POROver, arXiv:2410.12999) — a model that wins ABSENT by collapsing
to zero loses PRESENT, and the harmonic mean catches it.
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Read the accepted class set from the SDK at runtime, never hard-coded — RULES §8b.
# The list the organizers put INSIDE the prompt and the scoring registry disagree on their
# 10th element (`foreign object` vs `Absorbable Hemostatic Agent`), and FOType.from_name()
# RAISES on an unrecognised token.
def scoring_vocab() -> list[str]:
    from focus.foreign_objects import FOType

    return list(FOType.names())


# Plurals for the question stems. Only classes actually observed in our batch are probed;
# `Mesh` and `Absorbable Hemostatic Agent` have zero examples anywhere, so we cannot build
# a PRESENT arm for them and they are excluded from the paired design (they are exactly the
# open-vocabulary case, which is a different probe).
PLURAL = {
    "Sponge": "Sponges",
    "Clip": "Clips",
    "Specimen Bag": "Specimen bags",
    "Silicone Loop": "Silicone loops",
    "External Drain": "External drains",
    "Needle": "Needles",
    "Gallstone": "Gallstones",
    "Specimen": "Specimens",
}

_NUM_Q = "How many {plural} are there in this frame?"
_BIN_Q = "Is there a {singular} in this frame? Please answer with yes or no."

_PERCLASS_RE = re.compile(r"how many\s+([A-Za-z ]+?)\s*(?:are|is|can|do|does|appear|visible|\?)", re.I)
_NCLASS_RE = re.compile(r"foreign object classes|different .*classes", re.I)
_TOTAL_RE = re.compile(r"foreign object instances", re.I)


@dataclass
class ProbeQ:
    """One probe question. `arm` is ABSENT or PRESENT; `expect_zero` follows from it."""

    frame_key: str
    frame_path: str
    dataset: str
    video_id: str
    cls: str
    arm: str  # "absent" | "present"
    fmt: str  # "number" | "binary"
    question: str
    named: tuple[str, ...] = field(default_factory=tuple)

    @property
    def distribution(self) -> str:
        # heico = the OOD half, lapchole = ID. Matches the qID-derived split used everywhere.
        return "OOD" if self.dataset == "heico" else "ID"


def _norm(name: str) -> str:
    return name.strip().lower().rstrip(".")


def answer_format(ref) -> str:
    """The SDK's `Reference` carries the format as `_format`, NOT `answer_format`.

    Reading the wrong attribute is silent: `getattr(ref, "answer_format", "")` returns "",
    every branch below misses, and the probe set comes out EMPTY rather than raising. That
    is exactly how the first smoke run burned a model load on 0 questions. Fail loudly here
    instead — an unknown schema is a bug, not an empty result.
    """
    for attr in ("_format", "answer_format"):
        val = getattr(ref, attr, None)
        if val:
            return str(val).lower()
    raise AttributeError(
        f"{type(ref).__name__} exposes neither `_format` nor `answer_format` — "
        "the SDK schema changed; fix this accessor rather than defaulting to empty."
    )


def build_inventory(items) -> dict[str, dict]:
    """Per-frame gold inventory: which classes the annotation asserts are present.

    Sources, both gold, never the model:
      * `fo_class` answers — the comma-separated class list
      * per-class `number` answers with a POSITIVE count

    Returns {frame_key: {"named": set[str], "dataset", "video_id", "frame_index"}}.
    """
    from frame.data import frame_cache_name

    vocab_lower = {_norm(c): c for c in PLURAL}
    inv: dict[str, dict] = {}
    for it in items:
        key = frame_cache_name(it)
        e = inv.setdefault(
            key,
            {
                "named": set(),
                "dataset": it.dataset,
                "video_id": it.video_id,
                "frame_index": it.frame_index,
            },
        )
        ref, req = it.reference, it.request
        fmt = answer_format(ref)
        ans = str(getattr(ref, "answer", "") or "")
        if "fo_class" in fmt:
            for tok in ans.split(","):
                c = vocab_lower.get(_norm(tok))
                if c:
                    e["named"].add(c)
        elif "number" in fmt:
            q = str(getattr(req, "question", "") or "")
            if _NCLASS_RE.search(q) or _TOTAL_RE.search(q):
                continue
            m = _PERCLASS_RE.search(q)
            if not m:
                continue
            c = vocab_lower.get(_norm(m.group(1)))
            try:
                n = int(ans.strip())
            except ValueError:
                continue
            if c and n > 0:
                e["named"].add(c)
    return inv


def build_probe_set(
    items,
    n_frames_per_cell: int = 60,
    seed: int = 0,
    formats: tuple[str, ...] = ("number", "binary"),
) -> list[ProbeQ]:
    """Balanced ABSENT/PRESENT probe over ID and OOD frames.

    One frame contributes at most ONE absent question and ONE present question per format,
    so no frame is over-represented. Frames are sampled per (distribution) cell to keep ID
    and OOD balanced — the headline is read per distribution, never pooled.
    """
    from frame.data import frame_cache_name  # noqa: F401  (documents the key's provenance)

    rng = random.Random(seed)
    inv = build_inventory(items)
    probe_classes = list(PLURAL)

    by_dist: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, e in inv.items():
        if not e["named"]:
            continue  # nothing asserted -> no usable ABSENT label under closure
        if len(e["named"]) >= len(probe_classes):
            continue  # no absent class to ask about
        by_dist["OOD" if e["dataset"] == "heico" else "ID"].append((key, e))

    out: list[ProbeQ] = []
    for dist, frames in by_dist.items():
        frames = sorted(frames, key=lambda kv: kv[0])  # deterministic before sampling
        rng.shuffle(frames)
        for key, e in frames[:n_frames_per_cell]:
            named = sorted(e["named"])
            absent_pool = [c for c in probe_classes if c not in e["named"]]
            if not absent_pool:
                continue
            c_absent = rng.choice(absent_pool)
            c_present = rng.choice(named)
            for cls, arm in ((c_absent, "absent"), (c_present, "present")):
                for fmt in formats:
                    q = (
                        _NUM_Q.format(plural=PLURAL[cls])
                        if fmt == "number"
                        else _BIN_Q.format(singular=cls.lower())
                    )
                    out.append(
                        ProbeQ(
                            frame_key=key,
                            frame_path=f"/workspace/frames_cache/{key}",
                            dataset=e["dataset"],
                            video_id=e["video_id"],
                            cls=cls,
                            arm=arm,
                            fmt=fmt,
                            question=q,
                            named=tuple(named),
                        )
                    )
    logger.info(
        "probe set: %d questions over %d frames (%s)",
        len(out),
        len({p.frame_key for p in out}),
        {d: len(v[:n_frames_per_cell]) for d, v in by_dist.items()},
    )
    return out


# --- scoring -----------------------------------------------------------------

_YES = re.compile(r"^\s*yes\b", re.I)
_NO = re.compile(r"^\s*no\b", re.I)


def read_answer(raw: str, fmt: str) -> dict:
    """Parse one raw model answer. Never raises — an unparseable answer is recorded, not dropped."""
    s = (raw or "").strip()
    if fmt == "number":
        m = re.match(r"^\s*(\d+)\s*$", s)
        val = int(m.group(1)) if m else None
        return {"raw": s, "value": val, "is_zero": val == 0, "parsed": m is not None}
    yes, no = bool(_YES.match(s)), bool(_NO.match(s))
    return {
        "raw": s,
        "value": "yes" if yes else ("no" if no else None),
        "is_zero": no,  # "no" is the binary analogue of emitting 0
        "parsed": yes or no,
    }


def score(rows: list[dict]) -> dict:
    """Aggregate a probe run.

    Returns per (model, distribution, format) and pooled:
      emit_rate_absent  — how often the model says 0/no when the label says ABSENT  [NOISE-FREE]
      emit_rate_present — how often it says 0/no when the label says PRESENT  (should be LOW)
      acc_absent / acc_present — agreement with the closure-derived label  [~8.4% noisy]
      hmean — harmonic mean of acc_absent and acc_present (the two-axis discipline)
      malformed — fraction the parser could not read
    """

    def _agg(sel: list[dict]) -> dict:
        ab = [r for r in sel if r["arm"] == "absent"]
        pr = [r for r in sel if r["arm"] == "present"]
        er_ab = sum(r["is_zero"] for r in ab) / len(ab) if ab else float("nan")
        er_pr = sum(r["is_zero"] for r in pr) / len(pr) if pr else float("nan")
        acc_ab = er_ab  # for ABSENT, emitting zero IS the correct answer
        # for PRESENT: number -> any positive integer; binary -> "yes"
        ok_pr = [
            r for r in pr if (r["value"] not in (None, 0, "no")) and r["parsed"]
        ]
        acc_pr = len(ok_pr) / len(pr) if pr else float("nan")
        hm = (
            0.0
            if (acc_ab + acc_pr) == 0
            else 2 * acc_ab * acc_pr / (acc_ab + acc_pr)
        )
        mal = sum(not r["parsed"] for r in sel) / len(sel) if sel else float("nan")
        return {
            "n": len(sel),
            "n_absent": len(ab),
            "n_present": len(pr),
            "emit_rate_absent": er_ab,
            "emit_rate_present": er_pr,
            "acc_absent": acc_ab,
            "acc_present": acc_pr,
            "hmean": hm,
            "malformed": mal,
        }

    out = {"pooled": {}, "cells": {}}
    models = sorted({r["model"] for r in rows})
    for mdl in models:
        mrows = [r for r in rows if r["model"] == mdl]
        out["pooled"][mdl] = _agg(mrows)
        for dist in ("ID", "OOD"):
            for fmt in ("number", "binary"):
                sel = [r for r in mrows if r["distribution"] == dist and r["fmt"] == fmt]
                if sel:
                    out["cells"][f"{mdl}|{dist}|{fmt}"] = _agg(sel)
    return out


def run_probe(engine, probes: list[ProbeQ], model_tag: str, log_every: int = 50) -> list[dict]:
    """Run one loaded engine over the probe set. Returns flat rows ready for `score`."""
    from PIL import Image

    rows: list[dict] = []
    for i, p in enumerate(probes, 1):
        try:
            with Image.open(p.frame_path) as im:
                raw = engine.predict(im.convert("RGB"), p.question)
        except Exception as exc:  # a missing frame must be visible, not silently skipped
            logger.warning("probe %s failed: %s", p.frame_key, exc)
            raw = ""
        parsed = read_answer(raw, p.fmt)
        rows.append(
            {
                "model": model_tag,
                "frame_key": p.frame_key,
                "dataset": p.dataset,
                "video_id": p.video_id,
                "distribution": p.distribution,
                "cls": p.cls,
                "arm": p.arm,
                "fmt": p.fmt,
                "question": p.question,
                "named": ", ".join(p.named),
                **parsed,
            }
        )
        if i % log_every == 0:
            logger.info("  %s: %d/%d", model_tag, i, len(probes))
    return rows
