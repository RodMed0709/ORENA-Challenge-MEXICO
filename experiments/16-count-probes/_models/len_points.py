"""Probe 16c — derive the count as `len(predicted_points)` instead of verbalising it.

## The finding this replicates

Alghisi, Rizzoli, Mousavi, Riccardi 2026 (*Getting to the Point*, arXiv:2603.21746), on
Qwen2.5-VL-7B + LoRA r=32/α=64, exact-match count accuracy:

| setting | direct count | point-then-count, VERBALISED | **`#Coord` = len(points)** |
|---|---|---|---|
| synthetic ID | 32.00% | **94.96%** | — |
| synthetic OOD (10–18 objects) | 23.41% | 14.04% | **94.96%** |
| real OOD (11–20) | 8.82% | 25.68% | **33.53%** |

🟢 **The lever is not pointing. It is refusing to let the model state the number** — taking the
count as the length of its own coordinate list. On synthetic OOD that is **+80.9 points** over
the verbalised form. Their ID/OOD axis is count-range extrapolation, which is structurally our
shape: our gold reaches 12 while the model saturates at ~2 and scores 0% from gold ≥5.

This matters because rung 15 replicated Gautam's structured count *format* and returned a null.
It never tested this. And Gautam's own Table I (re-verified from the repo PDF, page 5) shows the
*joint* count+point objective **costs** counting accuracy — counting-only 0.26 vs count+point
1.52 — so the structured target is the trick and the joint objective is the tax. `len(points)`
is a third thing: a decoding-side derivation, not a joint objective.

## Caveats the authors report, which we must carry

- Replacing the image with a **black screen costs <2%** — the count is read off the model's own
  text, not re-derived from pixels. So a win here is not automatically a perception win.
- Coordinate↔count self-consistency is as low as **20.56%** ⇒ the deterministic `len()` fallback
  is **mandatory, not optional**. Never trust a verbalised number that accompanies points.
- **Prompt-only point-then-count is weak without fine-tuning.** A null in this probe is
  therefore NOT decisive — it bounds the prompt-only path, not the trained one.
- Distractor robustness varies by architecture, and some models degrade *faster* than direct
  counting as distractors increase. Our frames are all distractors.

## ⚠️ The coordinate convention is a trap

Qwen2-VL used `[0,1000]` with `<|box_start|>` tokens → **Qwen2.5-VL switched to absolute pixel
coordinates in JSON** → **Qwen3-VL switched back to `[0,1000]` normalised**, with points emitted
as `point_2d` inside JSON. Gautam and Alghisi both ran Qwen2.5-VL, so **their exact serialisation
is the wrong one for our backbone**. Qwen3-VL ships pretrained *counting* supervision in its own
convention; rung 15's bare-integer and `{"label","counts"}` targets used neither. Arm `a1` below
matches the native convention on purpose — that is the single variable.

## Arms (single variable = the answer target; the image and question are identical)

| arm | what the model is asked to emit | how the integer is recovered |
|---|---|---|
| `a0` | a bare integer (the rung-06 control) | the integer itself |
| `a1` | a `point_2d` JSON list in Qwen3-VL's native `[0,1000]` | **`len(points)`** |
| `a2` | numbered points, Molmo style | **the last index** |

`a2` comes from Molmo/PixMo (arXiv:2409.17146), which numbers its points so *"the total count
[is] always the number of the last point"* — a shorter serialisation than a1's and one that
removes the self-consistency failure by construction.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# a0 asks nothing extra — it is the control and MUST stay byte-identical to the scored engine's
# path, so the arm is defined by an empty suffix rather than by a different question.
ARM_SUFFIX = {
    "a0": "",
    "a1": (
        " Locate every one of them and reply ONLY with a JSON list of points in the form "
        '[{"point_2d": [x, y]}, ...], using normalized coordinates in the range 0-1000. '
        "One entry per object. If there are none, reply []."
    ),
    "a2": (
        " Point at every one of them and reply ONLY with a numbered list, one line per object, "
        'in the form "1: (x, y)" using normalized coordinates in the range 0-1000. '
        "If there are none, reply with nothing."
    ),
}

# a1/a2 need room for the coordinate list. Our counts top out at 12, so ~12 points is the worst
# case: ~170 chars, which also has to clear the SDK's 300-char auto-incorrect cap. Alghisi used
# 1,000-3,000 tokens for counts up to 18; we do NOT need that and must not pay for it.
ARM_MAX_NEW_TOKENS = {"a0": 8, "a1": 192, "a2": 192}

_POINT_2D = re.compile(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')
_NUMBERED = re.compile(r"^\s*(\d+)\s*[:.)]\s*\(?\s*-?\d", re.M)
_BARE_INT = re.compile(r"^\s*(\d+)\s*$")


def parse_count(raw: str, arm: str) -> dict:
    """Recover the integer from one raw answer. Never raises.

    Returns {value, n_points, parsed, method, raw}. `value` is None when unrecoverable —
    an unparseable answer is a FINDING (malformed rate is a reported metric), never a silent 0.
    """
    s = (raw or "").strip()
    if arm == "a0":
        m = _BARE_INT.match(s)
        return {
            "raw": s,
            "value": int(m.group(1)) if m else None,
            "n_points": None,
            "parsed": m is not None,
            "method": "bare_int",
        }

    if arm == "a1":
        # Prefer a real JSON parse; fall back to the regex, because a truncated list is common
        # and a regex still recovers every COMPLETE point in it.
        pts = None
        try:
            start, end = s.find("["), s.rfind("]")
            if start != -1 and end > start:
                obj = json.loads(s[start : end + 1])
                if isinstance(obj, list):
                    pts = [p for p in obj if isinstance(p, dict) and "point_2d" in p]
        except Exception:
            pts = None
        if pts is None:
            pts = _POINT_2D.findall(s)
            method = "regex_points"
        else:
            method = "json_points"
        n = len(pts)
        return {"raw": s, "value": n, "n_points": n, "parsed": True, "method": method}

    # a2 — the count is the LAST index, which is robust to a truncated tail in a way len() is not
    idxs = [int(m) for m in _NUMBERED.findall(s)]
    if not idxs:
        empty = s == "" or s.lower() in {"none", "no objects", "nothing"}
        return {
            "raw": s,
            "value": 0 if empty else None,
            "n_points": 0 if empty else None,
            "parsed": empty,
            "method": "numbered_empty" if empty else "numbered_unparsed",
        }
    return {
        "raw": s,
        "value": max(idxs),
        "n_points": len(idxs),
        "parsed": True,
        "method": "numbered_last_index",
    }


def select_number_items(items, template_re: str = r"how many\s+clips", split: str = "test"):
    """The `number` questions this probe scores.

    Default = the **Clips** template: 681 val questions, 12 distinct true values, and a margin
    over the template-aware trivial floor of only **+0.026** — the single largest hole in the
    exam. Pass a different regex to widen it.
    """
    pat = re.compile(template_re, re.I)
    out = []
    for it in items:
        ref, req = it.reference, it.request
        if "number" not in str(getattr(ref, "answer_format", "") or "").lower():
            continue
        if not pat.search(str(getattr(req, "question", "") or "")):
            continue
        try:
            gold = int(str(ref.answer).strip())
        except (ValueError, AttributeError):
            continue
        out.append((it, gold))
    logger.info("selected %d number items matching %r", len(out), template_re)
    return out


def run_arm(engine, selected, arm: str, log_every: int = 100) -> list[dict]:
    """Run one arm over the selected items. `engine` must already be loaded."""
    from PIL import Image

    from frame.data import frame_cache_name

    suffix = ARM_SUFFIX[arm]
    rows: list[dict] = []
    for i, (it, gold) in enumerate(selected, 1):
        key = frame_cache_name(it)
        path = f"/workspace/frames_cache/{key}"
        q = str(it.request.question) + suffix
        try:
            with Image.open(path) as im:
                raw = engine.predict(im.convert("RGB"), q)
        except Exception as exc:
            logger.warning("arm %s frame %s failed: %s", arm, key, exc)
            raw = ""
        p = parse_count(raw, arm)
        rows.append(
            {
                "arm": arm,
                "qid": getattr(it.request, "qID", None) or getattr(it.request, "qid", None),
                "frame_key": key,
                "dataset": it.dataset,
                "video_id": it.video_id,
                "distribution": "OOD" if it.dataset == "heico" else "ID",
                "gold": gold,
                "correct": p["value"] == gold,
                **p,
            }
        )
        if i % log_every == 0:
            logger.info("  arm %s: %d/%d", arm, i, len(selected))
    return rows


def template_floor(golds: list[int]) -> float:
    """The trivial floor for this template: always answer its modal gold.

    Read MARGIN over this, never raw accuracy — `acc_OOD > acc_ID` is an artifact of the OOD
    floor sitting ~12 points higher (data card §4b, RULES §12).
    """
    if not golds:
        return float("nan")
    from collections import Counter

    return Counter(golds).most_common(1)[0][1] / len(golds)


def score(rows: list[dict]) -> dict:
    """Per (arm, distribution): accuracy, the template floor, and the margin over it."""
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        for dist in ("ID", "OOD", "ALL"):
            sel = [
                r
                for r in rows
                if r["arm"] == arm and (dist == "ALL" or r["distribution"] == dist)
            ]
            if not sel:
                continue
            golds = [r["gold"] for r in sel]
            acc = sum(r["correct"] for r in sel) / len(sel)
            fl = template_floor(golds)
            preds = [r["value"] for r in sel if r["value"] is not None]
            out[f"{arm}|{dist}"] = {
                "n": len(sel),
                "n_videos": len({r["video_id"] for r in sel}),
                "acc": acc,
                "floor": fl,
                "margin": acc - fl,
                "malformed": sum(not r["parsed"] for r in sel) / len(sel),
                "mean_pred": (sum(preds) / len(preds)) if preds else float("nan"),
                "mean_gold": sum(golds) / len(golds),
                "bias": ((sum(preds) / len(preds)) - (sum(golds) / len(golds)))
                if preds
                else float("nan"),
            }
    return out
