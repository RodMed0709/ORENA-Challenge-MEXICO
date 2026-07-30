"""Rung 25 — individuation probe: does the model HAVE the objects it cannot count?

Importable engine; the notebook calls ``run_probe(engine, items)``. Nothing here is a launcher,
and nothing here trains.

## The measurement this rung exists for

Every counting rung so far scored **the number the model says**. None could score **whether the
model has the objects**, because our own gold is a bare integer. CholecInstanceSeg's instance masks
are the localization gold we have never had, so for the first time:

* ask the model to **point at every object**, and check each point against the masks;
* ask the model to **count**, on the same frame in the same pass;
* read the two together.

🎯 **The discriminating cell is `coverage high AND count accuracy low`.** That is
[[counting-is-a-mapping-failure]] — *"it can SEE the objects, it cannot EMIT the number"* — measured
on **our** checkpoint instead of borrowed from Alghisi's.

## What the numbers mean, precisely

* **precision** — emitted points that land inside *some* gold mask. Low precision means the points
  are decoration, and coverage stops meaning anything.
* **coverage** — gold instances containing ≥1 emitted point. **This is individuation.** It is the
  quantity nothing in this repo has ever measured.
* **`n_points` vs gold `N`** — the enumeration signal. Rung 16c measured the prompt-only model
  emitting **exactly 1 point on all 681** questions (`mean_pred` 1.0000); if that repeats here,
  everything downstream is moot and the lever dies for two GPU-hours.

## 🔴 Two traps this module is built around

**G-INFER (inherited, `cac9843`): a failed generation must NOT score as a wrong answer.** A parse
failure and a confident wrong answer are different findings. Rung 23 lost a whole smoke to this —
0/24 that was `max_new_tokens` truncation, not incapacity. Every row therefore carries a `status`,
and the scorer reports parse failures as their own category rather than folding them into zero.

**The ceiling is 3, and it is carried in the data, not in a comment.** CholecInstanceSeg has 4,914
frames at 0, 14,794 at 1, 16,715 at 2, 5,509 at 3 and exactly one at 4. We fail at **5–12**. So
every row this module writes carries `max_gold_n`, and `score()` refuses to emit a pooled headline
— per-stratum only. A pooled number would be 75% `N ∈ {1,2}` and would hide the only cell worth
looking at.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The two asks. Byte-identical images, one pass, so the comparison is within-frame.
# Kept as module constants because a prompt edited in a notebook cell is an unversioned variable.
POINT_PROMPT = (
    "Point at every surgical instrument visible in this frame. "
    "Respond with a JSON list of [x, y] coordinates normalised to [0, 1000], one per instrument. "
    "If there are none, respond with []."
)
COUNT_PROMPT = (
    "How many surgical instruments are visible in this frame? Please provide a number."
)

# Rung 16a measured that the fine-tuned checkpoint emits "1." — a trailing period — where
# `Number.verify` gates on `str.strip().isdigit()`, making it auto-incorrect. We parse leniently
# and RECORD the raw string, because the formatting defect is a finding of its own and must not be
# silently repaired into a score.
_INT_RE = re.compile(r"-?\d+")
_JSON_RE = re.compile(r"\[.*\]", re.S)


@dataclass(frozen=True)
class ProbeItem:
    """One frame with its instance gold. Built by the notebook from the dataset's annotations."""

    frame_id: str
    image_path: str
    gold_n: int
    # One mask per instance, as a callable so a 22 GB corpus is not held in memory. Each entry
    # answers "is (x, y) inside instance i?" in ORIGINAL image pixel coordinates.
    contains: tuple  # tuple[Callable[[int, int], bool], ...]
    width: int
    height: int


def parse_points(raw: str) -> dict:
    """Coordinates out of a generation. Returns `status` so G-INFER can hold.

    Statuses: `ok` (a list parsed, possibly empty), `unparseable` (no JSON list found),
    `malformed` (a list, but no usable pairs inside).
    """
    m = _JSON_RE.search(raw or "")
    if not m:
        return {"status": "unparseable", "points": [], "raw": raw}
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001 — a bad generation is data
        return {"status": "unparseable", "points": [], "raw": raw}

    pts = []
    for el in data if isinstance(data, list) else []:
        if isinstance(el, (list, tuple)) and len(el) >= 2:
            try:
                pts.append((float(el[0]), float(el[1])))
            except (TypeError, ValueError):
                continue
        elif isinstance(el, dict):
            x, y = el.get("x"), el.get("y")
            if x is not None and y is not None:
                try:
                    pts.append((float(x), float(y)))
                except (TypeError, ValueError):
                    continue
    if not pts and data != []:
        return {"status": "malformed", "points": [], "raw": raw}
    return {"status": "ok", "points": pts, "raw": raw}


def parse_count(raw: str) -> dict:
    """The integer, plus whether the SDK's own verifier would have accepted the raw string.

    `sdk_legal` is recorded but never used to score here: rung 16a found we CREATED the illegal
    `"1."` format by fine-tuning, and conflating "wrong number" with "unscoreable string" is what
    that probe existed to separate.
    """
    if raw is None:
        return {"status": "unparseable", "count": None, "sdk_legal": False, "raw": raw}
    m = _INT_RE.search(raw)
    if not m:
        return {"status": "unparseable", "count": None, "sdk_legal": False, "raw": raw}
    return {
        "status": "ok",
        "count": int(m.group(0)),
        "sdk_legal": raw.strip().isdigit(),
        "raw": raw,
    }


def _denorm(pt: tuple[float, float], w: int, h: int) -> tuple[int, int]:
    """[0,1000]-normalised -> pixel. Values already in pixel range are passed through.

    Qwen's native pointing convention is [0,1000]; a model that ignores the instruction and emits
    pixels would otherwise be scored as pointing at the top-left corner, which would read as a
    capability failure rather than a convention mismatch.
    """
    x, y = pt
    if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0 and (x > 1.0 or y > 1.0):
        return int(round(x / 1000.0 * w)), int(round(y / 1000.0 * h))
    return int(round(x)), int(round(y))


def score_item(item: ProbeItem, points: list[tuple[float, float]]) -> dict:
    """Precision and coverage of one frame's points against its instance masks."""
    px = [_denorm(p, item.width, item.height) for p in points]
    inside = [any(c(x, y) for c in item.contains) for x, y in px]
    covered = [any(c(x, y) for x, y in px) for c in item.contains]
    return {
        "n_points": len(px),
        "n_points_inside": sum(inside),
        "precision": (sum(inside) / len(px)) if px else None,
        "n_covered": sum(covered),
        "coverage": (sum(covered) / item.gold_n) if item.gold_n else None,
        # A frame with gold_n == 0 has no coverage to speak of; what matters there is whether the
        # model emitted nothing. Our supervision contains no numeric zero anywhere, and rung 16a
        # measured the model emitting `0` in 1 case out of 120 — so this cell is its own finding.
        "correct_empty": (item.gold_n == 0 and len(px) == 0) if item.gold_n == 0 else None,
    }


def run_probe(engine, items: list[ProbeItem], log_every: int = 50) -> list[dict]:
    """One pass: both prompts per frame, same pixels, `engine` already loaded.

    `engine.predict(image, question) -> str`, the interface rung 16's probes use.
    """
    from PIL import Image  # noqa: PLC0415

    rows: list[dict] = []
    for i, item in enumerate(items):
        with Image.open(item.image_path) as im:
            rgb = im.convert("RGB")
            try:
                raw_pts = engine.predict(rgb, POINT_PROMPT)
            except Exception as exc:  # noqa: BLE001 — a generation failure is a status, not a crash
                raw_pts = None
                logger.warning("point generation failed on %s: %s", item.frame_id, exc)
            try:
                raw_cnt = engine.predict(rgb, COUNT_PROMPT)
            except Exception as exc:  # noqa: BLE001
                raw_cnt = None
                logger.warning("count generation failed on %s: %s", item.frame_id, exc)

        p = parse_points(raw_pts or "")
        c = parse_count(raw_cnt)
        row = {
            "frame_id": item.frame_id,
            "gold_n": item.gold_n,
            "point_status": p["status"],
            "count_status": c["status"],
            "pred_count": c["count"],
            "count_correct": (c["count"] == item.gold_n) if c["status"] == "ok" else None,
            "sdk_legal": c["sdk_legal"],
            "raw_points": p["raw"],
            "raw_count": c["raw"],
        }
        row.update(score_item(item, p["points"]) if p["status"] == "ok"
                   else {"n_points": None, "precision": None, "coverage": None})
        rows.append(row)
        if i and i % log_every == 0:
            logger.info("probed %d/%d", i, len(items))
    return rows


def score(rows: list[dict]) -> list[dict]:
    """Per-stratum report. Deliberately returns NO pooled headline (PLAN §ceiling).

    75% of the corpus is `N ∈ {1,2}`, so a pooled number would be dominated by the easy strata and
    would hide the only cell this rung was built to see.
    """
    max_gold_n = max((r["gold_n"] for r in rows), default=0)
    out: list[dict] = []
    for n in sorted({r["gold_n"] for r in rows}):
        sub = [r for r in rows if r["gold_n"] == n]
        ok_pts = [r for r in sub if r["point_status"] == "ok"]
        ok_cnt = [r for r in sub if r["count_status"] == "ok"]
        cov = [r["coverage"] for r in ok_pts if r.get("coverage") is not None]
        prec = [r["precision"] for r in ok_pts if r.get("precision") is not None]
        npts = [r["n_points"] for r in ok_pts if r.get("n_points") is not None]
        out.append({
            "gold_n": n,
            "n_frames": len(sub),
            # G-INFER: parse failures are their own category, never folded into "wrong".
            "n_point_parse_fail": len(sub) - len(ok_pts),
            "n_count_parse_fail": len(sub) - len(ok_cnt),
            "mean_n_points": (sum(npts) / len(npts)) if npts else None,
            "mean_precision": (sum(prec) / len(prec)) if prec else None,
            "mean_coverage": (sum(cov) / len(cov)) if cov else None,
            "count_accuracy": (sum(1 for r in ok_cnt if r["count_correct"]) / len(ok_cnt))
            if ok_cnt else None,
            "sdk_illegal_rate": (sum(1 for r in ok_cnt if not r["sdk_legal"]) / len(ok_cnt))
            if ok_cnt else None,
            # G-RANGE travels with every row so no reader can quote this outside its bound.
            "max_gold_n": max_gold_n,
        })
    return out


def read_verdict(report: list[dict], collapse_atol: float = 0.25) -> dict:
    """The PRE-REGISTERED read (PLAN §ceiling). Kill-only, and it says so in its own output.

    * **collapse** — `mean_n_points` is flat across strata (the rung-16c signature: one point
      regardless of `N`). ⇒ individuation is absent, the trained-pointing lever dies here.
    * **individuates** — coverage rises with `N` and precision holds. ⇒ licenses the NEXT probe,
      **not** the lever: this instrument tops out at 3 and we fail at 5–12.
    """
    strata = [r for r in report if r["gold_n"] > 0 and r["mean_n_points"] is not None]
    if len(strata) < 2:
        return {"verdict": "inconclusive", "reason": "fewer than two usable strata"}

    spread = max(r["mean_n_points"] for r in strata) - min(r["mean_n_points"] for r in strata)
    collapsed = spread < collapse_atol
    return {
        "verdict": "collapse" if collapsed else "individuates",
        "n_points_spread": spread,
        "collapse_atol": collapse_atol,
        "max_gold_n": strata[0]["max_gold_n"],
        "reason": (
            "emitted points do not track gold N — the rung-16c signature. Individuation is absent "
            "and the trained-pointing lever dies here."
            if collapsed else
            "emitted points track gold N. ⚠️ This licenses the NEXT probe, NOT the lever: the "
            f"instrument tops out at {strata[0]['max_gold_n']} and our failure range is 5-12. "
            "It may not be written as 'pointing works'."
        ),
    }


__all__ = [
    "COUNT_PROMPT",
    "POINT_PROMPT",
    "ProbeItem",
    "parse_count",
    "parse_points",
    "read_verdict",
    "run_probe",
    "score",
    "score_item",
]
