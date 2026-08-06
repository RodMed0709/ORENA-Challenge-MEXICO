"""Step 6 — the SAM 2 temporal probe: is the sub-second gold instability label or scene?

August plan, step 6 (`local/tasks/plan-accion.md`); spec A.4 in `roadmap-percepcion-rl.md`,
design in [[covt-reduced-sam-route]]. Not a rung — the ladder is closed. This module measures;
the notebook states the pre-registered verdict.

**The claim being tested.** At <=1 s apart a SAM 2 track is the same physical instance by
construction, while the counting gold moves +/-0.86 between such frames. Where the track holds
and the gold jumps by 3 or more, the gold is wrong — a decomposition [[synthetic-counting-reconciled]]
records as unavailable without clinical adjudication. It needs none.

**Why the seeding is hand-rolled.** SAM 2 ships inside our pinned `transformers` (4.57.6) as
`Sam2VideoModel`, but the port carries no automatic mask generator — that lives in Meta's own
package, which we do not add. So instances are seeded from a lattice of point prompts and
de-duplicated here, where it can be audited.

🔴 **SAM 2 is class-agnostic and this is not a detail.** Measured on our own footage before this
file was written: one centre point on a `heico` frame returns 90% of the image. It segments
tissue. Every threshold below that touches area is therefore load-bearing, not hygiene.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "build_pairs",
    "seed_instances",
    "propagate_pair",
    "pair_stats",
    "control_c1",
    "control_c2",
    "verdict",
]


# ── population ────────────────────────────────────────────────────────────────

def build_pairs(
    items,
    *,
    gap_max: float = 1.0,
    min_gold: int = 5,
    jump: int = 3,
    read_gold=None,
):
    """Consecutive same-video annotated-frame pairs, with both golds attached.

    ``items`` are ``FrameItem``s; ``read_gold`` parses a reference into an int count and
    returns None for anything that is not a countable `number` answer — passed in rather
    than imported so the caller stays the single owner of the parsing rule (RULES EVAL).

    Pairs are consecutive **within a video, in time**, which is the only ordering under
    which "the previous annotated frame" means anything. Rows carry the gap so the two
    arms can be matched on it instead of assumed comparable.
    """
    import pandas as pd

    rows = []
    for it in items:
        g = read_gold(it.reference)
        if g is None:
            continue
        rows.append(
            {
                "qID": it.request.qID,
                "dataset": it.dataset,
                "video_id": it.video_id,
                "frame_index": it.frame_index,
                "t": float(it.request.start_time),
                "gold": int(g),
                "template": str(it.request.question),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    out = []
    for (ds, vid), g in df.groupby(["dataset", "video_id"], sort=True):
        g = g.sort_values("t").reset_index(drop=True)
        a, b = g.iloc[:-1].reset_index(drop=True), g.iloc[1:].reset_index(drop=True)
        gap = b["t"].to_numpy() - a["t"].to_numpy()
        # Same template on both sides: a "how many Clips" gold and a "how many Needles"
        # gold are not two readings of one quantity, and differencing them is nonsense.
        same_tpl = a["template"].to_numpy() == b["template"].to_numpy()
        keep = (gap > 0) & (gap <= gap_max) & same_tpl
        for i in np.flatnonzero(keep):
            ga, gb = int(a.gold[i]), int(b.gold[i])
            if max(ga, gb) < min_gold:
                continue
            d = abs(gb - ga)
            out.append(
                {
                    "dataset": ds,
                    "video_id": vid,
                    "qID_a": a.qID[i], "qID_b": b.qID[i],
                    "frame_a": int(a.frame_index[i]), "frame_b": int(b.frame_index[i]),
                    "gap_s": float(gap[i]),
                    "gold_a": ga, "gold_b": gb, "delta": d,
                    "arm": "J" if d >= jump else ("S" if d == 0 else "-"),
                }
            )
    return pd.DataFrame(out)


def match_arms(pairs, *, seed: int = 42):
    """Down-sample arm S to arm J's size, matched on the gap distribution.

    Without matching the two arms differ in how far apart their frames are, and gap is
    exactly what drives track survival — the comparison would then measure the sampling,
    not the scene.
    """
    import pandas as pd

    J = pairs[pairs.arm == "J"]
    S = pairs[pairs.arm == "S"]
    if J.empty or S.empty:
        return pd.concat([J, S])
    rng = np.random.default_rng(seed)
    picked = []
    pool = S.copy()
    for gap in J.gap_s:
        if pool.empty:
            break
        i = (pool.gap_s - gap).abs().idxmin()
        picked.append(i)
        pool = pool.drop(i)
    return pd.concat([J, S.loc[picked]])


# ── seeding: a hand-rolled automatic mask generator ───────────────────────────

def _grid(h: int, w: int, n: int) -> list[list[float]]:
    ys = (np.arange(n) + 0.5) * h / n
    xs = (np.arange(n) + 0.5) * w / n
    return [[float(x), float(y)] for y in ys for x in xs]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _nms(masks: list[np.ndarray], areas: list[int], thr: float) -> list[int]:
    """Keep the LARGEST surviving mask of each overlapping group.

    Order matters and largest-first is the deliberate choice: two grid points landing on
    one object return the object and a fragment of it, and keeping the fragment would
    inflate the instance count with pieces of things already counted.
    """
    order = list(np.argsort(areas)[::-1])
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(masks[i], masks[j]) < thr]
    return keep


def seed_instances(
    model,
    processor,
    frames: list[np.ndarray],
    *,
    frame_idx: int = 0,
    grid: int = 16,
    min_area_frac: float = 0.0005,
    max_area_frac: float = 0.25,
    nms_iou: float = 0.7,
    device: str = "cuda",
    dtype=torch.bfloat16,
):
    """Segment ``frames[frame_idx]`` into candidate instances from a point lattice.

    Returns ``(session, obj_ids, masks)`` — the session is returned because propagation
    must continue on the SAME session that holds the memory bank.

    ⚠️ ``max_area_frac`` is what stops the tissue background from being returned as the
    dominant "instance". Without it the largest mask is ~90% of the frame, measured.
    """
    h, w = frames[frame_idx].shape[:2]
    session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)

    pts = _grid(h, w, grid)
    cand_masks: list[np.ndarray] = []
    cand_area: list[int] = []
    # One prompt at a time: a shared session accumulates state per object id, and batching
    # distinct hypotheses into one call makes them compete instead of being independent.
    for k, p in enumerate(pts):
        processor.process_new_points_or_boxes_for_video_frame(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=[k],
            input_points=[[[p]]],
            input_labels=[[[1]]],
        )
        with torch.inference_mode():
            out = model(inference_session=session, frame_idx=frame_idx)
        m = processor.post_process_masks(
            [out.pred_masks], original_sizes=[[h, w]], binarize=True
        )[0]
        m = np.asarray(m[-1, 0].cpu()).astype(bool)
        a = int(m.sum())
        if min_area_frac * h * w <= a <= max_area_frac * h * w:
            cand_masks.append(m)
            cand_area.append(a)

    keep = _nms(cand_masks, cand_area, nms_iou) if cand_masks else []
    masks = [cand_masks[i] for i in keep]

    # Re-open a clean session carrying ONLY the kept instances, so propagation tracks the
    # de-duplicated set rather than every grid hypothesis.
    session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
    obj_ids = list(range(len(masks)))
    for oid, m in zip(obj_ids, masks):
        processor.process_new_mask_for_video_frame(
            inference_session=session, frame_idx=frame_idx, obj_ids=oid,
            input_masks=torch.from_numpy(m),
        )
    return session, obj_ids, masks


def propagate_pair(model, processor, session, obj_ids, frames, *, target_idx: int = 1):
    """Propagate the seeded instances to ``frames[target_idx]``; returns its masks."""
    h, w = frames[0].shape[:2]
    got: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(inference_session=session, start_frame_idx=0):
            if out.frame_idx != target_idx:
                continue
            m = processor.post_process_masks(
                [out.pred_masks], original_sizes=[[h, w]], binarize=True
            )[0]
            for j, oid in enumerate(obj_ids):
                got[oid] = np.asarray(m[j, 0].cpu()).astype(bool)
    return [got.get(o, np.zeros((h, w), bool)) for o in obj_ids]


# ── statistics ────────────────────────────────────────────────────────────────

def pair_stats(masks_a: list[np.ndarray], masks_b: list[np.ndarray], *, iou_survive: float = 0.5):
    """Track persistence across one pair — the probe's whole statistic.

    ``persistence`` is the fraction of seeded instances still present at the second frame.
    Reported with ``n_seed`` because a persistence of 1.0 over 2 instances and over 30 are
    not the same evidence.
    """
    ious = [_iou(a, b) for a, b in zip(masks_a, masks_b)]
    surv = [i >= iou_survive for i in ious]
    return {
        "n_seed": len(masks_a),
        "n_survived": int(sum(surv)),
        "persistence": float(np.mean(surv)) if surv else float("nan"),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
    }


def control_c2(stats_adjacent: list[dict], *, floor: float = 0.90) -> dict:
    """🔴 BLOCKING — can SAM 2 track this footage at all?

    Persistence one frame apart (~40 ms). A tracker that fails here cannot be read at 1 s,
    and both arms would be measuring the tracker instead of the scene.
    """
    p = [s["persistence"] for s in stats_adjacent if s["persistence"] == s["persistence"]]
    got = float(np.mean(p)) if p else float("nan")
    return {"persistence_adjacent": got, "floor": floor,
            "passes": bool(got >= floor), "n": len(p)}


def control_c1(k_high: list[int], k_one: list[int], *, min_sep: float = 0.5) -> dict:
    """🔴 BLOCKING — does the seeded instance count carry ANY count information?

    Rodrigo's negative control, restated. His literal test (`K == 1` on a `gold == 1`
    frame) cannot be used: SAM 2 is class-agnostic and segments tissue, so K >> 1 there is
    a property of the tool, not a defect — see the README. What must hold instead is
    SEPARATION: frames with gold >= 5 must seed measurably more instances than frames with
    gold == 1. If they do not, nothing SAM emits is count-bearing.

    ⚠️ A failure here kills the COUNT reading. It does not formally kill the persistence
    reading — and reading that fallback anyway would be moving the goalposts, so the
    notebook reports NO VERDICT and the decision goes back to the team.
    """
    a, b = np.asarray(k_high, float), np.asarray(k_one, float)
    sep = float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")
    return {
        "mean_k_gold_high": float(a.mean()) if len(a) else float("nan"),
        "mean_k_gold_one": float(b.mean()) if len(b) else float("nan"),
        "separation": sep,
        "min_separation": min_sep,
        "passes": bool(sep >= min_sep),
        "n_high": len(a), "n_one": len(b),
    }


def verdict(persist_J: list[float], persist_S: list[float], *, eps: float = 0.10,
            low: float = 0.50) -> dict:
    """The pre-registered three-way read on arm J (gold jumps) vs arm S (gold stable)."""
    j = float(np.mean(persist_J)) if len(persist_J) else float("nan")
    s = float(np.mean(persist_S)) if len(persist_S) else float("nan")
    d = j - s
    if j < low and s < low:
        call = "NO_VERDICT"
        reading = "persistence is low in BOTH arms — the probe measured its own tracker"
    elif abs(d) < eps:
        call = "LABEL"
        reading = "the scene did not change where the gold jumped — the jump is annotation noise"
    elif d < 0:
        call = "SCENE"
        reading = "tracks really do break where the gold jumps — the instability is scene, not label"
    else:
        call = "NO_VERDICT"
        reading = "tracks survive BETTER where the gold jumps — unmodelled, do not read it"
    return {
        "persistence_J": j, "persistence_S": s, "delta": d,
        "eps": eps, "n_J": len(persist_J), "n_S": len(persist_S),
        "verdict": call, "reading": reading,
    }
