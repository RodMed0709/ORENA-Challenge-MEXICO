"""Rung 49b — is the spurious `Clip` triggered by SPECULAR / METALLIC brightness?

Three `Sigma-5` false positives were opened and looked at before this was written (the
`RULES` habit that caught the Voxel51 mismatch). Two things stood out and neither is a clip:
one frame is dominated by a large metallic instrument shaft — sigmoid resection uses staplers,
which our cholecystectomy-heavy training set does not contain — and two are covered in small
bright specular highlights on wet, bloody tissue. A placed clip is exactly *a small bright
metallic blob*, so both would be the same shortcut: **brightness read as `Clip`**.

That is a hypothesis from three frames. This measures it on all of them.

🔴 The comparison is WITHIN VIDEO. Pooled, a correlation would only restate that Sigma videos
are bloodier and brighter than lapchole ones, which is already known and explains nothing —
[[pooled-screening-manufactures-winners]] is the same defect one rung earlier. The question is
whether, *inside one video*, the frames that draw a spurious `Clip` are brighter than the ones
that do not.

Statistics, per frame:
  `spec_frac`   fraction of pixels above a high-value threshold — specular highlight mass
  `blob_count`  small bright connected components, i.e. clip-shaped things
  `mean_v`      mean HSV value inside the endoscopic circle, as the brightness control
  `sat_low`     fraction of low-saturation pixels — metal is grey, tissue is not

Only the endoscopic circle counts: these frames are letterboxed with black borders, and
including them would make `mean_v` a function of the crop rather than the scene.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

V_SPEC = 240
V_BLOB = 200
S_LOW = 60
BLOB_MIN, BLOB_MAX = 8, 400


def frame_stats(path: Path) -> dict | None:
    """PIL + scipy only. `cv2` is absent from `orena-train`, which is the env every other
    number in this rung was produced in — importing a second env for a pure image statistic
    would be a confound for no gain."""
    from PIL import Image
    from scipy import ndimage

    try:
        with Image.open(path) as im:
            hsv = np.asarray(im.convert("HSV"))
    except Exception:  # noqa: BLE001
        return None
    h_, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    del h_
    # the endoscopic circle: anything not near-black. Black borders are not scene.
    inside = v > 12
    n = int(inside.sum())
    if n < 1000:
        return None
    bright = (v >= V_BLOB) & inside
    lab, n_lab = ndimage.label(bright, structure=np.ones((3, 3), dtype=int))
    if n_lab:
        areas = np.bincount(lab.ravel())[1:]
        blobs = int(((areas >= BLOB_MIN) & (areas <= BLOB_MAX)).sum())
    else:
        blobs = 0
    return {
        "spec_frac": float(((v >= V_SPEC) & inside).sum()) / n,
        "blob_count": blobs,
        "mean_v": float(v[inside].mean()),
        "sat_low": float(((s <= S_LOW) & inside).sum()) / n,
        "n_inside": n,
    }


def probe(d: pd.DataFrame, *, limit_per_video: int | None = None) -> pd.DataFrame:
    """One row per scored frame, with its stats and whether it drew a spurious Clip."""
    rows = []
    for vid, g in d.groupby("video"):
        g = g if limit_per_video is None else g.head(limit_per_video)
        for r in g.itertuples():
            p = Path(str(r.frame))
            st = frame_stats(p)
            if st is None:
                continue
            rows.append({"video": vid, "dataset": r.dataset, "qID": r.qID,
                         "fp": bool(r.fp), "gold_clip": bool(r.gold_clip), **st})
    return pd.DataFrame(rows)


def within_video(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per video: the FP frames' stats minus the non-FP frames', on clip-free rows only.

    Restricting to `gold_clip == False` matters — comparing FP frames against frames that
    really do contain a clip would measure the clip, not the shortcut.
    """
    d = f[~f.gold_clip]
    out = []
    for vid, g in d.groupby("video"):
        a, b = g[g.fp], g[~g.fp]
        if len(a) < 5 or len(b) < 5:
            continue
        row = {"video": vid, "dataset": g.dataset.iloc[0], "n_fp": len(a), "n_ok": len(b)}
        for c in ("spec_frac", "blob_count", "mean_v", "sat_low"):
            row[f"d_{c}"] = round(float(a[c].mean() - b[c].mean()), 5)
        out.append(row)
    w = pd.DataFrame(out)
    if w.empty:
        return w, {"videos": 0}
    summary = {"videos": int(len(w))}
    for c in ("spec_frac", "blob_count", "mean_v", "sat_low"):
        col = w[f"d_{c}"]
        summary[c] = {
            "mean_delta": round(float(col.mean()), 5),
            "videos_positive": int((col > 0).sum()),
            "sign_test_p_two_sided": _sign_p(int((col > 0).sum()), int((col != 0).sum())),
        }
    return w, summary


def _sign_p(k: int, n: int) -> float | None:
    """Exact two-sided sign test. n videos, k with a positive delta — no normal approx."""
    if not n:
        return None
    from math import comb

    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def run(inspect_csv: Path, out: Path, metrics, *, limit_per_video: int | None = None) -> None:
    valid = {n: n for n in tuple(metrics._load_fotype().names())}
    lower = {k.lower(): v for k, v in valid.items()}
    d = pd.read_csv(inspect_csv)
    d = d[d.answer_format == "fo_class"].copy()
    d["gold_set"] = d.ground_truth.map(lambda x: metrics.read_fo_class(x, lower))
    d["pred_set"] = d.our_answer.map(lambda x: metrics.read_fo_class(x, lower))
    d = d[d.gold_set.notna() & d.pred_set.notna()]
    d["gold_clip"] = d.gold_set.map(lambda s: "Clip" in s)
    d["fp"] = (~d.gold_clip) & d.pred_set.map(lambda s: "Clip" in s)

    f = probe(d, limit_per_video=limit_per_video)
    out.mkdir(parents=True, exist_ok=True)
    f.to_csv(out / "RESULTS_specular_frames.csv", index=False)
    w, summary = within_video(f)
    w.to_csv(out / "RESULTS_specular_within_video.csv", index=False)
    (out / "RESULTS_specular_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"frames measured: {len(f)}  videos usable: {summary.get('videos')}")
    print(json.dumps(summary, indent=1))
    if len(w):
        print(w.sort_values("n_fp", ascending=False).head(10).to_string(index=False))
