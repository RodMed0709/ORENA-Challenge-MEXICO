"""Rung 12d — cheap screening of candidate transforms, before any of them costs a training run.

Library only. No launcher: a notebook cell (or a throwaway driver) calls `screen`.

Why this exists
---------------
Branch A measured ONE transform (`unsharp`) at two doses and it was negative. The open
question is not "was that one good" but **"which transform is worth a training run at all"**,
and guessing costs ~7.5 h of pod per guess.

`build_frame_index.py` already established the screening method — its `image_stats` docstring
says each statistic "maps to a candidate transform" — and it earned its keep by killing two
candidates (global white-boost, CLAHE) for zero GPU. This module extends it in the two places
that index left open:

1. **Local, not global.** Every statistic there is computed over the WHOLE frame, and that is
   exactly how the gauze candidate died: the sponge occupies little area and a global mean
   buries it. Here each frame is tiled and the aggregation keeps `max` / `top-k`, so a small
   bright object survives the summary.
2. **A bank, not one transform.** Edge and boundary operators (Canny, Sobel, morphological
   gradient, top-hat) are the family the eye says is missing — `unsharp` amplifies fine texture
   and does not delimit regions.

What a high score here does and does not mean
---------------------------------------------
🔴 **Separability is a SCREEN, not a prediction of model benefit.** It ranks transforms by how
much they pull "class present" apart from "class absent" in a descriptor. It cannot tell you the
ViT will use that separation — `edge_density` already proved that a statistic can move a lot
(+669 %) while perceptual severity moves little. Use it to spend the training run on the best
candidate, never to claim a transform works.

⚠️ **AUC is computed WITHIN dataset.** `heico` and `lapchole` differ in brightness, contrast and
edge density, and their class priors differ too, so a pooled AUC would rediscover "which centre
is this" instead of "is the object present". This is the same confound that made the resolution
axis irresolvable (rung 11).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# flat sibling import: `_models/` is not a package (no __init__.py) — the driver puts it on sys.path
from enhance import specular, unsharp

# ── the bank ────────────────────────────────────────────────────────────────
# Every entry takes and returns a PIL RGB image, so descriptors are computed the
# same way regardless of what the operator did to the pixels.


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def _pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def identity(img: Image.Image) -> Image.Image:
    return img


def clahe(img: Image.Image, clip_limit: float = 2.0, tile: int = 8) -> Image.Image:
    """Contrast-limited adaptive histogram equalisation on L only (keeps colour)."""
    lab = cv2.cvtColor(_np(img), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return _pil(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def canny(img: Image.Image, lo: int = 50, hi: int = 150) -> Image.Image:
    """Binary contour map. REPLACES the image — screening only, never an inference arm alone."""
    return _pil(cv2.Canny(cv2.cvtColor(_np(img), cv2.COLOR_RGB2GRAY), lo, hi))


def sobel(img: Image.Image, ksize: int = 3) -> Image.Image:
    """Gradient magnitude, normalised per frame."""
    g = cv2.cvtColor(_np(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize),
                   cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize))
    return _pil(255.0 * mag / (mag.max() or 1.0))


def morph_gradient(img: Image.Image, k: int = 5) -> Image.Image:
    """Dilate minus erode — thick region boundaries, less noisy than Canny."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return _pil(cv2.morphologyEx(_np(img), cv2.MORPH_GRADIENT, ker))


def tophat(img: Image.Image, k: int = 15) -> Image.Image:
    """Bright structures SMALLER than the kernel — the gauze/clip size regime.

    This is the operator the failed global white-boost actually needed: it is
    scale-selective, so a small bright object is not buried by a large bright field.
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return _pil(cv2.morphologyEx(_np(img), cv2.MORPH_TOPHAT, ker))


def blackhat(img: Image.Image, k: int = 15) -> Image.Image:
    """Dark structures smaller than the kernel — sutures, dark loops, shadow gaps."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return _pil(cv2.morphologyEx(_np(img), cv2.MORPH_BLACKHAT, ker))


def bilateral(img: Image.Image, d: int = 9, sc: int = 75, ss: int = 75) -> Image.Image:
    """Edge-preserving smoothing — removes the haze/"nubosidad" without softening contours."""
    return _pil(cv2.bilateralFilter(_np(img), d, sc, ss))


def bilateral_unsharp(img: Image.Image) -> Image.Image:
    """Dehaze first, then sharpen — the combination the eyeball session asked for."""
    return unsharp(bilateral(img), radius=2.0, amount=1.0)


# ── surgical-domain operators (from the endoscopic-imaging literature) ──────
# The generic bank came back flat, and these differ from it in kind: they encode a
# fact about ENDOSCOPIC images specifically, not about images in general.


def _specular_mask(rgb: np.ndarray) -> np.ndarray:
    """Specular pixels by the endoscopic physics, not by an HSV rule of thumb.

    Under normal tissue the R channel dominates (haemoglobin absorbs G and B). Under a
    specular highlight the surface reflects the illuminant directly, so R, G and B come
    out nearly EQUAL and all high. That two-part test (`spread` small AND `level` high)
    is far more specific than our `v>0.85 & s<0.20`, which also fires on pale tissue.
    """
    f = rgb.astype(np.float32)
    return ((f.max(2) - f.min(2) < 30.0) & (f.mean(2) > 200.0)).astype(np.uint8)


def despecular(img: Image.Image, dilate: int = 3) -> Image.Image:
    """Detect specular highlights and INPAINT them (Navier-Stokes).

    🔴 This is the exact OPPOSITE of `specular()`, and both are in the bank on purpose.
    We amplify highlights because `specular_frac` was the one descriptor separating the
    metallic `Clip`; the literature removes them because they saturate and destroy the
    texture underneath. Both hypotheses are plausible; the screen is where they compete.
    """
    rgb = _np(img)
    mask = _specular_mask(rgb)
    if mask.sum() == 0:
        return img
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return _pil(cv2.inpaint(rgb, cv2.dilate(mask, ker), 3, cv2.INPAINT_NS))


def dehaze(img: Image.Image, w: float = 0.85, patch: int = 15, t0: float = 0.1) -> Image.Image:
    """Dark-channel-prior dehazing — for electrocautery smoke.

    The proper version of what `bilateral` was only approximating.
    """
    f = _np(img).astype(np.float32) / 255.0
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(f.min(2), ker)
    a = float(np.percentile(f.reshape(-1, 3).max(1), 99.9))
    a = max(a, 1e-3)
    t = np.clip(1.0 - w * cv2.erode((f / a).min(2), ker), t0, 1.0)[..., None]
    return _pil(255.0 * np.clip((f - a) / t + a, 0, 1))


def contrast_1s(img: Image.Image) -> Image.Image:
    """Multiply the RGB planes by the (1-S) plane of HSV.

    Saturated regions (tissue, blood) get pushed down; desaturated ones (metal, gauze,
    plastic — i.e. the foreign objects) survive. A cheap desaturation-based prior.
    """
    rgb = _np(img).astype(np.float32)
    s = cv2.cvtColor(_np(img), cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    return _pil(rgb * (1.0 - s)[..., None])


def homomorphic(img: Image.Image, sigma: float = 30.0,
                gl: float = 0.5, gh: float = 1.5) -> Image.Image:
    """Separate illumination (low frequency) from reflectance (high frequency) in log space."""
    lab = cv2.cvtColor(_np(img), cv2.COLOR_RGB2LAB)
    ln = np.log1p(lab[..., 0].astype(np.float32))
    low = cv2.GaussianBlur(ln, (0, 0), sigma)
    out = np.expm1(gl * low + gh * (ln - low))
    lab[..., 0] = np.clip(out, 0, 255).astype(np.uint8)
    return _pil(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


# ── null anchors ────────────────────────────────────────────────────────────
# 🔴 Without these the ranking is unreadable: `tophat` scored +0.0043 and there was no
# way to say whether that beats what a transform carrying NO class information scores by
# chance. These perturb the pixels about as much as the real operators and add nothing.


def null_jpeg(img: Image.Image, quality: int = 75) -> Image.Image:
    """Re-encode. Pixels move everywhere; no class information is created."""
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def null_noise(img: Image.Image, sigma: float = 4.0, seed: int = 20260720) -> Image.Image:
    """Deterministic mild Gaussian noise — the same frame always gets the same noise."""
    arr = _np(img).astype(np.float32)
    rng = np.random.default_rng(seed + (arr.shape[0] * 7919 + arr.shape[1]))
    return _pil(arr + rng.normal(0.0, sigma, arr.shape))


TRANSFORMS = {
    "identity": identity,
    # anchors: already measured on the model, so the screen can be calibrated against them
    "unsharp_x1": lambda im: unsharp(im, radius=2.0, amount=1.0),
    "unsharp_x3": lambda im: unsharp(im, radius=2.0, amount=3.0),
    "clahe": clahe,
    "specular": lambda im: specular(im, amount=1.0),
    # the untested family: edges, boundaries, scale-selective morphology
    "canny": canny,
    "sobel": sobel,
    "morph_gradient": morph_gradient,
    "tophat": tophat,
    "blackhat": blackhat,
    "bilateral": bilateral,
    "bilateral_unsharp": bilateral_unsharp,
    # surgical-domain operators from the endoscopic-imaging literature
    "despecular": despecular,
    "dehaze": dehaze,
    "contrast_1s": contrast_1s,
    "homomorphic": homomorphic,
    # null anchors — these MUST land near zero, or the screen has no resolution
    "null_jpeg": null_jpeg,
    "null_noise": null_noise,
}


# ── chained pipelines ───────────────────────────────────────────────────────
# Single operators screened flat, and the eyeball pass said why: each one shines in a
# DIFFERENT scenario, so averaging over all frames cancels them. Classical vision chains
# them for exactly that reason — normalise illumination, then remove glare, then extract
# structure — so that the structure step is not fed by artefacts the earlier steps remove.


def chain(*fns):
    """Compose transforms left-to-right: chain(a, b)(img) == b(a(img))."""
    def run(img: Image.Image) -> Image.Image:
        for f in fns:
            img = f(img)
        return img
    return run


def _homo(sigma: float, gl: float, gh: float):
    return lambda im: homomorphic(im, sigma=sigma, gl=gl, gh=gh)


COMBOS = {
    # baseline + anchors, carried so the paired ranking has its reference and its floor
    "identity": identity,
    "null_jpeg": null_jpeg,

    # ── legokna's proposals (eyeball pass, 2026-07-21) ──
    "clahe+despec+sobel": chain(clahe, despecular, sobel),
    "clahe+despec+morphgrad": chain(clahe, despecular, morph_gradient),
    "homo+sobel": chain(homomorphic, sobel),
    "homo+morphgrad": chain(homomorphic, morph_gradient),

    # ── homomorphic calibration: it ranked last but the eye found it useful, which is
    #    the signature of a badly tuned parameter rather than a dead operator ──
    "homo_soft": _homo(30.0, 0.8, 1.2),
    "homo_strong": _homo(15.0, 0.3, 2.0),
    "homo_soft+morphgrad": chain(_homo(30.0, 0.8, 1.2), morph_gradient),

    # ── mine, each with a reason ──
    # order matters: de-glare BEFORE equalising, or the highlights drive the histogram
    "despec+clahe": chain(despecular, clahe),
    # de-glare then scale-selective bright structure — the gauze/clip regime without the
    # specular false positives that made the global white-boost fail
    "despec+tophat": chain(despecular, tophat),
    # the eyeball pass said dehaze "mete mucho ruido"; denoise first, then take boundaries
    "bilateral+morphgrad": chain(bilateral, morph_gradient),
    # full classical pipeline
    "despec+homo+sobel": chain(despecular, homomorphic, sobel),
    # 🔴 the only chain that PRESERVES the image instead of replacing it, so it is the only
    # one usable as a direct inference arm rather than as a second channel (12c)
    "despec+bilateral+unsharp": chain(despecular, bilateral,
                                      lambda im: unsharp(im, radius=2.0, amount=1.0)),
}

# ── local descriptors ───────────────────────────────────────────────────────

_STATS = ("v_mean", "v_std", "specular_frac", "white_frac", "lap_var", "grad_mean")


def local_descriptors(img: Image.Image, grid: int = 8, topk: int = 4) -> dict[str, float]:
    """Tile the frame `grid`x`grid`, stat each tile, then aggregate.

    Returns, for each statistic, three aggregations:
      ``<stat>_mean``  — the whole-frame value (what `build_frame_index` computed)
      ``<stat>_max``   — the single most extreme tile
      ``<stat>_top{k}``— mean of the k most extreme tiles

    The last two are the point: a small object that a whole-frame mean dilutes to nothing
    still dominates the tile it sits in.
    """
    rgb = _np(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1].astype(np.float32) / 255.0, hsv[..., 2].astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    H, W = v.shape
    hs, ws = max(H // grid, 1), max(W // grid, 1)
    tiles: dict[str, list[float]] = {k: [] for k in _STATS}
    for y in range(0, grid * hs, hs):
        for x in range(0, grid * ws, ws):
            sl = (slice(y, y + hs), slice(x, x + ws))
            tv, ts, tl, tg = v[sl], s[sl], lap[sl], grad[sl]
            tiles["v_mean"].append(float(tv.mean()))
            tiles["v_std"].append(float(tv.std()))
            tiles["specular_frac"].append(float(((tv > 0.85) & (ts < 0.20)).mean()))
            tiles["white_frac"].append(float(((tv > 0.60) & (tv <= 0.90) & (ts < 0.25)).mean()))
            tiles["lap_var"].append(float(tl.var()))
            tiles["grad_mean"].append(float(tg.mean()))

    out: dict[str, float] = {}
    for k, vals in tiles.items():
        a = np.asarray(vals, dtype=np.float64)
        out[f"{k}_mean"] = float(a.mean())
        out[f"{k}_max"] = float(a.max())
        out[f"{k}_top{topk}"] = float(np.sort(a)[-topk:].mean())
    return out


def describe_frame(path: str | Path, transforms: dict | None = None,
                   grid: int = 8, topk: int = 4) -> dict[str, float]:
    """Every transform in the bank applied to one frame, each described locally."""
    transforms = transforms or TRANSFORMS
    img = Image.open(path).convert("RGB")
    row: dict[str, float] = {}
    for tname, fn in transforms.items():
        for dname, val in local_descriptors(fn(img), grid=grid, topk=topk).items():
            row[f"{tname}::{dname}"] = val
    return row


# ── separability ────────────────────────────────────────────────────────────


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U / n1n2). 0.5 = no separation; symmetric around it.

    Rank-based on purpose: descriptors live on wildly different scales (a Laplacian
    variance is in the thousands, a fraction in [0,1]) and some are heavy-tailed, so any
    mean-difference effect size would be comparing incomparable things.
    """
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def separability(features: pd.DataFrame, labels: pd.DataFrame,
                 min_per_side: int = 30, min_videos_per_side: int = 3) -> pd.DataFrame:
    """AUC of every (transform, descriptor) for every (class, dataset).

    `features` is indexed by frame key, columns ``"<transform>::<descriptor>"``.
    `labels` is indexed by frame key with ``dataset`` and ``video`` columns and one
    boolean column per class (``has_<class>``).

    Two gates, and the second is the one that matters:

    - `min_per_side` frames on each side — an AUC over 4 positives is noise with a number.
    - 🔴 `min_videos_per_side` **distinct videos** on each side. Frames are not independent:
      they cluster on videos (`RULES` §13, effective n ≈ 38). If a class appears in two
      videos, a descriptor that separates it perfectly may only be separating *those two
      videos* — their lighting, their scope, their centre. That is the confound that made
      the resolution axis irresolvable in rung 11, and here it would manufacture a winner.

    `n_vid_pos` is reported on every row so a cell that barely clears the gate can still be
    discounted by eye.
    """
    rows = []
    classes = [c for c in labels.columns if c.startswith("has_")]
    for ds, lab_ds in labels.groupby("dataset"):
        feat_ds = features.loc[features.index.intersection(lab_ds.index)]
        lab_ds = lab_ds.loc[feat_ds.index]
        vids = lab_ds["video"].to_numpy()
        for cls in classes:
            mask = lab_ds[cls].to_numpy(dtype=bool)
            n_vp, n_vn = len(set(vids[mask])), len(set(vids[~mask]))
            if mask.sum() < min_per_side or (~mask).sum() < min_per_side:
                continue
            if n_vp < min_videos_per_side or n_vn < min_videos_per_side:
                continue
            for col in feat_ds.columns:
                x = feat_ds[col].to_numpy(dtype=np.float64)
                ok = np.isfinite(x)
                a = auc(x[ok & mask], x[ok & ~mask])
                transform, descriptor = col.split("::", 1)
                rows.append({
                    "dataset": ds, "klass": cls[4:], "transform": transform,
                    "descriptor": descriptor, "auc": a,
                    # |AUC-0.5| is the separation regardless of sign: a descriptor that
                    # goes DOWN when the object is present separates just as well.
                    "sep": abs(a - 0.5) if a == a else float("nan"),
                    "n_pos": int(mask.sum()), "n_neg": int((~mask).sum()),
                    "n_vid_pos": n_vp, "n_vid_neg": n_vn,
                })
    return pd.DataFrame(rows)


def rank_transforms(sep: pd.DataFrame) -> pd.DataFrame:
    """Best descriptor per (transform, class, dataset), then the transform's mean over cells.

    A transform only needs ONE descriptor to expose the object, so the per-cell score is the
    max over descriptors. The headline is `delta_vs_identity` — a transform that does not beat
    `identity` has no claim on a training run.
    """
    best = (sep.groupby(["dataset", "klass", "transform"], as_index=False)["sep"].max()
               .rename(columns={"sep": "best_sep"}))
    # NB: `best["transform"]`, never `best.transform` — that attribute is DataFrame.transform,
    # and comparing the bound method to a string silently yields False.
    base = (best[best["transform"] == "identity"]
            .set_index(["dataset", "klass"])["best_sep"].rename("identity_sep"))
    best = best.join(base, on=["dataset", "klass"])
    best["delta_vs_identity"] = best["best_sep"] - best["identity_sep"]
    out = (best.groupby("transform")
               .agg(mean_sep=("best_sep", "mean"),
                    mean_delta=("delta_vs_identity", "mean"),
                    cells_better=("delta_vs_identity", lambda s: int((s > 0).sum())),
                    n_cells=("delta_vs_identity", "size"))
               .sort_values("mean_delta", ascending=False)
               .reset_index())
    return out


def within_video(features: pd.DataFrame, labels: pd.DataFrame,
                 classes: list[str] | None = None, min_per_side: int = 15) -> pd.DataFrame:
    """🔴 The control that decides. Separation measured INSIDE each video.

    Pooling videos lets a transform score by detecting *which scene this is* rather than
    *whether the object is there*: videos that contain a given object are videos that look
    different, so a whole-frame statistic separates them without ever touching the object.
    That is not a hypothetical — it is how `tophat` came first in the pooled ranking
    (+0.0043) and then fell to −0.0172 here.

    Inside one video the illumination, the scope, the patient and the centre are held
    fixed, so the only thing that differs between a positive and a negative frame is the
    object. Whatever survives this is about the object; whatever evaporates was scenery.

    Returns one row per (class, video, transform) with its best descriptor's separation.
    """
    classes = classes or [c[4:] for c in labels.columns if c.startswith("has_")]
    rows = []
    for klass in classes:
        has = labels[f"has_{klass}"].to_numpy(dtype=bool)
        for vid, g in labels.groupby("video"):
            m = has[labels.index.get_indexer(g.index)]
            if m.sum() < min_per_side or (~m).sum() < min_per_side:
                continue
            sub = features.loc[g.index]
            for col in sub.columns:
                x = sub[col].to_numpy(dtype=np.float64)
                ok = np.isfinite(x)
                transform, descriptor = col.split("::", 1)
                rows.append({"klass": klass, "video": vid, "transform": transform,
                             "descriptor": descriptor,
                             "sep": abs(auc(x[ok & m], x[ok & ~m]) - 0.5),
                             "n_pos": int(m.sum()), "n_neg": int((~m).sum())})
    return pd.DataFrame(rows)


def rank_within_video(wv: pd.DataFrame) -> pd.DataFrame:
    """Paired ranking: each transform against `identity` on the SAME (class, video) cell.

    Paired because the cells differ enormously in difficulty; an unpaired mean would be
    dominated by which cells each transform happened to have.
    """
    best = wv.loc[wv.groupby(["klass", "video", "transform"])["sep"].idxmax()]
    w = best.pivot(index=["klass", "video"], columns="transform", values="sep")
    if "identity" not in w.columns:
        raise ValueError("`identity` must be in the bank — it is the paired baseline")
    out = []
    for t in w.columns:
        d = (w[t] - w["identity"]).dropna()
        out.append({"transform": t, "wv_sep": float(w[t].mean()),
                    "wv_delta": float(d.mean()),
                    "wv_wins": int((d > 0).sum()), "wv_cells": int(len(d))})
    return (pd.DataFrame(out).sort_values("wv_delta", ascending=False)
              .reset_index(drop=True))


def null_band(features: pd.DataFrame, labels: pd.DataFrame, n_perm: int = 20,
              seed: int = 20260720, **kw) -> pd.DataFrame:
    """The noise floor of `mean_delta`, by permuting the class labels.

    Why this is not optional: `mean_delta` takes a MAX over 18 descriptors and then a mean
    over cells, so even a transform carrying zero class information scores above 0 by
    chance — the max of noise is positive. Reading a ranking without this band is reading
    a sign with no denominator.

    ⚠️ **This band is a LOWER bound.** It permutes labels at the FRAME level, which
    destroys the clustering of frames within videos; the real sampling noise is wider
    because the effective n is videos, not frames (`RULES` §13). A transform that fails
    to clear this optimistic band is dead a fortiori — that is exactly what it is for.
    """
    rng = np.random.default_rng(seed)
    classes = [c for c in labels.columns if c.startswith("has_")]
    rows = []
    for i in range(n_perm):
        perm = labels.copy()
        for ds in perm["dataset"].unique():
            m = perm["dataset"] == ds
            idx = rng.permutation(int(m.sum()))
            perm.loc[m, classes] = perm.loc[m, classes].to_numpy()[idx]
        r = rank_transforms(separability(features, perm, **kw))
        r["perm"] = i
        rows.append(r)
    allp = pd.concat(rows, ignore_index=True)
    return (allp.groupby("transform")["mean_delta"]
                .agg(null_mean="mean", null_sd="std",
                     null_p95=lambda s: float(np.percentile(s, 95)),
                     null_max="max")
                .reset_index())


# ── 12e: is the effect CONDITIONAL on what the model already gets right? ──────
# The screen ranks transforms by present-vs-absent separability averaged over cells, and
# `CONTEXT.md` records the objection that this "averages conditional effects away": the
# eyeball pass reports each operator shining in a different scenario — helping when the
# object is conspicuous, hurting when it is camouflaged. If that is true it would explain
# the whole rung, branch A's -0.056 included, for zero GPU.
#
# 🔴 The literal test — recompute the screen inside the model's right pile and its wrong
# pile — is NOT measurable. Splitting 235 (class, video) cells in two leaves **14** with
# both piles clearing `min_per_side=15` (19 if every answer format is used instead of just
# `fo_class`, because the model was asked about only 4,486 of the 15,213 indexed frames).
# Reporting a sign inversion over 14 cells would manufacture exactly the kind of winner
# this rung has already caught twice. That negative is recorded, not worked around.
#
# What IS measurable is the adjacent question, and it decides as well or better:
# **does the transform's descriptor separate the frames the model gets right from the ones
# it fails, inside each video?** 37 of 38 videos clear the gate. Crossed with the screen it
# gives a 2x2 — a transform that separates present/absent but NOT right/wrong carries a
# signal orthogonal to the model's actual failures and will not help it.


def _cell_summary(rows: list[dict]) -> pd.DataFrame:
    """Per (cell, transform): the BEST descriptor and the MEAN over descriptors.

    🔴 Both, deliberately. `wv_delta` reports only the best of 18 descriptors, and 12d bis
    showed that statistic is what made the edge family look "strongly negative": those
    pipelines raise every descriptor while lowering the top one, because they collapse 18
    diverse measurements onto a single axis and a maximum reads compression as loss. Any
    new metric on this page reports both or repeats the mistake.
    """
    df = pd.DataFrame(rows)
    return (df.groupby(["cell", "transform"])["sep"]
              .agg(sep_max="max", sep_mean="mean").reset_index())


def _paired_vs_identity(cells: pd.DataFrame) -> pd.DataFrame:
    """Rank transforms against `identity` on the SAME cells, paired (cells differ in difficulty)."""
    out = []
    for stat in ("sep_max", "sep_mean"):
        w = cells.pivot(index="cell", columns="transform", values=stat)
        if "identity" not in w.columns:
            raise ValueError("`identity` must be in the bank — it is the paired baseline")
        for t in w.columns:
            d = (w[t] - w["identity"]).dropna()
            out.append({"transform": t, "stat": stat, "value": float(w[t].mean()),
                        "delta": float(d.mean()), "wins": int((d > 0).sum()),
                        "cells": int(len(d))})
    return (pd.DataFrame(out).pivot(index="transform", columns="stat")
              .pipe(lambda x: x.set_axis([f"{b}_{a}" for a, b in x.columns], axis=1))
              .reset_index()
              .sort_values("sep_max_delta", ascending=False, ignore_index=True))


def right_wrong_by_video(features: pd.DataFrame, verdict: pd.DataFrame,
                         min_per_side: int = 15) -> pd.DataFrame:
    """(A) Inside each video, does each transform separate model-RIGHT from model-WRONG frames?

    `verdict` is indexed by frame key with a `video` column and a boolean `right`. The cell
    is the video, so illumination, scope, patient and centre are held fixed exactly as in
    `within_video` — what differs between the two piles is whatever made the model fail.

    ⚠️ This conditions on an OUTCOME. The two piles are not randomly composed: they differ
    in class mix and in scene difficulty, so a transform can separate them without touching
    the object at all. `right_wrong_null` is the control that says how much of any result is
    that composition, and it is not optional here.
    """
    rows = []
    for vid, g in verdict.groupby("video"):
        m = g["right"].to_numpy(dtype=bool)
        if m.sum() < min_per_side or (~m).sum() < min_per_side:
            continue
        sub = features.loc[features.index.intersection(g.index)]
        m = g.loc[sub.index, "right"].to_numpy(dtype=bool)
        for col in sub.columns:
            x = sub[col].to_numpy(dtype=np.float64)
            ok = np.isfinite(x)
            transform, descriptor = col.split("::", 1)
            rows.append({"cell": vid, "transform": transform, "descriptor": descriptor,
                         "sep": abs(auc(x[ok & m], x[ok & ~m]) - 0.5)})
    return _paired_vs_identity(_cell_summary(rows))


def right_wrong_within_class(features: pd.DataFrame, verdict: pd.DataFrame,
                             labels: pd.DataFrame, classes: list[str] | None = None,
                             min_per_side: int = 15) -> pd.DataFrame:
    """(B) The same, restricted to frames that CONTAIN the class — the closest to the literal question.

    ⚠️ **EXPLORATORY, NOT CONCLUSIVE.** Only 19 (class, video) cells clear the gate, against
    235 in the screen. It is reported so the literal hypothesis is on the record with a
    number, not so it can be cited as a result.
    """
    classes = classes or [c[4:] for c in labels.columns if c.startswith("has_")]
    rows = []
    for klass in classes:
        for vid, g in verdict.groupby("video"):
            idx = g.index.intersection(labels.index[labels[f"has_{klass}"]])
            gg = g.loc[idx]
            m = gg["right"].to_numpy(dtype=bool)
            if m.sum() < min_per_side or (~m).sum() < min_per_side:
                continue
            sub = features.loc[features.index.intersection(gg.index)]
            m = gg.loc[sub.index, "right"].to_numpy(dtype=bool)
            for col in sub.columns:
                x = sub[col].to_numpy(dtype=np.float64)
                ok = np.isfinite(x)
                transform, descriptor = col.split("::", 1)
                rows.append({"cell": f"{klass}|{vid}", "transform": transform,
                             "descriptor": descriptor,
                             "sep": abs(auc(x[ok & m], x[ok & ~m]) - 0.5)})
    return _paired_vs_identity(_cell_summary(rows))


def right_wrong_null(features: pd.DataFrame, verdict: pd.DataFrame, n_perm: int = 20,
                     seed: int = 20260720, **kw) -> pd.DataFrame:
    """Noise floor of (A), permuting the right/wrong verdict INSIDE each video.

    Permuting within video preserves each video's accuracy and its frame composition, so
    what the band measures is exactly the chance separation a max-over-18 buys — the same
    reason `null_band` exists. Not a lower bound in the same way that one is: the video
    clustering is kept, because the permutation happens inside it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_perm):
        v = verdict.copy()
        for vid in v["video"].unique():
            m = v["video"] == vid
            v.loc[m, "right"] = rng.permutation(v.loc[m, "right"].to_numpy())
        r = right_wrong_by_video(features, v, **kw)
        r["perm"] = i
        rows.append(r)
    allp = pd.concat(rows, ignore_index=True)
    return (allp.groupby("transform")[["sep_max_delta", "sep_mean_delta"]]
                .agg(["mean", "std", lambda s: float(np.percentile(s, 95))])
                .pipe(lambda x: x.set_axis(
                    [f"null_{a}_{'p95' if 'lambda' in b else b}" for a, b in x.columns], axis=1))
                .reset_index())
