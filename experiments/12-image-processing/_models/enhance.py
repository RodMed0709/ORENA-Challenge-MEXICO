"""Rung 12 — image enhancements applied before the vision encoder.

Library only. The flag lives in ``frame.config``; this module just does the pixels.

Design note — why unsharp and not a pure edge filter: a Sobel/Laplacian map REPLACES
the image with something else, which is maximally out-of-distribution for a ViT
trained on surgical frames. Rung 05 measured that a black frame degrades "by rarity,
not only by absence of information". Unsharp PRESERVES the image and only amplifies
its high-frequency content, so a null result is attributable to the hypothesis rather
than to appearance shock.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def unsharp(img: Image.Image, radius: float = 2.0, amount: float = 1.0) -> Image.Image:
    """``out = img + amount * (img - gaussian(img, radius))``.

    Works in float32 and clips once at the end. An aggressive unsharp saturates, and
    saturation would change the image statistics through a channel other than the one
    under test — so the clip is deliberate and singular, not incidental.
    """
    if amount == 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    blurred = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=radius, sigmaY=radius)
    out = arr + amount * (arr - blurred)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def specular(img: Image.Image, amount: float = 1.0) -> Image.Image:
    """Amplify bright, desaturated regions — the metallic-clip signature.

    `clip` is the most frequent class in the index (4,957 frames) and the only class
    whose `specular_frac` separates presence from absence in BOTH datasets.
    """
    if amount == 0:
        return img
    arr = np.asarray(img, dtype=np.float32)
    hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
    s, v = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    mask = ((v > 0.85) & (s < 0.20)).astype(np.float32)[..., None]
    out = arr + amount * mask * (255.0 - arr) * 0.5
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


_KINDS = {"unsharp": unsharp, "specular": specular}


def apply(img: Image.Image, kind: str, amount: float = 1.0) -> Image.Image:
    """Dispatch by name. Unknown names raise — a typo must not silently no-op."""
    if kind not in _KINDS:
        raise ValueError(f"unknown enhancement {kind!r}; expected one of {sorted(_KINDS)}")
    return _KINDS[kind](img, amount=amount)
