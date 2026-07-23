"""Rung 14 — appearance augmentation for the TRAINING images only.

Library only. No launcher, no GPU, no model. Every function here is pure
``PIL.Image -> PIL.Image`` and is verifiable on a laptop with synthetic pixels,
which is the point: the augmentation is the variable under test, so it must not
be the thing nobody could check.

──────────────────────────────────────────────────────────────────────────────
WHY THIS FAMILY AND NOT ANOTHER  (every dose below carries a citation —
CONSTITUTION-adjacent wave rule #9: no hand-invented parameters)
──────────────────────────────────────────────────────────────────────────────
Rung 12 applied transforms at INFERENCE on a model fine-tuned without them and
measured −0.026 / −0.056 (monotone in dose). Medeiros 2026 (`literature/
preprocessing/FICHAS.md` Tier-1 #3, `p03`) names that mechanism — inference-only
transforms cost up to −31.6 pts — and Jong 2025 (preprocessing FICHAS Tier-1 #1,
`p01`) shows the fix is to put the transform in TRAINING as augmentation: vendor
enhancement settings swing endoscopic-AI sensitivity 9 pts / specificity 18 pts, and enhancement-based
augmentation collapses that to 1–2 pts (P<0.001). This module is that transform.

Scope, and what is deliberately absent:

* **Colour + illumination only.**
* **No sharpening.** ``unsharp`` was measured −0.056 monotone on this exact
  model (rung 12). Re-introducing it as augmentation would confound the arm with
  a family we already know is negative here.
* **No geometric ops, no crops.** Ramesh 2023 (*MedIA*, preprocessing FICHAS Tier-2 #12, `p12`) is the only
  surgical augmentation ablation there is: low-resolution Multi-Crop views HURT
  on Cholec80 (−3.5 % phase F1 at 4 crops, −4.5 % at 8) because "discriminative
  cues may be scattered in the entire image". Our aggregation questions are
  exactly that case.
* **No blur.** Wang 2024 (preprocessing FICHAS Tier-2 #19, `p19`) separates "Illumination Variability" from
  "Optical Distortions"; we take the illumination family only. Blur is also the
  primary axis of the zero-GPU quality diagnostic in ``quality_diag.py`` —
  augmenting it would confound the diagnostic with the arm.
* **Colour is the one family surgical CV certifies.** Ramesh 2023: "colour
  augmentation consistently and significantly improves representation quality"
  on laparoscopic data.

──────────────────────────────────────────────────────────────────────────────
THE TWO OPERATORS AND EVERY NUMBER IN THEM
──────────────────────────────────────────────────────────────────────────────
1. **White-balance error emulation** (primary).
   Afifi & Brown, ICCV 2019 (preprocessing FICHAS Tier-1 #9, `p09`) — the doctrinal source for "augment,
   don't preprocess", and the explicit finding that *generic colour jitter does
   not model WB error*. Their emulator is driven by the WB-sRGB rendering set,
   whose five FIXED colour temperatures are, verbatim from the paper (§5,
   "Robustness Strategies"): **2850 K, 3800 K, 5500 K, 6500 K, 7500 K**. That
   five-point grid is ``AugPolicy.wb_temps_K`` unchanged, and 5500 K is the
   identity anchor (the paper: "in most cases, one of these five fixed colour
   temperatures will be visually similar to the correct WB").

   ⚠️ **Declared deviation, ours, not theirs.** Afifi's mapping is a *learned*
   nonlinear 3×9 polynomial retrieved by kNN over a 17,970-image rendering
   dataset — it cannot be bundled offline and it needs their released data. We
   substitute a **Bradford chromatic-adaptation transform applied in LINEAR
   light** (inverse sRGB EOTF → XYZ → CAT → XYZ → sRGB EOTF). This is the
   physically-grounded version of the same manipulation and is strictly closer
   to it than the alternative the paper rejects: Afifi's objection is to a
   *diagonal correction applied directly on sRGB-encoded values*, which he calls
   "similar to multiplicative colour jittering". Ours is diagonal in LMS on
   LINEARISED radiance, which is the von Kries model of an illuminant change.
   What we do NOT model is the ISP's post-WB photo-finishing nonlinearity —
   stated as a known limit in ``context/14-appearance-aug/CONTEXT.md``.

   CCT → xy uses the standard Kim et al. cubic approximation of the Planckian
   locus (a textbook colour-science formula, not a tuned parameter).

2. **Illumination variability** (secondary).
   Wang 2024 (preprocessing FICHAS Tier-2 #19, `p19`) builds the
   endoscopy-calibrated corruption taxonomy with five severity levels and states
   verbatim: *"We employ the severity settings from the imagecorruptions library"* — i.e. Hendrycks & Dietterich's
   ImageNet-C constants — and *"a severity level of 0 signifies that the
   original image remains uncorrupted"*. Its **Illumination Variability** family
   is (Brightness, Dark, Contrast). We take that family, at severities 1–2 only,
   which is the dose ``literature/preprocessing/FICHAS.md`` → "What to steal"
   §3(b) prescribes.

   ImageNet-C constants, copied exactly:
     - brightness  ``c = [0.1, 0.2, 0.3, 0.4, 0.5][sev-1]`` → additive on the
       HSV V channel in [0,1]. Severity 1 = +0.10, severity 2 = +0.20.
     - contrast    ``c = [0.4, 0.3, 0.2, 0.1, 0.05][sev-1]`` → scale about the
       per-image mean. Severity 1 = 0.40, severity 2 = 0.30.

   ⚠️ **Declared deviation, ours, not theirs.** ImageNet-C has no "dark"
   corruption; Wang's taxonomy names *Dark* as a distinct illumination
   corruption but publishes no constants for it. We obtain it by **negating the
   brightness constant** (−0.10 / −0.20). Sign-symmetrisation is OURS and is
   pre-registered as such.

──────────────────────────────────────────────────────────────────────────────
DETERMINISM
──────────────────────────────────────────────────────────────────────────────
The draw is a pure function of ``(policy.seed, row_key)`` via BLAKE2b — NOT of
Python's ``hash()`` (salted per process) and NOT of iteration order. The same
(seed, qID) yields the same image on any machine, any run, any OS. ``draw()``
returns the parameters so they can be written to a manifest and audited.

FLAG OFF: ``apply(img, None, key)`` returns **the same object** ``img``. It is a
branch at the top of the function, and ``gate_flag_off_identity`` checks it with
``is`` — gated, not asserted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# ── literature constants ─────────────────────────────────────────────────────
# Afifi & Brown ICCV 2019 §5: the five fixed colour temperatures of the WB-sRGB
# rendering dataset. 5500 K is the identity/anchor.
AFIFI_WB_TEMPS_K: tuple[int, ...] = (2850, 3800, 5500, 6500, 7500)
WB_ANCHOR_K: int = 5500

# ImageNet-C / `imagecorruptions` constants, as used by Wang 2024
# (literature/preprocessing/FICHAS.md Tier-2 #19 = p19).
IMAGENET_C_BRIGHTNESS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
IMAGENET_C_CONTRAST: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1, 0.05)

# Wang 2024's "Illumination Variability" family. "dark" is our sign-flip of
# ImageNet-C brightness (declared deviation, see module docstring).
ILLUM_OPS: tuple[str, ...] = ("brightness", "dark", "contrast")

# ── colour-science matrices (textbook, not tuned) ────────────────────────────
_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float64)
_BRADFORD_INV = np.linalg.inv(_BRADFORD)


@dataclass(frozen=True)
class AugPolicy:
    """The augmentation policy. Defaults ARE the pre-registered arm.

    Every field is a literature constant or a declared-ours choice; nothing here
    was tuned against a number, because no number exists yet.
    """

    seed: int = 42

    # Afifi's five-point grid, uniform. 5500 K is in the grid, so exactly 1/5 of
    # rows keep their original white balance — the analogue of Afifi keeping the
    # correct-WB image alongside the ten emulated variations.
    wb_temps_K: tuple[int, ...] = AFIFI_WB_TEMPS_K

    # Wang 2024 (preprocessing p19): severity 0 = uncorrupted; 1..5 progressive.
    # literature/preprocessing/FICHAS.md "What to steal"
    # §3(b) prescribes a severity-1..2 draw → {0,1,2}, uniform, so 1/3 of
    # rows take no illumination corruption. The uniform weighting is OURS.
    illum_severities: tuple[int, ...] = (0, 1, 2)
    illum_ops: tuple[str, ...] = ILLUM_OPS

    # JPEG quality of the materialised augmented frame. NOT a free parameter:
    # rung 02's `_export` writes the cache frame at quality=95
    # (02-lora-sft/_models/lora_sft_train.py:133) and the augmented frame must
    # be encoded identically or JPEG quality becomes a second variable.
    jpeg_quality: int = 95

    tag: str = field(default="wb_afifi5+illum_sev12")

    def describe(self) -> str:
        return (f"{self.tag}(seed={self.seed}, wb_K={list(self.wb_temps_K)}, "
                f"illum_sev={list(self.illum_severities)}, ops={list(self.illum_ops)})")


# ── deterministic draw ───────────────────────────────────────────────────────

def row_rng(seed: int, key: str) -> np.random.Generator:
    """A generator that depends only on (seed, key) — never on the machine.

    BLAKE2b, not ``hash()``: CPython salts ``hash()`` per process, so a policy
    keyed on it would silently produce a different dataset on every run and the
    A/B would not be reproducible.
    """
    digest = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def draw(policy: AugPolicy, key: str) -> dict:
    """The seeded parameter draw for one row. Pure, auditable, manifest-ready."""
    rng = row_rng(policy.seed, key)
    t = int(policy.wb_temps_K[int(rng.integers(0, len(policy.wb_temps_K)))])
    sev = int(policy.illum_severities[int(rng.integers(0, len(policy.illum_severities)))])
    op = str(policy.illum_ops[int(rng.integers(0, len(policy.illum_ops)))])
    if sev == 0:
        op = "none"
    return {"wb_K": t, "illum_op": op, "illum_severity": sev,
            "identity": (t == WB_ANCHOR_K and sev == 0)}


# ── operator 1: white-balance error (Afifi ICCV 2019, Bradford substitute) ───

def _cct_to_xy(t: float) -> tuple[float, float]:
    """Planckian-locus chromaticity for a correlated colour temperature.

    Kim et al.'s cubic approximation — the standard closed form used across
    colour-science libraries. Valid 1667–25000 K, which covers Afifi's grid.
    """
    t = float(t)
    if not (1667.0 <= t <= 25000.0):
        raise ValueError(f"CCT {t} K outside the 1667–25000 K validity range")
    if t <= 4000.0:
        x = (-0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910)
    else:
        x = (-3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390)
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return float(x), float(y)


def _white_xyz(t: float) -> np.ndarray:
    x, y = _cct_to_xy(t)
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _srgb_to_linear(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 0, None), 1 / 2.4) - 0.055)


def wb_error(img: Image.Image, t_kelvin: float, anchor_K: float = WB_ANCHOR_K) -> Image.Image:
    """Emulate a white-balance error: re-render as if lit at ``t_kelvin``.

    Bradford chromatic adaptation from ``anchor_K`` to ``t_kelvin``, computed in
    LINEAR light (inverse sRGB EOTF applied first). ``t_kelvin == anchor_K`` is
    the identity and returns the input object untouched — the adaptation matrix
    would be the identity anyway, and short-circuiting keeps the 1-in-5 "correct
    WB" rows byte-exact instead of merely round-trip-exact.
    """
    if float(t_kelvin) == float(anchor_K):
        return img
    src, dst = _white_xyz(anchor_K), _white_xyz(t_kelvin)
    rho_s, rho_d = _BRADFORD @ src, _BRADFORD @ dst
    cat = _BRADFORD_INV @ np.diag(rho_d / rho_s) @ _BRADFORD
    m = (_XYZ2RGB @ cat @ _RGB2XYZ).astype(np.float64)

    lin = _srgb_to_linear(np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0)
    out = _linear_to_srgb(np.clip(lin @ m.T, 0.0, 1.0))
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))


# ── operator 2: illumination variability (ImageNet-C via Wang 2024) ──────────

def brightness(img: Image.Image, severity: int, sign: int = +1) -> Image.Image:
    """ImageNet-C ``brightness``: additive on the HSV V channel.

    ``sign=-1`` is Wang 2024's *Dark* corruption, obtained by negating the
    ImageNet-C constant (declared deviation — see module docstring).
    """
    if severity == 0:
        return img
    c = IMAGENET_C_BRIGHTNESS[severity - 1] * float(sign)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)          # float32: H∈[0,360], S,V∈[0,1]
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0.0, 1.0)
    out = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


def contrast(img: Image.Image, severity: int) -> Image.Image:
    """ImageNet-C ``contrast``: scale about the per-image mean by ``c``."""
    if severity == 0:
        return img
    c = IMAGENET_C_CONTRAST[severity - 1]
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1), keepdims=True)
    out = np.clip((arr - means) * c + means, 0.0, 1.0)
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


_ILLUM = {
    "none": lambda im, sev: im,
    "brightness": lambda im, sev: brightness(im, sev, +1),
    "dark": lambda im, sev: brightness(im, sev, -1),
    "contrast": lambda im, sev: contrast(im, sev),
}


# ── the policy application ───────────────────────────────────────────────────

def apply_params(img: Image.Image, params: dict) -> Image.Image:
    """Apply an already-drawn parameter dict. WB first, then illumination.

    Order is fixed and declared: the WB error is a property of the *capture*
    (it happens in the camera's ISP), the illumination corruption is a property
    of the *scene*, so chaining scene→capture would be backwards. It is also
    the order in which the two citations describe their own pipelines.
    """
    op = params.get("illum_op", "none")
    if op not in _ILLUM:
        raise ValueError(f"unknown illumination op {op!r}; expected one of {sorted(_ILLUM)}")
    out = wb_error(img, params["wb_K"])
    return _ILLUM[op](out, int(params.get("illum_severity", 0)))


def apply(img: Image.Image, policy: AugPolicy | None, key: str) -> Image.Image:
    """THE entry point. ``policy is None`` → the flag is OFF → the SAME object.

    This is a gate-able branch, not an assertion: ``gate_flag_off_identity``
    checks it with ``is``. Nothing downstream may re-encode, re-save or copy an
    image on the OFF path, or the "byte-identical" claim stops being true.
    """
    if policy is None:
        return img
    return apply_params(img, draw(policy, key))


# ── gates (run from the notebook; no GPU, no data) ───────────────────────────

def gate_flag_off_identity(img: Image.Image) -> dict:
    """G-A1 — flag OFF returns the input object itself, for any key."""
    outs = [apply(img, None, k) for k in ("a", "b", "heico__0001", "")]
    return {"n": len(outs), "same_object": all(o is img for o in outs),
            "PASS": all(o is img for o in outs)}


def gate_determinism(img: Image.Image, policy: AugPolicy, keys: list[str]) -> dict:
    """G-A2 — same (seed, key) → byte-identical pixels; different keys → drift.

    The second half matters as much as the first: an augmentation that is
    deterministic *and constant* is not an augmentation.
    """
    first = {k: np.asarray(apply(img, policy, k)) for k in keys}
    second = {k: np.asarray(apply(img, policy, k)) for k in keys}
    stable = all(np.array_equal(first[k], second[k]) for k in keys)
    params = {k: draw(policy, k) for k in keys}
    distinct = len({(p["wb_K"], p["illum_op"], p["illum_severity"]) for p in params.values()})
    return {"n_keys": len(keys), "stable_across_calls": stable,
            "distinct_draws": distinct, "PASS": stable and distinct > 1}


def gate_no_forbidden_ops(policy: AugPolicy) -> dict:
    """G-A3 — the policy cannot smuggle in a family this rung excluded.

    Cheap, but it is the check that stops a later edit from quietly adding the
    operator (`unsharp`) that is already measured negative on this model, or a
    crop, which Ramesh 2023 measured hurting surgical tasks.
    """
    allowed = set(ILLUM_OPS) | {"none"}
    bad = sorted(set(policy.illum_ops) - allowed)
    sev_ok = all(0 <= s <= 2 for s in policy.illum_severities)
    return {"unexpected_ops": bad, "severities_within_1_2": sev_ok,
            "PASS": not bad and sev_ok}


def dose_table(policy: AugPolicy, keys: list[str]) -> dict:
    """The realised marginal distribution of the draw over a key sample.

    Reported before training so the pre-registered dose is a measurement, not a
    belief. Expected under the defaults: P(WB identity) = 1/5, P(illum
    severity 0) = 1/3, P(both identity) = 1/15 ≈ 0.067.
    """
    from collections import Counter
    d = [draw(policy, k) for k in keys]
    return {
        "n": len(d),
        "wb_K": dict(sorted(Counter(x["wb_K"] for x in d).items())),
        "illum_op": dict(sorted(Counter(x["illum_op"] for x in d).items())),
        "illum_severity": dict(sorted(Counter(x["illum_severity"] for x in d).items())),
        "frac_untouched": sum(x["identity"] for x in d) / max(1, len(d)),
    }


__all__ = [
    "AugPolicy", "AFIFI_WB_TEMPS_K", "WB_ANCHOR_K", "IMAGENET_C_BRIGHTNESS",
    "IMAGENET_C_CONTRAST", "ILLUM_OPS",
    "row_rng", "draw", "wb_error", "brightness", "contrast", "apply_params", "apply",
    "gate_flag_off_identity", "gate_determinism", "gate_no_forbidden_ops", "dose_table",
]
