"""Step 4 — the VCD gate: does the language prior assert `Clip` MORE when the image is gone?

August plan, week 1 (`local/tasks/plan-accion.md`; spec in `roadmap-percepcion-rl.md` B.3).
Not a rung — the ladder is closed ([[august-plan-closes-the-ladder]]). This module measures;
the notebook states the pre-registered verdict.

**What VCD would do.** ``logit_vcd = (1+alpha)*logit(real) - alpha*logit(degraded)``, applied at
decode time. It touches no weights, so flag-off is byte-identical and no rung is invalidated.

**The three outcomes, and why "unchanged" is the trap.** On frames where `Clip` is a false
positive, compare the SOFTMAX MASS SHARE of `Clip` (never the raw logit — what picks the token is
the ranking) between the real frame and a degraded one:

  p(Clip) SINKS with noise  -> the model does use the image; the error is elsewhere  -> DIES
  p(Clip) UNCHANGED         -> (1+a)*L - a*L = L, VCD is an exact no-op               -> DIES
  p(Clip) RISES with noise  -> the prior asserts Clip harder without pixels           -> BUILD

🔴 **Why `manipulation_check` exists and is blocking.** "Unchanged" is also what a degradation
too weak to affect the model looks like. Without evidence that the corruption moved the output
distribution *at all*, a null is uninterpretable and would read as a legitimate kill. This is the
same defect Rodrigo's negative-control amendment fixes in step 6.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

__all__ = ["class_mass", "degrade", "manipulation_check", "verdict"]


def degrade(image: Image.Image, sigma: float, seed: int = 42) -> Image.Image:
    """Additive Gaussian pixel noise — the pre-declared corruption.

    ``sigma`` is in 0-255 units. Chosen over blur or masking because it is the corruption
    VCD's own formulation assumes (a diffusion-style forward step) and because it has ONE
    scalar to pre-declare. **The strength is a pre-registered parameter, not a tuned one:**
    sweeping sigma until a verdict appears is the multiplicity problem wearing a lab coat.
    Report the sweep as exploratory, decide on the declared value.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(image, dtype=np.float32)
    out = a + rng.normal(0.0, sigma, a.shape).astype(np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _first_token_ids(processor, names: list[str]) -> dict[str, list[int]]:
    """Vocabulary ids whose decoded text starts a given class name.

    A class is not one token. `Clip` may be reachable as "Clip", " Clip", "Cl"… so the mass we
    want is the SUM over every first-token that could begin the name, in both the bare and
    space-prefixed forms. Taking a single id would understate the mass and bias the gate.
    """
    tok = processor.tokenizer
    out: dict[str, list[int]] = {}
    for n in names:
        ids = set()
        for variant in (n, " " + n, n.lower(), " " + n.lower()):
            enc = tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[n] = sorted(ids)
    return out


@torch.no_grad()
def class_mass(engine, image: Image.Image, question: str, names: list[str]) -> dict:
    """Softmax mass share of each class name at the FIRST generated position.

    Returns ``{name: share}`` plus ``_entropy`` and ``_probs_top`` for the manipulation check.
    Mass share — not the logit — because the decision is a ranking over the normalised
    distribution, and a logit shift that moves every candidate equally changes nothing.
    """
    from qwen_vl_utils import process_vision_info

    messages = engine._messages(image, question)
    text = engine.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = engine.processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(engine.model.device)

    logits = engine.model(**inputs).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    ids = _first_token_ids(engine.processor, names)
    out = {n: float(probs[torch.tensor(v, device=probs.device)].sum()) for n, v in ids.items()}
    p = probs[probs > 0]
    out["_entropy"] = float(-(p * p.log()).sum())
    out["_probs_top"] = float(probs.max())
    return out


def manipulation_check(pairs: list[tuple[dict, dict]], *, min_shift: float = 0.01) -> dict:
    """🔴 BLOCKING — did the corruption move the distribution at all?

    ``pairs`` is ``[(real, degraded), …]`` as returned by :func:`class_mass`.

    A verdict of "unchanged" is only readable if the degradation demonstrably perturbs the
    model. We check the mean absolute shift in output ENTROPY and in the top probability: a
    corruption the model ignores leaves both untouched. ``min_shift`` is pre-declared.
    """
    de = np.array([abs(d["_entropy"] - r["_entropy"]) for r, d in pairs])
    dp = np.array([abs(d["_probs_top"] - r["_probs_top"]) for r, d in pairs])
    moved = float(max(de.mean(), dp.mean()))
    return {
        "mean_abs_entropy_shift": float(de.mean()),
        "mean_abs_top_prob_shift": float(dp.mean()),
        "moved": moved,
        "min_shift": min_shift,
        "passes": bool(moved >= min_shift),
        "n": len(pairs),
    }


def verdict(real: np.ndarray, degraded: np.ndarray, *, eps: float = 0.02) -> dict:
    """The three-way pre-registered read on paired p(Clip) values.

    ``eps`` is the band within which the two are called UNCHANGED — pre-declared, because
    "identical" never happens in floating point and choosing the band afterwards chooses the
    verdict. Paired over the same frames, so the statistic is the mean of the differences.
    """
    d = np.asarray(degraded, dtype=float) - np.asarray(real, dtype=float)
    mean_d = float(d.mean())
    # video-clustered CIs belong to the notebook; this is the point estimate and the band
    if abs(mean_d) < eps:
        call, reading = "DIES", "unchanged -> (1+a)L - aL = L, VCD is an exact no-op"
    elif mean_d < 0:
        call, reading = "DIES", "sinks -> the model does use the image; the error is elsewhere"
    else:
        call, reading = "BUILD", "rises -> the prior asserts Clip harder without pixels"
    return {
        "mean_p_real": float(np.mean(real)),
        "mean_p_degraded": float(np.mean(degraded)),
        "mean_delta": mean_d,
        "eps": eps,
        "n": int(len(d)),
        "verdict": call,
        "reading": reading,
    }
