"""Token-weighted training loss — the ONE hook three roadmap levers need.

Pure ``torch``. No ms-swift import, no GPU, no model: everything here is a function of a
per-token loss vector, a label tensor and a weight mask, so it is unit-testable offline and the
framework adapter stays a thin, separately-validated shim (see ``make_compute_loss_func``).

──────────────────────────────────────────────────────────────────────────────
Why this module exists
──────────────────────────────────────────────────────────────────────────────
``transformers`` >= 4.46 (we pin 4.57) normalises cross-entropy by ``num_items_in_batch`` — the
answer-token count of the whole effective batch. **Every answer token carries the same gradient
weight, so a row's influence is proportional to how many tokens its answer has.** Measured on
rung 18's real ``train.jsonl`` ([[loss-mass-is-token-weighted]], 14,415 rows, 47,381 tokens):

    format        rows    % rows   % of gradient   tokens/row
    fo_class      7,343    50.9%       62.4%          4.03
    number        4,929    34.2%       20.8%          2.00     <- under-weighted 1.64x
    open/MC         913     6.3%       11.6%          6.02
    binary        1,230     8.5%        5.2%          2.00

``number`` is **80.4% of the `aggregation` bucket** and carries the whole scoring deficit — and it
is the format the optimiser attends to least, by construction, and no rung ever knew.

Three roadmap levers are the same edit through this one hook:

  * **2a SCALe** (FICHAS §v02) — weight ``<answer>`` against ``<think>`` on a schedule. Without
    it a CoA scaffold puts ~99% of the gradient on prose: a 70–120-token trace against a ~2.4-token
    gold. That is [[coa-sft-published-null]]'s mechanism, at ~60x the magnitude of the format skew.
  * **2b NTL** (FICHAS §v12) — an auxiliary numeric-proximity term, masked to ``number`` rows.
    🔒 Gated on `number-probe`: a proximity loss needs a distribution to widen.
  * **2d non-dilution** (FICHAS §v05) — per-format weights, to stop discounting ``number`` 1.64x.

──────────────────────────────────────────────────────────────────────────────
Two invariants this module enforces, both from measurement
──────────────────────────────────────────────────────────────────────────────
1. **OFF is byte-identical.** ``make_compute_loss_func`` returns ``None`` when disabled, so the
   trainer takes its ordinary branch and no code of ours runs. Repo rule: a flag defaulting to
   OFF must be byte-identical, or the A/B has two variables.

2. 🔴 **Reweighting is MEAN-PRESERVING by default, because ``max_grad_norm`` is 1.0.** Rung 22
   measured that clipping is on (transformers' default; ``_swift_args`` never sets it), and
   gradient clipping is *the* channel a loss rescale is not invariant to. A weighting that also
   changes the loss MAGNITUDE silently changes the effective learning rate, and the arm then
   confounds "we reweighted the formats" with "we trained hotter" — which, after rung 21 showed a
   single lr flag is worth +0.058, is the one confound we can least afford. ``normalise=True``
   rescales the weights so their sum over supervised tokens equals the token count, i.e. the mean
   weight is exactly 1 and only the *distribution* of gradient changes.

⚠️ Not ``--loss_scale``. Rung 22 read the path: that flag multiplies the per-token loss
(``seq2seq_trainer.py:167-168``) and leaves the denominator alone — a ~3.3x magnitude shrink,
exactly the failure invariant 2 exists to prevent. The right hook is ``compute_loss_func``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

IGNORE_INDEX = -100


def _torch():
    import torch  # noqa: PLC0415 — keeps `frame.metrics` importable with no torch installed
    return torch


# ── weight construction ───────────────────────────────────────────────────────

def normalise_weights(weights, labels, *, ignore_index: int = IGNORE_INDEX):
    """Rescale ``weights`` so the MEAN weight over supervised tokens is exactly 1.0.

    Invariant 2 above. Returns the weights unchanged in shape; supervised positions are those
    where ``labels != ignore_index``. A batch with no supervised token is returned untouched
    (there is nothing to normalise and raising would kill a legitimate padding-only micro-batch).
    """
    torch = _torch()
    mask = labels.ne(ignore_index)
    n = mask.sum()
    if n == 0:
        return weights
    total = (weights * mask).sum()
    if total <= 0:
        raise ValueError(
            "total weight over supervised tokens is <= 0; every supervised token would be "
            "dropped from the gradient. Check the weight map."
        )
    return weights * (n.to(weights.dtype) / total.to(weights.dtype))


def weight_for_target_share(base_share: float, target_share: float) -> float:
    """Multiplier that moves a group from ``base_share`` of the gradient to ``target_share``.

    🔴 **It is the ODDS RATIO, not ``target/base``** — and getting this wrong is easy, because
    ``target/base`` is the obvious guess and it silently undershoots. Up-weighting a group also
    enlarges the denominator, so::

        w = target(1 - base) / (base(1 - target))

    Worked on our own measurement ([[loss-mass-is-token-weighted]]): ``number`` holds **20.8%** of
    the gradient off **34.2%** of the rows, which the note describes as *"under-weighted 1.64x"*.
    That 1.64 is the ratio of the two SHARES — it is **not** the multiplier that fixes it. The
    multiplier is ``0.342 x 0.792 / (0.208 x 0.658)`` = **~1.98**. Applying 1.64 would leave
    ``number`` at ~0.31, short of its row share. Do not read the diagnostic ratio as a dose.
    """
    if not (0 < base_share < 1) or not (0 < target_share < 1):
        raise ValueError(
            f"shares must be strictly between 0 and 1 (got base={base_share}, "
            f"target={target_share}); a group with no gradient cannot be rescaled into one"
        )
    return (target_share * (1.0 - base_share)) / (base_share * (1.0 - target_share))


def row_group_weights(labels, row_groups: Sequence, group_weights: dict, *,
                      ignore_index: int = IGNORE_INDEX, normalise: bool = True):
    """Per-token weights from a per-ROW group label — lever 2d (per-format weighting).

    ``row_groups[i]`` is the group of row ``i`` (e.g. ``"number"``); ``group_weights`` maps group
    → multiplier. A group absent from the map weighs 1.0. Every supervised token of a row gets
    that row's weight, so this reallocates gradient BETWEEN formats without touching how tokens
    inside a row compare.

    ⚠️ This is a **reallocation, not a free lunch** ([[loss-mass-is-token-weighted]] §Scope):
    up-weighting ``number`` takes gradient from ``fo_class``, which is 71% of
    ``object_recognition`` — the other scored bucket. Any arm using this MUST be read on both.
    """
    torch = _torch()
    if len(row_groups) != labels.shape[0]:
        raise ValueError(f"row_groups has {len(row_groups)} entries, labels has {labels.shape[0]} rows")
    per_row = torch.tensor(
        [float(group_weights.get(g, 1.0)) for g in row_groups],
        dtype=torch.float32, device=labels.device,
    )
    if (per_row < 0).any():
        raise ValueError("negative group weight — a negative weight reverses the gradient")
    w = per_row.unsqueeze(1).expand_as(labels).clone()
    return normalise_weights(w, labels, ignore_index=ignore_index) if normalise else w


def segment_weights(labels, answer_spans: Sequence[tuple[int, int]], answer_weight: float, *,
                    ignore_index: int = IGNORE_INDEX, normalise: bool = True):
    """Per-token weights that up-weight the ``<answer>`` span — lever 2a (SCALe).

    ``answer_spans[i] = (start, end)`` is the half-open token range of row ``i``'s answer span, in
    the same coordinates as ``labels``. Tokens inside get ``answer_weight``; supervised tokens
    outside (the reasoning trace) get 1.0. A row whose span is ``(0, 0)`` — no answer span found —
    is left uniform rather than silently zeroed.

    🔴 A missing span is a FINDING, not a default: SCALe's own headline result is that malformed
    outputs (a generation with no ``<answer>`` tag) drop 4.76% -> 2.78%, and a missing tag scores a
    silent 0 for us. Callers should count how many rows came back ``(0, 0)`` and gate on it.
    """
    torch = _torch()
    if answer_weight < 0:
        raise ValueError("answer_weight must be >= 0")
    if len(answer_spans) != labels.shape[0]:
        raise ValueError(f"answer_spans has {len(answer_spans)} entries, labels has {labels.shape[0]} rows")

    w = torch.ones_like(labels, dtype=torch.float32)
    for i, (start, end) in enumerate(answer_spans):
        if end > start:
            w[i, start:end] = answer_weight
    return normalise_weights(w, labels, ignore_index=ignore_index) if normalise else w


def scale_schedule(step: int, total_steps: int, *, w_start: float, w_end: float) -> float:
    """SCALe's cosine anneal of the answer weight: reasoning-heavy first, answer-heavy later.

    v02's finding is that the SCHEDULE matters, not merely the reweighting — the model first
    learns the structure, then learns to get the answer right. Clamped at both ends so a resumed
    or over-run trainer cannot extrapolate past ``w_end``.
    """
    if total_steps <= 0:
        return float(w_end)
    t = min(max(step / total_steps, 0.0), 1.0)
    return float(w_end + (w_start - w_end) * 0.5 * (1.0 + math.cos(math.pi * t)))


# ── the loss itself ───────────────────────────────────────────────────────────

def weighted_ce(logits, labels, weights, num_items_in_batch=None, *,
                ignore_index: int = IGNORE_INDEX):
    """Cross-entropy with per-token ``weights``, normalised the way our trainer normalises.

    Shift-by-one is applied here (``logits[..., :-1, :]`` against ``labels[..., 1:]``), matching
    causal-LM convention; ``weights`` is indexed in LABEL coordinates and shifted with them.

    ``num_items_in_batch`` is the denominator the trainer passes — the supervised-token count of
    the WHOLE effective batch, which rung 22 verified to the token across 48 ``compute_loss``
    calls. Passing it through unchanged is what keeps this a pure reweighting: the same
    denominator, a different numerator. When it is ``None`` we fall back to the batch's own
    supervised-token count, which is what the trainer's own fallback does.
    """
    torch = _torch()
    import torch.nn.functional as F  # noqa: PLC0415

    sl = logits[..., :-1, :].contiguous()
    tl = labels[..., 1:].contiguous()
    tw = weights[..., 1:].contiguous()

    per_token = F.cross_entropy(
        sl.view(-1, sl.size(-1)).float(), tl.view(-1),
        ignore_index=ignore_index, reduction="none",
    )
    mask = tl.ne(ignore_index).view(-1)
    weighted = per_token * tw.view(-1) * mask

    denom = num_items_in_batch if num_items_in_batch is not None else mask.sum()
    denom = torch.as_tensor(denom, dtype=weighted.dtype, device=weighted.device)
    return weighted.sum() / denom.clamp(min=1)


def make_compute_loss_func(weight_fn=None, *, enabled: bool = False):
    """The trainer callback, or ``None`` when disabled.

    🔴 **``enabled=False`` returns ``None``, and that is the point.** ms-swift checks
    ``compute_loss_func`` *before* its default branch (``seq2seq_trainer.py:189``), so a ``None``
    means the trainer takes exactly the path it takes today and **no code of ours executes** —
    byte-identical, which is what makes the A/B single-variable.

    ``weight_fn(labels, **kwargs) -> Tensor`` builds the per-token weights; compose it from
    ``row_group_weights`` / ``segment_weights``. The returned callable follows the
    ``(outputs, labels, num_items_in_batch=None, **kwargs)`` shape transformers calls with.

    ⚠️ **The adapter is the part this repo cannot verify.** ms-swift is not installed locally
    (checked 2026-07-29: torch 2.13 and transformers 4.57.6 are, `swift` is not), so the exact
    call signature and the shape ms-swift hands over must be confirmed **on the pod** before any
    training run — a smoke of 3 steps printing the received args, exactly as rung 22 did to
    discharge its own precondition. Everything ABOVE this function is pure torch and is tested
    offline; this function is the seam.
    """
    if not enabled:
        return None
    if weight_fn is None:
        raise ValueError("enabled=True needs a weight_fn; an enabled no-op hook is a silent variable")

    def compute_loss(outputs, labels, num_items_in_batch=None, **kwargs):
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        weights = weight_fn(labels, **kwargs)
        return weighted_ce(logits, labels, weights, num_items_in_batch)

    return compute_loss


__all__ = [
    "IGNORE_INDEX", "make_compute_loss_func", "normalise_weights", "row_group_weights",
    "scale_schedule", "segment_weights", "weighted_ce",
]
