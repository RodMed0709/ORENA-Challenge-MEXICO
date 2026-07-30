"""Offline tests for ``src/frame/loss.py`` — the token-weighting hook.

Lives inside rung 22's ``_tools/`` because rung 22 is what measured the mechanism this module
acts on (repo rule: never a top-level ``tests/``). Pure torch, no GPU, no ms-swift, no model.

The two invariants under test are the ones that make the A/B single-variable:

  1. OFF is byte-identical — the factory returns ``None`` and no code of ours runs.
  2. Reweighting is MEAN-PRESERVING, because ``max_grad_norm`` is 1.0 and clipping is the one
     channel a loss rescale is not invariant to. A weighting that changed the magnitude would
     silently change the effective learning rate, and after rung 21 that is the confound we can
     least afford.

Run:  python experiments/22-loss-mass/_tools/test_loss_weighting.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "src", _ROOT / "vendor" / "orena-focus" / "src"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from frame import loss as L  # noqa: E402

IGN = L.IGNORE_INDEX


def _batch(rows, vocab=7, seed=0):
    """(logits, labels) for `rows` = list of supervised-token counts, padded to the longest."""
    torch.manual_seed(seed)
    width = max(rows) + 1  # +1 because weighted_ce shifts by one
    labels = torch.full((len(rows), width), IGN)
    for i, n in enumerate(rows):
        labels[i, 1:n + 1] = torch.randint(0, vocab, (n,))
    return torch.randn(len(rows), width, vocab), labels


def test_off_is_byte_identical():
    assert L.make_compute_loss_func() is None
    assert L.make_compute_loss_func(lambda lb: torch.ones_like(lb, dtype=torch.float32)) is None
    try:
        L.make_compute_loss_func(enabled=True)
    except ValueError:
        pass
    else:
        raise AssertionError("enabled=True with no weight_fn must raise -- a no-op hook is a silent variable")
    print("  [1] OFF returns None (byte-identical); enabled with no weight_fn raises  PASS")


def test_uniform_weights_reproduce_plain_ce():
    """The anchor. All-ones weights must give EXACTLY the trainer's own number."""
    logits, labels = _batch([3, 5, 2])
    w = torch.ones_like(labels, dtype=torch.float32)
    n_items = int(labels.ne(IGN).sum())

    got = L.weighted_ce(logits, labels, w, n_items)
    sl, tl = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    want = F.cross_entropy(sl.view(-1, sl.size(-1)).float(), tl.view(-1),
                           ignore_index=IGN, reduction="sum") / n_items
    assert torch.allclose(got, want, atol=1e-6), (got.item(), want.item())

    # ...and normalising all-ones changes nothing, so ON-with-neutral-weights is also identical.
    wn = L.normalise_weights(w, labels)
    assert torch.allclose(wn * labels.ne(IGN), w * labels.ne(IGN), atol=1e-6)
    assert torch.allclose(L.weighted_ce(logits, labels, wn, n_items), want, atol=1e-6)
    print("  [2] uniform weights reproduce plain CE exactly; normalising ones is a no-op  PASS")


def test_normalisation_is_mean_preserving():
    """Invariant 2: whatever the weight map, the MEAN weight over supervised tokens is 1.0."""
    _, labels = _batch([4, 1, 6, 2])
    mask = labels.ne(IGN)
    for gw in ({"number": 1.64}, {"number": 0.25, "fo_class": 3.0}, {"number": 10.0}):
        w = L.row_group_weights(labels, ["number", "fo_class", "number", "binary"], gw)
        mean = float((w * mask).sum() / mask.sum())
        assert abs(mean - 1.0) < 1e-5, (gw, mean)
    # unnormalised must NOT be mean-preserving -- otherwise the flag does nothing
    raw = L.row_group_weights(labels, ["number"] * 4, {"number": 3.0}, normalise=False)
    assert abs(float((raw * mask).sum() / mask.sum()) - 3.0) < 1e-5
    print("  [3] normalise=True keeps mean weight at 1.0 across weight maps; raw does not  PASS")


def test_reallocates_the_measured_164x_skew():
    """The actual lever: rung 22 measured `number` at 20.8% of gradient off 34.2% of rows."""
    # A batch shaped like the measured corpus: fo_class 4 tokens/row, number 2, binary 2.
    rows = [4] * 5 + [2] * 3 + [2] * 1
    groups = ["fo_class"] * 5 + ["number"] * 3 + ["binary"] * 1
    _, labels = _batch(rows)
    mask = labels.ne(IGN)

    def share(w, g):
        sel = torch.tensor([x == g for x in groups]).unsqueeze(1) & mask
        return float((w * sel).sum() / (w * mask).sum())

    flat = torch.ones_like(labels, dtype=torch.float32)
    base = share(flat, "number")                     # token share == gradient share today
    row_share = sum(1 for g in groups if g == "number") / len(groups)
    assert base < row_share, (base, row_share)       # the skew is present in the fixture

    # THE TRAP: `row_share / base` is the obvious multiplier and it UNDERSHOOTS, because
    # up-weighting a group also enlarges the denominator. Shown here so it stays shown.
    naive = L.row_group_weights(labels, groups, {"number": row_share / base})
    assert share(naive, "number") < row_share - 0.03, share(naive, "number")

    fixed = L.row_group_weights(labels, groups, {"number": L.weight_for_target_share(base, row_share)})
    assert abs(share(fixed, "number") - row_share) < 1e-5, share(fixed, "number")
    # and the reallocation came OUT of the other formats -- not a free lunch
    assert share(fixed, "fo_class") < share(flat, "fo_class")
    # And on the REAL measured numbers the diagnostic ratio is not the dose either:
    assert abs(L.weight_for_target_share(0.208, 0.342) - 1.98) < 0.01
    for bad in ((0.0, 0.3), (0.2, 1.0), (-0.1, 0.3)):
        try:
            L.weight_for_target_share(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"weight_for_target_share{bad} must raise")
    print(f"  [4] number gradient share {base:.3f} -> {share(fixed, 'number'):.3f} (= its row "
          f"share); naive target/base undershoots to {share(naive, 'number'):.3f}; the measured "
          f"1.64x diagnostic needs a 1.98x dose  PASS")


def test_segment_weights_and_missing_span():
    _, labels = _batch([6, 6])
    w = L.segment_weights(labels, [(5, 7), (0, 0)], answer_weight=8.0, normalise=False)
    assert torch.allclose(w[0, 5:7], torch.tensor(8.0)) and torch.allclose(w[0, 1:5], torch.tensor(1.0))
    # a row with NO answer span stays uniform -- never silently zeroed
    assert torch.allclose(w[1], torch.ones_like(w[1])), w[1]
    try:
        L.segment_weights(labels, [(0, 1)], answer_weight=2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("span/label row-count mismatch must raise")
    try:
        L.segment_weights(labels, [(0, 1), (0, 1)], answer_weight=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative answer_weight must raise")
    print("  [5] segment_weights up-weights the answer span; a missing span stays uniform  PASS")


def test_schedule_and_refusals():
    s = L.scale_schedule
    assert abs(s(0, 100, w_start=1.0, w_end=9.0) - 1.0) < 1e-9
    assert abs(s(100, 100, w_start=1.0, w_end=9.0) - 9.0) < 1e-9
    mid = s(50, 100, w_start=1.0, w_end=9.0)
    assert 4.9 < mid < 5.1, mid
    # clamped at both ends: a resumed/over-run trainer cannot extrapolate
    assert abs(s(500, 100, w_start=1.0, w_end=9.0) - 9.0) < 1e-9
    assert abs(s(-5, 100, w_start=1.0, w_end=9.0) - 1.0) < 1e-9
    assert abs(s(7, 0, w_start=1.0, w_end=9.0) - 9.0) < 1e-9  # total_steps 0 -> the end value
    # monotone toward the answer-heavy end, which is SCALe's whole claim
    vals = [s(i, 100, w_start=1.0, w_end=9.0) for i in range(0, 101, 10)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals

    _, labels = _batch([3])
    try:
        L.row_group_weights(labels, ["number"], {"number": -1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("a negative group weight must raise -- it reverses the gradient")
    try:
        L.row_group_weights(labels, ["number", "extra"], {})
    except ValueError:
        pass
    else:
        raise AssertionError("row_groups/labels length mismatch must raise")
    try:
        L.normalise_weights(torch.zeros_like(labels, dtype=torch.float32), labels)
    except ValueError:
        pass
    else:
        raise AssertionError("an all-zero weight map must raise, not drop every token")
    print("  [6] cosine schedule endpoints/clamping/monotonicity + four refusals  PASS")


def test_end_to_end_callback():
    """The factory's callable, with the shape transformers calls it with."""
    logits, labels = _batch([3, 4])
    fn = L.make_compute_loss_func(
        lambda lb, **kw: L.row_group_weights(lb, ["number", "fo_class"], {"number": 2.0}),
        enabled=True,
    )
    out = type("O", (), {"logits": logits})()
    n_items = int(labels.ne(IGN).sum())
    v_obj = fn(out, labels, num_items_in_batch=n_items)
    v_dict = fn({"logits": logits}, labels, num_items_in_batch=n_items)
    assert torch.allclose(v_obj, v_dict) and torch.isfinite(v_obj), v_obj
    # differs from the unweighted number -- the hook is actually doing something
    flat = L.weighted_ce(logits, labels, torch.ones_like(labels, dtype=torch.float32), n_items)
    assert not torch.allclose(v_obj, flat), "weighted loss must differ from the flat one"
    # None denominator falls back to the batch's own supervised-token count
    assert torch.isfinite(fn(out, labels))
    print("  [7] callback accepts obj/dict outputs, honours num_items_in_batch, falls back  PASS")


def main() -> int:
    print("frame.loss - offline (pure torch, no ms-swift, no GPU):")
    test_off_is_byte_identical()
    test_uniform_weights_reproduce_plain_ce()
    test_normalisation_is_mean_preserving()
    test_reallocates_the_measured_164x_skew()
    test_segment_weights_and_missing_span()
    test_schedule_and_refusals()
    test_end_to_end_callback()
    print("\nOK - all offline. The ms-swift ADAPTER is NOT covered here and must be smoked "
          "on the pod (3 steps, print the received args) before any training run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
