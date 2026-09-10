"""Rung 59 — what is the layer-24 cardinality probe WORTH, in score?

Rung 55 measured that the probe reads set cardinality better than the head emits it
(0.8849 vs 0.8504, cluster-balanced delta +0.0304, CI [+0.0046, +0.0569], 7 of 10 videos).
Its verdict closed with the gap this file exists to fill:

    "the probe is a DIAGNOSTIC, not a deployable, and the translation from cardinality
     accuracy to `bucket_mean` is UNMEASURED"

🔴 Cardinality is not the answer. `fo_class` is scored by EXACT SET EQUALITY, so knowing
that a frame holds two classes buys nothing unless it changes WHICH classes get emitted.
This measures that translation and nothing else. Zero GPU: rung 55's per-row probe output
and rung 47's per-question answers both already exist on disk.

## The asymmetry that decides how big this can be

A cardinality probe can only ever do ONE of two things to an emitted set:

  emitted LARGER than the probe's size   ->  TRUNCATE. Actionable: we know what to drop.
  emitted SMALLER than the probe's size  ->  EXTEND.  NOT actionable: cardinality says how
                                             many are missing, never which. Nothing in
                                             rung 55 identifies a class.

So the reachable ceiling is the truncation half alone, and that is what `truncatable`
reports. The extension half is the measurement rung 55 never made — whether layer 24
encodes class IDENTITY and not just count — and it is the next probe, not this one.

Rung 55's own OOD confusion says the truncation half is where the model errs: on gold
size 1 it emitted size 2 thirty-seven times and size 3+ five times; on gold size 2 it
emitted 3+ twenty times. The probe never predicted 3+ at all.

## Truncation rule, and why this one

Keep the FIRST n classes in the model's own emission order. That order is the model's
own ranking — it is what an autoregressive head puts first — and using it means the
correction adds no information the deployed system would not have. `oracle` reports what
a perfect chooser would get from the same truncation budget, which bounds how much of the
gap is the RULE and how much is the cardinality signal itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# rung 55's per-row probe output, committed
PROBE_ROWS = "experiments/55-cardinality-probe/RESULTS_ood_rows.csv"
# A2 ep3 on the full 6,252 — the SAME checkpoint rung 55 probed, on the box
INSPECT = ("/mnt/storage/uaq_user/rung47/runs/47_a2_ep5_v1/eval_full6252/"
           "47_a2_ep5_v1_ep3_bridge/inspect.csv")


def _reader():
    """`metrics.read_fo_class` and its class list — the ONLY legal reader (RULES §8b)."""
    from frame.metrics import _load_fotype, read_fo_class

    valid_names = tuple(_load_fotype().names())
    valid_lower = {n.lower(): n for n in valid_names}
    return read_fo_class, valid_lower


def _ordered_classes(text: str, valid_lower: dict[str, str]) -> list[str]:
    """The emitted classes in EMISSION ORDER.

    `read_fo_class` returns a frozenset, which is exactly right for scoring and exactly
    wrong here: truncation needs the order the model produced. So the set decides
    membership (one reader, no second parser) and the raw string decides order.
    """
    from frame.metrics import read_fo_class

    parsed = read_fo_class(text, valid_lower)
    if parsed is None:
        return []
    low = str(text).lower()
    return sorted(parsed, key=lambda c: (low.find(c.lower()) if c.lower() in low else 10**6, c))


def main() -> int:
    read_fo_class, valid_lower = _reader()

    probe = pd.read_csv(PROBE_ROWS)
    insp = pd.read_csv(INSPECT, usecols=["qID", "answer_format", "ground_truth", "our_answer"])
    df = probe.merge(insp, on="qID", how="inner")
    assert len(df) == len(probe), f"joined {len(df)} of {len(probe)} probe rows"

    rows = []
    for r in df.itertuples():
        gold = read_fo_class(r.ground_truth, valid_lower)
        emitted_set = read_fo_class(r.our_answer, valid_lower)
        if gold is None or emitted_set is None:
            continue
        emitted = _ordered_classes(r.our_answer, valid_lower)
        n = int(r.probe_pred)
        # the correction, exactly as a deployed decoder would apply it
        corrected = frozenset(emitted[:n]) if len(emitted) > n else emitted_set
        # what a perfect chooser gets from the SAME budget — bounds the rule, not the signal
        oracle = frozenset(list(gold)[:n]) if len(emitted) > n else emitted_set
        rows.append({
            "qID": r.qID, "video": r.video,
            "gold_size": int(r.gold_size), "probe_pred": n, "emitted_size": len(emitted),
            "base_correct": emitted_set == gold,
            "corr_correct": corrected == gold,
            "oracle_correct": oracle == gold,
            "truncatable": len(emitted) > n,
        })

    out = pd.DataFrame(rows)
    base, corr, orac = out.base_correct.mean(), out.corr_correct.mean(), out.oracle_correct.mean()
    tr = out.truncatable

    print(f"rows scored                {len(out)} over {out.video.nunique()} videos")
    print(f"exact-set, model alone     {base:.4f}")
    print(f"exact-set, probe-corrected {corr:.4f}   delta {corr - base:+.4f}")
    print(f"exact-set, oracle truncate {orac:.4f}   delta {orac - base:+.4f}  <- ceiling of the RULE")
    print()
    print(f"truncatable rows           {tr.sum()} ({tr.mean():.1%})  <- the only half a count can fix")
    print(f"  of those, fixed          {(out.corr_correct & ~out.base_correct & tr).sum()}")
    print(f"  of those, broken         {(~out.corr_correct & out.base_correct & tr).sum()}")
    print(f"extension-only rows        {(out.emitted_size < out.probe_pred).sum()}"
          "  <- needs class IDENTITY, which rung 55 never measured")

    dest = Path("experiments/59-cardinality-decoder/RESULTS_translation.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
