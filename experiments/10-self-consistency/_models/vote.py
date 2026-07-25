"""Rung 10 engine — self-consistency voting over sampled `number` answers.

Idea 11 was unblocked by [[latency-budget-is-pooled]]: the FRAME cap is POOLED
(``120 s + B × 5 s``), not per question, so k candidates are affordable against a
measured p99 of 0.352 s.

🔴 **The risk this module exists to measure first.** Self-consistency votes toward
the mode of the model's own distribution, and ours is measured to be *biased*:
rung 05b found **81.5 % of errors are under-counts** with the scale saturating at
~2, and rung 05c found that true values **2, 3 and 4 all share the same modal
prediction (1)**. If mass already sits on the wrong answer, voting reinforces the
error rather than correcting it — the exact mechanism that killed post-hoc
calibration ([[count-calibration-dead]]).

So ``t0_diversity`` runs *before* any scoring run, and can close the rung for the
cost of ~20 min of GPU. Only ``mode_moves_toward_truth`` licenses spending the
full pass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from frame.parsing import parse_number

__all__ = ["diversity", "t0_diversity", "vote_number"]


def vote_number(samples: Sequence[str]) -> float:
    """Majority vote over parsed integers. Ties break to the LOWEST value.

    The tie rule is deliberately the *conservative* one: the model under-counts
    (05b), so breaking ties upward would smuggle in a second variable — a bias
    correction — under cover of the voting rule. Any gain must come from the
    voting, not from a thumb on the scale.
    """
    values = [v for v in (parse_number(s) for s in samples) if not np.isnan(v)]
    if not values:
        return float("nan")
    counts = Counter(values)
    best = max(counts.values())
    return min(v for v, c in counts.items() if c == best)


def diversity(samples: Sequence[str]) -> dict[str, float]:
    """Per-question spread of the k samples — the T0 quantities.

    ``mode_share`` near 1.0 means sampling produced nothing to vote on, and the
    rung dies there.
    """
    values = [v for v in (parse_number(s) for s in samples) if not np.isnan(v)]
    if not values:
        return {"n_unique": 0.0, "entropy": 0.0, "mode_share": float("nan")}
    counts = np.array(list(Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return {
        "n_unique": float(len(counts)),
        "entropy": float(-(p * np.log2(p)).sum()),
        "mode_share": float(counts.max() / counts.sum()),
    }


def t0_diversity(records: pd.DataFrame) -> pd.DataFrame:
    """T0 summary per temperature. Produces numbers, NOT a verdict.

    ``records`` needs ``temperature``, ``true``, ``greedy`` and ``samples``
    (a list of k strings per row).

    The decisive column is ``mode_closer_than_greedy``: the fraction of questions
    where the voted answer lands nearer the truth than greedy did. Diversity alone
    is not evidence — a model can be diverse and still centred on the wrong value,
    which is the scenario 05b and 05c predict for us.
    """
    rows = []
    for temp, g in records.groupby("temperature"):
        div = pd.DataFrame([diversity(s) for s in g["samples"]])
        voted = g["samples"].map(vote_number).to_numpy()
        greedy = g["greedy"].map(parse_number).to_numpy()
        true = g["true"].to_numpy(dtype=float)
        rows.append(
            {
                "temperature": temp,
                "n": len(g),
                "n_unique_mean": div.n_unique.mean(),
                "entropy_mean": div.entropy.mean(),
                "mode_share_mean": div.mode_share.mean(),
                "acc_greedy": float((greedy == true).mean()),
                "acc_voted": float((voted == true).mean()),
                "mode_closer_than_greedy": float(
                    (np.abs(voted - true) < np.abs(greedy - true)).mean()
                ),
                "mode_farther_than_greedy": float(
                    (np.abs(voted - true) > np.abs(greedy - true)).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)
