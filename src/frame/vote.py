"""Sampling spread over k candidate answers — voting (rung 10) and the entropy gate (step 5).

Promoted from ``experiments/10-self-consistency/_models/vote.py`` on 2026-08-05 because the
August plan's **step 5** needs it and `_models/` is folder-private by
`EXPERIMENT_REPO_STRUCTURE_SPEC`. Rung 10's module is now a re-export shim so its notebook
stays reproducible byte-for-byte.

**What changed in the promotion, and why.** Rung 10 only ever asked about ``number``, so
``diversity`` hard-coded ``parse_number``. Step 5 requires the two gate numbers **broken down
by format — ``number``, ``fo_class`` and ``binary`` separately** — so the spread functions now
take a ``key`` that maps a raw generation to a hashable canonical answer (or ``None`` when it
does not parse). ``number`` keeps the old parser and its behaviour is unchanged.

🔴 **The measurement rung 10 could not make.** ``t0_diversity`` averages the per-question
spread and throws the rows away, so no per-question record has ever existed
(`local/tasks/plan-accion.md`, step 5). ``per_question`` is the same computation **without the
mean**; the CSV it feeds is the artifact step 5 is actually for.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from frame.parsing import parse_number

__all__ = [
    "diversity",
    "key_binary",
    "key_fo_class",
    "key_number",
    "pass_at_k",
    "per_question",
    "t0_diversity",
    "vote_number",
    "zero_advantage",
]

# ── keys: raw generation -> canonical answer, or None when it does not parse ──────
# One per answer_format. `None` means "unparseable", which is NOT the same as "wrong":
# the SDK scores a format failure as incorrect, so an unparseable sample still counts as
# a rollout with reward 0 and must not be silently dropped from the denominator.


def key_number(text: str) -> float | None:
    """``number`` — rung 10's parser, unchanged."""
    v = parse_number(text)
    return None if np.isnan(v) else float(v)


def key_fo_class(text: str, valid_lower: dict[str, str]) -> frozenset[str] | None:
    """``fo_class`` — the scorer compares **sets**, so the canonical answer is a frozenset.

    Delegates to ``metrics.read_fo_class`` so there is exactly one reader; comparing raw
    strings here would count ``"Clip, Specimen"`` and ``"Specimen, Clip"`` as two distinct
    rollouts and manufacture diversity that the reward function does not see.
    """
    from frame.metrics import read_fo_class

    return read_fo_class(text, valid_lower)


def key_binary(text: str) -> str | None:
    """``binary`` — first yes/no token, case- and punctuation-insensitive."""
    t = text.strip().lower().lstrip("*# ").rstrip(".,!*# ")
    for token in ("yes", "no", "true", "false"):
        if t == token or t.startswith(token + " ") or t.startswith(token + ","):
            return {"true": "yes", "false": "no"}.get(token, token)
    return None


# ── voting (rung 10) ──────────────────────────────────────────────────────────────


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


# ── spread ────────────────────────────────────────────────────────────────────────


def diversity(
    samples: Sequence[str], key: Callable[[str], Hashable | None] = key_number
) -> dict[str, float]:
    """Per-question spread of the k samples — the T0 quantities.

    ``mode_share`` near 1.0 means sampling produced nothing to vote on.
    ``n_unparsed`` is reported rather than hidden: it is the share of rollouts the reward
    function will score 0 on format alone.
    """
    parsed = [key(s) for s in samples]
    values = [v for v in parsed if v is not None]
    n_unparsed = float(len(parsed) - len(values))
    if not values:
        return {
            "n_unique": 0.0,
            "entropy": 0.0,
            "mode_share": float("nan"),
            "n_unparsed": n_unparsed,
        }
    counts = np.array(list(Counter(values).values()), dtype=float)
    p = counts / counts.sum()
    return {
        "n_unique": float(len(counts)),
        # max(0, ...) only to kill the -0.0 that a single-value distribution produces;
        # entropy is non-negative by construction and a "-0.000000" in a committed CSV
        # reads like a bug.
        "entropy": float(max(0.0, -(p * np.log2(p)).sum())),
        "mode_share": float(counts.max() / counts.sum()),
        "n_unparsed": n_unparsed,
    }


# ── the two step-5 gate numbers ───────────────────────────────────────────────────


def zero_advantage(correct: Sequence[bool]) -> bool:
    """True when all k rollouts earn the SAME reward — i.e. GRPO gets no gradient here.

    🔴 **Defined on the reward, not on the answer text, and the difference decides the
    gate.** Two rollouts that disagree textually but are both wrong produce identical
    rewards and therefore zero advantage. Measuring answer-identity instead would report
    more usable signal than GRPO can actually see — an error in the *optimistic*
    direction, which for a kill gate is the dangerous one.
    """
    return len(set(bool(c) for c in correct)) <= 1


def pass_at_k(correct: Sequence[bool]) -> bool:
    """True when at least one of the k rollouts is correct — the ceiling sampling can reach."""
    return any(bool(c) for c in correct)


def per_question(
    records: pd.DataFrame,
    keys: dict[str, Callable[[str], Hashable | None]] | None = None,
) -> pd.DataFrame:
    """One row per question — the artifact rung 10 averaged away.

    ``records`` needs ``qID``, ``answer_format``, ``samples`` (list of k strings),
    ``samples_correct`` (list of k bools, from the SDK verifier) and ``greedy_correct``.

    🔴 ``samples_correct`` must come from the **SDK's** verifier, not from a local string
    comparison. `RULES` §EVAL: score ONLY via the vendor path; a hand-rolled comparison
    here would make the gate measure our parser instead of the reward.
    """
    keys = keys or {"number": key_number, "binary": key_binary}
    required = {"qID", "answer_format", "samples", "samples_correct", "greedy_correct"}
    missing = required - set(records.columns)
    assert not missing, f"per_question needs {sorted(missing)}"

    rows: list[dict[str, Any]] = []
    for r in records.itertuples(index=False):
        fmt = r.answer_format
        key = keys.get(fmt)
        div = (
            diversity(r.samples, key)
            if key is not None
            else {"n_unique": np.nan, "entropy": np.nan, "mode_share": np.nan, "n_unparsed": np.nan}
        )
        rows.append(
            {
                "qID": r.qID,
                "answer_format": fmt,
                "k": len(r.samples),
                **div,
                "zero_advantage": zero_advantage(r.samples_correct),
                "pass_at_k": pass_at_k(r.samples_correct),
                "greedy_correct": bool(r.greedy_correct),
                "n_correct": int(sum(bool(c) for c in r.samples_correct)),
            }
        )
    return pd.DataFrame(rows)


def gate_by_format(pq: pd.DataFrame) -> pd.DataFrame:
    """Step 5's two numbers, per ``answer_format`` — reported, never averaged into one.

    ``zero_advantage_frac`` >= 0.60 kills the arm (no gradient).
    ``pass_at_k - greedy`` < +0.05 kills the arm (no ceiling).
    Both thresholds are the plan's, pre-declared; this function does NOT apply them —
    it produces the numbers and the notebook states the verdict.
    """
    g = pq.groupby("answer_format", dropna=False)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "zero_advantage_frac": g.zero_advantage.mean(),
            "pass_at_k": g.pass_at_k.mean(),
            "greedy": g.greedy_correct.mean(),
            "entropy_mean": g.entropy.mean(),
            "mode_share_mean": g.mode_share.mean(),
            "unparsed_mean": g.n_unparsed.mean(),
        }
    )
    out["headroom"] = out["pass_at_k"] - out["greedy"]
    return out.reset_index()


# ── rung 10's summary, kept so its notebook still runs ────────────────────────────


def t0_diversity(records: pd.DataFrame) -> pd.DataFrame:
    """T0 summary per temperature. Produces numbers, NOT a verdict.

    ``records`` needs ``temperature``, ``true``, ``greedy`` and ``samples``
    (a list of k strings per row).

    The decisive column is ``mode_closer_than_greedy``: the fraction of questions
    where the voted answer lands nearer the truth than greedy did. Diversity alone
    is not evidence — a model can be diverse and still centred on the wrong value,
    which is the scenario 05b and 05c predict for us.

    ⚠️ **This function averages and discards the rows.** That is what left rung 10 with no
    per-question record; use ``per_question`` for anything new.
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
