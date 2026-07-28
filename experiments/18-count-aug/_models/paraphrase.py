"""Rung 18 · L2 — surface variation on the `number` question, and the format-tail dropout.

## What this lever is, and what it is NOT

**No new labels, therefore no new label noise.** Same frames, same golds; only the wording
of the question moves. That is the whole reason L2 can ride in the same run as L1 without
compounding the risk: L1 adds rows whose labels can be wrong, L2 adds none.

## Why the tail dropout is the point

Probe 16a found the fine-tuned checkpoint emits **`"1."`** — with a trailing period — where
`Number.verify` gates on `str.strip().isdigit()`, making every such answer **auto-incorrect**
(base rate 0.0000: we created it). Probe 16d then **narrowed the trigger**, and the
correction matters here:

> 16d asked the model REAL corpus questions under six surface variants and measured
> **0.0000 illegal answers for base, ep2 and ep3, on every variant and every format.**
> Paraphrasing is NOT the trigger; the question being out of distribution is.

So this module is deliberately two-sided:

* the **stem variants** are the cheap, measured-safe half — 16d says the model already
  tolerates them, so they cost nothing and buy robustness against the organizers'
  generator, which is not ours ([[open-class-vocabulary]] already records a possible
  train/test regime mismatch);
* the **format-tail dropout** is the half with a mechanism behind it. The format is glued to
  the literal string *"Please provide a number."*; strip it and the model emits `"1."`.
  Dropping the tail on a fraction of rows — while the target stays a bare integer — is the
  only way to teach that the ANSWER SHAPE is a property of the question's meaning rather
  than of one trailing sentence.

## Applied to `number` rows only — on purpose

L1 is `number`-only ([[zero-is-format-localized]]: the model already says "no" fluently, so
`binary` needs nothing). Scoping L2 the same way keeps `binary` and `fo_class` **untouched
inside this very run**, which turns them into a within-run control for probe 16d's readout:
if the illegal rate moves on `number` and not on the other two, the move is ours.

## Replacement, not duplication

A variant REPLACES the wording of an existing row; it never appends a row. The dataset size,
the steps per epoch and the optimizer trajectory therefore stay a function of L1's dose
alone, and the comparison to rung 06 stays readable.
"""

from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

# The tail the format is glued to. Written exactly as the corpus writes it, leading space
# included, so removing it leaves no double space and no orphaned punctuation.
NUMBER_TAIL = " Please provide a number."

# Every `number` question in the corpus is `How many <NP> appear in this frame?` — the two
# aggregate templates (`different foreign object instances` / `classes`) and the eight
# per-class ones all share it. Capturing the noun phrase is what lets one rule cover all ten.
_STEM_RE = re.compile(r"^How many\s+(?P<np>.+?)\s+appear in this frame\?$", re.I)


def _variant_are_there(np: str) -> str:
    return f"How many {np} are there in this frame?"


def _variant_visible(np: str) -> str:
    return f"How many {np} are visible in this frame?"


def _variant_frame_first(np: str) -> str:
    return f"In this frame, how many {np} appear?"


def _variant_count(np: str) -> str:
    return f"Count the {np} that appear in this frame."


def _variant_total(np: str) -> str:
    return f"What is the total number of {np} that appear in this frame?"


# Meaning-preserving by construction: each rewrites only the interrogative frame around an
# UNCHANGED noun phrase. A variant that touched the noun phrase would change what is asked —
# and would silently measure a different question.
STEM_VARIANTS: dict[str, callable] = {
    "s1_are_there": _variant_are_there,
    "s2_visible": _variant_visible,
    "s3_frame_first": _variant_frame_first,
    "s4_count": _variant_count,
    "s5_total": _variant_total,
}


def split_tail(question: str) -> tuple[str, str]:
    """``(stem, tail)`` — the tail is ``""`` when the question does not carry it."""
    q = str(question)
    if q.endswith(NUMBER_TAIL):
        return q[: -len(NUMBER_TAIL)], NUMBER_TAIL
    return q, ""


def paraphrase_stem(stem: str, variant: str) -> tuple[str, bool]:
    """Rewrite one stem. Returns ``(new_stem, matched)``.

    ``matched=False`` means the stem did not fit the corpus's ``How many ... appear in this
    frame?`` shape and was returned untouched. Callers MUST gate on the aggregate match rate:
    a rewrite rule that silently stops matching is indistinguishable from a lever that was
    never switched on — the rung-16 papermill trap, one layer down.
    """
    m = _STEM_RE.match(stem)
    if not m:
        return stem, False
    return STEM_VARIANTS[variant](m.group("np")), True


def _unit(key: str, salt: str, seed: int) -> float:
    """A stable uniform in [0,1) for one row and one decision.

    Keyed on the row's identity rather than on iteration order, so adding, removing or
    reordering rows never reshuffles the decisions taken for the others — the property that
    makes the flag-off/flag-on diff readable and the whole build reproducible.
    """
    h = hashlib.sha256(f"{seed}|{salt}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def apply(question: str, key: str, *, stem_rate: float, tail_dropout: float,
          seed: int = 42) -> dict:
    """Apply L2 to ONE `number` question.

    ``key`` is the row's stable identity (a qID for a real row, ``frame_key|class`` for a
    minted one). The two decisions draw from independent streams, so the realized rates are
    independent and each is separately auditable.

    Returns the rewritten question plus the flags a provenance table needs.
    """
    stem, tail = split_tail(question)
    names = sorted(STEM_VARIANTS)

    u_stem = _unit(key, "stem", seed)
    matched = False
    variant = "s0_verbatim"
    if u_stem < stem_rate:
        # Pick the variant from a SECOND stream: reusing u_stem would correlate "which
        # variant" with "how close to the threshold", concentrating one variant at the edge.
        variant = names[int(_unit(key, "which", seed) * len(names)) % len(names)]
        stem, matched = paraphrase_stem(stem, variant)
        if not matched:
            variant = "s0_verbatim"  # unmatched shape: recorded as untouched, never as done

    drop_tail = _unit(key, "tail", seed) < tail_dropout
    out_tail = "" if drop_tail else tail
    return {
        "question": stem + out_tail,
        "variant": variant,
        "stem_changed": matched,
        "tail_dropped": bool(drop_tail and tail),
        "had_tail": bool(tail),
    }


def assert_rates(rows: list[dict], *, stem_rate: float, tail_dropout: float,
                 tol: float = 0.05) -> dict:
    """GATE — the REALIZED rates must match the requested ones.

    RAISES. This is the lever's own "did it take effect" proof: a rewrite rule that stopped
    matching the corpus, a `key` that is not actually unique, or a rate wired to the wrong
    variable all land here instead of in a silently-unchanged dataset.
    """
    n = len(rows)
    if n == 0:
        raise AssertionError("L2 gate: zero `number` rows were offered to the paraphraser")
    got_stem = sum(r["stem_changed"] for r in rows) / n
    with_tail = [r for r in rows if r["had_tail"]] or rows
    got_tail = sum(r["tail_dropped"] for r in rows) / len(with_tail)
    problems = []
    if abs(got_stem - stem_rate) > tol:
        problems.append(f"stem {got_stem:.4f} vs requested {stem_rate:.4f}")
    if abs(got_tail - tail_dropout) > tol:
        problems.append(f"tail-dropout {got_tail:.4f} vs requested {tail_dropout:.4f}")
    if problems:
        raise AssertionError(
            "L2 RATE GATE FAILED (tol " + f"{tol}): " + "; ".join(problems)
            + f" over n={n} number rows. A lever that fails to take effect is otherwise "
            "indistinguishable from a successful run."
        )
    from collections import Counter

    mix = dict(Counter(r["variant"] for r in rows).most_common())
    logger.info("L2 gate OK: stem %.4f, tail-dropout %.4f over %d number rows | mix %s",
                got_stem, got_tail, n, mix)
    return {"n": n, "stem_rate": got_stem, "tail_dropout": got_tail, "variant_mix": mix}


__all__ = [
    "NUMBER_TAIL", "STEM_VARIANTS", "split_tail", "paraphrase_stem", "apply", "assert_rates",
]
