"""Build the paired arms for rung 26 part 2 — the same frame and gold, asked without the shortcut.

Engine for the GPU half of roadmap phase 0.4. Importable; the notebook calls ``build_arms`` and
the three gates. Pure string manipulation over the question text — no model, no gold change.

## What part 1 left open

``shortcut_taxonomy`` measured that shortcut-carrying ``fo_class`` questions score ~2x the margin
of shortcut-free ones. It cannot say whether the model *exploits* the shortcut, because the two
cells are DIFFERENT QUESTIONS. Part 2 removes that ambiguity the way SurgCheck does: hold the
frame and the gold fixed, vary only the phrasing, and read the paired delta.

## The stratum, read off the corpus rather than assumed

Within ``fo_class`` the shortcut is **one template**, not a family::

    673  "There is one surgical foreign object visible in the frame. What surgical foreign
          object is visible in this video frame? Please provide a class name."

Its gold is a single class on all 673 rows, so the premise is true *and* informative: it collapses
an exact-SET task into a 1-of-10 pick. ``leak_named_entity`` is essentially absent here for a
structural reason — naming the class would hand over the answer of a ``fo_class`` question
outright, so the corpus does not do it. That is why part 2 is scoped to the cardinality premise.

## The two de-shortcut arms

SurgCheck keeps a de-named question well-defined with one of four grounding cues: bounding box,
arrow, spatial position, periphrasis. 🔴 **Three of the four need localization we do not have** —
no boxes, no masks (that gap is rung 25's). Only periphrasis is available without new annotation,
which is what both arms below use, and it is the reason part 2 cannot simply re-anchor the 673 by
object position.

* ``premise_dropped`` — the minimal edit: delete the leading cardinality sentence, keep everything
  else byte-identical. Strictly single-variable, but a weak manipulation: the residual *"What
  surgical foreign object"* still frames the answer as singular.
* ``set_framed`` — re-ask with the corpus's OWN shortcut-free ``fo_class`` template (976 rows of
  it). A stronger manipulation, and the phrasing is not invented by us, so an effect cannot be
  blamed on out-of-distribution wording. It moves two things at once (premise removed *and* set
  framing), which is why it does not replace the minimal arm.

Both are reported. The pre-registered decision rule in ``PLAN.md`` is read against
``premise_dropped`` as primary — it is the one that changes a single thing.

## Why the gates are hard

A malformed rewrite produces a wrong answer that is not the model's fault, and it would read as
"the margin was phrasing" — the exact conclusion part 2 exists to test. So every arm is verified
by ``shortcut_taxonomy.classify``: the same detector that defined part 1 must agree that the
rewritten text carries no shortcut. RULES §9b — gates raise, they are never softened.
"""

from __future__ import annotations

import re

import pandas as pd

from shortcut_taxonomy import CARDINALITY_RE, classify

# The corpus's own shortcut-free ``fo_class`` template. Declared here but ASSERTED against the
# gold in ``assert_arms_wellformed`` — if the organizers reword it, the gate fails loudly rather
# than us silently asking a question the corpus never asks.
SET_FRAMED_QUESTION = (
    "List all foreign objects that are visible in this video frame. "
    "Please provide the class names or answer with none."
)

ARMS = ("original", "premise_dropped", "set_framed")

# Sentence split that keeps the terminator, so a rebuilt question is byte-identical to the
# original minus the dropped sentence.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]\s*")


def strip_cardinality_premise(question: str) -> str:
    """Drop leading sentences that assert a count; leave the rest untouched.

    Returns the question unchanged when no leading sentence carries a premise, so callers can
    apply this to a whole format and select afterwards.
    """
    sentences = _SENTENCE_RE.findall(question)
    if not sentences:
        return question
    kept = list(sentences)
    while kept and CARDINALITY_RE.search(kept[0]):
        kept.pop(0)
    if not kept:  # the premise was the entire question — refuse rather than emit an empty ask
        return question
    return "".join(kept).strip()


def shortcut_stratum(gold: pd.DataFrame, *, answer_format: str = "fo_class") -> pd.DataFrame:
    """The rows part 2 manipulates: one format, carrying the cardinality premise.

    ``gold`` needs ``question`` and ``answer_format``; every other column is carried through so
    the caller keeps ``qID``/``video``/``answer`` for pairing and scoring.
    """
    df = gold[gold["answer_format"] == answer_format]
    return df[df["question"].str.contains(CARDINALITY_RE, regex=True)].copy()


def build_arms(gold: pd.DataFrame, *, answer_format: str = "fo_class") -> pd.DataFrame:
    """One row per (question, arm) with the rewritten ``question``; the gold is never touched.

    The ``original`` arm is included so all three are inferred in the SAME process on the SAME
    GPU. That is not redundancy: ~0.5% of answers change on a GPU swap
    ([[archived-results-not-bit-reproducible]]), so reusing an archived control would put drift
    inside a paired delta that is being read at a 0.05 threshold.
    """
    base = shortcut_stratum(gold, answer_format=answer_format)
    frames = []
    for arm in ARMS:
        a = base.copy()
        a["arm"] = arm
        if arm == "premise_dropped":
            a["question"] = a["question"].map(strip_cardinality_premise)
        elif arm == "set_framed":
            a["question"] = SET_FRAMED_QUESTION
        frames.append(a)
    return pd.concat(frames, ignore_index=True)


def assert_arms_wellformed(arms: pd.DataFrame, gold: pd.DataFrame) -> None:
    """Four blocking checks. Every failure mode here is silent, which is why they raise.

    1. the ``set_framed`` wording is one the corpus actually uses;
    2. the de-shortcut arms carry NO shortcut under part 1's own detector;
    3. the ``original`` arm is untouched;
    4. the gold is identical across arms — we change the question, never the answer.
    """
    n_corpus = int((gold["question"] == SET_FRAMED_QUESTION).sum())
    if n_corpus == 0:
        raise AssertionError(
            "SET_FRAMED_QUESTION does not appear in the gold — the template was reworded and the "
            "arm would be asking something the corpus never asks."
        )

    flags = classify(arms["question"])
    for arm in ("premise_dropped", "set_framed"):
        leaking = arms[(arms["arm"] == arm) & flags["shortcut"].to_numpy()]
        if len(leaking):
            raise AssertionError(
                f"{arm}: {len(leaking)} of {(arms['arm'] == arm).sum()} rewritten questions still "
                f"carry a shortcut under shortcut_taxonomy.classify — e.g. "
                f"{leaking['question'].iloc[0]!r}"
            )

    original = arms[arms["arm"] == "original"]
    src = shortcut_stratum(gold).set_index("qID")["question"]
    if not original.set_index("qID")["question"].equals(src.loc[original["qID"]].rename(None)):
        raise AssertionError("the 'original' arm is not byte-identical to the corpus question")

    per_arm = arms.groupby("arm")["answer"].apply(lambda s: tuple(s.tolist()))
    if len({*per_arm}) != 1:
        raise AssertionError("gold differs across arms — part 2 varies phrasing only")


def eyeball_sample(arms: pd.DataFrame, *, n: int = 30, seed: int = 26) -> pd.DataFrame:
    """The n rows a human reads before any GPU time, as rung 09 stage 1 does.

    Sampled on the question, not on the arm, so the reviewer sees all three phrasings of the same
    frame side by side and can judge whether the rewrite is still answerable.
    """
    qids = (
        arms.loc[arms["arm"] == "original", "qID"]
        .drop_duplicates()
        .sample(n=min(n, arms["qID"].nunique()), random_state=seed)
    )
    out = arms[arms["qID"].isin(qids)].copy()
    out["arm"] = pd.Categorical(out["arm"], categories=ARMS, ordered=True)
    return out.sort_values(["qID", "arm"])[["qID", "video", "arm", "question", "answer"]]


__all__ = [
    "ARMS",
    "SET_FRAMED_QUESTION",
    "assert_arms_wellformed",
    "build_arms",
    "eyeball_sample",
    "shortcut_stratum",
    "strip_cardinality_premise",
]
