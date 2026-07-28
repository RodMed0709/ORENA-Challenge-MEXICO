"""Rung 18 · L1 — minted zero-count rows, for ``number`` and ONLY for ``number``.

## Why this exists, and why it is half the size the original proposal was

[[zero-is-format-localized]] (probe 16a, n=960) measured that our fine-tuned checkpoint
says **"no"** fluently — `emit_absent` 0.82 on `binary`, with only 0.10–0.12 false
negatives on the PRESENT control — and **never says "0"**: 1 case in 120 (ID), 2 in 120
(OOD) on `number`. That maps one-to-one onto the supervision: `binary` carries 1,008 `no`
golds; `number` carries **zero zeros in 2,495 per-class examples**.

So the deficit is FORMAT-LOCALIZED. The `binary`/`fo_class` half of the original minting
proposal is unnecessary — the model already holds the concept of absence — and dropping it
halves the rows, the cost and the label noise. This module mints for `number` only.

The same probe also killed the cheaper hypothesis: the base model is a **zero attractor**
(it emits `0`/`no` on 83.1% of PRESENT cases), so the capability never existed in this
domain and there is nothing for regularisation to restore.

## The label, and its noise

The ABSENT label is derived under a **closure assumption** — that the `fo_class` gold names
every class present in the frame. Measured non-circularly against the co-occurrence
binaries: **0.00% false positives**, **8.4% under-naming** (n=333; a second cross-check
agrees at 10.7%). So ~1 in 12 raw ABSENT labels is wrong, always in the direction of zero,
on a model that already carries a −0.66 count bias.

This module attacks that noise directly rather than budgeting for it, using the ONE channel
independent of the `fo_class` annotation — the 1,230 co-occurrence binaries:

* ``Do <A> and <B> co-occur in this frame?`` → **yes** ⇒ A and B are BOTH present. This
  both **refutes** a minted zero for A or B *and* **repairs** the inventory (it recovers
  exactly the under-naming the 8.4% measures).
* → **no** ⇒ ¬(A∧B). If the inventory already asserts A present, then B is absent and the
  minted zero for B is **positively confirmed** by a channel that never read `fo_class`.
* ``Are all visible foreign objects in this frame of the same class?`` → **yes** with an
  inventory naming exactly one class ⇒ every other class is absent (confirmation). The same
  answer against an inventory naming ≥2 classes is a **gold self-contradiction** and is
  reported, never silently absorbed.

## Adversarial sampling (POPE), derived from the corpus and not hard-coded

Random minting ("how many Meshes?" on a sponge frame) is the easy negative and will not move
an attractor. Weights come from the MEASURED per-class training counts, at both ends:

* the **dominant** class (`Clip`, 1,390 of 1,667 per-class count rows) — the model's prior
  answer is a positive integer, so a true `0` there is the hardest negative we can write;
* the **rarely-counted** classes — 16a's own rows show `sdk_invalid` = **1.0000** for
  External Drain / Gallstone / Needle / Silicone Loop / Specimen, i.e. those questions are
  auto-incorrect today regardless of the count;
* the **never-seen** classes (zero examples anywhere) get an explicitly capped share, since
  a large dose would teach "unfamiliar word ⇒ 0" rather than perception.

🔴 The class vocabulary is read from ``FOType.names()`` at runtime and NEVER hard-coded
(RULES §8b): the list the organizers put inside the prompt and the scoring registry disagree
on their 10th element, and ``FOType.from_name()`` RAISES on an unrecognised token.
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The corpus's OWN per-class count template, verbatim — including the
# `"Please provide a number."` tail. A minted row must be in-distribution: 16a reached an
# 87% SDK-illegal rate precisely because it asked an off-template phrasing, and
# [[the-gap-is-the-number-format]] records that the tail is what the format is glued to.
# Surface variation is L2's job (`paraphrase.py`) and is applied afterwards, to real and
# minted rows alike, so the two levers stay separable.
NUMBER_Q = "How many {plural} appear in this frame? Please provide a number."
NUMBER_TAIL = " Please provide a number."

# The corpus's co-occurrence binary and its "all one class" sibling — the only channels
# independent of the `fo_class` annotation.
_COOCCUR_RE = re.compile(
    r"^Do\s+(.+?)\s+and\s+(.+?)\s+co-occur in this frame\?", re.I
)
_ALL_SAME_RE = re.compile(
    r"^Are all visible foreign objects in this frame of the same class\?", re.I
)
# The per-class count question, used to measure each class's training exposure.
_PERCLASS_RE = re.compile(r"^How many\s+(.+?)\s+appear in this frame\?", re.I)


# ── vocabulary → the corpus's plural surface form ─────────────────────────────

def plural_of(name: str) -> str:
    """The corpus's plural surface for a class name, by RULE rather than by table.

    The corpus writes ``Clip -> Clips``, ``External Drain -> External drains``,
    ``Specimen Bag -> Specimen bags``, ``Silicone Loop -> Silicone loops``: sentence-case
    the whole name, then pluralise. ``assert_plurals_match_corpus`` proves that rule against
    every class the corpus actually asks about, so it is a measurement and not a guess — and
    it is the only way to build a question for ``Mesh`` / ``Absorbable Hemostatic Agent``,
    which appear in ``FOType.names()`` and in no gold anywhere.
    """
    s = name.strip()
    s = s[:1].upper() + s[1:].lower()
    return s + ("es" if re.search(r"(s|sh|ch|x|z)$", s, re.I) else "s")


def class_plurals() -> dict[str, str]:
    """``{class_name: plural}`` over the SCORING vocabulary, read at runtime (RULES §8b)."""
    from focus.foreign_objects import FOType

    return {c: plural_of(c) for c in FOType.names()}


def corpus_plurals(items) -> dict[str, int]:
    """``{plural_as_the_corpus_writes_it: n_rows}`` from the real per-class count questions."""
    out: Counter = Counter()
    for it in items:
        if _fmt(it.reference) != "number":
            continue
        m = _PERCLASS_RE.match(str(it.request.question))
        if not m:
            continue  # the two aggregate templates (instances / classes) are not per-class
        noun = m.group(1).strip()
        if noun.lower().startswith("different foreign object"):
            continue
        out[noun] += 1
    return dict(out)


def assert_plurals_match_corpus(items) -> dict[str, int]:
    """GATE — the derived plural must equal the corpus's own for every class it asks about.

    RAISES on any mismatch. A minted question whose noun differs from the corpus's by one
    character is an OFF-TEMPLATE question, which is exactly the condition 16a measured an
    87% auto-incorrect rate under: the row would then teach format fragility instead of
    curing it, and nothing in the scored eval would show it.
    """
    derived = {p: c for c, p in class_plurals().items()}
    observed = corpus_plurals(items)
    unknown = {p: n for p, n in observed.items() if p not in derived}
    if unknown:
        raise AssertionError(
            f"corpus asks per-class counts with plural(s) the rule does not produce: "
            f"{unknown}. Derived surfaces: {sorted(derived)}. Fix `plural_of` — do NOT "
            "hand-map the class, or minted rows go off-template silently."
        )
    logger.info(
        "plural gate OK: %d corpus plurals all reproduced by rule (%s)",
        len(observed), ", ".join(f"{p}={n}" for p, n in sorted(observed.items())),
    )
    return observed


# ── item accessors ────────────────────────────────────────────────────────────

def _fmt(ref) -> str:
    """The SDK exposes the format as ``_format``, NOT ``answer_format``.

    Reused from probe 16a rather than re-implemented: reading the wrong attribute returns
    ``""`` silently, every branch misses, and the built set comes out EMPTY instead of
    raising — which is how 16a's first smoke burned a 17 GB model load on 0 questions.
    """
    from zero_probe import answer_format

    return answer_format(ref)


def _ans(ref) -> str:
    return str(getattr(ref, "answer", "") or "").strip()


# ── the independent verification channel: the co-occurrence binaries ──────────

@dataclass
class BinaryEvidence:
    """What the binaries assert per frame, independently of ``fo_class``."""

    present: dict[str, set]        # frame_key -> classes a `yes` co-occurrence proves present
    absent: dict[str, set]         # frame_key -> classes a binary positively proves absent
    contradictions: list[dict]     # gold-vs-gold conflicts, reported and never absorbed
    n_cooccur: int = 0
    n_all_same: int = 0


def binary_evidence(items, inventory: dict[str, dict]) -> BinaryEvidence:
    """Read the 1,230 co-occurrence binaries into per-frame presence / absence facts.

    ``inventory`` is the ``fo_class``-derived closure inventory (probe 16a's
    ``build_inventory``); it is consulted only to turn a ``no`` co-occurrence into a
    one-sided absence fact. The presence facts derived here are independent of it.
    """
    from frame.data import frame_cache_name

    by_plural = {p.lower(): c for c, p in class_plurals().items()}
    ev = BinaryEvidence(present=defaultdict(set), absent=defaultdict(set), contradictions=[])

    for it in items:
        if _fmt(it.reference) != "binary":
            continue
        key = frame_cache_name(it)
        q, a = str(it.request.question), _ans(it.reference).lower()
        if a not in {"yes", "no"}:
            continue

        m = _COOCCUR_RE.match(q)
        if m:
            ev.n_cooccur += 1
            a_cls = by_plural.get(m.group(1).strip().lower())
            b_cls = by_plural.get(m.group(2).strip().lower())
            if a_cls is None or b_cls is None:
                # An unmapped noun means the template moved. Loud, not silent: the whole
                # point of this channel is that it is independent, so a half-read channel
                # would quietly weaken the filter it exists to provide.
                raise AssertionError(
                    f"co-occurrence binary names class noun(s) outside the derived "
                    f"vocabulary: {m.group(1)!r} / {m.group(2)!r} in {q!r}"
                )
            if a == "yes":
                ev.present[key] |= {a_cls, b_cls}
            else:
                named = inventory.get(key, {}).get("named", set())
                # ¬(A∧B): if the inventory already asserts one side present, the OTHER is
                # proven absent by a channel that never read `fo_class`.
                if a_cls in named and b_cls not in named:
                    ev.absent[key].add(b_cls)
                if b_cls in named and a_cls not in named:
                    ev.absent[key].add(a_cls)
            continue

        if _ALL_SAME_RE.match(q) and a == "yes":
            ev.n_all_same += 1
            named = inventory.get(key, {}).get("named", set())
            if len(named) == 1:
                ev.absent[key] |= set(class_plurals()) - named
            elif len(named) > 1:
                ev.contradictions.append(
                    {"frame_key": key, "kind": "all_same_class_but_inventory_multi",
                     "named": sorted(named)}
                )

    # A `yes` co-occurrence that names a class the inventory calls absent IS the 8.4%
    # under-naming, caught in the act. Count it — it is this batch's own noise estimate.
    repaired = 0
    for key, present in ev.present.items():
        repaired += len(present - inventory.get(key, {}).get("named", set()))
    logger.info(
        "binary evidence: %d co-occurrence + %d all-same-class rows read | "
        "%d frames with presence facts (%d classes the fo_class inventory had missed) | "
        "%d frames with proven absences | %d gold self-contradictions",
        ev.n_cooccur, ev.n_all_same, len(ev.present), repaired, len(ev.absent),
        len(ev.contradictions),
    )
    return ev


def repair_inventory(inventory: dict[str, dict], ev: BinaryEvidence) -> tuple[dict, int]:
    """Fold the binaries' presence facts INTO the closure inventory.

    This is not cosmetic: every class recovered here is one minted zero that would have been
    wrong, in the one direction the model is already biased toward. Returns the repaired
    inventory and how many (frame, class) presences it added.
    """
    added = 0
    for key, present in ev.present.items():
        entry = inventory.get(key)
        if entry is None:
            continue  # a frame the inventory never saw has no usable absence label anyway
        new = present - entry["named"]
        entry["named"] |= new
        added += len(new)
    return inventory, added


# ── the adversarial sampler ───────────────────────────────────────────────────

@dataclass
class MintConfig:
    """How many minted zeros, and how they are spread over classes and frames.

    ``dose_ratio`` is **real per-class count rows : minted zeros**. The PLAN's band is
    3:1–2:1, deliberately below LRV-Instruction's 1:1 optimum because the residual label
    noise points toward zero on a model already at a −0.66 count bias, so the EFFECTIVE dose
    exceeds the nominal one. 2.5 sits in the middle of that band and lands the realized count
    inside the PLAN's stated 600–830 row window against the 1,667 per-class rows the TRAIN
    split actually carries (the PLAN's 2,495 is the corpus-wide figure, train + val).
    """

    dose_ratio: float = 2.5
    unseen_share: float = 0.10      # capped share for classes with zero examples anywhere
    max_per_frame: int = 2          # no frame may dominate the minted set (16a's discipline)
    w_dominant: float = 5.0         # extra weight for the most-counted class (the attractor)
    w_rare: float = 2.0             # extra weight for rarely-counted classes (16a: illegal 1.0)
    rare_max_rows: int = 100        # "rarely counted" = fewer than this many training rows
    seed: int = 42


def class_weights(observed: dict[str, int], cfg: MintConfig) -> dict[str, float]:
    """Adversarial weights, derived from the MEASURED per-class training counts.

    Both tails get the boost and the middle does not: the dominant class because the model's
    prior there is a positive integer (a true `0` is the hardest negative available), the
    rare classes because 16a measured their questions auto-incorrect at rate 1.0000.
    """
    plurals = class_plurals()
    counts = {c: observed.get(p, 0) for c, p in plurals.items()}
    seen = {c: n for c, n in counts.items() if n > 0}
    top = max(seen.values()) if seen else 0
    w = {}
    for c, n in seen.items():
        w[c] = 1.0 + (cfg.w_dominant if n == top else 0.0) + (
            cfg.w_rare if n < cfg.rare_max_rows else 0.0
        )
    return w


def _weighted_sample(pool: list, weights: list[float], k: int, rng: random.Random) -> list:
    """Deterministic weighted sampling WITHOUT replacement (Efraimidis–Spirakis keys).

    ``key = u ** (1/w)``, take the top k. Seeded and order-independent given a sorted pool,
    so the same config reproduces the same minted set on any machine.
    """
    if k >= len(pool):
        return list(pool)
    keyed = [(rng.random() ** (1.0 / max(w, 1e-9)), x) for x, w in zip(pool, weights)]
    keyed.sort(key=lambda t: -t[0])
    return [x for _, x in keyed[:k]]


def mint(items, *, cfg: MintConfig | None = None) -> tuple[list[dict], dict]:
    """Build the minted zero-count rows from TRAIN items only.

    Returns ``(rows, stats)``. Each row carries everything a provenance CSV and every gate
    needs: the frame's identity (so the val-leak gate can key on ``video_id``), the class,
    the verbatim corpus question, the gold ``"0"``, and WHICH channel licensed it.

    ⚠️ Pass **train-split items only**. This function does not know the split; the leak gate
    in ``build_train_jsonl`` is what proves no minted row touches a val video.
    """
    from frame.data import frame_cache_name
    from zero_probe import build_inventory

    cfg = cfg or MintConfig()
    rng = random.Random(cfg.seed)

    observed = assert_plurals_match_corpus(items)
    plurals = class_plurals()
    n_perclass = sum(observed.values())
    target = int(round(n_perclass / cfg.dose_ratio))

    # 1. the closure inventory (probe 16a's own builder — reused, never copied)
    inventory = build_inventory(items)
    # 2. the independent channel: repair it, then read its absence proofs
    ev = binary_evidence(items, inventory)
    inventory, n_repaired = repair_inventory(inventory, ev)
    ev_after = binary_evidence(items, inventory)   # re-read: repairs create new `no`-side facts

    # 3. candidates — one per (frame, absent class), refuted ones already gone
    weights = class_weights(observed, cfg)
    unseen = [c for c in plurals if observed.get(plurals[c], 0) == 0]
    frame_meta = {}
    cands_seen: list[tuple[str, str]] = []
    cands_unseen: list[tuple[str, str]] = []
    for key, e in sorted(inventory.items()):
        if not e["named"]:
            continue  # nothing asserted → no usable absence label under closure
        frame_meta[key] = e
        # `named` already carries the binaries' presence facts (repair_inventory ran above),
        # so a class the independent channel proved present is already excluded here.
        for c in plurals:
            if c in e["named"]:
                continue
            (cands_unseen if c in unseen else cands_seen).append((key, c))

    # 4. sample: the never-seen classes take an explicitly capped share, so a large dose
    #    cannot teach "unfamiliar word ⇒ 0" in place of perception.
    n_unseen = min(int(round(target * cfg.unseen_share)), len(cands_unseen))
    n_seen = min(target - n_unseen, len(cands_seen))
    picked = _weighted_sample(cands_seen, [weights[c] for _, c in cands_seen], n_seen, rng)
    picked += _weighted_sample(cands_unseen, [1.0] * len(cands_unseen), n_unseen, rng)

    # 5. per-frame cap, applied deterministically after sampling
    per_frame: Counter = Counter()
    kept, capped = [], 0
    for key, c in sorted(picked):
        if per_frame[key] >= cfg.max_per_frame:
            capped += 1
            continue
        per_frame[key] += 1
        kept.append((key, c))

    rows = []
    for key, c in kept:
        e = frame_meta[key]
        confirmed = c in ev_after.absent.get(key, set())
        rows.append({
            "frame_key": key,
            "dataset": e["dataset"],
            "video_id": e["video_id"],
            "frame_index": e["frame_index"],
            "cls": c,
            "question": NUMBER_Q.format(plural=plurals[c]),
            "answer": "0",
            "named": ", ".join(sorted(e["named"])),
            "seen_class": c not in unseen,
            # the honest provenance: "binary" = a channel independent of `fo_class` proved
            # this absence; "closure" = only the fo_class closure assumption backs it.
            "evidence": "binary" if confirmed else "closure",
        })
    rows.sort(key=lambda r: (r["dataset"], r["video_id"], r["frame_index"], r["cls"]))

    stats = {
        "n_perclass_rows": n_perclass,
        "dose_ratio": cfg.dose_ratio,
        "target": target,
        "realized": len(rows),
        "n_candidates_seen": len(cands_seen),
        "n_candidates_unseen": len(cands_unseen),
        "n_unseen_minted": sum(1 for r in rows if not r["seen_class"]),
        "n_confirmed_by_binary": sum(1 for r in rows if r["evidence"] == "binary"),
        # Every presence the binaries recovered is one minted zero that would have been
        # WRONG, in the exact direction the model is already biased toward. This is the
        # 8.4% under-naming caught in the act, on this batch, rather than budgeted for.
        "n_refuted_by_binary": n_repaired,
        "n_capped_by_frame": capped,
        "n_frames": len({r["frame_key"] for r in rows}),
        "class_mix": dict(Counter(r["cls"] for r in rows).most_common()),
        "class_weights": weights,
        "contradictions": len(ev_after.contradictions),
    }
    logger.info("minted %d/%d zeros over %d frames | mix %s",
                stats["realized"], target, stats["n_frames"], stats["class_mix"])
    return rows, stats


def assert_dose(stats: dict, *, tol: float = 0.10) -> None:
    """GATE — the realized dose must land within ``tol`` of the target (PLAN §gates).

    RAISES. A minted set that silently came out half the intended size is a different
    experiment wearing this one's name, and nothing downstream would show it.
    """
    target, realized = stats["target"], stats["realized"]
    if target <= 0:
        raise AssertionError(f"dose target is {target} — the per-class row count was 0")
    off = abs(realized - target) / target
    if off > tol:
        raise AssertionError(
            f"DOSE GATE FAILED: realized {realized} vs target {target} "
            f"({off:.1%} off, tol {tol:.0%}). Candidate pool: "
            f"{stats['n_candidates_seen']} seen + {stats['n_candidates_unseen']} unseen."
        )
    logger.info("dose gate OK: %d minted vs target %d (%.1f%% off)", realized, target, off * 100)


def assert_class_mix(stats: dict, *, min_classes: int = 4, max_share: float = 0.60) -> None:
    """GATE — the minted set must be adversarial, not one class wearing a costume.

    RAISES if fewer than ``min_classes`` classes appear or any single class exceeds
    ``max_share``. The dominant class is deliberately oversampled; a set that is ONLY the
    dominant class would test a different hypothesis (and would not touch the never-seen
    exposure at all).
    """
    mix = stats["class_mix"]
    total = sum(mix.values())
    if len(mix) < min_classes:
        raise AssertionError(f"CLASS MIX GATE FAILED: only {len(mix)} classes minted: {mix}")
    top_cls, top_n = max(mix.items(), key=lambda kv: kv[1])
    if total and top_n / total > max_share:
        raise AssertionError(
            f"CLASS MIX GATE FAILED: {top_cls} is {top_n}/{total} = {top_n/total:.1%} of the "
            f"minted set (> {max_share:.0%}). Lower `w_dominant`."
        )
    logger.info("class mix gate OK: %d classes, top %s at %.1f%%",
                len(mix), top_cls, 100 * top_n / max(total, 1))


__all__ = [
    "NUMBER_Q", "NUMBER_TAIL", "MintConfig", "mint", "plural_of", "class_plurals",
    "corpus_plurals", "assert_plurals_match_corpus", "binary_evidence", "repair_inventory",
    "class_weights", "assert_dose", "assert_class_mix",
]
