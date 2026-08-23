# Rung 53 — the cross-question constraint, audited before it cost anything

> **Status: AUDITED 2026-08-23, NOT BUILT.** Zero GPU. The audit is the result.

## The idea

[[cross-question-constraints]] (Leo, 2026-07-28, never used): the organizers ask several
questions about ONE frame, so the answers constrain each other — **979 frames** carry both a
`number` and an `fo_class` question, **495** pair a per-class count with the total. Free,
already-labelled enumeration supervision, legitimate at train time.

It is the right shape for [[fo-class-and-number-are-one-front]]: not a scalar target — which
[[counting-is-a-mapping-failure]] shows collapses **93.02 % → 32.00 %** the moment it leaves the
training range, on our own backbone family — but a **structural constraint**, which is the half
that note says survives.

## 🔴 What the audit found, and why the arm does not get built as designed

Measured on the corpus we would actually train (**rung 47's 14,415 rows**, 8,661 distinct frames),
not across all 20,000 public questions:

| pairing | frames | verdict |
|---|---:|---|
| `fo_class` + *"how many different **classes**"* | **218** | 🔴 **golds disagree 11.0 %** |
| `fo_class` + *"how many different **instances**"* | 303 | not directly checkable (a set of classes ≠ a count of instances) |
| per-class count + total instances | **260** | 🟢 **0 violations** — clean but only an inequality |

**1. It is 3–4× smaller than the note suggests.** Leo's 979 spans all 20,000 public questions;
our training corpus holds **218–303**. Against 14,415 rows that is **~2 %** — a dose so small
that even a perfect effect would be hard to read.

**2. 🔴 The golds break the constraint 11 % of the time, and they break it in ONE direction.**
24 of 218 disagree, and **all 24 have the `fo_class` set SMALLER than the class count** (−1 in
22 cases, −2 in 2). Never larger.

```
fo_class gold "External drain"  -> 1 class,  but "how many classes" gold = 2
fo_class gold "Silicone loop"   -> 1 class,  but "how many classes" gold = 3
fo_class gold "Needle"          -> 1 class,  but "how many classes" gold = 3
```

⇒ **Training a consistency objective here would teach the model to satisfy a constraint its own
labels violate, one-directionally, on a ninth of the population.** That is how a rung produces a
confident null and blames the idea.

**3. The one clean constraint is the weak one.** Per-class counts never exceed the total —
**260 / 260** — so that relation is sound, but `≤` carries far less signal than `=`.

## 🔑 What the audit found that is worth more than the arm

The asymmetry is not noise-shaped. Noise scatters; this is **24 of 24 in the same direction**.
Either the two templates ask different things, or **`fo_class` golds systematically UNDER-LIST**
— and `fo_class` is **42.8 % of the eval**. If it is under-listing, every `fo_class` measurement
we have, including rung 51's whole anatomy, sits on a target that omits classes.

⚠️ **Not established here.** n=24, and this rung cannot separate "different question scope" from
"inconsistent annotation" — that needs the frames opened by eye, which is the cheap next step and
costs no GPU.

## Verdict

**Do not build the consistency arm on the `=` constraint.** 218 frames at 11 % broken labels is
not a lever. Revisit only if the eyeball pass shows the disagreement is scope, not noise — in
which case the constraint is repairable and the dose is still only ~2 %.
