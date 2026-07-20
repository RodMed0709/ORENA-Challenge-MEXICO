---
question: Does the +77% secondary-label supervision pool actually reach the `number` format?
verdict: NO — it is 89.6% `fo_class` and 0% `number`; the gap's format gains nothing directly
status: MEASURED
date: 2026-07-19
measured_in: external_data/orena-data/*/data/frame/train.parquet
amends: [unused-metadata]
---

# Finding: the +77% secondary-label pool is `fo_class`, not `number`

- **Status:** MEASURED · 2026-07-19 · zero GPU, from the released parquets
- **Applies when:** costing the "reweight toward `aggregation` using secondary labels" lever —
  which was the **top-ranked candidate** before [[the-gap-is-the-number-format]] localised the gap.
- **Origin:** [[the-gap-is-the-number-format]] closed with an explicit caveat — *"[[unused-metadata]]'s
  +77% aggregation pool is the exception; it is labelled by capability, not format, and needs its
  own check."* This is that check.

## Question

[[unused-metadata]] measured that `aggregation` appears as a **secondary** label on 4,238 more
training questions (+77% over the 5,524 primary ones) at zero annotation cost. Since the gap is
now known to live in the **`number` format** (80.4% of `aggregation × ID`), does that extra
supervision reach `number`?

## What it gave us

```
aggregation PRIMARY      (n=5,524)        77.2% number · 22.3% binary · 0.6% open_ended
aggregation SECONDARY-only (n=4,238, +77%) 89.6% fo_class · 10.4% open_ended · 0.0% number
```

🔴 **Zero `number`.** The extra pool is almost entirely `fo_class` — the format that sits
**entirely inside `object_recognition`**, the bucket we already lead by +14.9.

(The 5,524 / 4,238 counts reproduce [[unused-metadata]] exactly, which validates the recomputation.)

## Verdict — re-scoped, not dead

**The lever's original rationale is wrong.** It was sold as "+77% more supervision for the bucket
we need to lift, at zero cost". In the format that owns the gap it is **+0%**.

**But it is not worthless, and the reason is measured.** 05b found the multiplicity compression is
**shared across formats**: at truth=2 the model names **1.76** classes in `fo_class` while saying
**1.69** in `number`; at truth=3, 1.98 vs 2.01. Two formats with nothing in common saturate at the
same place — which is why 05b concluded the failure is *upstream of the format*, in perception.

So the lever survives with a **different and much narrower claim**:

> Does multiplicity perception trained through `fo_class` transfer to `number`?

That is a testable hypothesis with a pre-registrable target (mean prediction at truth=2 in
**`number`**, not overall accuracy), not a data-volume argument. **It must be judged on transfer,
and a run that only lifts `fo_class` is a failure of this lever, not a partial success** — it would
merely widen a lead we already have.

## Method note

The first attempt at this measurement returned zero rows because `primary_capability` was filtered
by **group** name against a column holding **leaf** values — the exact leaf→group defect
[[eval-canonical]] exists to prevent. It was caught only because zero was absurd. The numbers above
use `Capability.from_any(leaf).group.value`. **The canonical mapping is not optional even for a
throwaway check.**

## Sources

- `external_data/orena-data/{heico,lapchole}/data/frame/train.parquet` (n=13,748)
- Related: [[unused-metadata]] · [[the-gap-is-the-number-format]] · [[eval-canonical]] ·
  `experiments/05-bottleneck-audit/README.md` §5b (the shared-compression evidence)
