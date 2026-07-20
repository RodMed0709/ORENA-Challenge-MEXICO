---
question: What question metadata are we not reading?
verdict: `secondary_capabilities` (89.8% of train) — aggregation supervision is +77% larger; `clinical_relevance` is all-False
status: MEASURED
date: 2026-07-19
measured_in: experiments/08-data-card/
question_derived: true
---
# Finding: 90% of questions carry secondary capability labels we have never used

- **Status:** MEASURED (zero GPU, from the released parquets) · 2026-07-19
- **Applies when:** sizing the training pool for a capability, reweighting a data mix, or
  filtering on a metadata column.
- **Origin:** legokna asked what annotations could be *injected* into the data to lift the
  metric. The best ones turned out to be already there and unread.

## What the parquets actually carry
`id · video · procedure_type · question · answer · answer_format · track · generation ·
clinical_relevance · ood · timestamp_start · timestamp_end · primary_capability ·
secondary_capabilities`

Everything downstream (`frame.metrics`, the split, the buckets) reads **`primary_capability`
only**. Three columns are unused.

## 🔴 1. `secondary_capabilities` — 89.8% of train, 89.2% of val

Bucketing is by PRIMARY capability (ours and the organizers' — the challenge design PDF says
"the test case is mapped to **one** primary capability"). But most questions are multi-label,
and the second label is often aggregation:

| | aggregation as PRIMARY | aggregation as SECONDARY | |
|---|---|---|---|
| TRAIN | 5,524 | **+4,238** | **+77% more** |
| VAL | 2,830 | **+1,749** | **+62% more** |

**The supervision pool for aggregation is 9,762 train questions (71%), not 5,524 (40%).**

This matters directly for [[aggregation-is-the-gap]]: if `aggregation` is the bucket to lift,
the relevant training pool is 77% larger than we have been treating it, and no annotation is
required to access it.

⚠️ The **ranking** stays primary-only — a secondary-labelled question is scored in its primary
bucket. This changes what we can TRAIN on, not what we are SCORED on. The organizers do use
the overlap, but for analysis: *"analyses will explicitly account for label overlap... we will
report both marginal performance per category and conditional performance within key
co-occurrence strata."*

## 2. `generation` — three tiers, not two

| | automatic | anchor | manual |
|---|---|---|---|
| TRAIN | 10,760 (78%) | 2,104 (15%) | 884 (6.4%) |
| VAL | 4,934 (79%) | 969 (15%) | 349 (5.6%) |

The annotation pipeline's Stage 5 generates part of the questions automatically from instance
data and adds the rest via **expert surgeons**. This is an annotation-quality axis nobody has
used. Note the proportions are **the same in train and val**, so val is representative here.

## 🔴 3. `clinical_relevance` is all-False — a second landmine, sister to `ood`

```
clinical_relevance: False on all 13,748 train rows AND all 6,252 val rows
```

Exactly like `ood` (`CONSTITUTION §I.5`). The challenge design PDF stratifies analyses by
**"clinical impact levels"**, so the private test almost certainly populates it and the public
release does not. **Filtering on it locally returns an empty set, silently.**

Both all-False columns should be treated the same way: **never filter or stratify on
`ood` or `clinical_relevance` from public data.**

## 4. Capability leaves present (FRAME)
`1a` object_identification · `3a` object_aggregation · `1d` spatial_camera ·
`1c` attributes · `1e` spatial_situs · `2a` temporal (train n=2, val n=1 — the known orphan).

## Next
1. When reweighting toward `aggregation`, select on **primary OR secondary** — 77% more data,
   zero annotation cost.
2. Consider `generation` as a quality weight (expert-written vs auto-generated).
3. Add `clinical_relevance` to the "never trust on public data" rule beside `ood`.

## Sources
- `external_data/orena-data/*/data/frame/{train,test}.parquet` (14 columns).
- Challenge design PDF §"Statistics - Overview" (multi-label handling) and §Stage 5.
- Related: [[aggregation-is-the-gap]], [[eval-canonical]], `CONSTITUTION §I.5`.
