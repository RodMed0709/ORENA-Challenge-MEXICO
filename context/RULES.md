# context/RULES.md — operating rules the brain enforces

> The DO / DON'T rules that keep results **consistent** and stop recurring mistakes —
> human OR Claude. Read with `context/INDEX.md` before touching evaluation, training,
> or results. A rule here is not advice; it is how we avoid re-breaking what we already
> fixed. **When a rule and a number disagree, the number is the suspect — not the rule.**
>
> These complement `CONSTITUTION.md` (broad binding law): CONSTITUTION is the project's
> constitution; this file is the operational checklist for the traps we have actually hit.

## EVAL — how metrics are computed (canonical, non-negotiable)

1. **One source of truth.** Score ONLY via `frame.metrics.stratified_report`. NEVER
   re-derive buckets / floors / accuracy inline in a notebook. Ad-hoc re-derivation is
   exactly how the same bug came back in rung 07. If the module lacks something, EXTEND
   the module — do not reimplement beside it.

2. **Leaf→group ALWAYS via `Capability.group`** (`split._group`). `results_df["primary"]`
   is a taxonomy LEAF (e.g. `object_identification`); the scored bucket is a GROUP
   (`object_recognition`). NEVER filter group names against the leaf column — it silently
   drops rows (964 lost in rung 07).

3. **ID/OOD ALWAYS from the qID prefix** (`heico`=OOD, `lapchole`=ID). NEVER trust
   `results_df["ood"]` — it is all-False on the public data (organizers populate it only
   in their private test split).

4. **Headline = `bucket_mean`** (mean over the 4 real buckets; drop `temporal_grounding`
   n=1). `pre_evaluation_score` is REFERENCE ONLY — it is broken on our split: an
   unweighted bucket mean that a single n=1 question inflates (it lifted rung-02 from an
   honest **0.550** to a reported **0.708**).

5. **Trivial floors: vs the EVAL set, split ID/OOD.** Never against the train prior, never
   read off a suffix-collided merged dataframe.

6. **Checkpoint selection by acc_OOD, per-epoch.** Never last-epoch-by-default. Discard any
   checkpoint that wins ID but drops OOD (chole-overfit / OOD collapse). OOD = 50% of score.

7. **Gates RAISE — never disable one.** A gate that fires is a FINDING, not an obstacle.
   NEVER change an expected value to make a broken number pass; fix the code. (Lesson from
   commit `881d057` — the reverted "fix" that papered over a real 964-row drop.)

8. **SMOKE must be stratified** across `heico`+`lapchole`, never a single-dataset prefix.

9. **Every result → the ledger.** Regenerate root `RESULTS.md` via `frame.ledger`; every
   number must be reproducible from a commit.

## READING results (from the data card, rung 08 — read `experiments/08-data-card/`)
A raw accuracy is meaningless without its trivial floor. Read numbers this way:

10. **Judge by MARGIN over the template-aware floor, not raw accuracy.** The floor = what a dumb
    constant (the per-template modal answer) scores by exploiting the answer distribution alone.
    Margin = accuracy − floor = the real skill added. A high floor makes a raw number look good.
11. **`acc_OOD > acc_ID` does NOT mean better generalization.** The OOD floor is ~12 pts higher
    (answers are easier to guess), so by margin the model adds *less* on OOD. Never read the raw
    OOD>ID gap as "generalises well".
12. **`acc_number` is NOT interpretable — do not quote it.** It averages 8 templates with floors
    from 0.24 to 1.00 (4 degenerate). Use the SDK hierarchical estimate + per-template margins.
13. **Effective n ≈ 38 videos, not 6252.** Questions are not independent (they cluster on 38
    videos); trust the video-level hierarchical CI for "will this hold on a new video?".

## How a rule changes
A rule changes ONLY by: (a) a new measurement that contradicts it, recorded as a
`context/decisions/*.md` note, then (b) editing this file in the SAME commit, linking that
decision. Never silently, never mid-experiment to make a number look better.
