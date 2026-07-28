# Handoff — rung 19 (external counting data), 2026-07-28

Paste the block at the bottom into a fresh session. Everything below it is context that
session can re-derive from the repo; the block is what it needs to start.

---

## Where things actually stand

**Branch `task/r3-rung16`, everything pushed.** ⚠️ A second Claude session works in the same
tree on rungs 16/18 (counting probes with our own data) — coordinate, don't clobber.

### The strategic picture (settled 2026-07-27/28)

- Leaderboard: **12/13, score 0.4710**. A **4B model beats us at 0.5163**; 1st is 0.5591 (Qwen3.6).
- **The entire deficit is `aggregation`** — 67 questions of 754. `object_recognition` is a
  two-question tie (607 vs 609). Proven on the *identical* 2000-question set via shared
  denominators. See `context/decisions/leaderboard-metric-vs-our-headline.md`.
- **Their metric ≠ ours.** `pre_evaluation_score` = unweighted mean of *populated* buckets; only 2
  populate on pre-eval, both ID. Our `bucket_mean` averages 4. Right comparator = **mean-ID**.
- **We need ~+0.07.** Best local checkpoint, minus the measured −0.057 transfer loss, lands ~0.489
  — still under the 4B. **Swapping checkpoints cannot close this; only a step change can.**
- Latency is a non-issue: 0.79 s/q against a 5.06 ceiling, **6.4× headroom**. Never re-raise it.
- 🔒 **We do not know where the baselines sit** — beating both is the gate to the final stage and
  no baseline row is visible. Worth asking the organizers.

### Rung 19 — what has been measured, all with ZERO GPU

| dataset | verdict | evidence |
|---|---|---|
| **SAR-RARP50** | 🔴 **DEAD** — presence, not counts | semantic masks; connected components splits one elongated object into "2–3". Its `clamps` (our clip analogue) is 0 or 1 in 84/101 frames |
| **CholecInstanceSeg** | 🟡 **narrow** — tops out at **3** instances (exactly 1 frame reaches 4) | measured over all 41,933 annotations. We fail in the **5–12** range, so it cannot teach it. **BUT it ships 4,914 zero-object frames (11.7%)** and our training set has **no numeric gold equal to zero anywhere** |
| **ROBUST-MIS** | ⏳ **IN FLIGHT** — the histogram was running when the session ended | download permission **confirmed working** after the user registered on Synapse |
| **MedMultiPoints** | license resolved **CC BY-NC 4.0**, not yet measured | the only dataset with a *published counting win on our own model family* (Qwen2.5-VL-7B + LoRA, Count MAE 9.86 → 0.26) |

🔴 **The fact that frames the whole rung:** no public dataset annotates applied surgical clips.
Every option is a **transfer bet on a generic enumeration prior**. But we measured that this is
less damaging than it sounds — **the largest single template is "how many different foreign object
INSTANCES" at 830 of 2094 (39.6 %), class-agnostic**, with "how many classes" adding 436. So 60 %
of counting questions are not clip-specific. (Clips are 681. Bags/drains/needles/specimens are 64
and **degenerate** — always answer 1 — so there is no headroom there.)

### Credentials and access (all working, all stored)

`.secrets.env` (gitignored, verified) holds `SYNAPSE_AUTH_TOKEN`, `RUNPOD_API_KEY`, `HF_TOKEN`,
`GITHUB_TOKEN`, `RUNPOD_S3_*`. Synapse download permission on ROBUST-MIS is **granted**.

⚠️ **Local disk is at 99 % — ~14 GB free.** Do not download CholecInstanceSeg's 22 GB image set
locally; masks/annotations only, or send it to the pod volume.

⚠️ **Synapse gotcha:** `syn.get(..., downloadLocation=...)` returns `path=None` and downloads
nothing. Use the presigned URL instead — `GET /repo/v1/entity/{id}/file?redirect=false` — and fetch
it with a **non-default User-Agent** (plain `urllib` gets 403 from S3). Working implementation in
`scratchpad/robustmis_probe.py`.

### The one hard prohibition

**Never pull `syn21891314/Stage_3`** — it is Sigmoid Resection = our `val_ood`. ROBUST-MIS's
Training + Stage_1 + Stage_2 are all procto/rectal, already inside our `train` split, so they are
safe by construction. `G-NO-OOD-BLEED` exists to catch this mechanically.

### Obligations if ROBUST-MIS is used

Cite **Maier-Hein et al. 2021 (Sci Data)** and **Roß et al. 2021 (MedIA)**, and ship any derived
QA under **CC BY-NC-SA** ("new creations must use the exact same terms"). Compatible with the
challenge's own publish-your-annotations rule.

---

## THE PROMPT — paste this into the new session

```
Continuing the ORENA FRAME challenge on branch task/r3-rung16.

Read these first, in order:
  1. HANDOFF_RUNG19.md              (this handoff — the full state)
  2. context/NOW.md                  (the 2026-07-27/28 entries)
  3. context/decisions/leaderboard-metric-vs-our-headline.md
  4. experiments/19-external-count/README.md   (the three dataset measurements)
  5. context/RULES.md §3, §4b, §4c, §4d

Note: a second Claude session works in this same tree on rungs 16/18. Do not clobber it.

IMMEDIATE TASK — finish the ROBUST-MIS kill-check.

scratchpad/robustmis_probe.py was mid-run when the last session ended (659 masks
downloaded, ~2 KB each, working). Re-run it, then answer ONE question:

  Does ROBUST-MIS's instance-count distribution resemble our own `number` gold?

Compare its per-frame counts (`unique(mask) - 1`, range 0-7) against the gold
distribution of our `number` questions from external_data/orena-data/*/data/frame/*.parquet.
Report both histograms side by side. Then move the probe into
experiments/19-external-count/_tools/ and commit the finding — negative or positive,
the same way SAR-RARP50 and CholecInstanceSeg were recorded.

THEN, depending on the answer:

  If the distributions are comparable -> ask me before spending pod. The next step is
  19a: score rung 06 ep3 (checkpoint-2580) zero-shot on ROBUST-MIS frames asking
  "How many surgical instruments are visible?" against the exact counts. That is ~2
  GPU-hours and it decides whether a ~20 GPU-hour training rung is worth running.
  Report bias AND correlation, not just accuracy — a model emitting a prior and a
  model perceiving badly fail differently.

  If they are not comparable -> say so plainly, close the external-data lever, and
  re-open the other path I want: a bigger/newer backbone. Qwen3-VL-32B is already on
  the volume, and today's latency finding (0.79 s/q against a 5.06 ceiling, 6.4x
  headroom) removed the objection that had gated the size ladder.

Rules that bind you: single-variable A/B against the rung 06 ep3 control; the eval set
and the organizers' ID/OOD split stay untouched; gates RAISE and are never disabled;
score only via frame.metrics. Do not propose resubmitting a checkpoint to the
leaderboard and do not propose seed-repeat runs — I rejected both.
```
