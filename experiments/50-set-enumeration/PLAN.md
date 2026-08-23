# Rung 50 — set enumeration: two mechanisms, one diagnosis

> **Status: PRE-REGISTERED, NOT LAUNCHED.** Written before any GPU. Two arms, one variable each,
> both against the SAME control. Heavy artifacts in `/mnt/storage/uaq_user/rung50/` (UNAM) and
> `/workspace/rung50/` (RunPod).

## The diagnosis this rung acts on

[[fo-class-and-number-are-one-front]], measured 2026-08-22 on six arms, zero GPU:

- `fo_class` accuracy by classes-in-gold: **0.801 / 0.616 / 0.175 / 0.000** for 1/2/3/4.
- Across videos, accuracy correlates **−0.694** with mean gold set size.
- On the 364 frames carrying both question types, `fo_class` and `number` **fail together**:
  odds ratio **2.38**, **z = 3.01** against a within-video null, replicating **6/6** in direction.
- ⇒ one **enumeration** deficit in two output formats, **4,769 of 6,252** rows.

🔴 **And exposure is NOT the problem.** A2's corpus is already **30.6 %** multi-class `fo_class`
against **22.8 %** in the eval — the model sees more conjunctions than it is asked and still
scores 0.616/0.175/0.000. That kills the obvious arm (oversample multi-class rows) before it is
built, and it is why both arms below act on the **learning signal**, not on the data mix.

📌 It also kills the naive read of rung 19b: its 5,718 CholecT50 rows are **all single-class**,
so they diluted the multi-class share **30.6 % → 16.0 %** — the wash has a mechanism.

## The control, and why it is rung 47 and not rung 42

**Control = rung 47 ep4**, `bucket_mean` **0.6468** on the untouched 6,252 / 38 videos.

Not rung 42 ep4 (0.6744) even though that is what shipped: rung 42's corpus promoted **30 of the
38 public test videos into training**, so its 6,252 number is **contaminated**. Rung 47 trained
A2's own 14,415 rows and the eval is clean for it. Rung 19b used exactly this control and its
gate G5 is written against rung 47's literal argv — reused, not re-derived.

⚠️ Any arm must beat **0.6468** on the clean eval to be a candidate, and would still have to
clear rung 42's **0.6744** before it could cost a submission slot.

## Arm A — the TARGET: commit to the cardinality first (UNAM, GPU 0)

**One variable: `--dataset`.** Same 14,415 rows, same recipe, same seed, same schedule. Only the
`fo_class` assistant string changes.

```
before   "Clip, Sponge"                 (free list, 4.03 tokens on average)
after    "2: Clip, Sponge"              (the set SIZE first, then the same list)
```

**Why the count prefix and not oversampling.** The false-`Clip` rate is **4× higher** on
questions asking for a *combination* (0.410) than on ones presupposing a single object (0.108) —
with a free list the model commits to a first class and `Clip` is the default filler. Stating
**N first** forces a cardinality commitment and then binds the list to it, which constrains
**both** measured directions: filler (296 spurious `Clip`) and omission (**123** rows
`missed_a_class`). And it makes the coupling explicit — the model must produce a count in the
`fo_class` format, which is [[fo-class-and-number-are-one-front]]'s whole claim, and it gives the
head a shorter path to the quantity rung 34 found present at layer 24 and lost on the way out.

**Why not ten fixed slots** (`"N Y N N N N N N N Y"`), which was the first design: the token
arithmetic kills it. Priced before writing any code —

| variant | tokens/row | `fo_class` share of gradient | `number` falls to |
|---|---:|---:|---:|
| today, free list | 4.03 | 62.4 % | 20.8 % |
| named slots | ~35 | **93.5 %** | 3.6 % 🔴 |
| bare slots | ~10 | 80.5 % | 10.8 % |
| **count prefix** | **~6** | **71.2 %** | 15.9 % ✅ |

Named slots strangle `number`'s gradient **5.8×** and the arm would be unreadable; bare slots add
a *second* failure mode (positional class confusion) that could not be separated from the
hypothesis. **Slots are the follow-up if the count prefix works, not the first shot.**

**Inference:** `predict()` strips the `N: ` prefix with a deterministic rule. The container emits
exactly the string shape the SDK scores today.

🔴 **DECLARED CONFOUND — this arm is not free of one, and pretending otherwise would be the
defect this repo keeps catching.** Even at ~6 tokens the `fo_class` gradient share moves
**62.4 % → ~71 %**, taking it from `number`. Since the two are one deficit, **a win on
`object_recognition` paid for by a matching loss on `aggregation` is a WASH, not a win.**
Pre-registered: **read both buckets, and the arm does not pass on `bucket_mean` alone.**

⚠️ **A prefix the model gets wrong is a new way to be wrong.** If it emits `"3: Clip, Sponge"`
the parser must still return `{Clip, Sponge}` — the prefix is stripped, never used to truncate
or pad. Disagreement between the stated N and the list length is **counted and reported** as its
own diagnostic, and it is the single most interesting number this arm can produce.

### Gates before the GPU (arm A)
| | gate | dies if |
|---|---|---|
| **A-G1** | **round-trip identity**: every gold → prefixed string → stripped → `metrics.read_fo_class` equals the original set | **any** row differs. 100 %, not 99.9 % |
| A-G2 | the converter is total: every gold in the corpus parses, `none` included | one unparsed row |
| A-G3 | row count, image paths and every non-`fo_class` row are byte-identical to rung 47's corpus | any other row changed |
| A-G4 | single variable vs rung 47: only `--dataset` differs in the argv | anything else moves |
| A-G5 | measured token length of the new answers, and the recomputed per-format gradient share, recorded **before** the run | not recorded |

## Arm B — the GRADIENT: weight the continuation classes (RunPod)

**One variable: `compute_loss_func`.** Same corpus byte-for-byte, same recipe, same seed.

🔴 **Both existing weight functions are IDENTITIES in our configuration — verified in the
source, not assumed.** `normalise_weights` (`src/frame/loss.py:69`) rescales so the mean weight
over supervised tokens is exactly 1.0. At `per_device_train_batch_size=1`:
- `row_group_weights` gives every supervised token in the batch the same row weight ⇒ normalises
  to 1.0 ⇒ identity. **This is the real rung-22 defect**, and it is not "the trigger never
  fired".
- `segment_weights` is an identity too whenever the prompt is masked, because then the
  supervised tokens *are* the answer span and again all share one weight.

⇒ Arm B needs a weight that varies **inside** the answer. That is new code, and it is exactly
what the diagnosis asks for: on rung 19b ep4, **123 rows are `missed_a_class`** — the model names
the first class and stops.

```
answer   "Clip, Sponge"
weights   1  1     w        # w > 1 on the tokens of the 2nd..Nth class only
```

📌 **This is not exposure with a multiplier.** It changes the gradient *within* an example, not
how often the example is seen — so the "exposure is not the problem" finding does not apply to
it, while it does apply to (and kills) an oversampling arm.

🔴 **And it is the OPPOSITE sign to rung 22's planned variable.** Rung 22 proposed per-SAMPLE
normalisation, which equalises rows and therefore *removes* gradient from long answers — i.e.
from exactly the multi-class rows rung 49 identified as the deficit. **Rung 22's variable is
contraindicated by rung 49 and is not revived here.**

### Gates before the GPU (arm B)
| | gate | dies if |
|---|---|---|
| **B-G1** | **OFF is byte-identical**: `enabled=False` returns `None` and the loss matches the unhooked trainer to the last bit | any drift |
| **B-G2** | **ON is NOT an identity at `per_device=1`** — the produced weights must differ from all-ones on a real multi-class row, after normalisation | weights normalise to 1.0, which is how rung 22 died |
| B-G3 | span-finding is exact: the 2nd..Nth class token ranges are located on ≥99 % of multi-class rows; a miss is COUNTED, never defaulted | miss rate >1 % unreported |
| B-G4 | `grad_norm` is non-zero and differs from the control's at step 1 ([[rc-zero-is-not-evidence]]) | it matches the control |
| B-G5 | single variable vs rung 47: the argv is identical, only `compute_loss_func` is set | anything else moves |

## How both are read — pre-registered, before any number exists

**Primary:** `bucket_mean` on the untouched 6,252 / 38 videos, **ep4**, paired against rung 47
ep4 through `frame.metrics`, CI clustered on **VIDEO** (`RULES` §13).

**Both buckets, always.** `object_recognition` AND `aggregation`, reported side by side. An arm
that lifts one and drops the other by a comparable amount is a **WASH** and is recorded as one.

**Mechanism check, not a promotion signal:** re-run rung 49a's anatomy
(`experiments/49-clip-attractor/_tools/fo_class_anatomy.py`) on the arm. **The cardinality curve
(0.801 / 0.616 / 0.175 / 0.000) must flatten.** If the headline moves and the curve does not,
the arm won for some other reason and the reason is unknown.

**Declared NO-GOs**
- Neither arm ships on a local number alone. Local **orders** 3/3 and gets magnitudes 0/3.
- An arm that beats 0.6468 but not rung 42's 0.6744 is a **result, not a submission**.
- `pred_illegal` must stay **0**. Arm A deliberately touches the format; if the converter leaks
  a single illegal string into the eval, the arm is void, not "slightly worse".
- Epochs are NOT a variable here — ep4 is the pre-registered read on both arms, and the epoch
  effect is already established and exhausted ([[rung42-gain-was-epochs-not-corpus]]).

## Cost, and where each runs

| | machine | wall clock | cost |
|---|---|---|---|
| A | UNAM GPU 0 (RTX 6000 Ada) | ~21 h (4,505 steps at rung 47's 17.1 s/it) | $0 |
| B | RunPod, EU-RO-1 volume `ORENA-CHALLENGE` | ~21 h | ~$30 at $1.39/h |

⚠️ **UNAM has two idle cards and could host both for free.** RunPod is used for arm B as
**insurance, not throughput**: UNAM rebooted on 2026-08-18 and killed rung 47 mid-run, and with
pre-eval closing **2026-09-01** a single-machine failure costs a day we do not have. That is the
whole argument for spending the $30; it is not a speed argument.

⚠️ **Pod hygiene** ([[pod-operations-lessons-20260815]]): accept a pod only after reading
`s/it` deltas between consecutive `[step]` lines for 10 minutes — never tqdm's cumulative mean —
and check `utilization.memory` (32 % healthy, 4 % power-capped). Rung 40 lost 8.5 h and $26 to a
capped card that never wrote a checkpoint.
