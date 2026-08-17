# Rung 46 — cross-model debate at inference: two VLMs, one frame, an argument

**Status:** PRE-REGISTERED 2026-08-17, not yet run. Everything below is declared BEFORE any
number exists. `RULES §S3` (one primary cell, declared first) and `§S7` (a faithful null is
published, no re-cutting afterwards) are the reason this file is written first.

## The question

Two fine-tuned VLMs both look at the same frame, exchange positions, and one issues a final
answer. **Does arguing beat answering alone?**

Not *"does the trace contain the answer"* — that is [[trace-extractor-is-a-coin-flip]], a
text-only extractor, measured and closed. Here **both models see the pixels**, which is the
difference that matters: a second visual encoder is new information, not a re-encoding of the
first one's distribution. [[self-consistency-dead]]'s mechanism — *aggregating a distribution
that lacks the information creates none* — does not reach this design, and saying so is the
whole justification for spending the GPU.

## What we already know, both directions

**For.** On identical frames the 27B enumerates at **0.670** where the 8B scores **0.317**
(Spearman 0.692 vs 0.497, [[counting-is-enumeration-not-small-object-perception]]), and `number`
is **71 %** of rung 42's errors. The two models are *not* redundant on the defect that costs us
the most.

**The ceiling, measured 2026-08-17 on rung 19a's paired predictions** (both models, same
questions, both seeing the image), zero GPU:

| | SurgSigma n=1105 | MISAW n=2080 |
|---|---|---|
| best single model | 0.4905 | 0.3913 |
| **union ceiling** (≥1 model right) | **0.5430** | **0.4192** |
| **oracle headroom** | **+0.0525** | **+0.0279** |
| both wrong | 0.4570 | 0.5808 |
| identical prediction | 69.5 % | 62.6 % |

🔴 **That headroom is what a PERFECT arbiter would win.** Any protocol — debate, adversarial
critique, voting, a judge — is bounded by it, which is why it was computed before the protocol
was designed. A real protocol captures a fraction. On external enumeration data that ceiling sits
right at the `§S1` ship bar of 0.03 *before* any protocol loss.

**Against.** Four papers in `literature/vlm-techniques/` agree that explicit reasoning degrades
grounding and object counting specifically. [[margin-is-vision-not-phrasing]] measured that our
`object_recognition` deficit is discrimination, not phrasing — a verbal exchange is aimed at the
half that is not broken.

📌 **The lead has read the ceiling and elected to run it anyway.** Recorded as a decision, not as
an oversight: the ceiling is on external enumeration data, the FRAME ceiling is unmeasured, and a
protocol-independent bound is worth owning either way.

## The arms — one variable

| arm | turns | who speaks last |
|---|---|---|
| **A_alone** (control) | T1 only | A |
| **A_selfrevise** (confound control) | T1 → T3′ | A |
| **A_debate** (subject) | T1 → T2 → T3 | A |

- **T1** — each model answers from the frame alone, standard `SYSTEM_PROMPT`, greedy. A's T1 **is**
  the control, so arm and control share one generation: zero drift between them.
- **T2** — B receives the frame, the question and A's answer, and must AGREE or DISAGREE with one
  short reason plus its own answer.
- **T3** — A receives the frame, the question, its own T1 answer and B's critique, and issues the
  final answer.
- **T3′** — identical to T3 with the critique removed: *"you answered X, reconsider."*

🔑 **`A_debate` vs `A_selfrevise` is the single variable: the presence of B's critique.** Without
that second control a gain is unattributable — a second look at the frame is itself an
intervention, and it is the cheaper explanation. `A_debate` vs `A_alone` measures the whole
pipeline and is reported beside it, never instead of it.

**A is the 27B** (`conn4e5_ep23_fp8`, gen-3.6, thinking OFF), **B is the 8B**
(`r42_ep4_merged`). A speaks last because the control is A alone: keeping the final voice fixed is
what makes the comparison single-variable.

## Population, and why it is not the usual one

**The 8 videos rung 42 held out — 1,283 questions.** `RESULTS_split_42.json`: heico `0023`,
`0027` + six `lapchole`.

🔴 **Forced, not chosen.** B (`r42_ep4`) trained on 30 of the 38 public test videos, so on those
its "opinion" is recall, not perception, and a debate scored there measures memorisation.
A2 ep3 — the uncontaminated 8B the brief asked for — **exists on no machine we hold**: not local,
not UNAM; it lived on a dead RunPod volume. The held-out 8 is the only population clean for both
models, and it is already the team's declared primary eval for anything touching rung 42.

⚠️ Price, stated up front: effective n is **8 videos** (`RULES §13`), two of them heico. **No OOD
CI here is readable** — a jackknife moves `acc_OOD` by a median 0.024 on *38* videos; on 2 it is
not an instrument. This is the same limit `merged-corpus-buys-the-id-half` carries.

## 🔴 Declared BEFORE the run

**Primary cell (`§S3`): `ALL_ID` accuracy, `A_debate` − `A_selfrevise`, paired and clustered by
video (6 videos).** NOT local `bucket_mean` — that overstates the judge by +0.12 and inverts the
bucket ordering.

**Pre-declared secondary:** `aggregation_ID` — where the enumeration asymmetry that motivates the
whole rung lives. Exploratory under `§S5`; it cannot grant the win.

**Win (`§S8`, all three):** (a) the primary cell's paired CI excludes zero in the arm's favour;
(b) it holds on ID and OOD jointly — ⚠️ **unreachable here by construction**, since no OOD CI is
readable on 2 videos, so **the strongest available verdict for this rung is `PROMISING, NOT
SHIPPABLE`**; (c) no cell anywhere shows significant harm.

**Veto:** any cell whose CI excludes zero in the control's favour kills it, including `fo_class`
and macro-F1.

**Scale (`§S1`):** below +0.03 on the primary cell there is no evidence of transfer and nothing
ships; it becomes a team call, not a run.

**A faithful null is published.** No re-cutting to find a cell that clears the bar.

## Gates that RAISE (`RULES §7` — none may be disabled)

| gate | fires when | why it exists |
|---|---|---|
| `G-INFER` | error sentinels > 1 % | `§8c` — a dead engine scores as incapacity |
| `G-FORMAT` | A's T3 parse-failure rate exceeds A's T1 | a debate that makes answers verbose loses points to `Number.verify`, not to reasoning |
| `G-CLASS` | any emitted class outside `FOType.names()` | `§8b` — `FOType.from_name()` RAISES, taking the whole answer down |
| `G-F1` | `fo_class` scored without macro-F1 | `§9b`, `assert_class_f1_reported` |
| `G-DRIFT` | A's T1 disagrees with the archived `r43_ep23_nothink` on > 2 % of shared qIDs | [[archived-results-not-bit-reproducible]] — same box, so this should be ~0 |

## Cost

1,283 questions × 3 batched passes, three sequential model loads on **GPU 0 only** (`tmux`,
`CUDA_VISIBLE_DEVICES=0`). Both cards were free at launch (rung 45's R00/R0 both report
`VERDICT OK`); GPU 1 is left alone regardless. Estimated **~45–70 min**, $0 — UNAM.

Memory is a non-issue for the *experiment* because the passes are sequential: one model resident
at a time, 34 GiB peak of 48. 🔴 It is **not** a non-issue for a submission — 27B FP8 (33.5 GiB) +
8B bf16 (16 GiB) = 50 GiB on a 47.4 GiB card, and the 27B's cold start alone is 126.7 s against a
120 s allowance. **Deployability is explicitly out of scope here and is not evidence either way;
this rung measures whether the idea works, not whether it ships.**

## Order of operations

build → **smoke (60 questions, stratified across both datasets per `§8`)** → read the gates →
full. A smoke that trips `G-FORMAT` stops the rung before the full run is spent.
