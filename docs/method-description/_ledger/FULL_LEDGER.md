# Full ablation ledger — every rung, single-variable, with citation

Compiled 2026-09-07 from `context/NOW.md`, `context/decisions/*`, `ATTACK_LADDER.md`,
`RESULTS.md` and each `experiments/*/README.md` + `RESULTS*.csv`.

**Conventions.** `BM` = `bucket_mean`. `NR` = not recorded in the repo. A **faithful negative**
is a valid test that did not beat its baseline — these are results, not failures. Deltas computed
from two recorded values are rounded to four decimals.

Every rung is single-variable against a **named** prior rung; flags default OFF so that a
disabled flag is byte-identical to the control.

---

## Rungs 00–20

| rung / run | single variable | internal metric | Δ vs named baseline | verdict | citation |
|---|---|---:|---:|---|---|
| 00 | Qwen3-VL-8B zero-shot, 1 frame, greedy | BM 0.2557 | anchor | baseline | `RESULTS.md:54`; `ATTACK_LADDER.md:8-13` |
| 01 | Video/procedure split | 28 lapchole / 2,252 q ID; 10 heico / 4,000 q OOD | n/a | infrastructure | `experiments/01-ood-split/RESULTS.csv:2`; `context/RULES.md:23-40` |
| 02 ep1 | LoRA on LLM only, 1 epoch | BM 0.5282 | +0.2725 vs 00 | **WIN** | `RESULTS.md:45` |
| 02 ep2 | LoRA on LLM only, 2 epochs | BM 0.5486 | **+0.2929 vs 00** | **WIN** | `experiments/06-vit-lora/RESULTS_arms.csv:2`; `RESULTS.md:38` |
| 03 a0 | Remove FO grounding from prompt | acc_ID 0.1808 | −0.0613 vs a1 | faithful negative | `RESULTS.md:56-57` |
| 03 a1 | Original FO prompt | ID 0.2421; OOD 0.2735 | control | baseline | `RESULTS.md:57` |
| 03 a2 | "Decisive" wording | acc_ID 0.2505 | +0.0084 | faithful negative | `RESULTS.md:58` |
| 03 a3 | "Instruments ≠ FO" emphasis | acc_ID 0.2234 | −0.0187 | faithful negative | `RESULTS.md:59` |
| 03 a4 | Negative exemplar | ID 0.2602; OOD 0.2640 | +0.0181 ID; −0.0095 OOD | faithful negative | `RESULTS.md:60` |
| 03 a5 | Bare-digit answer instruction | acc_ID 0.2136 | −0.0285 | faithful negative | `RESULTS.md:61` |
| 04 | Vendor/tutorial baseline | NR — never run | NR | unrun | `experiments/04-vendor-baseline/README.md:13-34` |
| 05 real | Correct image | BM 0.5503 | control | baseline | `RESULTS.md:37` |
| 05 black | Black image | BM 0.2752 | **−0.2751** | NO-GO for "text alone suffices" | `RESULTS.md:37,53` |
| 05 shuffled | Image from another question | BM 0.3338 | **−0.2165** | NO-GO for pure shortcut | `RESULTS.md:37,51` |
| 05 count LUT | Post-hoc integer calibration | oracle +0.0263; ID→OOD −0.0164 | −0.0164 transfer | NO-GO | `experiments/05-bottleneck-audit/RESULTS_count_confusion.csv:2` |
| 06 ep1 | Open LoRA on the ViT | BM 0.5345 | +0.0063 vs 02 ep1 | inconclusive | `experiments/06-vit-lora/RESULTS_epoch1.csv:2-3` |
| 06 ep2 | LoRA ViT + LLM | BM 0.5667 | **+0.0181 vs 02 ep2** | PARTIAL | `experiments/06-vit-lora/RESULTS_arms.csv:2-3` |
| 06 ep3 | One extra epoch | BM 0.5724 | +0.0057 vs ep2 | inconclusive | `experiments/06-vit-lora/RESULTS_epoch3.csv:2` |
| 07 | Reasoned enumeration | NR — no candidate | NR | method closed | `experiments/07-enumeration/README.md:1-5` |
| 08 | Data card + per-template floors | 6,252 q; 4,486 frames; 38 videos | n/a | diagnostic | `experiments/08-data-card/README.md:28-35` |
| 09 | Chain-of-Adaptation SFT (local) | header only | NR | unrun | `experiments/09-coa-sft/RESULTS.csv:1` |
| 09 published control | Scaffold-SFT without RL | 62.0 vs bare-gold 65.7 | −3.7 pp (other dataset) | NO-GO as external evidence | `context/decisions/coa-sft-published-null.md:47-74` |
| 10 A | Self-consistency k=8, T=1 | acc 0.4088 | ID −0.0053; OOD −0.0195 | faithful negative | `experiments/10-self-consistency/RESULTS.csv:2` |
| 10 B | k=8, T=1.3 | acc 0.4112 | ID −0.0172; OOD −0.0074 | faithful negative | `experiments/10-self-consistency/RESULTS.csv:3` |
| 10 C | k=16, T=1 | acc 0.4150 | ID +0.0284; **OOD −0.0430** | **NO-GO** (OOD CI excludes 0) | `experiments/10-self-consistency/RESULTS.csv:4` |
| 11a | Native-resolution gate | 130/130 videos single-resolution; non-monotonic | n/a | NO-GO | `experiments/11-resolution/README.md:21-34` |
| 12 arm1 | Unsharp ×1, inference only | fo ID −0.0259; fo OOD −0.0109 | negative | faithful negative | `experiments/12-image-processing/RESULTS.csv:2-3` |
| 12 arm2 | Unsharp ×3, inference only | fo ID −0.0558; OOD −0.0582 | negative | NO-GO | `experiments/12-image-processing/RESULTS.csv:4-5` |
| 12c control | 25 % corpus, plain image | BM 0.4792 | control | baseline | `RESULTS.md:50` |
| 12c composite | Image + trained edge map | BM 0.4819 | **+0.0027** | faithful negative | `RESULTS.md:49-50` |
| 12d screen | Ranking of 32 transforms | tophat +0.0043 pooled → −0.0172 within-video | inverts | NO-GO; confounded instrument | `context/decisions/pooled-screening-manufactures-winners.md:23-35` |
| 13 α=.85 | WiSE-FT weight interpolation | BM 0.5642 | −0.0025 vs 06 ep2 | faithful negative | `RESULTS.md:30,32` |
| 13 α=.70 | WiSE-FT | BM 0.5533 | −0.0134 | faithful negative | `RESULTS.md:36` |
| 13 α=.50 | WiSE-FT | BM 0.5008 | −0.0659 | NO-GO | `RESULTS.md:48` |
| 14 ep1–3 | Appearance augmentation | BM 0.5291 / 0.5642 / 0.5611 | −0.0376 / −0.0025 / **−0.0114** | faithful negative | `experiments/14-appearance-aug/RESULTS_epochs.csv:2-4` |
| 15 ep1–3 | Structured count target | BM 0.5274 / 0.5612 / 0.5699 | −0.0072 / −0.0055 / **−0.0025** | faithful negative | `experiments/15-count-target/RESULTS_epochs.csv:2-4` |
| 16a | Teach "0"/absence via prompt | FT emits absent-0 on 0.8–1.7 %; SDK-invalid 86.7–87.5 % | collapse | NO-GO | `experiments/16-count-probes/RESULTS_16a.csv:6-9` |
| 16c a0 | Direct count | acc 0.2761 | control | baseline | `experiments/16-count-probes/RESULTS_16c.csv:2-4` |
| 16c a1 | Enumerate/reason before counting | acc 0.1865 | **−0.0896** | NO-GO | `experiments/16-count-probes/RESULTS_16c.csv:5-7` |
| 16c a2 | Pointing / coordinates prompt | acc 0.1410 | **−0.1351** | NO-GO | `experiments/16-count-probes/RESULTS_16c.csv:8-10` |
| 17 | Qwen3-VL-32B as blind teacher | acc 0.080 vs floor 0.295 | −0.215 below floor | NO-GO (kills distillation) | `context/INDEX.md:95`; `context/NOW.md:1918-1925` |
| 18 ep1–3 | Count augmentation (minted zeros + paraphrase) | BM 0.5255 / 0.5488 / 0.5721 | −0.0469 / −0.0236 / **−0.0003** | faithful negative | `experiments/18-count-aug/RESULTS.csv:2-4` |
| 19a | Counting on large external instruments | both models acc 0.040 at count=3 | worse than 0.200 on 4 mm clips | NO-GO for "small object" hypothesis | `context/NOW.md:653-661` |
| 19b ep1–5 | +5,718 Strasbourg frames | BM 0.5465 → 0.6557 | ALL_ID −0.0089; ALL_OOD +0.0075, both CIs cross 0 | faithful negative | `experiments/19b-external-recognition/RESULTS_epochs_full6252.csv:2-6` |
| 20 control | Substitute judge Qwen3-4B | BM 0.5724 | control | baseline | `experiments/20-judge-swap/RESULTS.csv:2` |
| 20 official | Official judge Qwen3.5-4B | BM 0.5710 | **−0.0014** | faithful negative: the judge does not explain the gap | `experiments/20-judge-swap/RESULTS.csv:3` |

---

## Rungs 21–40

| rung / run | single variable | internal metric | Δ vs named baseline | verdict | citation |
|---|---|---:|---:|---|---|
| 21 A | LR 2e-5 → 1e-4 | BM 0.6305 | **+0.0585 vs 18 ep3** | **WIN** | `experiments/21-recipe-sweep/RESULTS_A_lr.csv:4` |
| 21 A2 | LR 1e-4 → 2e-4 | BM 0.6496 | **+0.0191 vs A** | **WIN — shipped** | `experiments/21-recipe-sweep/RESULTS_A2_lr.csv:4` |
| 21 A3 | Lower `vit_lr` | BM 0.6185 | **−0.0311 vs A2** | NO-GO | `experiments/21-recipe-sweep/RESULTS_A3_vitlr.csv:4` |
| 21 B | Rank 8 → 32 | BM 0.6478 | +0.0173 vs A; −0.0018 vs A2 | inconclusive | `experiments/21-recipe-sweep/RESULTS_B_rank.csv:4` |
| 21 D | Gradient clipping 1 → 10 | BM 0.6463 | −0.0033 vs A2 | faithful negative | `experiments/21-recipe-sweep/RESULTS_D_clip.csv:4` |
| 22 | Loss-mass equalisation by format | pre-flight only: `number` 34.2 % rows → 20.8 % gradient | intervention NR | NO-GO; arm never ran | `context/decisions/loss-mass-is-token-weighted.md:25-38` |
| 23a | 8B gen-3 → 27B gen-3.6 zero-shot | BM 0.2913 | +0.0356 vs 00, still below floor | NO-GO | `context/decisions/backbone-generation-is-not-the-lever.md:17-26` |
| 23b | 35B-A3B FP8 | NR; gate 23a failed | NR | unrun | ibid.:49-56 |
| 24 ep1–3 | Label-aware horizontal flip p=0.25 | BM 0.5440 / 0.6206 / 0.6333 | −0.0167 / +0.0022 / **+0.0028** | faithful negative | `experiments/24-geometric-aug/RESULTS_flip_p25.csv:2-4` |
| 24 position probe | class→quadrant dependence | interaction +0.0608 [+0.0023, +0.1225] | mechanism only | no net win | `context/NOW.md:2013-2025` |
| 25 | Individuation / instance segmentation | NR; blocking gate never ran | NR | closed-unrun | `context/INDEX.md:98` |
| 26 part1 | Linguistic shortcut audit | 40.6 % exposed; fo_class ≈ +0.22, number −0.02 | diagnostic | inconclusive | `experiments/26-deshortcut-eval/README.md:17-25` |
| 26 part2 | Drop cardinal premise / "list all" | premise-dropped +0.0021 ID / +0.0064 OOD; set-framed −0.0647 | equivalent / negative | faithful negative for phrasing | ibid.:27-41 |
| 27 | Decoupled high `vit_lr` | NR — closed-unrun | NR | unrun | `context/INDEX.md:98` |
| 28 | VCD (visual contrastive decoding) gate | pClip 0.7794 → 0.6198 | **−0.1596** against its own premise | NO-GO; arm never built | `context/decisions/vcd-has-nothing-to-subtract.md:2-6` |
| 29 | SAM2 temporal | corrected gold movement 0.384; model error 2.6×; n=11 | no model score | closed-unrun | `experiments/29-sam2-temporal/README.md:3-13` |
| 30 | **GRPO (RL)** on exact-match, `number` only | `aggregation_ID` 0.4932 vs 0.5026; BM 0.6478 vs 0.6513 | **−0.0094 primary; −0.0035 BM** | **NO-GO** | `experiments/30-grpo-number/README.md:182-195` |
| 31 | Visual-attention mass across checkpoints | 0.0519 → 0.0937 → 0.0802 → 0.1283 | diagnostic | inconclusive | `experiments/31-attention-probe/RESULTS_attention.csv:2-5` |
| 32 | `freeze_aligner=false` | 720 tensors = 504 LLM + 216 ViT + **0 aligner** | +0 reachable tensors | NO-GO in transformers 4.57 | `experiments/32-aligner-unfreeze/RESULTS_reachability.csv:2-3` |
| 33 | Decode P(number) distribution | greedy 0.4680; mask-0 0.4804; E[value] 0.4647 | honest headline ≈ +0.010–0.015 | NO-GO — below action bar | `experiments/33-number-logit-probe/README.md:36-60` |
| 34 | Linear probe on hidden-state count | OOD **0.5264** vs head 0.4680, layer 24 | **+0.0584** | positive diagnostic | `experiments/34-hidden-state-probe/README.md:9-10` |
| 35 | **NTL-WAS ordinal loss** | `aggregation_ID` −0.0262; `object_recognition_ID` −0.0880; BM −0.0431 | negative | **NO-GO** | `experiments/35-ntl-was/README.md:95-110` |
| 36 KTB | Did the base forget Clip/Sponge? | 24/148 = 0.1622 base-correct | n/a | NO-GO: not forgetting | `experiments/36-clip-sponge-probes/README.md:39` |
| 36 KTA | Top-2 class-set reranking | gold-top2 0.5917 vs chance 0.5359; Δ +0.0558 [+0.0149, +0.0961] | imported gate invalid | inconclusive | ibid.:60-80 |
| 37 | Attention vs SAM-mask gate | coverage 0.9714; clean-given-covered 0.3529 [0.206, 0.529] | fails the 0.70 threshold | NO-GO | `experiments/37-attention-vs-masks/README.md:3-12` |
| 38 | A2 recipe verbatim on the 27B | BM 0.5302; paired ALL −0.0334 [−0.0642, −0.0027] | negative | NO-GO for recipe transfer | `experiments/38-gen36-ft-screen/RESULTS.csv:2` |
| 39 | **Connector LoRA on the 8B** | `obj_rec_ID` +0.0361 [−0.0019, +0.0781]; ALL_ID +0.0227, CI crosses 0 | not significant | inconclusive | `experiments/39-connector-lora/RESULTS_paired_ci_ep1_full.csv:2-9` |
| 40 A | 27B `lora_alpha` 32 → 16 | BM 0.5453; paired ALL +0.0251 [+0.0030, +0.0498] | positive | WIN, 27B only | `experiments/40-gen36-recipe-connector/RESULTS.csv:2` |
| 40 B | 27B connector LR 4e-5 | BM 0.5763; paired ALL **+0.0540** [+0.0312, +0.0784] | positive | WIN, 27B only | `experiments/40-gen36-recipe-connector/RESULTS.csv:3` |

---

## Rungs 42–61

| rung / run | single variable | internal metric | Δ vs named baseline | verdict | citation |
|---|---|---:|---:|---|---|
| 42 ep1 | Merged corpus + connector, epoch 1 | BM 0.5649 | −0.0693 vs A2 ep3 | inconclusive | `experiments/42-merged-corpus/RESULTS.csv:2` |
| 42 ep2 | epoch 2 | BM 0.6242 | −0.0100 | inconclusive | ibid.:3 |
| 42 ep3 | epoch 3 | BM 0.6262 | −0.0079 | confounded by schedule | ibid.:4 |
| **42 ep4** | **epoch 4** | **BM 0.6744** | **+0.0402 vs A2 ep3** | **WIN (bundle) — SHIPPED as submission 03** | ibid.:5; `submissions/03-rung42-connector-ood/README.md:75-93` |
| 42 ep5 | epoch 5 | BM 0.6592 | +0.0250 vs A2; **−0.0152 vs ep4** | faithful negative (memorisation) | ibid.:6 |
| 43 | Thinking at inference | acc 0.4188 vs 0.6485; 9.888 vs 0.515 s/q | **−0.2297** | **NO-GO** | `context/NOW.md:663-675` |
| 43 trace | Second model extracts the number from the trace | 0.1786 / 0.2233 vs random 0.2189; breaks 24.2–37.7 % correct | ≈ chance | NO-GO | `context/decisions/trace-extractor-is-a-coin-flip.md:14-29` |
| 44 | 27B FP8 + vLLM, `enforce_eager` | 30/50 exact, identical to HF; 0.498 s/q; startup 126.7 s | 0 quality cost; 2.3× speedup | WIN for deployability, no score lift | `experiments/44-fp8-deployability/RESULTS_fp8_eager.json:2-12` |
| 45 R00 | 27B NF4, original corpus | BM 0.4315 | control | baseline | `experiments/45-gen36-data-and-reg/RESULTS.csv:2` |
| 45 R0 | + merged corpus on the 27B | BM 0.3170 | **−0.1145** | NO-GO | ibid.:3 |
| 46 A alone | 27B single pass | BM 0.5916 | control | baseline | `experiments/46-cross-model-debate/RESULTS_scored.json:3-28` |
| 46 self-revise | 27B revises its own answer | BM 0.5893 | −0.0022 | faithful negative | ibid.:47-73 |
| 46 debate | 8B critiques the 27B | BM 0.6314; ALL_ID +0.0497 | +0.0421 vs self-revise | **NO-GO for shipping — the 8B alone scores 0.6727** | `context/NOW.md:534-542` |
| 47 ep3 | A2 corpus, 3 epochs | BM 0.6142 | −0.0120 vs 42 ep3 | inconclusive | `experiments/47-epochs-vs-corpus/RESULTS.csv:2` |
| 47 ep4 | A2 corpus, 4 epochs | BM 0.6468 | **+0.0327 vs ep3**; −0.0276 vs 42 ep4 | **WIN on epochs**; corpus inconclusive | ibid.:2-3; `context/decisions/rung42-gain-was-epochs-not-corpus.md:32-46` |
| 47 ep5 | A2 corpus, 5 epochs | BM 0.6466 | −0.0002 vs ep4 | faithful negative | ibid.:4 |
| 48 | Centre proxy on CholecT50 (Strasbourg) | bag-F1 orders 3/3 anchors; macro only 7/15 | diagnostic | inconclusive | `experiments/48-centre-probe/RESULTS_probe_v2.csv:2-10` |
| 49 heldout | Horizontal-flip equivariance | Δacc −0.0076 [−0.107, +0.087] | unreadable | inconclusive | `experiments/49-flip-equivariance/README.md:86-101` |
| 49 memorized | same, on train | Δacc **−0.0467** [−0.088, −0.013] | negative | NO-GO | ibid.:204-218 |
| 50 A | Explicit cardinality prefix | BM 0.6480; format learned 100 % | +0.0012 | faithful negative | `experiments/50-set-enumeration/RESULTS.csv:2` |
| 50 B | Continuation-loss weight | BM 0.6594 | +0.0126, CIs cross 0 | faithful negative | ibid.:3 |
| 51a | Price of the Clip attractor | loose ceiling +0.0878; attributable **+0.0387** | diagnostic | inconclusive | `experiments/51-clip-attractor/README.md:64-86` |
| 51c | Strasbourg positives against Clip false positives | fp_rate −0.0405, misses the pre-registered 0.05 line | not attributable | faithful negative | ibid.:269-300 |
| 51 specular | Specular/metal cue hypothesis | 5/13 videos, p = 0.58 | null | NO-GO (refuted) | ibid.:235-252 |
| 52 | Routing 27B/8B by format | 27B − 8B: −0.1402, −0.0710, −0.0450, −0.1189 | all four negative | NO-GO | `experiments/52-model-routing/RESULTS_routing_cells.csv:2-5` |
| 53 | Cross-question consistency constraints | 218 equality pairs, **11.0 % gold violations**; 260 inequality pairs, 0 violations | dose ≈ 2 % | NO-GO; never built | `experiments/53-cross-question/README.md:22-33` |
| 54 | Stack: count target + appearance aug + LoRA+ λ=4 | BM 0.3391 vs fresh r42 0.6716 | **−0.335**, 6/6 cells significant | **NO-GO** | `context/decisions/rung54-stack-collapsed.md:2-3,21-31` |
| 55 | Hidden-state cardinality probe | OOD 0.8849 vs model 0.8504 | +0.0345 | positive diagnostic | `experiments/55-cardinality-probe/RESULTS.csv:2` |
| 56 off | Greedy, K=1 | bag-F1 0.8937 | control | baseline | `experiments/48-centre-probe/RESULTS_rung56_arms.csv:2` |
| 56 sample | Sampling K=5 | bag-F1 0.8899 | −0.0038 | faithful negative | ibid.:3 |
| 56 temporal | Temporal K=5 | bag-F1 0.8986 | +0.0049 | sub-threshold | ibid.:4 |
| 59 | Hidden-state **identity** decoder | best layer 26 = 0.5243 vs model 0.6752 | **−0.1509** | NO-GO | `experiments/59-cardinality-decoder/RESULTS_identity_full.json:6-11` |
| 60 | Vision-side identity probe | best visual-max layer 28 = 0.5652 vs model 0.6752 | **−0.1100** | NO-GO — identity is built by the LLM with depth, not present in the encoder | `experiments/60-vision-side-probe/RESULTS_vision_full.csv:1-12` |
| 61 | Specialist vision head | design/code only | NR | unrun | `experiments/61-specialist-head/_tools/specialist.py:1-19` |

There are no FRAME experiment folders numbered 41, 57 or 58. "Rung 56" is recorded as
`RESULTS_rung56_arms.csv` inside experiment 48.

---

## SEGMENT track (side branch — not the FRAME submission)

| rung / run | variable | metric | Δ / bar | verdict | citation |
|---|---|---:|---:|---|---|
| SEG-01 probe | Harness, token budget, temporal grid | 480 tokens/frame; K ≤ 36 | n/a | infrastructure | `experiments_segment/01-viability/RESULTS_probe.md:11-30` |
| SEG-01 arm A | FRAME A2 + LoRA on SEGMENT clips | BM **0.4964** | **−0.0154** vs bar 0.5118 | faithful negative — one broken bucket | `context/decisions/segment-arm-a-is-one-broken-bucket.md:34-50` |
| SEG-01 temporal | `time` answer format | temporal ID 0.0351; OOD 0.0457 | floors 0.4392 / 0.3299 | NO-GO | ibid.:46-77 |
| SEG-01 recognition | FRAME → SEGMENT transfer | obj_rec ID 0.5792; OOD 0.6139 | +0.029 / +0.064 over floor | bucket win, not headline | ibid.:36-40 |
| SEG-02 premise gate | relative → absolute time target | zero constant 0.0256; midpoint 0.0363; oracle grid 1.0 | premise false | NO-GO; never run | `experiments_segment/01-viability/RESULTS_time_target_premise.json:12-26` |

---

## Known stale artifacts — do not cite these without checking

1. **`RESULTS.md`** declares only 24 runs with canonical stratified data and its table stops
   around rung 24/40. It is **not** the exhaustive ledger.
2. **Rung 17's README** says "PRE-REGISTERED, UNRUN", but the 32B probe was later run and scored
   0.080. The later result prevails.
3. **Rung 39's README** says "NOT RUN", but a full result with CIs exists
   (`RESULTS_paired_ci_ep1_full.csv`).
4. **Rung 24** was marked closed-unrun in an earlier decision; the geometric experiment did
   subsequently run.
5. **Rung 32's connector finding is version-dependent**: 0 tensors under transformers 4.57, eight
   modules under 5.12.1. Not a contradiction — a version difference.
6. **Rung 42 attribution**: "the corpus caused +0.0402" is too strong. The current decomposition
   is +0.0327 epochs / +0.0276 corpus, and only the epoch effect is established.
7. **The old `proxy_leaderboard` ID-only notes are retracted.** The platform metric has always
   been `bucket_mean` (`context/RULES.md:65-78`).
8. **The SEGMENT "zero attractor" explanation is withdrawn.** The `00:00:00` observation is real;
   the "target concentrated near zero" explanation was refuted — the real median offset is
   44–47 s, so it is a temporal-localisation failure, not a formatting one.
