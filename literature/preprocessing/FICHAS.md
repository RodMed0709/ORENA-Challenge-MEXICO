# FICHAS — Image Preprocessing / Enhancement for Laparoscopic & Endoscopic Imagery

One structured ficha per **Tier-1** and **Tier-2** paper in `INDEX.md`. Tier-3 papers are
catalogued in the index only.

**Reading key.** `Evaluated on` is the field that decides whether a paper is usable evidence for
us: a *perceptual/no-reference* evaluation (PSNR, SSIM, NIQE, human rating, restoration error)
tells us nothing about whether a learned model benefits; a *downstream-task* evaluation
(mAP, Dice, accuracy, F1) does. `Train/test consistency` is the axis our own rung got wrong.
`Verdict` ∈ {STEAL, TEST, CONTEXT, REFUTES-US, IRRELEVANT}.

Anything not stated in the source is marked *not reported*. Nothing here is inferred silently.

---

# TIER 1

---

## 1. Impact of standard enhancement settings of endoscopy systems on performance of endoscopic AI systems

- **Paper** — Jong MR, Kusters CHJ, van Bokhorst QNE, Jukema JB, et al. (BONS-AI Consortium)
- **Venue, year** — *Endoscopy* 57(6):602–610, 2025 (CC BY)
- **file** — `pdfs/p01_jong_2025_enhancement_settings_endoscopic_ai.pdf`

**Problem it attacks.** Endoscopy processors ship with vendor image-enhancement settings that
change the pixels a CAD system sees. Nobody had measured whether these clinically invisible
settings move AI output.

**Method.** 16 enhancement settings on Olympus EXERA III CV190 processors — type A (A1–A8,
contrast/sharpness on fine patterns) and type B (B1–B8, subtler fine-pattern enhancement).
Two systems evaluated: CADe (Barrett's neoplasia detection) and CADx (colorectal polyp
characterisation). Two training regimes compared: standard data augmentation vs
**image-enhancement-based data augmentation** (i.e. train on the enhancement settings).

**Data.** White-light endoscopy. CADe: 6,223 images / 906 patients. CADx: 3,288 images /
969 patients.

**Evaluated on.** **Downstream task metric** — CADe/CADx sensitivity and specificity. No PSNR/SSIM.

**Reported effect.** With standard augmentation, output swings with the setting: CADe sensitivity
83–92 % (9-pt range), CADe specificity 84–91 % (7 pt), CADx sensitivity 78–85 % (7 pt), CADx
specificity 45–63 % (18 pt). With **enhancement-based augmentation** the ranges collapse: CADe
sens. 89–91 % (2 pt, P<0.001), CADe spec. 90–91 % (1 pt, P=0.003), CADx sens. 78–80 % (2 pt,
P=0.03), CADx spec. 55–63 % (8 pt, P=0.19).

**Train/test consistency.** This *is* the paper's subject. Models trained without the enhancement
degrade and become setting-dependent at test; the fix that works is putting the transform in the
**training** distribution as augmentation. Preprocessing at test only is exactly the failure mode.

**Transfer to us.** Extremely high. It is the strongest published statement of the rule our
negative #1 rediscovered the hard way: an input-side transform applied only at inference on a
model fine-tuned without it is biased negative, and the correct experiment is transform-as-
augmentation during LoRA training. It also warns that our HeiCo vs lapchole appearance gap may
itself be a processor/enhancement-settings gap — i.e. an *augmentation* problem, not an operator
problem. Caveat: their models are single-task CNN classifiers, not a 8B VLM with a frozen-ish ViT.

**Verdict** — **STEAL** (the protocol: enhancement-as-augmentation, never test-only).

---

## 2. UIT-Saviors at MEDVQA-GI 2023: Improving Multimodal Learning with Image Enhancement for Gastrointestinal VQA

- **Paper** — Thai TM, Vo AT, Tieu HK, Bui LNP, Nguyen TTB
- **Venue, year** — ImageCLEF 2023 / CLEF Working Notes (arXiv 2307.02783v2), 2023
- **file** — `pdfs/p02_thai_2023_image_enhancement_gi_vqa.pdf`

**Problem it attacks.** Whether endoscopy-specific preprocessing in front of a multimodal VQA
model improves VQA accuracy.

**Method.** Two hand-crafted operators, both *removal-by-inpainting*, not intensity stretching:
(a) **specular-highlight removal** — RGB→greyscale, **fixed global threshold** (explicitly chosen
over adaptive thresholding, to avoid information loss on text/instruments/over-exposed regions),
average smoothing, then **Telea fast-marching inpainting**; (b) **black-mask removal** — synthesise
an artificial black-frame mask from border width, then inpaint; deliberately *preserves* the
black-box artefact because one of the 18 questions asks about it. VQA model: BERT text encoder ⊕
one of 8 vision encoders (ResNet152, Inception-v4, MobileNetV2, EfficientNet-B3, ViT-B/16, DeiT-B,
Swin-B, BEiT-B), concatenation fusion, multi-label classification head, threshold 0.5,
BCEWithLogitsLoss, 15 epochs, batch 64, Adam lr 5e-5.

**Data.** ImageCLEFmed-MEDVQA-GI-2023: 2,000 gastroscopy/colonoscopy images × 18 questions →
28,800 train / 3,600 val / 3,600 test QA pairs, plus a private test set.

**Evaluated on.** **Downstream task metric** — VQA accuracy, precision, recall, F1 (sample-averaged).

**Reported effect.** Best model BEiT-B: accuracy 0.8647 → **0.8725**, F1 0.9074 → **0.9185**
(+1.11 pt F1). Swin-B: F1 0.9090 → 0.9168. Across 8 backbones: **6/8 improved, 2/8 got worse**
(Inception-v4 0.9067→0.9071 ≈ flat; MobileNetV2 0.8906→**0.8867**, EfficientNet-B3
0.9023→0.9046). Authors summarise the range as "+0.4 % to +1.11 % F1". Private test accuracy
0.8201. Per-question, the model still fails on multi-answer colour ("What color is the
abnormality?" 0.58 dev / 0.497 private) and localisation questions.

**Train/test consistency.** **Both.** Explicitly: *"All images from development set and private
test set are first passed into an image enhancement block… The enhanced results are then used as
input in the training and testing of the proposed VQA model."*

**Transfer to us.** This is the single closest published setting to ours — endoscopy + VQA +
preprocessing — and it is a *qualified positive*. But calibrate the prize: ~1 point F1 on a
CNN/ViT classifier with 28.8k training pairs, applied consistently at train and test, with a third
of backbones not benefiting. On our `bucket_mean` (4 cells, 5 answer formats, ~38 videos), an
effect that size would be indistinguishable from noise — which is consistent with our null #3.
Note also *which* operators won: **artefact removal by inpainting**, not contrast/sharpening.
Our screen tested despecular but paired it with CLAHE and judged it by AUC separability, not by
end-task accuracy after joint training.

**Verdict** — **TEST** (specular-inpainting + border/mask handling, applied at train **and** test,
scored by `bucket_mean`, not by a separability proxy).

---

## 3. I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image Classification

- **Paper** — Medeiros DN
- **Venue, year** — arXiv 2604.09697, 2026
- **file** — `pdfs/p03_medeiros_2026_tta_hurts_medical_classification.pdf`

**Problem it attacks.** Whether test-time augmentation — the canonical "apply a transform only at
inference" intervention — is the free lunch it is assumed to be in medical imaging.

**Method.** TTA (prediction aggregation over augmented copies) across 4 architectures spanning
21K–11M parameters, with geometric vs intensity-only augmentation arms, with and without the
original unaugmented image in the ensemble.

**Data.** Three MedMNIST v2 benchmarks (pathology, dermatology, +1).

**Evaluated on.** **Downstream task metric** — classification accuracy.

**Reported effect.** Standard TTA *consistently degrades* accuracy; worst case **−31.6 percentage
points** (ResNet-18, pathology). Sole exception: +1.6 % (ResNet-18, dermatology). Degradation is
most severe for conv nets with batch-norm. Intensity-only augmentations preserve more performance
than geometric; including the original image partially — but not fully — mitigates.

**Train/test consistency.** The paper's whole point: it isolates the **test-only** case and names
the mechanism as *distribution shift between augmented and training-time inputs*, amplified by
BatchNorm statistics mismatch.

**Transfer to us.** Directly validates our negative #1's *interpretation*, independently.
Our unsharp-at-inference result (−0.026 at ×1, −0.056 at ×3, monotone in dose) is textbook: the
model was LoRA-tuned on un-sharpened frames, so sharpening at test moves the input off the
training manifold and the dose-response is the shift magnitude. One caveat that *helps* us: their
mechanism leans on BatchNorm, which Qwen3-VL's ViT (LayerNorm/RMSNorm) does not have — so the
mechanism in our case is representational drift rather than running-statistic mismatch, and our
effect size (−0.026) is correspondingly far milder than their −31.6 pts.

**Verdict** — **REFUTES-US** (specifically: refutes the *design* of our unsharp rung, not its
result; the result was correctly measured and the paper explains it).

---

## 4. Revisiting Underwater Image Enhancement for Object Detection: A Unified Quality–Detection Evaluation Framework

- **Paper** — Awad A, Saleem A, Paheding S, Lucas E, Al-Ratrout S, Havens TC
- **Venue, year** — *Journal of Imaging* (MDPI), 2025
- **file** — `pdfs/p04_awad_2025_underwater_enhancement_vs_detection.pdf`

**Problem it attacks.** Whether image enhancement, chosen and tuned by image-quality metrics,
actually improves a downstream detector — in a domain (underwater) with the same degradation
physics as laparoscopy: scattering medium, non-uniform illumination, colour cast, low contrast.

**Method.** 9 SOTA enhancement methods across 3 paradigms — non-physical (ACDC, TEBCF, BayesRet),
physics-based (PCDE, ICSP), deep-learning (AutoEnh, Semi-UIR, USUIR, TUDA) — each evaluated both
by no-reference quality metrics **and** by detector mAP, at dataset level and per-image.

**Data.** RUOD (14,000 images, 10 object classes) and CUPDD (414 images, 3 plant categories).

**Evaluated on.** **Both, deliberately paired** — no-reference image quality metrics *and*
**downstream detection mAP**. This pairing is the paper's contribution.

**Reported effect.** *"Increases in enhancement metric values do not correlate with increases in
mAP."* Dataset-level: the **original, un-enhanced** detector wins — 0.62 mAP on RUOD, 0.38 on
CUPDD — and most enhancement methods degrade overall detection. Per-image analysis reveals hidden
gains: **selective** enhancement (enhance only the low-quality images) yields +13 % relative on
RUOD and +56 % on CUPDD over the fully-enhanced set.

**Train/test consistency.** Enhancement applied at **inference to the detector's input** (detectors
trained on original data). This is the same design as our negative #1 — and it produces the same
sign. Their rescue is not train-time consistency but **conditional application**.

**Transfer to us.** Very high, and it reframes our screen. Two lessons. (a) Our zero-GPU screen
ranked operators by an image-side separability proxy — this paper shows that class of proxy is
*known* not to predict downstream gain; our screen was measuring the wrong thing, so its null is
weakly informative rather than decisive. (b) The one configuration that worked was **selective /
per-image**, i.e. gate on a quality estimate and enhance only bad frames. Our rung applied every
operator globally to every frame. That arm is untried.

**Verdict** — **REFUTES-US** (kills "global uniform enhancement, ranked by an image metric") **+
STEAL** (the selective/gated protocol).

---

## 5. MLLMs Know Where to Look: Training-Free Perception of Small Visual Details with Multimodal LLMs

- **Paper** — Zhang J, Khayatkhoei M, Chhikara P, Ilievski F (USC / VU Amsterdam)
- **Venue, year** — ICLR 2025 (arXiv 2502.17422)
- **file** — `pdfs/p05_zhang_2025_mllms_know_where_to_look.pdf`

**Problem it attacks.** Whether MLLMs perceive small visual subjects as well as large ones, and if
not, whether the failure is *access* or *localisation*.

**Method.** (1) Sensitivity study: vary the size of the question's visual subject, measure answer
probability. (2) **Intervention study** establishing the effect is causal, not correlational.
(3) Attention/gradient analysis: show the model attends to the right region *even when it answers
wrong*. (4) Training-free visual interventions that use the model's own attention and gradient
maps to produce a focused crop, re-fed to the same model. No training, no external detector.

**Data.** Two widely-used MLLMs × seven VQA benchmarks (natural images).

**Evaluated on.** **Downstream task metric** — VQA accuracy.

**Reported effect.** Significant accuracy improvements across benchmarks without any training;
the sensitivity to subject size is shown causal by intervention.

**Train/test consistency.** N/A in the preprocessing sense — this is a **test-time** method, but
one that changes *framing/geometry*, not *intensity statistics*. That distinction matters: a crop
of the original pixels stays on the training manifold in a way a sharpened image does not (cf. #3).

**Transfer to us.** The highest-value transfer in this corpus. Our foreign objects (clips, needles,
silicone loops) occupy a tiny fraction of a 960×540 frame — precisely the regime this paper says
MLLMs fail in. Crucially it also **reframes our negative #4**: our diagnostic concluded "the
bottleneck is discrimination, not access", because an aux edge map reached the model but moved
accuracy at chance. This paper's finding is compatible with a third option — the model *localises*
fine but its pooled visual tokens do not carry enough resolution at that location. That is fixed by
zoom/crop, not by an aux channel and not by contrast. Risks for us: (a) it needs an extra forward
pass (attention extraction + re-query), which pressures the 5 s/question budget; (b) our questions
are often frame-level ("which classes are present", counts), not single-referent, so a single crop
may lose the aggregation questions — this argues for crop-**and**-original as two images rather
than crop-only.

**Verdict** — **STEAL** (highest priority experiment in this corpus).

---

## 6. The Impact of Image Resolution on Biomedical Multimodal Large Language Models

- **Paper** — Chen L, Burgess J, Nirschl JJ, Zohar O, Yeung-Levy S (Stanford)
- **Venue, year** — Machine Learning for Healthcare / PMLR 298, 2025 (arXiv 2510.18304)
- **file** — `pdfs/p06_chen_2025_image_resolution_biomedical_mllm.pdf`

**Problem it attacks.** MLLMs are built for low-resolution natural images; biomedical images are
native-high-resolution. What does resolution — and resolution *mismatch* — do to performance?

**Method.** Compare native-resolution vs downscaled training and inference across biomedical tasks;
introduce mixed-resolution training as a mitigation.

**Data.** Biomedical imaging tasks/datasets (multiple; specific set not enumerated in the abstract
we verified — full-text tables not transcribed here).

**Evaluated on.** **Downstream task metric** — per-task MLLM performance.

**Reported effect.** Three findings, stated qualitatively in the abstract we verified: (i)
native-resolution training *and* inference significantly improves performance across multiple
tasks; (ii) **train/inference resolution discrepancy causes substantial degradation**; (iii)
mixed-resolution training mitigates the mismatch and trades off compute.

**Train/test consistency.** This is the paper's second finding, on the *resolution* axis rather
than the *intensity* axis. Same doctrine as #1 and #3: whatever transform the input undergoes must
be the same at train and test, or be represented in training as variation.

**Transfer to us.** Two direct implications. (a) Our negative #3 fed the aux edge map **at half
resolution** while the primary frame stayed native — a resolution mismatch *within* one sample. If
this paper is right, that design choice alone could have absorbed the effect it was trying to
measure. Any re-run of the aux-view arm must hold resolution constant. (b) It supports the
`max_pixels` finding being genuinely a non-lever *downwards* (no frame exceeds the cap) while
leaving open the *upwards* question — mixed-resolution LoRA training is a cheap robustness
insurance we have not tried.

**Verdict** — **STEAL** (resolution-consistency rule) / **TEST** (mixed-resolution training).

---

## 7. SegSTRONG-C: Segmenting Surgical Tools Robustly On Non-adversarial Generated "Corruptions" — EndoVis'24 Challenge

- **Paper** — Ding H, Zhang Y, Lu T, Liang R, Shu H, Seenivasan L, Long Y, Dou Q, Gao C, … Unberath M
- **Venue, year** — EndoVis'24 / MICCAI challenge report, arXiv 2407.11906v3 (2026 revision)
- **file** — `pdfs/p07_ding_2024_segstrongc_corruption_robustness.pdf`

**Problem it attacks.** How badly surgical DNNs degrade under plausible non-adversarial appearance
corruptions, and which mitigations the community can actually demonstrate.

**Method.** Dataset built by **counterfactual robotic replay** — paired clean and corrupted samples
of the same scene, so the corruption is the only variable. Corruption types in the unreleased test
set: **bleeding, smoke, low brightness**. Participants train binary robot-tool segmentation on
uncorrupted data and are evaluated on corrupted domains.

**Data.** Robotic surgery video (da Vinci), paired clean/corrupted; multiple participating teams.

**Evaluated on.** **Downstream task metric** — DSC and NSD on tool segmentation.

**Reported effect.** Winners reach **0.9394 DSC / 0.9301 NSD** averaged over the corrupted test
sets. Organisers' conclusion: prior knowledge, customised training strategies and architectural
choice are what buy robustness — while noting *"most approaches rely on conventional techniques
that have known limitations"* and calling for paradigms beyond data augmentation.

**Train/test consistency.** Train clean, test corrupted — deliberately the mismatched setting. The
successful entries closed the gap **from the training side** (augmentation / training strategy),
not by preprocessing the corrupted test images back to clean.

**Transfer to us.** This is the best-controlled surgical analogue of our OOD problem. The
methodological lesson is the one we keep hitting: when appearance shifts, the community's working
answer is *train through it*, not *filter it out at inference*. It also tells us the ceiling of
that answer is real but not spectacular. Limitation for us: binary tool segmentation on da Vinci
data is a much easier, much denser task than foreign-object VQA on 38 videos, so the achievable
DSC numbers do not transfer — only the protocol does.

**Verdict** — **STEAL** (corruption-as-augmentation protocol; also the corruption taxonomy:
bleeding, smoke, low brightness).

---

## 8. A deep learning framework for quality assessment and restoration in video endoscopy

- **Paper** — Ali S, Zhou F, Bailey A, Braden B, East J, Lu X, Rittscher J (Oxford)
- **Venue, year** — arXiv 1904.07073 → *Medical Image Analysis*, 2019/2021
- **file** — `pdfs/p08_ali_2019_endoscopy_quality_restoration.pdf`

**Problem it attacks.** Endoscopy video is polluted by *multiple simultaneous* artefacts — motion
blur, bubbles, specular reflections, floating debris, pixel saturation, contrast loss — and prior
work handled one at a time.

**Method.** Three-part framework: (1) fast **multi-scale single-stage CNN detector** for six
primary artefact classes; (2) a **per-frame quality metric** that also predicts whether restoration
will succeed; (3) **GAN-based restoration** with regularisation for blind deblurring, saturation
correction and inpainting — applied only to *mildly* corrupted frames.

**Data.** Multi-centre endoscopy video; evaluation on 10 test videos.

**Evaluated on.** **Mixed** — artefact detection is a downstream metric (**mAP 49.0** at 5 %
threshold, 88 ms/frame); restoration quality is reported against prior restoration methods
(perceptual); and the *system-level* claim is a **frame-retention metric**: the pipeline preserves
an average of **68.7 % of frames, 25 % more than raw video**.

**Train/test consistency.** Restoration is applied as an inference-time repair, but it is
**conditional on a learned quality score** — i.e. selective, not global, which is the same shape
as the only configuration that worked in #4.

**Transfer to us.** This paper supplies the lever our rung never tested: instead of transforming
every frame, **score every frame and choose which one to sample**. Our pipeline already samples
1–3 frames per clip from a frame cache; the sampling policy is currently position-based, not
quality-based. Swapping in a quality/artefact-aware selector is (a) zero-cost at inference if
precomputed into the cache, (b) does not touch the pixels the model sees, so it cannot trigger the
train/test-shift penalty of #3, and (c) is directly supported by Tier-2 #27 which shows
quality-sorted training data beats random-sorted. The artefact detector itself is 2019-vintage;
we would reimplement the *idea* (cheap per-frame artefact/blur/specular scores) rather than the
model.

**Verdict** — **STEAL** (frame selection by quality score — the cheapest untried lever we have).

---

## 9. What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance

- **Paper** — Afifi M, Brown MS (York University / Samsung AI Toronto)
- **Venue, year** — ICCV 2019 (arXiv 1912.06960)
- **file** — `pdfs/p09_afifi_2019_color_constancy_fools_dnn.pdf`

**Problem it attacks.** Global (not local/adversarial) image manipulation — specifically incorrect
white balance — silently degrading DNN classification and segmentation, and the fact that standard
augmentation does not model it.

**Method.** Three arms compared: (a) baseline DNN on WB-correct data; (b) **novel WB-error
augmentation** that emulates realistic colour-constancy degradation during training; (c)
**pre-processing** train and test images with a WB-correction algorithm (their Deep WB line, Tier-3
#37). Explicit argument that generic colour jitter is *not* a substitute for physically-faithful
WB emulation.

**Data.** CIFAR-10, CIFAR-100 (classification), ADE20K (semantic segmentation); DNNs: ResNet,
RefineNet.

**Evaluated on.** **Downstream task metric** — classification accuracy and segmentation quality.

**Reported effect.** "Notable improvements" on all three datasets from the augmentation route;
pre-processing train+test also helps but is presented as the secondary strategy.

**Train/test consistency.** Both routes are consistent by construction — augmentation puts the
degradation in training; pre-processing is applied to train *and* test. The paper never proposes a
test-only correction.

**Transfer to us.** The doctrinal source for "augment, don't preprocess". Its concrete relevance:
HeiCo (colorectal, OR1 rig) and our lapchole set almost certainly differ in white balance and
colour rendering — a global colour cast, exactly the manipulation this paper says DNNs are
fragile to and that generic jitter under-models. That is a plausible mechanism for our
appearance-rarity/OOD gap that our rung never tested, because our screen contained no
colour-constancy operator at all (unsharp, CLAHE, despecular, bilateral, homomorphic, dehaze,
tophat, sobel, morph-gradient — all luminance/structure, none chromatic).

**Verdict** — **STEAL** (WB-error augmentation during LoRA training as an OOD-bucket lever).

---

## 10. Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI)

- **Paper** — Akyon FC, Altinuc SO, Temizel A (OBSS AI / METU)
- **Venue, year** — IEEE ICIP 2022 (arXiv 2202.06934)
- **file** — `pdfs/p10_akyon_2022_sahi_small_object.pdf`

**Problem it attacks.** Small objects occupy too few pixels after the detector's fixed-size resize;
detail is destroyed before the network sees it.

**Method.** **Slicing-aided hyper inference** — partition the full-resolution image into
overlapping slices, run the detector on each slice at its native scale, merge detections (plus the
full-image pass). Plus **slicing-aided fine-tuning**: build an augmented training set from slices so
the model is trained at the same effective object scale it will be tested at. Detector-agnostic;
integrated with Detectron2, MMDetection, YOLOv5.

**Data.** VisDrone and xView aerial detection benchmarks.

**Evaluated on.** **Downstream task metric** — AP.

**Reported effect.** Inference-only slicing: **+6.8 / +5.1 / +5.3 AP** for FCOS / VFNet / TOOD.
Adding slicing-aided fine-tuning: cumulative **+12.7 / +13.4 / +14.5 AP**.

**Train/test consistency.** Both variants documented, and the numbers make the point cleanly: the
test-only version gives ~half the gain, the train+test version roughly doubles it. Same doctrine
as #1/#3/#9, but here quantified on a *geometric* rather than *intensity* transform — and note
that unlike intensity transforms, the test-only version is still strongly positive, because slicing
preserves pixel statistics.

**Transfer to us.** Medium-high but needs translation: we do not run a detector, we run a VLM that
must answer "which classes are present" / "how many". A tiling scheme maps onto our setting as
*multiple image inputs per question* (full frame + N tiles), which multiplies visual tokens and
therefore latency — the binding constraint at 5 s/question. But the underlying claim ("the fix for
small objects is scale, not contrast") is precisely the axis our 32-operator screen never touched,
and it is corroborated independently by Tier-1 #5 and Tier-2 #22/#23. A 2-tile or centre-crop
variant may be affordable.

**Verdict** — **TEST** (tiled/multi-crop input as an fo_class + count lever, latency-budgeted).

---

## 11. What does CLIP know about a red circle? Visual prompt engineering for VLMs

- **Paper** — Shtedritski A, Rupprecht C, Vedaldi A (VGG, Oxford)
- **Venue, year** — ICCV 2023 (arXiv 2304.06712)
- **file** — `pdfs/p11_shtedritski_2023_red_circle_visual_prompt.pdf`

**Problem it attacks.** Whether a VLM can be steered by *editing the image* rather than the prompt.

**Method.** Draw a **red circle** (and variants) around the region of interest in the input image;
query the VLM zero-shot. Analyse which encoders respond and why.

**Data.** Referring-expression comprehension and keypoint-localisation benchmarks; CLIP, SigLIP,
DeiT, DINOv2 encoders.

**Evaluated on.** **Downstream task metric** — zero-shot referring-expression comprehension
accuracy and keypoint localisation.

**Reported effect.** SOTA zero-shot referring-expression comprehension and strong keypoint
localisation from the marking alone. Critically: the sensitivity **exists in CLIP/SigLIP but not
in DeiT/DINOv2**, and the authors attribute it to red circles being present in the encoders'
web-scale training data.

**Train/test consistency.** Test-time only — but this is a *semantic* edit the encoder was
pre-trained to understand, not a distribution shift in intensity statistics. That is why it
survives test-only application where unsharp masking does not.

**Transfer to us.** Conditional. It only pays if (a) Qwen3-VL's encoder carries the same
red-circle prior (must be probed, not assumed — the paper explicitly shows it is encoder-specific)
and (b) we have something to circle, which requires a localiser we do not currently have. Also
note: after LoRA fine-tuning on un-marked frames, marks become an unseen input pattern, so this
would likely need to be in training too — pushing it from cheap to expensive. Keep as a probe,
not a plan.

**Verdict** — **TEST** (cheap zero-GPU probe on the base model; do not build on it before probing).

---

# TIER 2

---

## 12. Dissecting Self-Supervised Learning Methods for Surgical Computer Vision

- **Paper** — Ramesh S, Srivastav V, Alapatt D, Yu T, Murali A, Mascagni P, Padoy N, et al. (CAMMA/IHU Strasbourg)
- **Venue, year** — *Medical Image Analysis*, 2023 (doi:10.1016/j.media.2023.102844; arXiv 2207.00449v3)
- **file** — `pdfs/p12_ramesh_2023_dissecting_ssl_surgical_cv.pdf`

**Problem it attacks.** Whether general-CV SSL recipes — and in particular their **augmentation
policies** — transfer to surgical video.

**Method.** ~200 experiments / 7,000 GPU-hours. Four SSL methods (MoCo v2, SimCLR, SwAV, DINO) ×
a factorial sweep of five design settings, of which the relevant one is augmentation: **Multi-Crop
∈ {2,4,8} crops × Color {on/off} × Geometric {on/off} × Strong-Color {on/off} = 24 configurations**.
Downstream: phase recognition and tool-presence detection on Cholec80.

**Data.** Cholec80 (laparoscopic cholecystectomy) plus 5 other surgical datasets across 10 tasks.

**Evaluated on.** **Downstream task metric** — F1 (phase recognition), mAP (tool presence).

**Reported effect.** (1) **Colour augmentation consistently and significantly improves**
representation quality — analogous to natural images, and attributed to colour distribution being
an easy shortcut otherwise. (2) **Multi-Crop low-resolution views HURT in surgery**: for MoCo v2,
2→4 crops costs **−3.5 % phase-recognition F1**, 2→8 costs **−4.5 %** — an explicit *deviation*
from natural-image results, explained as "discriminative cues may be scattered in the entire image
and be significant only if considered as a whole". (3) Colour and geometric augmentations feature
consistently in top-performing settings; strong-colour and extra low-res views are unclear-to-
harmful. (4) ImageNet-supervised initialisation is a critical positive.

**Train/test consistency.** Augmentation-only study (train-side by definition).

**Transfer to us.** Two sharp, opposite-signed transfers. **Positive:** colour augmentation is the
one augmentation family this paper certifies for laparoscopic data — reinforcing Tier-1 #9 and
pointing at a chromatic lever our screen entirely lacked. **Caution:** it is direct evidence
*against* naive low-resolution multi-crop views in the surgical domain, which tempers the
crop-and-zoom enthusiasm from Tier-1 #5 / #10. The reconciliation is that #5's crop is
*query-conditioned and high-resolution* while Multi-Crop's are *random and low-resolution* — but
our aggregation questions ("which classes are present in the scene") are exactly the "cues
scattered across the whole image" case this paper warns about. Any crop lever must keep the full
frame as a second view.

**Verdict** — **STEAL** (colour augmentation) **+ REFUTES-US-adjacent** (warns against low-res
crop views for scene-level questions).

---

## 13. Specular Reflections Detection and Removal for Endoscopic Images Based on Brightness Classification

- **Paper** — Nie C, Xu C, Li Z, Chu L, Hu Y
- **Venue, year** — *Sensors* 23(2), MDPI, 2023
- **file** — `pdfs/p13_nie_2023_specular_brightness_classification.pdf`

**Problem it attacks.** Colour-space specular detectors need per-scene threshold tuning; endoscopic
images vary in global brightness, so one global threshold either misses or over-segments.

**Method.** Classify the image by overall **brightness class** first, then apply class-conditioned
detection thresholds; restore detected regions with texture-structure-preserving inpainting
optimised for high-resolution throughput.

**Data.** Endoscopic images across a range of brightness levels (public + collected).

**Evaluated on.** **Perceptual / component-level** — detection quality and restoration fidelity,
plus runtime. **No downstream-task metric.**

**Reported effect.** Improved detection across brightness regimes and better texture-structure
restoration versus fixed-threshold baselines, at higher operating efficiency. (Numbers are
component-level; we did not transcribe them because they are not comparable to our metric.)

**Train/test consistency.** N/A — classical, training-free operator.

**Transfer to us.** This is the honest calibration reference for the `despecular` operator our
screen used. Our screen's best chain was `despec+clahe` (+0.0052 AUC, p=0.045, dies under
Bonferroni) — plausibly under-powered rather than dead, and plausibly mis-calibrated the same way
our `homomorphic` was (where softening the dose was monotone: −0.032 → −0.0087 → +0.0012). If
despecular gets a second run, the brightness-conditioned thresholding here is the fix. But note
the ceiling: even Tier-1 #2, which *did* run specular inpainting in front of an endoscopy VQA
model at train+test, bought ~1 F1 point. Expect small.

**Verdict** — **CONTEXT** (method calibration; not worth a rung on its own).

---

## 14. SpecReFlow: an algorithm for specular reflection restoration using flow-guided video completion

- **Paper** — Yin H, Eimen R, Moyer D, Bowden AK (Vanderbilt)
- **Venue, year** — *Journal of Medical Imaging* 11(2):024012, 2024
- **file** — `pdfs/p14_yin_2024_specreflow_specular_restoration.pdf`

**Problem it attacks.** Single-frame specular inpainting invents tissue that was never observed.
Video gives the real content from neighbouring frames.

**Method.** Three stages: (1) contrast-enhancement preprocessing; (2) deep-learning **detection**
of specular regions; (3) **restoration by optical-flow-guided video completion**, propagating
colour and structure from other frames of the same clip. Described by the authors as the first
complete deep-learning detect-and-restore solution with spatial *and temporal* coherence.

**Data.** Endoscopy video (oesophageal/GI).

**Evaluated on.** **Component-level + perceptual** — detection **Dice 82.8 %, sensitivity 94.6 %**;
restoration compared against single-frame methods. **No downstream-task metric.**

**Reported effect.** As above; multi-frame restoration is more accurate than single-frame.

**Train/test consistency.** Inference-time restoration; the detector is trained separately.

**Transfer to us.** **Blocked by our architecture.** Our FRAME pipeline deliberately samples 1–3
frames from a clip and feeds them as images — the whole latency strategy is *not* to decode video.
SpecReFlow needs a temporal neighbourhood and an optical-flow pass, both of which we spend our
5 s budget avoiding. It is also the strongest argument that single-frame specular *inpainting*
(Tier-1 #2's operator, and our screen's `despec`) is filling holes with fabricated texture — which
would explain why it does not help discrimination. Useful as an explanation, not as a tool.

**Verdict** — **CONTEXT** (explains why single-frame despeckling under-delivers; not runnable for us).

---

## 15. SurgiATM: A Physics-Guided Plug-and-Play Model for DL-Based Smoke Removal in Laparoscopic Surgery

- **Paper** — Sheng M, Fan J, Liu D, Zheng G, Kikinis R, Cai W (Sydney / SJTU / Harvard)
- **Venue, year** — arXiv 2511.05059v2, 2025/2026
- **file** — `pdfs/p15_sheng_2025_surgiatm_smoke_removal.pdf`

**Problem it attacks.** Deep desmoking models are accurate but generalise poorly; physics-based
atmospheric models generalise but are inaccurate.

**Method.** A **Surgical Atmospheric Model** derived by statistically optimising a
Mixture-of-Experts at the *output end* of an arbitrary desmoking network, using a **Laplacian-like
error distribution** to model surgical smoke. Adds **two hyperparameters and zero trainable
weights**; architecture-preserving, plug-and-play.

**Data.** Three public surgical datasets, ten desmoking methods, procedures spanning
cholecystectomy, partial nephrectomy, diaphragm dissection.

**Evaluated on.** **Perceptual / restoration error** (with cross-dataset generalisation framing).
The paper motivates itself by downstream tasks — phase recognition, tool detection, segmentation,
depth — but **does not itself measure them**.

**Reported effect.** Commonly reduces restoration error of existing models and relatively improves
their cross-dataset generalisability, with no added parameters.

**Train/test consistency.** Applied at the output of the restoration network; the restoration
network is trained, the downstream consumer is not retrained.

**Transfer to us.** Low as a lever, useful as a diagnosis. Our data (HeiCo colorectal + lapchole)
does contain cautery smoke, and Tier-2 #20 quantifies what smoke costs a surgical model
(IoU 88.49→80.46). But desmoking as a *preprocessor in front of a fine-tuned VLM* would be another
inference-only intervention — exactly the design Tier-1 #3 and #4 say is biased negative. If smoke
matters for us, the literature-consistent move is **smoke augmentation during LoRA training**
(Tier-1 #7's protocol), not a desmoking front-end.

**Verdict** — **CONTEXT** (best-engineered desmoking module if we ever need one; wrong pipeline
position for us).

---

## 16. Self-Supervised Video Desmoking for Laparoscopic Surgery (Self-SVD)

- **Paper** — Wu R, Zhang Z, Zhang S, Gou L, Chen H, Zhang L, Chen H, Zuo W (HIT / SUSTech)
- **Venue, year** — ECCV 2024 (arXiv 2403.11192v2)
- **file** — `pdfs/p16_wu_2024_self_supervised_video_desmoking.pdf`

**Problem it attacks.** Paired smoky/clean laparoscopic data does not exist in vivo; synthetic
smoke does not match real appearance.

**Method.** Use **pre-smoke frames of the same surgical video** as (misaligned) supervision for
smoky frames; a pretrained optical-flow network aligns the output to the pre-smoke reference.
Self-supervised, real-data.

**Data.** Real laparoscopic surgery video.

**Evaluated on.** **Perceptual** — restoration metrics against the pre-smoke reference. No
downstream-task metric.

**Train/test consistency.** Restoration model trained self-supervised; applied at inference.

**Transfer to us.** Low directly. Its value is the **data-construction trick**: within a single
laparoscopic video, temporally adjacent frames give you naturally paired appearance variants.
That trick is reusable by us for a completely different purpose — building appearance-augmented
training pairs (same scene, different smoke/illumination state) for LoRA, which is the
augmentation route the Tier-1 papers keep endorsing. Requires video decoding, which our frame
cache currently bypasses.

**Verdict** — **CONTEXT** (data-construction idea, not a preprocessing lever).

---

## 17. Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset (STANet)

- **Paper** — Liang Q, Li J, Han Z, Wang X, Wang Z, Mei B (Wuhan University / Zhongnan Hospital)
- **Venue, year** — AAAI-26 (arXiv 2512.02780), 2025
- **file** — `pdfs/p17_liang_2025_stanet_smoke_type_aware.pdf`

**Problem it attacks.** Existing desmoking treats smoke as one phenomenon; it has at least two
distinct motion regimes with different spatio-temporal statistics.

**Method.** Splits smoke into **Diffusion Smoke** (local, directional, pre-collision) and **Ambient
Smoke** (global, directionless, post-collision). A mask-segmentation sub-network jointly predicts
smoke mask *and* smoke type via attention-weighted mask aggregation, with a coarse-to-fine
disentanglement module using smoke-type-aware cross-attention between entangled and non-entangled
regions; a reconstruction sub-network desmokes guided by the two mask types. Releases the first
large-scale synthetic video desmoking dataset with smoke-type annotations.

**Data.** Synthetic smoky laparoscopic video (STSVD) with type annotations, plus public benchmarks.

**Evaluated on.** **Both** — restoration quality metrics **and**, per the abstract, *"superior
generalization across multiple downstream surgical tasks"*. This is one of very few enhancement
papers in this corpus that claims a downstream evaluation; we flag that we verified the *claim* in
the abstract and introduction but did **not** transcribe the downstream tables.

**Reported effect.** Outperforms SOTA on quality evaluation; downstream generalisation claimed but
not quantified here.

**Train/test consistency.** Restoration model trained on synthetic pairs, applied at inference to
the downstream consumer.

**Transfer to us.** Low as a lever (same pipeline-position objection as #15), but this is the paper
to read for **how to evaluate an enhancement against a downstream surgical task** — the exact
measurement protocol our own rung lacked when it ranked 32 operators by AUC separability.

**Verdict** — **CONTEXT** (protocol source for downstream-task evaluation of enhancement).

---

## 18. LighTDiff: Surgical Endoscopic Image Low-Light Enhancement with T-Diffusion

- **Paper** — Chen T, Lyu Q, Bai L, Guo E, Gao H, Yang X, Ren H, Zhou L (Sydney / CUHK)
- **Venue, year** — MICCAI 2024 (arXiv 2405.10550)
- **file** — `pdfs/p18_chen_2024_lightdiff_lowlight_endoscopic.pdf`

**Problem it attacks.** DDPMs give the best low-light enhancement quality but are far too slow for
medical/intraoperative use.

**Method.** A lightweight DDPM with a **T-shaped architecture**: capture global structure from a
low-resolution pass, then recover detail over subsequent denoising steps. Adds a **Temporal Light
Unit (TLU)**, a plug-and-play module for training stability and quality.

**Data.** Surgical endoscopic images (low-light).

**Evaluated on.** **Perceptual** — image quality metrics vs. low-light enhancement baselines.
No downstream-task metric.

**Train/test consistency.** Standalone restoration model; inference-time application.

**Transfer to us.** Low. Our frames (HeiCo OR1, lapchole) are lit by the endoscope and are not
chronically dark — low-light is not our failure mode, and the diffusion inference cost is
incompatible with 5 s/question anyway. Catalogued so that "did we consider learned illumination
correction?" has a documented answer: yes, and it does not fit the budget or the problem.

**Verdict** — **IRRELEVANT** (for our setting; kept as the reference for the family).

---

## 19. Benchmarking Robustness of Endoscopic Depth Estimation with Synthetically Corrupted Data

- **Paper** — Wang A, Yin H, Cui B, Xu M, Ren H (CUHK)
- **Venue, year** — MICCAI workshop 2024 (arXiv 2409.16063)
- **file** — `pdfs/p19_wang_2024_endoscopic_depth_corruption_benchmark.pdf`

**Problem it attacks.** Endoscopic models are benchmarked on clean data; real clinical frames are
corrupted, and no endoscopy-calibrated corruption suite existed.

**Method.** Define an endoscopy-specific corruption taxonomy with **five severity levels** across
~18 corruption types — illumination variability (brightness, darkness, contrast), optical
distortions (defocus blur, motion blur, zoom blur, Gaussian blur), noise, compression — and
re-evaluate depth models on the corrupted sets.

**Data.** Endoscopic depth-estimation datasets, synthetically corrupted.

**Evaluated on.** **Downstream task metric** — depth estimation error under corruption.

**Reported effect.** Substantial degradation of otherwise-strong models under clinically plausible
corruption; ranking of methods changes under corruption.

**Train/test consistency.** Train clean / test corrupted — the mismatched setting, used
diagnostically.

**Transfer to us.** Medium-high, but as an **augmentation vocabulary** rather than a result.
If we implement appearance augmentation for the OOD bucket (the route Tier-1 #1, #7, #9 and
Tier-2 #12 all endorse), this paper's parameterised, endoscopy-calibrated corruption list is the
right menu to sample from — it is severity-graded, so we can control dose, which our unsharp rung
showed matters (monotone in dose) and our homomorphic result showed is recoverable
(−0.032 → −0.0087 → +0.0012 as the operator softened).

**Verdict** — **STEAL** (the corruption taxonomy + severity grading, as an augmentation menu).

---

## 20. Synthetic and Real Inputs for Tool Segmentation in Robotic Surgery

- **Paper** — Colleoni E, Edwards P, Stoyanov D (WEISS, UCL)
- **Venue, year** — MICCAI 2020 (arXiv 2007.09107v2)
- **file** — `pdfs/p20_colleoni_2020_synthetic_real_inputs_tool_seg.pdf`

**Problem it attacks.** Tool-segmentation training data is scarce; can synthetic renderings
substitute, and how robust is the result to real-world appearance degradation?

**Method.** Train tool segmentation with mixed synthetic/real inputs; stress-test by superimposing
**Fractional Brownian Motion noise to simulate cautery smoke** on test frames.

**Data.** Robotic surgery video + synthetic renderings.

**Evaluated on.** **Downstream task metric** — IoU on tool segmentation.

**Reported effect.** Median IoU **88.49 %** on the clean test set, dropping to **80.46 %** with
simulated smoke — i.e. an ~8-point appearance penalty on a strong surgical model.

**Train/test consistency.** Clean train / degraded test, measured deliberately.

**Transfer to us.** Calibration value. It gives us a published number for "how much does a
laparoscopic appearance corruption cost a surgical model that was not trained on it" (~9 % relative
IoU). Our own OOD-bucket gap is of a comparable qualitative flavour. It also shows a cheap way to
*synthesise* the corruption (fBm noise) if we want a smoke-augmentation arm without a desmoking
dataset.

**Verdict** — **CONTEXT** (calibration + cheap smoke-augmentation recipe).

---

## 21. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution

- **Paper** — Wang P, Bai S, Tan S, Wang S, Fan Z, Bai J, Chen K, Liu X, Wang J, Ge W, Fan Y, Dang K, et al. (Qwen Team, Alibaba)
- **Venue, year** — arXiv 2409.12191v2, 2024
- **file** — `pdfs/p21_wang_2024_qwen2vl_dynamic_resolution.pdf`

**Problem it attacks.** Fixed-resolution visual encoders throw away detail and distort aspect ratio.

**Method.** **Naive Dynamic Resolution** — images of arbitrary aspect ratio are converted into a
variable-length sequence of patches **without padding**; a ViT encoder extracts features which are
compressed **4× by a 2×2 pooling/merger** before entering the LLM. Plus M-RoPE (temporal /
vertical / horizontal positional decomposition).

**Data.** Qwen2-VL pretraining corpus + standard multimodal benchmarks.

**Evaluated on.** **Downstream task metric** — multimodal benchmark suite.

**Reported effect.** State-of-the-art at the time across resolutions and aspect ratios.

**Train/test consistency.** N/A (backbone design paper), but it defines the mechanism by which
*our* `max_pixels` setting, resizes, tiles, and aux-image inputs translate into visual tokens and
therefore into latency.

**Transfer to us.** This is our backbone family's contract and should be read before any
geometry-side experiment. Three operational facts it fixes for us: (i) there is no fixed input
resolution to "hit" — token count scales with pixels, so a crop is *cheaper*, not more expensive,
than the full frame; (ii) the 2×2 merger means the effective spatial sampling of a small object is
coarser than the raw resolution suggests — the mechanistic reason Tier-1 #5's zoom lever exists;
(iii) a second image (our aux edge map, negative #3) costs a *full second budget* of visual tokens,
which is why halving its resolution was tempting and why that halving then created the
train/inference resolution mismatch Tier-1 #6 warns about.

**Verdict** — **CONTEXT** (mandatory reading before any resolution/crop/aux-image experiment).

---

## 22. LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images

- **Paper** — Xu R, Yao Y, Guo Z, Cui J, Ni Z, Ge C, Chua T-S, Liu Z, Sun M, Huang G (Tsinghua / NUS)
- **Venue, year** — ECCV 2024 (arXiv 2403.11703)
- **file** — `pdfs/p22_xu_2024_llava_uhd_any_resolution.pdf`

**Problem it attacks.** Naive high-resolution handling in LMMs (fixed grids like AnyRes) wastes
tokens and distorts images with mismatched aspect ratios.

**Method.** Three components: an **image-modularisation** strategy that divides native-resolution
images into **variable-sized slices**; a **compression module** that condenses visual tokens from
the LLM's perspective; and a **spatial schema** that tells the LLM how slices are arranged.

**Data.** Standard LMM benchmarks, including high-resolution and OCR-heavy tasks.

**Evaluated on.** **Downstream task metric** — multimodal benchmark accuracy, with token-budget
accounting.

**Reported effect.** Higher accuracy at lower visual-token cost than fixed-grid AnyRes baselines.

**Train/test consistency.** Slicing is present in both training and inference (the LLM is trained
to read the spatial schema) — which is exactly the property that makes it a *train-with* method,
not a *preprocess-at-test* method.

**Transfer to us.** Medium. It is the principled version of the tiling idea in Tier-1 #10, adapted
to LMMs, and it makes the essential point that tiling only works if the model is **trained to
consume the slice layout**. That is the same lesson as Tier-2 #24 (Perception Tokens) and the same
reason our aux edge map — dropped in as a second image with no training signal about what it was —
moved answers without moving accuracy (our negative #4). If we tile, we must LoRA-train with tiles.

**Verdict** — **TEST** (tiling with spatial schema, but only as a train-with arm).

---

## 23. ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration

- **Paper** — Shen H, Zhao K, Zhao T, Xu R, Zhang Z, Zhu M, Yin J (Zhejiang University / Om AI)
- **Venue, year** — arXiv 2411.16044v2, 2024
- **file** — `pdfs/p23_shen_2024_zoomeye_tree_image_exploration.pdf`

**Problem it attacks.** Answering questions about small regions of high-resolution images with an
MLLM that has a fixed token budget.

**Method.** **Training-free tree search** over the image: recursively subdivide, score each
candidate region for question-relevance with the MLLM itself, descend into the promising branch,
answer from the zoomed view. No model modification.

**Data.** High-resolution VQA benchmarks.

**Evaluated on.** **Downstream task metric** — VQA accuracy.

**Reported effect.** Substantial accuracy gains on high-resolution small-detail benchmarks over
single-pass baselines.

**Train/test consistency.** Test-time only; geometry-preserving (crops of the original pixels), so
it does not induce the intensity-shift penalty of Tier-1 #3.

**Transfer to us.** Same family as Tier-1 #5 but multi-pass, so **latency-prohibitive at
5 s/question**: each tree level is at least one extra forward pass. Catalogued to bound the design
space: if a single attention-guided crop (#5) does not pay, a search-based zoom will not pay
either at our budget. Also worth noting for the SEGMENT/PROCEDURE tracks where the budget differs.

**Verdict** — **CONTEXT** (bounds the zoom family; too slow for FRAME).

---

## 24. Perception Tokens Enhance Visual Reasoning in Multimodal Language Models

- **Paper** — Bigverdi M, Luo Z, Hsieh C-Y, Shen E, Chen D, Shapiro LG, Krishna R (UW / Google Research)
- **Venue, year** — arXiv 2412.03548v1, 2024
- **file** — `pdfs/p24_bigverdi_2024_perception_tokens.pdf`

**Problem it attacks.** Some visual reasoning needs intermediate representations (e.g. depth) that
language cannot express and that a VLM's token stream does not naturally contain.

**Method.** Introduce **perception tokens** — intrinsic image representations (e.g. a depth map)
emitted *as tokens* by the model and consumed as auxiliary reasoning steps, analogous to
chain-of-thought. The model is trained to generate and use them.

**Data.** Depth-dependent and counting-style visual reasoning benchmarks.

**Evaluated on.** **Downstream task metric** — task accuracy on reasoning benchmarks requiring the
intermediate representation.

**Reported effect.** Improvements on tasks where the auxiliary representation is causally required.

**Train/test consistency.** The auxiliary map is present in **training** — the model is explicitly
trained to produce and consume it. It is not injected as an extra input at inference.

**Transfer to us.** **This is the closest published analogue to our negatives #3 and #4, and it
explains them.** We concatenated an edge map as a second image at half resolution, on a 25 % data
subsample, with no supervision telling the model what the second image *was for*. This paper's
result is that an auxiliary intrinsic map pays only when the model is trained to *emit and reason
over* it — the representation must be in the reasoning chain, not merely in the input. Our
diagnostic (aux map changes 16.2 % of answers, moves accuracy at chance) is precisely the
signature of "the tokens arrived but carry no learned semantics". Two consequences: (a) our
conclusion "the bottleneck is discrimination, not access" is under-determined — a third
possibility is "the aux channel was never given a job"; (b) re-running the aux-view arm without
either (i) equal resolution and (ii) an explicit training signal about the aux view's role would
reproduce the same null.

**Verdict** — **REFUTES-US** (refutes the *design* of negative #3/#4) **+ TEST** (a properly
supervised aux-view arm is a legitimately different experiment).

---

## 25. HQG-Net: Unpaired Medical Image Enhancement with High-Quality Guidance

- **Paper** — He C, Li K, Xu G, Yan J, Tang L, Zhang Y, Li X, Wang Y
- **Venue, year** — IEEE TNNLS (arXiv 2307.07829), 2023
- **file** — `pdfs/p25_he_2023_hqgnet_unpaired_medical_enhancement.pdf`

**Problem it attacks.** Medical enhancement has no paired low/high-quality ground truth; unpaired
translation methods distort content in ways that matter clinically.

**Method.** Unpaired low→high quality translation guided by high-quality reference images, with
content-preservation constraints; designed so that the enhanced output remains faithful for
downstream analysis.

**Data.** Multiple medical modalities (unpaired low/high-quality sets).

**Evaluated on.** **Both** — perceptual quality **and** downstream task performance on the enhanced
images. The downstream arm is why it is Tier 2 rather than Tier 3.

**Reported effect.** Improved perceptual quality with downstream benefit reported alongside
(numbers modality-specific; not transcribed).

**Train/test consistency.** Enhancement model trained unpaired; applied as a front-end to the
downstream model.

**Transfer to us.** Medium-low as a lever, high as a **template**. If we ever justify a learned
enhancement arm, this is the shape it should take: unpaired (we have no clean ground truth for
HeiCo/lapchole), content-preserving, and **scored by the task, not by PSNR**. The pipeline-position
objection from Tier-1 #3/#4 still applies unless the enhanced images are also used for LoRA
training.

**Verdict** — **CONTEXT** (template for a learned-enhancement arm we are not yet justified in running).

---

## 26. An empirical study of preprocessing techniques with CNNs for accurate detection of chronic ocular diseases using fundus images

- **Paper** — Mayya V, Kamath S S, Kulkarni U, Surya DK, Acharya UR
- **Venue, year** — *Applied Intelligence* 53:1548–1566, 2022/2023
- **file** — `pdfs/p26_mayya_2022_preprocessing_cnn_fundus.pdf`

**Problem it attacks.** Which preprocessing actually helps a CNN on medical images — contrast
operators, noise reduction, or something else — asked as a systematic ablation rather than as a
single proposal.

**Method.** Combine several preprocessing families (illumination/contrast normalisation, noise
reduction, **ROI segmentation**) with CNN classifiers and compare like-for-like on the same task.

**Data.** Colour fundus images; chronic ocular diseases (myopia, diabetic retinopathy, AMD,
glaucoma, cataract).

**Evaluated on.** **Downstream task metric** — classification performance.

**Reported effect.** The headline is not a contrast operator: *CNNs trained on **ROI-segmented**
images outperform models trained on the original input images by a substantial margin.* Intensity
preprocessing is reported as reducing irregular illumination/low contrast but is not the winning
factor.

**Train/test consistency.** Preprocessing applied consistently to train and test in all arms.

**Transfer to us.** Real, in the same direction as Tier-1 #5/#10 and Tier-2 #12: when a systematic
ablation is run, the winner is **where you point the model (framing/ROI)**, not **how you stretch
the histogram**. Different modality (fundus is a static, standardised, circular-FOV image), so the
ROI operation is far easier there than in laparoscopy — we have no ROI without a localiser. But it
is a third independent vote for the geometry axis over the intensity axis.

**Verdict** — **CONTEXT** (independent support for "geometry beats intensity").

---

## 27. Automated endoscopic image quality assessment for robust colorectal lesion detection

- **Paper** — Kim BS, Lee HJ, Park J, Jeong S-Y, Kim MJ, Ahn J-S, Park JW, Kim S (Seoul National University)
- **Venue, year** — *BMC Medical Imaging* 25:473, 2025 (open access)
- **file** — `pdfs/p27_kim_2025_endoscopic_image_quality_assessment.pdf`

**Problem it attacks.** Poor image quality degrades endoscopic lesion detection, and no
**reference-free** quality framework existed to act on it.

**Method.** A no-reference image-quality-assessment framework. Detections are grouped into
**tubelets** (≥10 consecutive frames with a detected lesion), classified as fast- or slow-motion.
The key experiment: compare a detector trained on **randomly selected** images against the same
detector trained on **quality-sorted** images.

**Data.** 6,228 still endoscopic images from 867 patients for training/validation; tested on
**100 colonoscopy videos**.

**Evaluated on.** **Downstream task metric** — positive predictive value of lesion detection on
video.

**Reported effect.** **Quality-Sorted models outperform Random models**; PPV **76.2 %** (fast
motion) and **80.9 %** (slow motion).

**Train/test consistency.** The intervention is on **data selection**, not on pixels — so the
train/test-consistency question does not arise, which is exactly why it is attractive.

**Transfer to us.** High and cheap. Our frame cache already exists; our sampling policy is
position-based. This paper (with Tier-1 #8) says: score frames for artefact load / blur /
specular coverage, and (a) prefer high-quality frames when sampling for **training**, and possibly
(b) prefer them at inference where the clip permits. It is a **zero-GPU-screen-able** intervention
whose failure mode is bounded: it cannot induce a train/test distribution shift, because it never
alters a pixel. It is also the one intervention in this corpus that our four negatives say nothing
against — none of them tested data selection.

**Verdict** — **STEAL** (highest value-per-cost item in the corpus).

---

# What to steal — ranked

The honest bottom line first, because it is the deliverable:

> **The literature does not support "preprocess the pixels" as a lever for a fine-tuned learned
> model on this kind of data.** Every paper here that (a) uses a downstream-task metric and (b)
> applies a hand-crafted intensity transform *at inference only* reports null or negative
> (Tier-1 #3 up to −31.6 pts; Tier-1 #4 with uniform enhancement below the raw-image baseline on
> both datasets; Tier-1 #9's whole thesis). The papers that do get a positive from an image-side
> intervention win on one of exactly three axes — **(i) put the transform in *training*
> (augmentation), (ii) change *geometry/framing* rather than intensity, (iii) change *which
> frames* you use rather than what is in them.** None of our four negatives tested axis (ii) or
> (iii) at all, and axis (i) was tested only for a single operator (edge map) in a design that
> Tier-2 #24 predicts would come out null regardless. So our teammate's rung is a *correct*
> result about the wrong three-quarters of the space. The remaining quarter is worth exactly
> three experiments, below.

---

### 1. Quality-aware frame selection (no pixel is modified)
- **(a) Source** — Kim et al. 2025 (Tier-2 #27) + Ali et al. 2019 (Tier-1 #8); protocol echo in
  Awad et al. 2025 (Tier-1 #4), whose only winning configuration was *selective* application.
- **(b) Configuration to try** — Precompute per-frame scores into the existing frame cache:
  Laplacian-variance blur score, specular-pixel fraction (Nie 2023 brightness-conditioned
  threshold, Tier-2 #13), mean-luminance and saturated-pixel fraction. Two arms, both single
  variable: **(A) training-side** — resample the LoRA training set toward high-quality frames of
  the same clip/question, byte-identical otherwise; **(B) inference-side** — among the 1–3 frames
  our `predict()` already samples, pick by score instead of by position. Score by `bucket_mean`
  with the canonical `frame.metrics` path.
- **(c) Why the four negatives do not refute it** — None of the four touched *which* frame is
  shown; they all touched *what the pixels look like*. It cannot trigger the inference-only
  distribution-shift penalty that killed the unsharp arm (negative #1), because the model still
  sees an unmodified in-distribution frame. It is orthogonal to the 32-operator screen
  (negative #2), which scored transforms, not frames. And it does not depend on the aux-channel
  mechanism that failed in #3/#4.
- **(d) Cost** — **Zero-GPU screen** for the scores (CPU pass over the frame cache, hours). Arm (B)
  is **inference-only and free**. Arm (A) **requires retraining** (one LoRA run).

### 2. Query-conditioned crop-and-zoom, with the full frame retained
- **(a) Source** — Zhang et al. ICLR 2025 (Tier-1 #5, primary); Akyon et al. 2022 (Tier-1 #10) for
  the train+test doubling effect; Xu et al. 2024 (Tier-2 #22) for the token accounting; Ramesh
  et al. 2023 (Tier-2 #12) for the mandatory caution.
- **(b) Configuration to try** — Two images per question: **full frame at native resolution +
  one crop**. Crop chosen either (i) training-free from Qwen3-VL's own attention/gradient map over
  the question tokens (Tier-1 #5's method), or (ii) as a fixed 2-tile / centre-biased crop as a
  cheaper ablation. Hold **resolution identical** between the two views (Tier-1 #6 — do *not*
  repeat negative #3's half-resolution aux). Run as a **train-with arm**: LoRA-tune with the
  two-view input, not as an inference-only bolt-on. Measure latency p99 first — this is the arm
  most likely to breach 5 s/question.
- **(c) Why the four negatives do not refute it** — The 32-operator screen (negative #2) contained
  **zero geometric operators** — every one of unsharp/CLAHE/despec/bilateral/homomorphic/dehaze/
  tophat/sobel/morph-gradient is an intensity or edge transform. Negative #3's aux view was an
  *edge map at half resolution*, i.e. a derived, information-destroying view; a crop is
  *original pixels at native resolution*, which is a different object entirely, and its
  mechanism (Tier-1 #5's causal size-sensitivity result) is independently established. Negative
  #1's inference-only penalty does not apply because a crop preserves pixel statistics
  (cf. Tier-1 #10, where even the test-only version was strongly positive).
- **(d) Cost** — **Inference-only** for the cheap probe (fixed centre/2-tile crop on the current
  checkpoint, one eval run — but expect it to under-read for the same reason negative #1 did);
  **requires retraining** for the real arm. Non-trivial latency risk.

### 3. Appearance augmentation during LoRA training (colour first)
- **(a) Source** — Jong et al. 2025 (Tier-1 #1: enhancement-as-augmentation collapses setting-
  induced variability from 9/18 pts to 1–2 pts); Afifi & Brown 2019 (Tier-1 #9: WB-error
  augmentation, and the explicit finding that generic jitter does not model it); Ramesh et al. 2023
  (Tier-2 #12: **colour augmentation is the one family certified on Cholec80**); Ding et al. 2024
  (Tier-1 #7: augmentation is what the SegSTRONG-C winners used); Wang et al. 2024 (Tier-2 #19:
  severity-graded endoscopic corruption menu); Colleoni et al. 2020 (Tier-2 #20: fBm smoke recipe).
- **(b) Configuration to try** — A single-variable LoRA arm, flag default OFF = byte-identical:
  train-time-only augmentation with **(i) WB/colour-cast emulation** (Afifi's degradation model,
  not plain colour jitter) as the primary, and **(ii)** a severity-1..2 draw from the endoscopic
  corruption menu (brightness, contrast, defocus/motion blur, mild smoke via fBm) as a secondary.
  **No transform at inference.** Target: the **OOD half** of `bucket_mean` — that is where the
  mechanism predicts the gain, and reporting it as a headline-only delta will hide it.
- **(c) Why the four negatives do not refute it** — Negative #1 tested a transform at *inference*,
  which is the opposite intervention; Tier-1 #1 and #3 both say the training-side version is the
  one that works. Negative #2's screen ranked operators by within-video AUC separability of
  class-present vs class-absent — a *discriminability* proxy that Tier-1 #4 shows empirically does
  not predict downstream gain, and which is anyway blind to augmentation (augmentation is not
  supposed to make a single frame more separable; it is supposed to make the *model* less
  brittle). Negatives #3/#4 concern an added input channel, a different mechanism entirely. And
  critically: **our screen contained no chromatic operator at all**, so the family the surgical
  literature actually certifies was never tested.
- **(d) Cost** — **Requires retraining** (one LoRA run per arm; augmentation is free at inference,
  so no latency risk). This is the arm with the strongest literature backing and the weakest
  existing evidence against it.

---

### Explicitly not worth an experiment (and why)

- **More hand-crafted intensity operators / re-tuned CLAHE, unsharp, homomorphic, tophat, dehaze.**
  Tier-1 #4 shows image-quality-metric-guided operator selection does not predict downstream mAP;
  Tier-1 #2 shows the ceiling for the best-case endoscopic version in front of a VQA model is
  ~1 F1 point with 2/8 backbones going backwards. Our screen's `despec+clahe` (+0.0052, p=0.045)
  is inside that noise band. Retuning `homomorphic` further is chasing a null.
- **A desmoking or low-light front-end (SurgiATM / Self-SVD / STANet / LighTDiff).** Right
  operator family, wrong pipeline position: bolting a restoration net in front of a model
  fine-tuned on unrestored frames is precisely the inference-only design Tier-1 #3 and #4 punish.
  If smoke matters, augment with it (item 3), do not remove it.
- **Multi-pass zoom search (ZoomEye, CropVLM).** Latency-incompatible with 5 s/question; the
  single-pass version is already item 2.
- **Red-circle / mark-on-image visual prompting.** Requires a localiser we do not have, and
  Tier-1 #11 shows the effect is encoder-specific and would likely need to be in training too.
  Worth a 30-minute zero-GPU probe on the base model out of curiosity; not worth a rung.
- **Re-running the aux-edge-map arm as-was.** Tier-2 #24 predicts the null: an auxiliary map helps
  only when the model is trained to emit and reason over it, and ours was dropped in at half
  resolution with no supervision about its role. Either redo it properly (equal resolution +
  explicit training signal) or drop the family.
