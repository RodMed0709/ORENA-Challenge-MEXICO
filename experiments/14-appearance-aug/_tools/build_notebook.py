"""Folder-private generator for ``14_appearance_aug.ipynb``.

The notebook is the launcher; this script only assembles its JSON so the cell
sources can be written and reviewed as plain text (and re-emitted verbatim if the
file is ever corrupted). It is not part of any run. Run it from anywhere:

    python experiments/14-appearance-aug/_tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

MD, CODE = "markdown", "code"

CELLS: list[tuple[str, str]] = [

(MD, r"""# Rung 14 — appearance augmentation during LoRA (`14-appearance-aug`)

**One variable against rung 06: the TRAINING images carry a seeded appearance
augmentation. The evaluation set is untouched.**

Half of `bucket_mean` is OOD, and by MARGIN the model adds *less* there
(rung 06: `margin_ID` +0.207 vs `margin_OOD` +0.148). Rung 12 only ever
transformed at inference, which Medeiros 2026 shows is structurally biased
negative (up to −31.6 pts) and which our own `unsharp` reproduced (−0.026 →
−0.056, monotone in dose). Jong 2025 measured the fix: putting the transform in
**training** as augmentation collapses vendor-setting-induced variability from
9/18 pts to 1–2 pts (P<0.001). This rung is that experiment.

**The augmentation is colour + illumination only.** No sharpening (`unsharp` is
measured negative on this model), no geometry, no crops (Ramesh 2023 measured
low-res multi-crop *hurting* surgical tasks). White-balance-error emulation over
Afifi's five-temperature grid is the primary operator; ImageNet-C
brightness/dark/contrast at severity 1–2, via Wang 2024's endoscopic corruption
menu, is the secondary. Every dose is cited in `_models/appearance.py`, against
**`literature/preprocessing/FICHAS.md`** (Jong 2025 = Tier-1 #1 `p01` · Medeiros 2026
= Tier-1 #3 `p03` · Afifi & Brown 2019 = Tier-1 #9 `p09` · Ramesh 2023 = Tier-2 #12
`p12` · Wang 2024 = Tier-2 #19 `p19`) — **not** the older, unrelated
`literature/FICHAS.md` at the repo root. What is ours rather than a paper's is
declared as ours in `context/14-appearance-aug/CONTEXT.md`.

🔴 **The pre-registration is `context/14-appearance-aug/CONTEXT.md` and it was
written before any number existed.** This notebook produces numbers. It decides
nothing.

🔴 **Save per epoch, report the `number` margin at EVERY epoch.** Training erases
counting (`number` margin OOD, rung 06: +0.015 → +0.013 → **+0.000**), and
`acc_OOD` checkpoint selection picks the checkpoint that erased more. Selecting
last-by-default is how that gets hidden.

**Ladder:** 00 → 0.256 · 02 → 0.549 · 06 → **0.5667** (the control) · 14 → ?
"""),

(CODE, r'''# papermill parameters
SMOKE = True            # True -> tiny export + 2 train steps; proves the chain, decides nothing
SMOKE_STEPS = 2

# The arm. rung 06 IS the control and is NOT retrained (WAVE_SPEC: "No control arm
# needs retraining"), so a full run here is always AUG_ON=True. AUG_ON=False exists
# only to run the byte-identity gate.
AUG_ON = True

# rung 06's committed artifacts on the pod volume.
REF_TRAIN_JSONL = "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/train.jsonl"
REF_RESULTS     = "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/eval_best/results.csv"
DATA_ROOT       = "/workspace/orena-data"
'''),

(CODE, r'''import json, logging, os, sys, time
from pathlib import Path
import pandas as pd

# `swift` is shelled out to; a papermill/Jupyter kernel does not inherit the env's
# bin/ on PATH, and it must be THIS interpreter's bin so CLI and kernel match.
_envbin = str(Path(sys.executable).parent)
if _envbin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _envbin + os.pathsep + os.environ.get("PATH", "")

EXP = Path.cwd()
REPO = EXP
while REPO != REPO.parent and not ((REPO / ".git").exists() or (REPO / "src").is_dir()):
    REPO = REPO.parent
for p in (REPO / "src", EXP / "_models",
          REPO / "experiments/02-lora-sft/_models", REPO / "experiments/06-vit-lora/_models"):
    if p.is_dir():
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S")

import appearance as A
from aug_export import (AppearanceAugConfig, main, diff_vs_rung06,
                        gate_flag_off_sha256, gate_aug_jsonl_shape, gate_eval_untouched,
                        list_checkpoints, merge_checkpoint)

# The policy: every field is a literature constant (see _models/appearance.py).
POLICY = A.AugPolicy(seed=42)

cfg = AppearanceAugConfig(
    data_root=Path(DATA_ROOT),
    exp_dir=REPO / "experiments" / "14-appearance-aug",
    # A separate run dir for the SMOKE, or its throwaway checkpoint-1/-2 land next to
    # the full run's and list_checkpoints() reads them as real epochs (rung 06's lesson).
    run_name="14_appearance_aug_smoke" if SMOKE else "14_appearance_aug_v1",
    smoke=SMOKE,
    smoke_max_steps=SMOKE_STEPS,
    aug=POLICY if AUG_ON else None,     # 🎯 THE VARIABLE
)
print("run_dir     :", cfg.run_dir)
print("policy      :", POLICY.describe() if AUG_ON else "OFF (flag-identity gate only)")
print("--dataset   :", cfg.dataset_jsonl)
print("swift on PATH:", (Path(_envbin) / "swift").exists())
'''),

(CODE, r'''# ── G-A — the augmentation itself. Zero GPU, zero data. Runs anywhere. ──────
# The full off-pod suite is `_tools/test_appearance.py`; these are the three checks
# that must be true for THIS cfg's policy, printed into the run's own record.
from PIL import Image
import numpy as np

_probe = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8"))

g_off  = A.gate_flag_off_identity(_probe)
g_det  = A.gate_determinism(_probe, POLICY, [f"lapchole__{i}" for i in range(32)])
g_ops  = A.gate_no_forbidden_ops(POLICY)
print("G-A1 flag-OFF returns the same object :", g_off)
print("G-A2 deterministic per (seed, qID)    :", g_det)
print("G-A3 no forbidden operator family     :", g_ops)
assert g_off["PASS"] and g_det["PASS"] and g_ops["PASS"], "G-A FAILED — do not export"

# The realised dose, measured rather than believed. Expected under Afifi's 5-point
# grid + Wang severities {0,1,2}: 1/5 of rows keep their WB, 1/3 take no illumination
# corruption, 1/15 are fully untouched.
dose = A.dose_table(POLICY, [f"lapchole__{i}" for i in range(20000)])
print("\nG-A4 dose:", json.dumps(dose, indent=1))
'''),

(CODE, r'''# ── G-D — ONE variable, measured. rung 06's argv is built by rung 06's own
# `_swift_args` from a stock ViTLoRAConfig and diffed against ours. A hand-typed
# copy of the baseline command would be a claim; this is a measurement, and it is
# what catches a silently-drifted lora_rank / LR / epoch count / freeze_vit.
diff = diff_vs_rung06(cfg)
for flag, (r06, r14) in sorted(diff.items()):
    print(f"  {flag:16s} rung06={r06!s:60s} -> rung14={r14}")

expected = {"--dataset"} if AUG_ON else set()
assert set(diff) == expected, (
    f"G-D FAILED: rung 14 differs from rung 06 in {sorted(diff)}; only {sorted(expected)} may "
    "differ. Anything else means this run answers a different question than the pre-registered one.")
print("\nOK G-D: exactly one variable" if AUG_ON else "\nOK G-D: flag OFF -> argv identical to rung 06")

# G-EVAL — the augmentation must never reach inference. That is the entire thesis.
g_eval = gate_eval_untouched(cfg)
print("G-EVAL eval-side appearance flags OFF :", g_eval)
assert g_eval["PASS"], "G-EVAL FAILED: an inference-side appearance flag is set"
'''),

(CODE, r'''# ── Export. rung 02's OWN exporter runs first and ALWAYS: on the OFF path it is the
# entire export, so "byte-identical to rung 06" is a property of the code path.
# Frames come from the shared /workspace/frames_cache — a frame exists ONCE there.
t0 = time.perf_counter()
ds = main(cfg, stage="export")
print(f"export done in {time.perf_counter()-t0:.0f}s -> {ds}")
print(f"  base train.jsonl : {cfg.train_jsonl} ({sum(1 for _ in open(cfg.train_jsonl))} rows)")
if AUG_ON:
    print(f"  aug  train.jsonl : {cfg.aug_train_jsonl} ({sum(1 for _ in open(cfg.aug_train_jsonl))} rows)")
    print(f"  aug frames       : {cfg.aug_frames_dir}")
'''),

(CODE, r'''# ── 🔴 G-OFF — the load-bearing identity gate, by sha256, not by assertion.
# The base export we just produced must be byte-identical to rung 06's train.jsonl.
# If it is not, the "single variable" claim is false and every number below is
# comparing two different datasets.
import hashlib
from aug_export import sha256

ref = Path(REF_TRAIN_JSONL)
if not ref.exists():
    print(f"⚠️  rung 06's train.jsonl is absent at {ref}.")
    print("    It is regenerable: run rung 06's OWN export (experiments/06-vit-lora, stage='export')")
    print("    into its own run dir and re-run this cell. Do NOT skip this gate.")
elif SMOKE:
    print("SMOKE: the export is a stratified subset, so it cannot equal the full rung-06 file.")
    print(f"       sha256(ours) = {sha256(cfg.train_jsonl)}  (recorded, not compared)")
    print("       🔴 G-OFF is only meaningful on a FULL export — it MUST be run before training.")
else:
    off_cfg = AppearanceAugConfig(data_root=Path(DATA_ROOT), exp_dir=cfg.exp_dir,
                                  run_name=cfg.run_name, aug=None)
    g = gate_flag_off_sha256(off_cfg, ref)
    print(json.dumps(g, indent=1))
    assert g["PASS"], "🔴 G-OFF FAILED: our OFF-path train.jsonl is NOT rung 06's. STOP."
    print("OK G-OFF: the data leg is byte-identical to rung 06")
'''),

(CODE, r'''# ── 🔴 G-SHAPE — the ON JSONL differs from the OFF JSONL in the image path ONLY.
# Catches a rewrite that dropped rows, reordered them or edited the text — any of
# which silently turns an appearance A/B into a data A/B.
if AUG_ON:
    g = gate_aug_jsonl_shape(cfg.train_jsonl, cfg.aug_train_jsonl, cfg.aug_frames_dir)
    print(json.dumps(g, indent=1))
    assert g["PASS"], "🔴 G-SHAPE FAILED: the augmented JSONL is not a path-only rewrite"

    man = pd.read_csv(cfg.aug_manifest_csv)
    print("\nrealised draw over the ACTUAL training rows:")
    print(man.groupby(["wb_K"]).size().to_string())
    print(man.groupby(["illum_op", "illum_severity"]).size().to_string())
    print(f"fully untouched rows: {man['identity'].mean():.4f}  (expect ~1/15)")
'''),

(CODE, r'''# ── The human dose check. NOT a gate — a contact sheet, plus the one number that
# makes "is 2850 K too strong?" answerable instead of arguable.
#
# 🔴 If the dose is judged wrong here it may be changed ONLY NOW, before training,
# and ONLY by editing the pre-registration in context/14-appearance-aug/CONTEXT.md
# with the reason. After the first eval number exists, the grid is frozen — a dose
# edited after seeing a score stops being a parameter and becomes a story.
if AUG_ON and not SMOKE:
    from PIL import Image
    import numpy as np

    man = pd.read_csv(cfg.aug_manifest_csv)
    src = Image.open(man.iloc[0]["src_frame"]).convert("RGB")
    rb = lambda im: float(np.asarray(im, float)[..., 0].mean() / max(1e-6, np.asarray(im, float)[..., 2].mean()))
    print("mean R/B ratio by colour temperature (5500 K = the identity anchor):")
    for t in A.AFIFI_WB_TEMPS_K:
        print(f"  {t:>5} K   R/B = {rb(A.wb_error(src, t)):.3f}")

    strip = [src] + [A.wb_error(src, t) for t in A.AFIFI_WB_TEMPS_K] + \
            [A.brightness(src, 2, +1), A.brightness(src, 2, -1), A.contrast(src, 1)]
    w, h = strip[0].size
    sheet = Image.new("RGB", (w * len(strip) // 3, h * 3))
    for i, im in enumerate(strip):
        sheet.paste(im, ((i % 3) * w, (i // 3) * h))
    out = cfg.run_dir / "dose_eyeball.png"
    sheet.save(out)
    print(f"\ncontact sheet -> {out}  (original, then 2850/3800/5500/6500/7500 K, "
          f"then brightness+2, dark+2, contrast s1)")
'''),

(CODE, r'''# ── Train. rung 06's OWN `_train` runs, with only its argv builder swapped, so the
# run guard, the stdout tee and the train.log that read_g1 parses are identical.
#
# STOP RULES (pre-registered): OOM -> STOP. Do NOT lower max_pixels / batch / LR —
# each is a second variable. Loss diverges -> do NOT touch the LR; report it.
t0 = time.perf_counter()
main(cfg, stage="train")
print(f"\ntrain done in {(time.perf_counter()-t0)/60:.1f} min")
print("checkpoints:", [c.name for c in list_checkpoints(cfg)])
'''),

(CODE, r'''# ── G1 — the same log-read rung 06 used: did the LoRA reach the ViT, and is this
# still LoRA (few M trainable) rather than a full fine-tune?
from vit_lora_train import read_g1

g1 = read_g1(cfg)
print(json.dumps(g1, indent=1))
assert g1["targets_vision_tower"], "G1 FAILED: LoRA did not reach the ViT — not rung 06's recipe"
assert g1["trainable_params_M"] is None or g1["trainable_params_M"] < 100, \
    "G1 FAILED: this is a full fine-tune, not LoRA — a second variable"
'''),

(MD, r"""## Per-epoch merge → eval → canonical scoring

🔴 **Every epoch is merged, evaluated and reported. Never last-by-default.**
`checkpoint-selection-vs-number` measured that `acc_OOD` selection takes the
checkpoint that erased more counting — so the `number` margin is printed for
EVERY epoch, ID and OOD, and the trajectory is the deliverable.

The ONE row that goes to the shared ledger is still selected by **`acc_OOD`**
(`context/RULES.md` §6, and this rung's pre-registration): the selection rule is
rung 06's, unchanged, because changing it would be a second variable. Selecting
by `bucket_mean` instead would silently promote the very checkpoint this rung
exists to flag.

⚠️ **Disk budget.** 3 epochs × ~17 GB merged = ~51 GB if nothing is deleted, on
top of ~14k augmented JPEGs. Each merge is removed in a `finally` right after its
`results.csv` is scored, so the steady state is **one** merged checkpoint (~17 GB).

Scoring is `frame.metrics.stratified_report` and nothing else (RULES §EVAL).
Read **margins over the template-aware floor**, never raw accuracy.
"""),

(CODE, r'''# ── Per-epoch merge + eval + canonical score. Resumable: progress is a file read
# (frame.runstate), not a scrollback.
#
# 🔴 DISK. Each merged 8B bf16 checkpoint is ~17 GB and there are 3 epochs — 51 GB on
# top of ~14k pre-materialised augmented JPEGs. `merge_checkpoint` namespaces the merge
# per epoch (02-lora-sft/_models/lora_sft_train.py:214-223) precisely so that deleting
# it is the CALLER's job, and nothing in the engine deletes on its own. So each merge is
# removed the moment its results.csv is scored and gated — and the removal sits in a
# `finally`, because the failure that fills the volume is an eval that RAISED (CUDA OOM,
# judge crash) and left 17 GB behind. Rung 12 hit exactly this, mid-run.
import shutil

from frame.config import BaselineConfig
from frame.run import run_baseline
from frame import ledger, metrics, runstate

cks = list_checkpoints(cfg)
rs = runstate.RunState(root=cfg.run_dir, pipeline="14-appearance-aug",
                       stages=[f"eval_{c.name}" for c in cks])
gold = ledger.gold_from_frame_parquets(Path(DATA_ROOT))

strats = {}
for ck in cks:
    tag = f"eval_{ck.name}"
    rs.start_stage(tag)
    m = cfg.merged_dir / ck.name
    if not (m.is_dir() and any(m.iterdir())):
        m = merge_checkpoint(cfg, ck)
    try:
        bcfg = BaselineConfig(data_root=Path(DATA_ROOT), model_path=m, out_dir=cfg.run_dir,
                              run_name=tag, max_pixels=cfg.max_pixels, seed=cfg.seed,
                              n_eval=40 if SMOKE else None)
        t0 = time.perf_counter()
        run_baseline(bcfg)
        rs.heartbeat(note=f"{tag} eval {(time.perf_counter()-t0)/60:.1f} min")

        res = pd.read_csv(cfg.run_dir / tag / "results.csv")
        missing = set(res["qID"]) - set(gold.dropna(subset=["answer"])["qID"])
        assert not missing, f"GATE 0 {tag}: {len(missing)} qIDs without gold; margins would inflate"
        metrics.assert_no_dup_qid(res); metrics.assert_ood_from_qid(res)
        metrics.assert_all_rows_grouped(res)
        s = metrics.stratified_report(res, gold=gold)
        metrics.assert_floors_vs_eval_set(s)
        strats[ck.name] = s
        if not SMOKE:
            ledger.register_run(cfg.run_dir / tag, s, experiment="14-appearance-aug",
                                run=f"{cfg.run_name}__{tag}",
                                model=f"Qwen3-VL-8B + LoRA r8 ViT+LLM + {POLICY.tag} ({ck.name})")
    finally:
        # ~17 GB, on the happy path AND on the exception path. `results.csv`,
        # `predictions.json` and the registered `stratified.json` are already on disk,
        # so nothing below this loop needs the weights; a re-merge is one `swift export`.
        shutil.rmtree(m, ignore_errors=True)
        print(f"merged weights deleted: {m}")
    rs.finish_stage()
rs.finish("done")
print("scored:", list(strats))
'''),

(CODE, r'''# ── The epoch table. 🔴 `number` margin is reported at EVERY epoch, ID and OOD.
# rung 06's own trajectory, for reference: number margin OOD +0.015 -> +0.013 -> +0.000.
R06 = {"bucket_mean": 0.566707285167315, "acc_ID": 0.544404973357016,
       "acc_OOD": 0.60775, "floor_ID": 0.3370337477797513, "floor_OOD": 0.45975,
       "margin_ID": 0.20737122557726467, "margin_OOD": 0.148}

rows = []
for name, s in strats.items():
    bf = pd.DataFrame(s["by_format"]); bb = pd.DataFrame(s["by_bucket"])
    r = {"checkpoint": name, "bucket_mean": s["bucket_mean"],
         "d_bucket_mean_vs_06": s["bucket_mean"] - R06["bucket_mean"],
         # acc_ID/acc_OOD are carried because acc_OOD is the SELECTION criterion
         # (context/RULES.md §6) — the table the checkpoint is picked from must show
         # the quantity it is picked by.
         "acc_ID": s["acc_ID"], "acc_OOD": s["acc_OOD"],
         "margin_ID": s["margin_ID"], "margin_OOD": s["margin_OOD"],
         "d_margin_OOD_vs_06": s["margin_OOD"] - R06["margin_OOD"]}
    for fmt in ("number", "fo_class", "binary", "open_ended", "multiple_choice"):
        sub = bf[bf.answer_format == fmt].set_index("distribution")
        for d in ("ID", "OOD"):
            r[f"{fmt}_margin_{d}"] = sub.loc[d, "margin"] if d in sub.index else float("nan")
    for _, b in bb.iterrows():                      # the four bucket_mean cells
        r[f"cell_{b['capability_group']}_{b['distribution']}"] = b["margin"]
    rows.append(r)
epochs = pd.DataFrame(rows)
if not SMOKE:
    epochs.to_csv(EXP / "RESULTS_epochs.csv", index=False)
print(epochs.to_string(index=False))
'''),

(CODE, r'''# ── 🔴 The pre-registered decision input: the PAIRED, VIDEO-CLUSTERED CI on the
# OOD delta vs rung 06. Effective n is ~38 videos, not 6252 questions, so an
# unclustered CI would be roughly 10x too narrow and would manufacture significance.
#
# The floor is a property of the eval set and is IDENTICAL across arms, so
# Δmargin_OOD == Δacc_OOD and this CI is exactly the CI on the pre-registered
# quantity. Reported for ID too, and per format, either way.
ref = pd.read_csv(REF_RESULTS)[["qID", "video", "correctness"]].rename(
    columns={"correctness": "correct_a"})

ci_rows = []
for name in strats:
    ours = pd.read_csv(cfg.run_dir / f"eval_{name}" / "results.csv")[
        ["qID", "correctness", "answer_format"]].rename(columns={"correctness": "correct_b"})
    j = ref.merge(ours, on="qID", how="inner")
    # SMOKE evaluates a subset, so the control legitimately has rows the arm does not.
    # Strict either way: nothing the arm scored may be missing from the join.
    _need = len(ours) if SMOKE else len(ref)
    assert len(j) == _need, f"{name}: arms not scored on the same questions ({len(j)} vs {_need})"
    j["dist"] = j["qID"].astype(str).str.split("__").str[0].map({"heico": "OOD", "lapchole": "ID"})
    for dist in ("ID", "OOD"):
        for fmt in [None, "number", "fo_class"]:
            sub = j[j["dist"] == dist]
            sub = sub if fmt is None else sub[sub["answer_format"] == fmt]
            ci = metrics.paired_delta_ci(sub, n_boot=4000, seed=42)
            ci_rows.append({"checkpoint": name, "distribution": dist,
                            "answer_format": fmt or "ALL", **ci})
ci_df = pd.DataFrame(ci_rows)
ci_df["excludes_zero"] = (ci_df.ci_low > 0) | (ci_df.ci_high < 0)
if not SMOKE:
    ci_df.to_csv(EXP / "RESULTS_paired_ci.csv", index=False)
print(ci_df.to_string(index=False))
print("\nPre-registered WIN rule: d(margin_OOD) >= +0.02 AND its paired video-clustered CI")
print("excludes 0. The rule is read by a human against context/14-appearance-aug/CONTEXT.md.")
'''),

(MD, r"""## Folded-in ZERO-GPU diagnostic — does frame quality predict rung 06's errors?

Not an arm; no training, no eval. It scores the frames already in the cache and
joins them to rung 06's committed per-question correctness.

🔴 **The primary statistic is WITHIN-VIDEO.** Pooled over questions, quality and
difficulty are both properties of the video, so a pooled correlation is
confounded by scene — that is exactly how rung 12d manufactured a winner. The
pooled number is printed too, labelled CONFOUNDED, so the gap between the two is
visible rather than hidden.

⚠️ There is **no inference-side frame-selection lever** in FRAME: the track hands
us one extracted frame (`vendor/orena-focus/src/focus/enums.py:23`). Whatever
this motivates can only act on training.
"""),

(CODE, r'''import quality_diag as Q
from aug_export import qid_index

scores = Q.scan_frames(sorted(cfg.frames_dir.glob("*.jpg")))
print(f"scored {len(scores)} cached frames")
print(scores[list(Q.SCORES)].describe().to_string())

# qID -> cache filename, from the same index the exporter uses.
qid2frame = {q: k[0] for k, qs in qid_index(cfg).items() for q in qs}
ref_res = pd.read_csv(REF_RESULTS)
joined = Q.join_scores(ref_res, qid2frame, scores)
print(f"\njoined {joined['blur_lapvar'].notna().sum()}/{len(joined)} questions to a scored frame")

diag = Q.report(joined, n_boot=4000, seed=42)
if not SMOKE:
    diag.to_csv(EXP / "RESULTS_quality_diag.csv", index=False)
    scores.to_csv(cfg.run_dir / "frame_scores.csv", index=False)
print("\nPRIMARY = within-video delta (correct - wrong). pooled_r is CONFOUNDED BY SCENE.")
print(diag.to_string(index=False))
'''),

(CODE, r'''# ── RESULTS.csv — ledger-shaped, ONE row (WAVE_SPEC #6: frame.ledger reads an
# `arm` column as a run name, so per-arm detail goes to its own file).
if not SMOKE:
    # Checkpoint selection = rung 06's rule, unchanged: max acc_OOD (context/RULES.md §6,
    # and this rung's own pre-registration). Changing the criterion would be a SECOND
    # variable. Every epoch is reported above regardless, so a reader can see what any
    # other rule would have picked.
    #
    # 🔴 It was `max(bucket_mean)` here, and that is not a cosmetic difference: `number`
    # margin is MEASURED to regress across epochs while other buckets still rise (rung 02
    # +0.032 → +0.004, rung 06 +0.015 → +0.000). Selecting by the headline can promote to
    # the shared ledger exactly the checkpoint this rung exists to flag as bad, and no
    # existing gate would catch it.
    best = epochs.sort_values("acc_OOD", ascending=False).iloc[0]["checkpoint"]
    s = strats[best]
    bf = pd.DataFrame(s["by_format"])
    fmt_acc = bf.groupby("answer_format").apply(
        lambda g: (g["accuracy"] * g["n"]).sum() / g["n"].sum()).to_dict()
    row = {
        "run": cfg.run_name,
        "model": f"Qwen3-VL-8B + LoRA r8 ViT+LLM, train-time {POLICY.tag} ({best})",
        "bucket_mean": s["bucket_mean"], "acc_ID": s["acc_ID"], "acc_OOD": s["acc_OOD"],
        "floor_ID": s["floor_ID"], "floor_OOD": s["floor_OOD"],
        "margin_ID": s["margin_ID"], "margin_OOD": s["margin_OOD"],
        **{f"acc_{f}": fmt_acc.get(f, float("nan")) for f in
           ("fo_class", "number", "binary", "open_ended", "multiple_choice")},
        "n_questions": len(ref_res),
        "date": time.strftime("%Y-%m-%d"),
        "notes": ("single variable vs 06-vit-lora: seeded train-time appearance augmentation "
                  f"({POLICY.describe()}); eval set untouched. Per-epoch table in "
                  "RESULTS_epochs.csv, paired video-clustered CIs in RESULTS_paired_ci.csv, "
                  "zero-GPU quality diagnostic in RESULTS_quality_diag.csv."),
    }
    out = EXP / "RESULTS.csv"
    df = pd.read_csv(out)
    df = pd.concat([df[df["run"] != cfg.run_name], pd.DataFrame([row])], ignore_index=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
'''),
]


def build() -> Path:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {},
             **({"source": src.splitlines(keepends=True)} if t == MD else
                {"execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)})}
            for t, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parents[1] / "14_appearance_aug.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
