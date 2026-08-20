"""Rung 19b — does EXTERNAL recognition data in the corpus buy anything rung 47 did not?

Rung 47 trained A2's own corpus (14,415 rows) for 5 epochs. This trains the SAME recipe, the
SAME schedule and the SAME seed on A2's corpus PLUS 5,718 Strasbourg rows built from the 35
CholecT50 videos the split froze for training (`experiments/splits/cholect50_split_v1.csv`).

⇒ Against rung 47 the ONLY difference is `--dataset`. Both anneal a cosine over 5 epochs, so
ep4 reads against ep4 with the corpus as the single variable — the comparison rung 42 could
never make, because it moved the corpus AND ran an epoch its control never ran.

🔴 The 15 held-out CholecT50 videos are NOT in this corpus, and gate G3 proves it per row.
The moment they were, the rung-48 probe would stop measuring CENTRE and start measuring SCENE.

⚠️ WALL CLOCK. 20,133 rows / effective batch 16 = 1,259 steps per epoch = 6,295 for five.
At rung 47's measured 17.11 s/it that is ~29.9 h, and ep4 — the epoch we read — lands at
~23.9 h. `--save_strategy epoch` means every epoch stands alone, so a kill after ep4 costs
only ep5.
"""
import csv, hashlib, json, os, subprocess, sys, time
from collections import Counter
from dataclasses import replace
from pathlib import Path

# 🔴 /data is a DANGLING symlink since the 18-Aug reboot (no fstab entry). Real paths only.
STORE = Path("/mnt/storage/uaq_user")
REPO = STORE / "repo_leo"
RUNG = STORE / "rung19b"
MODEL = STORE / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

CORPUS = STORE / "rung48/corpus/train_merged.jsonl"
CORPUS_SHA = "5f26927b3f65555a1fe089f92818fe9428296faa6ee412fc49767af1cf7fc5ae"
R47_CORPUS = STORE / "rung47/corpus/train_mntpaths.jsonl"
R47_SHA = "374c2a236c08e6d5b7b17c978f6c1a2cef3cedd33667691751588257d97dc9e2"
SPLIT = REPO / "experiments/splits/cholect50_split_v1.csv"

SMOKE = os.environ.get("RUNG19B_SMOKE", "1") == "1"

os.environ.setdefault("HF_HOME", str(STORE / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 🔴 The `swift` console script carries `#!/data/uaq_user/envs/...` from pip install time, and
# /data has not existed since the 18-Aug reboot — exec fails with a bare ENOENT that reads as
# "swift is not installed". A shim with a LIVE shebang, ahead of it on PATH, for this run only:
# patching the env or `vit_lora_train.py` would change a shared artifact every rung compares to.
SHIM = RUNG / "code" / "bin"
SHIM.mkdir(parents=True, exist_ok=True)
_sw = SHIM / "swift"
_sw.write_text(f"#!{sys.executable}\nimport sys\nfrom swift.cli.main import cli_main\n"
               "sys.exit(cli_main())\n")
_sw.chmod(0o755)
os.environ["PATH"] = (f"{SHIM}{os.pathsep}{Path(sys.executable).parent}"
                      + os.pathsep + os.environ.get("PATH", ""))

for p in (REPO / "src", REPO / "vendor/orena-focus/src",
          REPO / "experiments/21-recipe-sweep/_models",
          REPO / "experiments/18-count-aug/_models",
          REPO / "experiments/16-count-probes/_models",
          REPO / "experiments/06-vit-lora/_models",
          REPO / "experiments/02-lora-sft/_models"):
    if p.is_dir():
        sys.path.insert(0, str(p))

import recipe_sweep_train as rst

cfg = rst.RecipeSweepConfig(
    exp_dir=RUNG,
    run_name="19b_merged_ep5_smoke" if SMOKE else "19b_merged_ep5_v1",
    model_path=MODEL,
    data_root=STORE / "orena-data",
    manifest_path=REPO / "experiments/splits/frame_ood_v1.csv",
    control_train_jsonl=CORPUS,
    control_sha256=CORPUS_SHA,
    # ── rung 47's recipe, restated so a drift is visible rather than inherited ──────────
    learning_rate=2e-4, lora_rank=8, lora_alpha=32, max_grad_norm=1.0, vit_lr=None,
    num_train_epochs=5,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    seed=42, smoke=SMOKE, smoke_max_steps=4,
)
print(f"run_dir {cfg.run_dir}\nsmoke   {SMOKE}", flush=True)

rows = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]

# ── G1 — the corpus is the bytes we declared ────────────────────────────────────────────
ds = rst.assert_dataset_is_the_controls(cfg)
assert ds["n_rows"] == 20133, f"expected 20,133 rows, got {ds['n_rows']}"
print(f"OK G1 dataset: {ds['n_rows']} rows, sha256 {ds['sha256'][:12]}..., "
      f"{ds['steps_per_epoch']:.1f} steps/epoch", flush=True)

# ── G2 — every image on disk. A missing frame dies 20 h in, not at step 0 ───────────────
imgs = [i for r in rows for i in r.get("images", [])]
bad = [i for i in imgs if not i.startswith("/mnt/") or not Path(i).exists()]
assert not bad, f"{len(bad)} unusable image paths, first: {bad[:3]}"
print(f"OK G2 images: {len(imgs)} paths, all absolute under /mnt, all present", flush=True)

# ── G3 — LEAK GATE. Not one frame from the 15 held-out CholecT50 videos ─────────────────
with SPLIT.open(newline="", encoding="utf-8") as fh:
    split_rows = list(csv.DictReader(fh))
hold = {r["video_id"] for r in split_rows if r["split"] == "hold"}
train_v = {r["video_id"] for r in split_rows if r["split"] == "train"}
assert len(hold) == 15 and len(train_v) == 35, f"split is {len(train_v)}/{len(hold)}, expected 35/15"
seen = Counter(Path(i).name.split("__")[1] for i in imgs if Path(i).name.startswith("cholect50__"))
leaked = sorted(set(seen) & hold)
assert not leaked, f"🔴 HOLD LEAK — the rung-48 probe would stop measuring centre: {leaked}"
assert set(seen) <= train_v, f"unknown CholecT50 videos in corpus: {sorted(set(seen) - train_v)}"
print(f"OK G3 leak: {sum(seen.values())} CholecT50 rows over {len(seen)} videos, "
      f"0 of the {len(hold)} held out", flush=True)

# ── G4 — provenance. The challenge half IS rung 47's corpus, row for row ────────────────
assert hashlib.sha256(R47_CORPUS.read_bytes()).hexdigest() == R47_SHA, "rung 47's corpus moved"
r47 = {json.dumps(json.loads(l), sort_keys=True) for l in R47_CORPUS.open(encoding="utf-8")}
mine = {json.dumps(r, sort_keys=True) for r in rows}
missing = len(r47 - mine)
assert missing == 0, f"🔴 {missing} of rung 47's 14,415 rows are NOT in this corpus"
print(f"OK G4 provenance: all {len(r47)} rung-47 rows present; "
      f"{len(mine - r47)} rows are new (Strasbourg)", flush=True)

# ── G5 — SINGLE VARIABLE vs RUNG 47, which is the control. Not vs A2 ────────────────────
# 🔴 `rst.diff_vs_control` cannot do this: `control_cfg` shares `--dataset` by construction,
# so it would report an empty diff and certify an arm whose ONLY variable it cannot see.
r47_cfg = replace(cfg, control_train_jsonl=R47_CORPUS, control_sha256=R47_SHA, smoke=False)
a = rst._as_map(rst.swift_args_21(r47_cfg))
b = rst._as_map(rst.swift_args_21(replace(cfg, smoke=False)))
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
for k, (x, y) in sorted(diff.items()):
    print(f"  {k:28s} r47={x}  ->  this={y}")
assert set(diff) == {"--dataset"}, f"NOT single-variable vs rung 47: {sorted(diff)}"
print("OK G5 single-variable: only --dataset moves", flush=True)

cfg.run_dir.mkdir(parents=True, exist_ok=True)
(cfg.run_dir / "argv.json").write_text(json.dumps(rst.swift_args_21(replace(cfg, smoke=False)), indent=1))
(cfg.run_dir / "diff_vs_r47.json").write_text(json.dumps({k: list(v) for k, v in diff.items()}, indent=1))
(cfg.run_dir / "gates.json").write_text(json.dumps(
    {"corpus_sha256": ds["sha256"], "n_rows": ds["n_rows"], "n_images": len(imgs),
     "cholect50_rows": sum(seen.values()), "cholect50_videos": len(seen),
     "hold_leaked": [], "r47_rows_present": len(r47), "new_rows": len(mine - r47),
     "steps_per_epoch": ds["steps_per_epoch"]}, indent=1))
(cfg.run_dir / "pip_freeze.txt").write_text(
    subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

t0 = time.perf_counter()
ckpt = rst._train(cfg)
print(f"\ntrained in {(time.perf_counter() - t0) / 3600:.2f} h -> {ckpt}", flush=True)
