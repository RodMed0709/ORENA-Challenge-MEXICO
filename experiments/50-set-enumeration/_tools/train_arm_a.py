"""Rung 50 arm A — rung 47's recipe on the count-prefix corpus. ONE variable: `--dataset`.

Rung 47 trained A2's 14,415 rows for 5 epochs. This trains the SAME recipe, the SAME schedule
and the SAME seed on the SAME 14,415 rows with the `fo_class` answers rewritten to state the set
size first. Row count, order, images and every other answer are byte-identical — proven per row
by `count_prefix.round_trip`, not asserted.

⇒ Against rung 47 the only difference is `--dataset`. Both anneal a cosine over five epochs, so
ep4 reads against ep4.

⚠️ WALL CLOCK. 14,415 rows / effective batch 16 = 901 steps per epoch = 4,505 for five. Rung 47
measured 17.11 s/it on this box ⇒ ~21.4 h, with ep4 (step 3,604) at ~17.1 h. `--save_strategy
epoch` means a kill after ep4 costs only ep5.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

STORE = Path("/mnt/storage/uaq_user")
REPO = STORE / "repo_rod"
RUNG = STORE / "rung50"
MODEL = STORE / "hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

CORPUS = RUNG / "corpus/train_countprefix.jsonl"
R47_CORPUS = STORE / "rung47/corpus/train_mntpaths.jsonl"
R47_SHA = "374c2a236c08e6d5b7b17c978f6c1a2cef3cedd33667691751588257d97dc9e2"

SMOKE = os.environ.get("RUNG50_SMOKE", "1") == "1"

os.environ.setdefault("HF_HOME", str(STORE / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# The `swift` console script carries a dead `#!/data/...` shebang since the 18-Aug reboot; a shim
# with a live one goes ahead of it on PATH, for this run only (rung 19b's fix, reused verbatim).
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

import recipe_sweep_train as rst  # noqa: E402

CORPUS_SHA = hashlib.sha256(CORPUS.read_bytes()).hexdigest()

cfg = rst.RecipeSweepConfig(
    exp_dir=RUNG,
    run_name="50a_countprefix_smoke" if SMOKE else "50a_countprefix_v1",
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
assert ds["n_rows"] == 14415, f"expected 14,415 rows, got {ds['n_rows']}"
print(f"OK G1 dataset: {ds['n_rows']} rows, sha256 {ds['sha256'][:12]}..., "
      f"{ds['steps_per_epoch']:.1f} steps/epoch", flush=True)

# ── G2 — every image on disk. A missing frame dies 17 h in, not at step 0 ───────────────
imgs = [i for r in rows for i in r.get("images", [])]
bad = [i for i in imgs if not i.startswith("/mnt/") or not Path(i).exists()]
assert not bad, f"{len(bad)} unusable image paths, first: {bad[:3]}"
print(f"OK G2 images: {len(imgs)} paths, all absolute under /mnt, all present", flush=True)

# ── G3 — the control is the corpus we think it is, and ONLY answers differ ──────────────
assert hashlib.sha256(R47_CORPUS.read_bytes()).hexdigest() == R47_SHA, "rung 47's corpus moved"
ctrl = [json.loads(l) for l in R47_CORPUS.open(encoding="utf-8")]
assert len(ctrl) == len(rows), f"{len(ctrl)} control rows vs {len(rows)}"
n_diff = 0
for a, b in zip(ctrl, rows):
    assert a.get("images") == b.get("images"), "an image list moved"
    assert a["messages"][:-1] == b["messages"][:-1], "a prompt moved"
    n_diff += a["messages"][-1]["content"] != b["messages"][-1]["content"]
assert n_diff == 6294, f"expected 6,294 rewritten answers, found {n_diff}"
print(f"OK G3 provenance: control sha matches, {n_diff} answers rewritten, nothing else", flush=True)

# ── G4 — SINGLE VARIABLE vs RUNG 47. `rst.diff_vs_control` cannot see --dataset ─────────
r47_cfg = replace(cfg, control_train_jsonl=R47_CORPUS, control_sha256=R47_SHA, smoke=False)
a = rst._as_map(rst.swift_args_21(r47_cfg))
b = rst._as_map(rst.swift_args_21(replace(cfg, smoke=False)))
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
for k, (x, y) in sorted(diff.items()):
    print(f"  {k:28s} r47={x}  ->  this={y}")
assert set(diff) == {"--dataset"}, f"NOT single-variable vs rung 47: {sorted(diff)}"
print("OK G4 single-variable: only --dataset moves", flush=True)

cfg.run_dir.mkdir(parents=True, exist_ok=True)
(cfg.run_dir / "argv.json").write_text(json.dumps(rst.swift_args_21(replace(cfg, smoke=False)), indent=1))
(cfg.run_dir / "diff_vs_r47.json").write_text(json.dumps({k: list(v) for k, v in diff.items()}, indent=1))
(cfg.run_dir / "gates.json").write_text(json.dumps(
    {"corpus_sha256": ds["sha256"], "n_rows": ds["n_rows"], "n_images": len(imgs),
     "n_answers_rewritten": n_diff, "control_sha256": R47_SHA,
     "steps_per_epoch": ds["steps_per_epoch"]}, indent=1))
(cfg.run_dir / "pip_freeze.txt").write_text(
    subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

t0 = time.perf_counter()
ckpt = rst._train(cfg)
print(f"\ntrained in {(time.perf_counter() - t0) / 3600:.2f} h -> {ckpt}", flush=True)
