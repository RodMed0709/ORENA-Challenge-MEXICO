"""Rung 58-C — turn the fo_class-only delta into the number the team actually tracks.

58-A/58-B measured the arms on `fo_class` alone, which is correct (no other format's
answer can change) but is NOT `bucket_mean`. This runs the CANONICAL eval on rung 42's
8 held-out videos for the shipped checkpoint, then recomputes the headline with the
arm's fo_class answers substituted in. The judge never sees a changed answer, so its
verdicts on the other formats are reused rather than re-run.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

S = Path(os.environ.get("STORAGE", "/mnt/storage/uaq_user"))
W = S / "rung58"; REPO = S / "repo_rod"
sys.path.insert(0, str(REPO / "experiments" / "45-gen36-data-and-reg" / "_tools"))
from eval_arm45 import EvalConfig, held_out_split, assert_cache_covers, score, arm_results_csv, ensure_paths

os.environ.setdefault("HF_HOME", str(S / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

cfg = EvalConfig(
    arm="r42_ep4_heldout",
    merged_dir=str(W / "merged_r42"),
    out_dir=str(W / "headline"),
    run_name="58C_r42_ep4",
    data_root=str(S / "orena-data"),
    repo_root=str(REPO),
    split_json=str(REPO / "experiments" / "42-merged-corpus" / "RESULTS_split_42.json"),
    frames_cache=str(S / "frames_cache"),
    use_vllm=False,          # the 8B loads fine on the HF path; vLLM is the 27B's crutch
)
# 🔴 `score()` swaps in GenericVLMEngine whenever use_vllm is False -- that class exists
# for gen-3.6, which cannot load under the 4.57 pin. For the 8B the right engine is the
# default QwenFrameEngine, the one every 8B rung was scored through. Patch the flag off
# by running run_baseline directly rather than letting score() reroute the engine.
ensure_paths(cfg.repo_root)
from eval_arm45 import build_baseline_config, assert_eval_set_is_legal, assert_inference_path_unmoved
from frame.run import run_baseline
from frame.metrics import leaderboard_proxy

t0 = time.time()
split = held_out_split(cfg)
print(f"[58C] held-out: {split['n_held']} items, {len(split['held_videos'])} videos", flush=True)
assert_eval_set_is_legal(cfg)
cfg_eval = build_baseline_config(cfg)
assert_inference_path_unmoved(cfg_eval)
report = run_baseline(cfg_eval, video_filter=split["held_videos"])
report.update(leaderboard_proxy(report)); report["proxy_leaderboard"] = report.pop("proxy")
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
(W / "RESULTS_58C_headline.json").write_text(json.dumps(
    {k: v for k, v in report.items() if not isinstance(v, (list, dict))} |
    {"bucket_mean": report.get("bucket_mean"), "proxy_leaderboard": report.get("proxy_leaderboard"),
     "minutes": round((time.time()-t0)/60, 1), "results_csv": str(arm_results_csv(cfg))}, indent=1, default=str))
print("[58C] bucket_mean:", report.get("bucket_mean"), "proxy:", report.get("proxy_leaderboard"), flush=True)
print("[58C] per-question table:", arm_results_csv(cfg), flush=True)
