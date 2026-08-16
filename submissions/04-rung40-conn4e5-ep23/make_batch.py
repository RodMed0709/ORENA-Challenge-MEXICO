"""Turn a slice of our labelled eval set into a platform-shaped batch.

Why this and not a smoke test: a run of the sample batch proves the container starts. It
cannot tell you whether the submission REPRODUCES the score our pipeline measured, which is
the only thing worth knowing before spending one of ten submissions. Feeding labelled
questions through the submission's own harness and scoring the result against the same gold
does both at once.

⚠️ One fidelity gap, stated rather than hidden: the platform mounts lossless PNGs at the
source video's native resolution. Our frames_cache holds JPEGs (q95) already resampled by
our extraction. Converting those to PNG does not restore what JPEG discarded, so this
measures our pipeline-vs-harness agreement, not the platform's exact pixels.
"""
import json, sys, shutil
from pathlib import Path
import pandas as pd
from PIL import Image

N = None if (len(sys.argv)>1 and sys.argv[1]=="all") else (int(sys.argv[1]) if len(sys.argv)>1 else 500)
OUT = Path("/workspace/submission/test/input/eval_full")
CACHE = Path("/workspace/frames_cache")
INSPECT = Path("/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/eval_best/inspect.csv")

d = pd.read_csv(INSPECT)
d["key"] = d.frame.str.split("/").str[-1].str.replace(".jpg", "", regex=False)
# stratify by answer_format so every format is represented in proportion
# N=None -> the WHOLE evaluation set. A subset cannot answer "what does this score";
# it only widens the interval around a number we already have.
slice_ = d if N is None else (d.groupby("answer_format", group_keys=False)
            .apply(lambda g: g.sample(max(1, round(N * len(g) / len(d))), random_state=42)))
print(f"slice: {len(slice_)} questions | formats {slice_.answer_format.value_counts().to_dict()}")

shutil.rmtree(OUT, ignore_errors=True)
(OUT / "frames").mkdir(parents=True)
reqs = []
for r in slice_.itertuples():
    Image.open(CACHE / f"{r.key}.jpg").convert("RGB").save(OUT / "frames" / f"{r.qID}.png")
    reqs.append({"qID": r.qID, "videoID": r.video, "start_time": float(r.timestamp_s),
                 "end_time": float(r.timestamp_s),
                 "procedure_type": "laparoscopic cholecystectomy", "question": r.question})
(OUT / "request.json").write_text(json.dumps(reqs, indent=1))
(OUT / "batch.json").write_text(json.dumps(
    {"qIDs": [r["qID"] for r in reqs], "batch_size": len(reqs),
     "layout": {"frames": "frames/<qID>.png"}}, indent=1))
from focus.foreign_objects import FO_DEFINITIONS_FILE
(OUT / "FO_definitions.json").write_text(json.dumps(FO_DEFINITIONS_FILE.read_text()))
# the gold + our pipeline's verdict on exactly these questions, for the comparison
slice_[["qID", "ground_truth", "our_answer", "correct", "answer_format"]].to_csv(
    OUT.parent / "eval_slice_gold.csv", index=False)
print(f"wrote {len(reqs)} requests + frames -> {OUT}")
print(f"our pipeline's accuracy on this exact slice: {slice_.correct.mean():.4f}")
