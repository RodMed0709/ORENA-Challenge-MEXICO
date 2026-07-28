"""ROBUST-MIS kill-check: is its instance-count distribution shaped like our gold?

Downloads ONLY `instrument_instances.png` (never raw.png, never the 10s clips) from the
TRAINING release, which contains Proctocolectomy + Rectal resection — our own train
videos. syn21891314 (Testing / Sigmoid = our val_ood) is never touched.

Count is `unique(mask) - 1` — instance IDs are in the annotation, nothing is inferred.
That is the property that killed SAR-RARP50 and promoted this one.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import synapseclient
from PIL import Image

REPO = Path(r"C:\Users\medro\Documents\_PERSONAL_PROJECTS\ORENA_CHALLENGE")
OUT = Path(__file__).parent / "robustmis_masks"
OUT.mkdir(exist_ok=True)
N_VIDEOS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

env = dict(
    l.split("=", 1)
    for l in (REPO / ".secrets.env").read_text(encoding="utf-8", errors="ignore").splitlines()
    if "=" in l and not l.strip().startswith("#")
)
import requests
SESS = requests.Session()
SESS.headers.update({"User-Agent": "curl/8.4.0"})
AUTH = {"Authorization": "Bearer " + env["SYNAPSE_AUTH_TOKEN"].strip()}
syn = synapseclient.Synapse(silent=True)
syn.login(authToken=env["SYNAPSE_AUTH_TOKEN"].strip())

TRAINING = "syn21870038"
groups = list(syn.getChildren(TRAINING))
assert all(g["name"] in ("Proctocolectomy", "Rectal resection") for g in groups), \
    f"unexpected folders in Training: {[g['name'] for g in groups]} — Sigmoid must never appear"
print("Training folders:", [g["name"] for g in groups], flush=True)

hist = Counter()
per_video = {}
n_masks = 0

for g in groups:
    for vid in list(syn.getChildren(g["id"]))[:N_VIDEOS]:
        key = f"{g['name']}/{vid['name']}"
        vh = Counter()
        for frame in syn.getChildren(vid["id"]):
            for f in syn.getChildren(frame["id"]):
                if f["name"] != "instrument_instances.png":
                    continue
                dest = OUT / f"{g['name'][:5]}_{vid['name']}_{frame['name']}.png"
                if not dest.exists():
                    _u = SESS.get(
                        f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{f['id']}/file?redirect=false",
                        headers=AUTH, timeout=60).text.strip().strip('"')
                    dest.write_bytes(SESS.get(_u, timeout=180).content)
                m = np.array(Image.open(dest))
                c = int(len(np.unique(m)) - 1)  # 0 = background
                hist[c] += 1
                vh[c] += 1
                n_masks += 1
        per_video[key] = vh
        print(f"  {key}: {sum(vh.values())} masks  {dict(sorted(vh.items()))}", flush=True)

tot = sum(hist.values())
print(f"\n=== ROBUST-MIS instances per frame ({tot} masks, {len(per_video)} videos) ===")
for k in sorted(hist):
    print(f"  {k}: {hist[k]:5d}  {100*hist[k]/tot:5.1f}%  " + "#" * int(50 * hist[k] / tot))
print(f"\nzero-object frames: {hist.get(0,0)} ({100*hist.get(0,0)/tot:.1f}%)")
print(f"frames with >=2   : {sum(v for k,v in hist.items() if k>=2)} "
      f"({100*sum(v for k,v in hist.items() if k>=2)/tot:.1f}%)")
print(f"max instances     : {max(hist) if hist else 0}")
print(f"\nmasks on disk: {sum(f.stat().st_size for f in OUT.glob('*.png'))/1e6:.1f} MB")
