"""SAR-RARP50 kill-check: do multiple FOREIGN OBJECTS co-occur in one frame?

Masks are semantic (pixel value = class), so an instance count needs connected
components. Classes 4/5/8/9 are foreign objects under the ORENA definition;
1/2/3/6/7 are instruments and are excluded from the taxonomy.

If clamps/needle are almost always 0 or 1 per frame, this dataset yields PRESENCE,
not COUNTING, and its rank collapses. That is the whole question.
"""
import zipfile
from collections import Counter

import numpy as np
from PIL import Image
from scipy import ndimage

FO = {4: "suturing needle", 5: "thread", 8: "clamps", 9: "catheter"}
INSTR = {1: "tool clasper", 2: "tool wrist", 3: "tool shaft", 6: "suction tool", 7: "needle holder"}
MIN_PIX = 50  # ignore specks; a real object is not 3 pixels

z = zipfile.ZipFile("video_34.zip")
masks = sorted(n for n in z.namelist() if n.startswith("segmentation/") and n.endswith(".png"))
print(f"masks: {len(masks)}\n")

counts = {c: Counter() for c in list(FO) + list(INSTR)}
fo_total_per_frame = Counter()
present_px = Counter()

for name in masks:
    with z.open(name) as fh:
        m = np.array(Image.open(fh))
    fo_total = 0
    for c in list(FO) + list(INSTR):
        binary = m == c
        px = int(binary.sum())
        if px:
            present_px[c] += px
        if px == 0:
            n = 0
        else:
            lab, n_raw = ndimage.label(binary)
            sizes = ndimage.sum(binary, lab, range(1, n_raw + 1))
            n = int((sizes >= MIN_PIX).sum())
        counts[c][n] += 1
        if c in FO:
            fo_total += n
    fo_total_per_frame[fo_total] += 1

print("=== per-class instance-count histogram (frames at each count) ===")
print(f"{'class':22s} {'kind':6s}  " + "  ".join(f"n={i}" for i in range(6)) + "   max  >1 frames")
for c, label in list(FO.items()) + list(INSTR.items()):
    h = counts[c]
    kind = "FO" if c in FO else "instr"
    row = "  ".join(f"{h.get(i, 0):3d}" for i in range(6))
    mx = max(h) if h else 0
    gt1 = sum(v for k, v in h.items() if k > 1)
    print(f"{label:22s} {kind:6s}  {row}   {mx:3d}   {gt1:3d}")

print("\n=== TOTAL foreign objects per frame (classes 4+5+8+9) ===")
tot = sum(fo_total_per_frame.values())
for k in sorted(fo_total_per_frame):
    v = fo_total_per_frame[k]
    print(f"  {k} FO(s): {v:3d} frames  {100*v/tot:5.1f}%  {'#'*int(40*v/tot)}")
multi = sum(v for k, v in fo_total_per_frame.items() if k >= 2)
print(f"\nframes with >=2 foreign objects: {multi}/{tot} = {100*multi/tot:.1f}%")
print(f"frames with 0 foreign objects  : {fo_total_per_frame.get(0,0)}/{tot} = "
      f"{100*fo_total_per_frame.get(0,0)/tot:.1f}%")
