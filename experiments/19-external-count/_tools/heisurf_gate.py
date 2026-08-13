"""HeiSurF kill-check: is there enough clip supervision, and is it OUR label?

Folder-private glue. Importable; a notebook cell calls ``stage0()`` / ``stage1()``.

HeiSurF = HeiChole Surgical Workflow Analysis and Full Scene Segmentation, EndoVis
2021, Synapse ``syn25101790``. Laparoscopic cholecystectomy, pixel-level, 21 classes
including **clips**, **specimen bags** and **drains** — three of our own foreign-object
labels rather than a transfer proxy. Public since 2021 ⇒ clears the 2026-07-15 gate.
CC BY-NC-SA, permitted by [[external-data-policy]].

🔴 **This is NOT the rung.** It is the kill-check that decides whether a clip-supervision
arm exists at all, pre-registered before any training is proposed, so that a faithful
negative costs metadata calls instead of a GPU week.

## The four pre-registered measures, and their kill lines

===  =============================================  ==================================
 #   measure                                        dies if
===  =============================================  ==================================
 1   frames whose mask contains >=1 clip            ``n_clip_frames < 300``
 2   clip instances per frame                       (shape only, no line)
 3   median clip area as a fraction of the frame    (shape only, no line)
 4   clip masks covering clips still INSIDE the     ``> 20 %``
     applier jaws, which ORENA explicitly excludes
===  =============================================  ==================================

Measure 4 is the one that matters and the one no script can settle: it is target-side
noise, priced at ~21 % damage by [[target-noise-is-the-harmful-kind]], and telling an
applied clip from one still in the jaws is an eye call. ``stage1`` therefore *exports* a
sample for a human pass rather than guessing — the same move that killed rung 29's C1
interpretation after it had already passed its numeric threshold.

## Two stages, because access is split

``stage0`` needs **metadata only** and runs today. ``stage1`` needs DOWNLOAD access,
which this account does not have — see the module note below.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent
OUT_JSON = EXP / "RESULTS_heisurf_gate.json"
MASK_DIR = Path(__file__).parent / "heisurf_masks"

PROJECT = "syn25101790"
SEGMENTATION = "syn25824653"
CLIPPING = "syn26041604"

#: Kill lines, fixed before any mask was opened.
MIN_CLIP_FRAMES = 300
MAX_JAW_VIOLATION = 0.20
#: Our frames_cache resolution; HeiSurF is 720x576, which matches exactly.
FRAME_WH = (720, 576)


def _session():
    """Synapse client + a plain requests session.

    The REST round-trip is not redundant: ``syn.get(downloadFile=True)`` returns an
    entity whose ``.path`` is ``None`` when the caller lacks DOWNLOAD, so the failure
    surfaces as an ``AttributeError`` on a missing file rather than as the permission
    error it actually is. Asking for the pre-signed URL returns the real reason.
    """
    import requests
    import synapseclient

    env = dict(
        l.split("=", 1)
        for l in (REPO / ".secrets.env").read_text(encoding="utf-8", errors="ignore").splitlines()
        if "=" in l and not l.strip().startswith("#")
    )
    tok = env["SYNAPSE_AUTH_TOKEN"].strip()
    syn = synapseclient.Synapse(silent=True)
    syn.login(authToken=tok)
    s = requests.Session()
    s.headers.update({"User-Agent": "curl/8.4.0", "Authorization": "Bearer " + tok})
    return syn, s


def can_download(s, entity: str = PROJECT) -> bool:
    r = s.get(
        f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{entity}/permissions",
        timeout=30,
    )
    return bool(r.status_code == 200 and r.json().get("canDownload"))


def stage0() -> dict:
    """Volume and structure, from metadata alone. Bounds measure 1 without a download."""
    syn, s = _session()
    seg = list(syn.getChildren(SEGMENTATION))
    main = [v for v in seg if v["name"] != "Clipping"]
    clip_root = [v for v in seg if v["name"] == "Clipping"][0]

    per_video = {v["name"]: len(list(syn.getChildren(v["id"]))) for v in main}
    per_clip = {c["name"]: len(list(syn.getChildren(c["id"])))
                for c in syn.getChildren(clip_root["id"])}
    n_main, n_clip = sum(per_video.values()), sum(per_clip.values())

    result = {
        "stage": 0,
        "needs_download": False,
        "project": PROJECT,
        "annotated_frames_main": n_main,
        "annotated_frames_clipping_set": n_clip,
        "annotated_frames_total": n_main + n_clip,
        "videos_main": len(per_video),
        "clipping_subfolders": len(per_clip),
        "frames_per_video_main": sorted(per_video.values()),
        "frames_per_clipping_subfolder": sorted(per_clip.values()),
        "min_clip_frames_kill_line": MIN_CLIP_FRAMES,
        "can_download": can_download(s),
        # 🔑 The bound that makes stage 1 optional for the volume question: measure 1
        # cannot exceed the total number of annotated frames, whatever the masks hold.
        "ceiling_on_measure_1": n_main + n_clip,
        "fraction_of_all_frames_that_must_show_a_clip_to_pass":
            MIN_CLIP_FRAMES / (n_main + n_clip),
        "dedicated_clipping_set_alone_clears_line": n_clip >= MIN_CLIP_FRAMES,
    }
    result["reading"] = (
        f"The whole segmentation release carries {result['annotated_frames_total']} annotated "
        f"frames. To clear a {MIN_CLIP_FRAMES}-frame line, "
        f"{result['fraction_of_all_frames_that_must_show_a_clip_to_pass']:.0%} of every "
        "annotated frame in the dataset would have to contain a clip. Clipping is one phase of "
        f"a cholecystectomy, and the release's own dedicated Clipping set holds {n_clip} frames "
        "-- below the line on its own. The gate is at genuine risk of dying on volume, and that "
        "is known before a single mask is opened."
    )
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def stage1(limit_per_folder: int | None = None, export_n: int = 40) -> dict:
    """Measures 1-4. Requires DOWNLOAD access; raises with the reason if absent.

    Writes masks to ``_tools/heisurf_masks/`` (folder-private, gitignored) and exports
    ``export_n`` clip crops for the human pass that measure 4 needs.
    """
    import numpy as np
    from PIL import Image

    syn, s = _session()
    if not can_download(s):
        raise PermissionError(
            "canDownload=False on " + PROJECT + ". This is NOT a data-use agreement: the "
            "entity carries ZERO access requirements, and the SAME token downloads "
            "ROBUST-MIS (syn21870038) fine. It is the project ACL -- HeiSurF grants "
            "download on challenge registration. Fix it at https://www.synapse.org/heisurf "
            "with the RodMed0709 account, then re-run. Nothing here can be measured until "
            "then, and no workaround is attempted on purpose."
        )

    MASK_DIR.mkdir(exist_ok=True)
    base = "https://repo-prod.prod.sagebase.org/repo/v1/entity"

    def grab(fid: str, dest: Path) -> Path:
        if not dest.exists():
            url = s.get(f"{base}/{fid}/file?redirect=false", timeout=60).text.strip().strip('"')
            dest.write_bytes(s.get(url, timeout=180).content)
        return dest

    # 🔴 The clip label id is NOT hardcoded: it is read from the release's own legend on
    # first run and asserted, because a guessed palette index is exactly the kind of
    # silent wrong-target that target-side noise punishes hardest.
    palette = _resolve_clip_label(syn, s, grab)

    inst_hist, areas, clip_frames, total = Counter(), [], 0, 0
    exported = []
    for root in (SEGMENTATION, CLIPPING):
        for folder in syn.getChildren(root):
            kids = list(syn.getChildren(folder["id"]))
            if kids and kids[0]["type"].endswith("Folder"):
                kids = [k for sub in kids for k in syn.getChildren(sub["id"])]
            for f in kids[:limit_per_folder]:
                m = np.array(Image.open(grab(f["id"], MASK_DIR / f["name"])))
                total += 1
                mask = _clip_mask(m, palette)
                n = _instances(mask)
                inst_hist[n] += 1
                if n:
                    clip_frames += 1
                    areas.append(float(mask.sum()) / (FRAME_WH[0] * FRAME_WH[1]))
                    if len(exported) < export_n:
                        exported.append(f["name"])

    areas.sort()
    result = {
        "stage": 1,
        "frames_scanned": total,
        "n_clip_frames": clip_frames,
        "instances_per_frame": dict(sorted(inst_hist.items())),
        "median_clip_area_frac": areas[len(areas) // 2] if areas else None,
        "exported_for_eye_pass": exported,
        "measure_1_passes": clip_frames >= MIN_CLIP_FRAMES,
        "measure_4_jaw_violation": None,
        "verdict": (
            "INCOMPLETE -- measure 4 (clips still inside the applier jaws) is an eye call. "
            f"{len(exported)} frames exported; a human scores them and fills "
            "measure_4_jaw_violation before this gate has a verdict."
        ),
    }
    prev = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    OUT_JSON.write_text(json.dumps({**prev, "stage1": result}, indent=2), encoding="utf-8")
    return result


def _resolve_clip_label(syn, s, grab):
    raise NotImplementedError(
        "Resolve the clip label from the release legend on the first run with download "
        "access, then assert it against a frame from the dedicated Clipping set, where a "
        "clip is known to be present. Do not guess a palette index."
    )


def _clip_mask(mask, palette):
    raise NotImplementedError("Depends on the encoding, which cannot be read without download access.")


def _instances(mask):
    raise NotImplementedError("Depends on the encoding, which cannot be read without download access.")


if __name__ == "__main__":
    r = stage0()
    print(json.dumps({k: v for k, v in r.items() if k != "frames_per_video_main"}, indent=2))
