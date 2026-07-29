"""Probe 16b — is the gold count something ANY visual system can see in the frame?

## Why this is the probe that bounds everything else

Rung 06 ep3 is our best checkpoint and its **`number_margin_OOD` is 0.0 — exactly the trivial
floor.** It adds literally nothing over "always answer the mode" on OOD counting. Three
explanations are on the table and they license opposite work:

1. the model cannot count (→ better recipe: `len(points)`, counting-only hyper-parameters);
2. the model was never taught to (→ more/better supervision: minted zeros, pointing);
3. **the label is not a count of frame-visible instances at all** (→ every `number` lever is
   aimed at a ceiling and the whole branch should be abandoned).

Reading (3) is not idle. `context/ERROR_ANATOMY.md` records that the gold moves **±0.86 between
frames ≤1 s apart**, that "total instances" contradicts the sum of per-class counts on **≥11%**
of frames, and that a **blind human scores r = −0.17** against the gold. Our own model scores
**r = +0.43**. If an independent, open-vocabulary detector also fails to correlate, the honest
conclusion is that the quantity is not recoverable from the pixels, and rung 15's
pre-registered annotation-ceiling question is answered.

## Design

Frozen **OWLv2** (`google/owlv2-base-patch16-ensemble`, Apache-2.0, downloaded to the pod —
🔒 no challenge frame ever leaves the machine, per the DUA) over the **681 val `Clips`
questions**. For each frame, `n_det` = detections above threshold. Report Spearman r and MAE
against the gold, **stratified by gold value**, next to the two reference points already in the
repo.

## 🔴 Pre-register the threshold before looking

| outcome | verdict |
|---|---|
| **r ≥ 0.5 and MAE < 1.0** | the gold IS a frame-visible instance count → pointing is learnable, pseudo-labels viable → proceed to 16c and a rung |
| **r ≈ 0.2–0.5** | ambiguous → escalate to ~100 clinician-adjudicated frames before spending training compute |
| **r < 0.2** | the gold is not what a detector *or* a human can see → **permanent NO-GO on pointing**, and every `number` lever is re-aimed |

## ⚠️ The confound, stated up front

A detector failure is also a detector **domain** failure — OWLv2 has never seen laparoscopic
tissue. That is exactly the trap rung 12 fell into three times (an inference-only intervention
on an out-of-domain instrument reads negative for reasons that have nothing to do with the
hypothesis). Two mitigations, both mandatory:

* **Report the detector's own precision on a 40-frame eyeball subset** — if it cannot find a
  clip a human can point at, the probe measured the detector, not the label.
* **Run a positive control**: the same detector on a prompt it must be able to do
  (`"surgical instrument"`). If the control also collapses, the instrument is blind here and
  the probe returns "not measurable" rather than a false NO-GO. A faithful "not measurable" is
  a valid result (rung 12e set that precedent).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Open-vocabulary text queries. OWLv2 is prompted with natural phrases, not class ids.
# Several phrasings per concept because open-vocab detectors are phrasing-sensitive, and a
# single unlucky wording would look like the label is unlearnable.
QUERIES = {
    "clip": [
        "a metal surgical clip",
        "a small metallic clip on tissue",
        "a surgical clip applied to a duct",
    ],
    # positive control — if THIS collapses too, the detector is blind in this domain and the
    # probe cannot speak to the label at all.
    "control": ["a surgical instrument", "a laparoscopic grasper"],
}


class Owlv2Counter:
    """Frozen OWLv2 wrapper: `count(image, concept, threshold) -> (n, boxes, scores)`."""

    def __init__(self, model_dir: str, device: str = "cuda") -> None:
        self.model_dir, self.device = model_dir, device
        self.model = self.processor = None

    def load(self) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        logger.info("loading OWLv2 from %s", self.model_dir)
        self.processor = Owlv2Processor.from_pretrained(self.model_dir)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(self.model_dir)
            .to(self.device)
            .eval()
        )
        self._torch = torch

    def count(self, image, concept: str, threshold: float = 0.15):
        """Detections above `threshold` for the best-scoring phrasing of `concept`.

        Taking the BEST phrasing (not the union) avoids double-counting the same object under
        two wordings — the tile-border problem in a different guise. It is also the generous
        reading: we want the detector's best shot, so a low correlation cannot be blamed on a
        bad prompt.
        """
        torch = self._torch
        phrases = QUERIES[concept]
        inputs = self.processor(text=[phrases], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        target = torch.tensor([[image.height, image.width]], device=self.device)
        res = self.processor.post_process_grounded_object_detection(
            outputs=out, target_sizes=target, threshold=threshold
        )[0]
        labels = res["labels"].tolist()
        scores = res["scores"].tolist()
        boxes = res["boxes"].tolist()
        best_n, best_i = 0, None
        for i in range(len(phrases)):
            n = sum(1 for lb in labels if lb == i)
            if n > best_n:
                best_n, best_i = n, i
        keep = [j for j, lb in enumerate(labels) if best_i is not None and lb == best_i]
        return best_n, [boxes[j] for j in keep], [scores[j] for j in keep]


def run(counter: Owlv2Counter, selected, threshold: float = 0.15, log_every: int = 100):
    """Run the detector over the selected (item, gold) pairs. One row per frame-question."""
    from PIL import Image

    from frame.data import frame_cache_name

    rows = []
    for i, (it, gold) in enumerate(selected, 1):
        key = frame_cache_name(it)
        path = f"/workspace/frames_cache/{key}"
        row = {
            "frame_key": key,
            "dataset": it.dataset,
            "video_id": it.video_id,
            "distribution": "OOD" if it.dataset == "heico" else "ID",
            "gold": gold,
        }
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
                for concept in ("clip", "control"):
                    n, _, sc = counter.count(img, concept, threshold)
                    row[f"n_{concept}"] = n
                    row[f"maxscore_{concept}"] = max(sc) if sc else 0.0
        except Exception as exc:
            logger.warning("frame %s failed: %s", key, exc)
            row.update(n_clip=None, n_control=None, maxscore_clip=0.0, maxscore_control=0.0)
        rows.append(row)
        if i % log_every == 0:
            logger.info("  detector: %d/%d", i, len(selected))
    return rows


def correlate(rows: list[dict], field: str = "n_clip") -> dict:
    """Spearman r + MAE of detector count vs gold, pooled and per distribution.

    Spearman (rank) rather than Pearson: the detector's absolute scale is arbitrary — it is
    threshold-dependent — while the ORDERING is the thing the label would have to share for a
    counting objective to be learnable at all.
    """
    from scipy import stats

    def _agg(sel):
        pairs = [(r[field], r["gold"]) for r in sel if r.get(field) is not None]
        if len(pairs) < 3:
            return {"n": len(pairs), "spearman_r": float("nan"), "p": float("nan")}
        det = [a for a, _ in pairs]
        gold = [b for _, b in pairs]
        r, p = stats.spearmanr(det, gold)
        mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
        return {
            "n": len(pairs),
            "n_videos": len({x["video_id"] for x in sel}),
            "spearman_r": float(r),
            "p": float(p),
            "mae": mae,
            "mean_det": sum(det) / len(det),
            "mean_gold": sum(gold) / len(gold),
            "det_zero_rate": sum(1 for d in det if d == 0) / len(det),
        }

    out = {"ALL": _agg(rows)}
    for dist in ("ID", "OOD"):
        sel = [r for r in rows if r["distribution"] == dist]
        if sel:
            out[dist] = _agg(sel)
    return out


def by_gold_value(rows: list[dict], field: str = "n_clip") -> dict:
    """Detector mean per gold value — where a flat line means the label carries no signal."""
    buckets: dict[int, list[int]] = {}
    for r in rows:
        if r.get(field) is None:
            continue
        buckets.setdefault(r["gold"], []).append(r[field])
    return {
        g: {"n": len(v), "mean_det": sum(v) / len(v)} for g, v in sorted(buckets.items())
    }


VERDICT = """\
r >= 0.5 and MAE < 1.0  -> gold IS frame-visible; pointing learnable; proceed to 16c + a rung
0.2 <= r < 0.5          -> ambiguous; escalate to ~100 adjudicated frames before training spend
r < 0.2                 -> gold is not frame-visible; NO-GO on pointing; re-aim every number lever
(and if the CONTROL also collapses, the detector is blind here: report NOT MEASURABLE)"""
