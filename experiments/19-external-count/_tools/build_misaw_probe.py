"""Build the rung-19a question package out of MISAW-Seg's COCO annotations.

19a asks one thing: can the model enumerate at all, or does it only fail on small objects?
The way to find out is to hold the question fixed and change the object. So the wording here is
lifted **verbatim** from the challenge's own `number` template —
``How many {Class}s appear in this frame? Please provide a number.`` — and only the class moves.

Gold comes from COCO instance annotations, so the count is `len(annotations of that category)`:
an exact instance count with nothing inferred. That is the property that disqualified SAR-RARP50
(semantic masks needing connected components) and it is why this dataset can carry the probe.

Two classes, chosen for what they contrast:
  * `Needle holder` — large, high-contrast, unambiguous instruments. Mass at 1-3, which is exactly
    the band where our own model is 0.814 / 0.406 / 0.200 on Clips.
  * `Wire` — thin and small, and the only class here with a tail (up to 9), so it also gives a
    second curve on the same frames.

⚠️ MISAW is microsurgical anastomosis, not laparoscopy. A flat failure could be domain shift
rather than an enumeration limit, which is why the pre-registered read is about the SHAPE of the
curve against our own, never the absolute level. See PLAN.md.
"""
import argparse, collections, json
from pathlib import Path

TEMPLATE = "How many {plural} appear in this frame? Please provide a number."
GROUPS = {
    "needle_holder": (["Left needle holder", "Right needle holder"], "Needle holders"),
    "wire": (["Wire"], "Wires"),
}


def build(coco_path: Path, out_path: Path) -> dict:
    coco = json.loads(coco_path.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for a in coco["annotations"]:
        counts[str(a["image_id"])][cats[a["category_id"]]] += 1

    rows = []
    for img_id, im in images.items():
        per = counts.get(str(img_id), collections.Counter())
        for key, (members, plural) in GROUPS.items():
            rows.append({
                "qID": f"misaw__{key}__{img_id}",
                "image": im["file_name"],
                "question": TEMPLATE.format(plural=plural),
                "answer": str(sum(per[m] for m in members)),
                "answer_format": "number",
                "group": key,
            })
    out_path.write_text(json.dumps(rows, indent=1) + "\n")

    summary = {k: dict(sorted(collections.Counter(
        int(r["answer"]) for r in rows if r["group"] == k).items())) for k in GROUPS}
    return {"n_questions": len(rows), "n_images": len(images), "gold_distribution": summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coco"); ap.add_argument("out")
    a = ap.parse_args()
    print(json.dumps(build(Path(a.coco), Path(a.out)), indent=1))
