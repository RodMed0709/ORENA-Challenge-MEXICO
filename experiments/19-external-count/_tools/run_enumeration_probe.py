"""Rung 19a — can the model enumerate at all, or does it only fail on small objects?

Holds the question fixed (the challenge's own `number` template) and changes the object: instead
of our 4 mm clips, ask for large, unmistakable surgical instruments on external frames whose
count is an exact instance annotation.

Two substrates, run because their confounds are orthogonal and neither is clean on its own:
  * SurgSigma-DB — laparoscopic cholecystectomy, our own domain, counts from named bounding
    boxes. But its frames are `desmoke` outputs, i.e. processed, and
    [[inference-only-input-tests-biased]] says an input transform met only at inference is
    biased toward the negative.
  * MISAW-Seg — raw frames and COCO instance masks, but microsurgical anastomosis, a domain the
    model has never seen.
A finding that holds in both survives either objection; one that appears in only one does not.

Scoring is exact match on the emitted integer — no judge, because the gold IS an integer. The
report is the bias/correlation decomposition rung 19 pre-registered, not accuracy alone: a model
emitting a prior and a model perceiving badly both score low and must be told apart.
"""
import argparse, collections, json, os, re, statistics, sys, time
from pathlib import Path

REPO = os.path.expanduser("~/storage/repo")
for p in (f"{REPO}/src", f"{REPO}/vendor/orena-focus/src",
          f"{REPO}/experiments/43-thinking-at-inference/_tools"):
    sys.path.insert(0, p)

NUM = re.compile(r"-?\d+")


def parse_int(text: str):
    m = NUM.search(text or "")
    return int(m.group()) if m else None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main(model: str, questions: Path, frames_dir: Path, out: Path, tag: str,
         limit: int = 0, thinking: bool = False):
    from PIL import Image
    from vllm_engine import VLLMBatchConfig, VLLMBatchEngine

    rows = json.loads(questions.read_text())
    if limit:
        rows = rows[:limit]
    imgs = []
    for r in rows:
        # The two substrates land on disk in different shapes and neither matches the path the
        # annotation carries, so resolve explicitly rather than guessing: SurgSigma frames came
        # out of the archive flattened as `<frame>__<video>.png`, MISAW's mirror the repo's
        # `images/<prefix>/<file>` nesting. A miss raises here instead of at question 900.
        name = r.get("frame") or r.get("image")
        stem, parent = Path(name).stem, Path(name).parent.name
        for cand in (frames_dir / f"{stem}__{parent}.png",
                     frames_dir / "_".join(stem.split("_")[:2]) / f"{stem}.png",
                     frames_dir / f"{stem}.png",
                     frames_dir / name):
            if cand.exists():
                imgs.append(Image.open(cand).convert("RGB"))
                break
        else:
            raise FileNotFoundError(f"{r['qID']}: no frame for {name} under {frames_dir}")

    # With thinking on, the trace is left UNCAPPED inside a 4096 window and the engine splits on
    # `</think>` — rung 43 measured that a guessed cap truncates the trace, and a truncated trace
    # emits no answer at all, which scores as a wrong answer rather than as the artefact it is.
    cfg = (VLLMBatchConfig(model_path=model, enable_thinking=True, max_new_tokens=None,
                           answer_char_cap=1_000_000, max_model_len=4096,
                           gpu_memory_utilization=0.88)
           if thinking else
           VLLMBatchConfig(model_path=model, enable_thinking=False,
                           max_new_tokens=64, answer_char_cap=300))
    eng = VLLMBatchEngine(cfg, "You are an expert surgical assistant. Answer with a number and "
                               "nothing else.")
    answers, wall = eng.predict_batch(imgs, [r["question"] for r in rows])

    preds, golds, recs = [], [], []
    for r, a in zip(rows, answers):
        g, p = int(r["answer"]), parse_int(a)
        recs.append({"qID": r["qID"], "gold": g, "raw": a, "pred": p, "group": r.get("group", "")})
        if p is not None:
            preds.append(p); golds.append(g)

    err = [p - g for p, g in zip(preds, golds)]
    by_gold = collections.defaultdict(list)
    for rec in recs:
        by_gold[rec["gold"]].append(rec["pred"] == rec["gold"])
    res = {
        "tag": tag, "model": model, "n": len(recs),
        "n_unparseable": sum(1 for r in recs if r["pred"] is None),
        "accuracy": round(sum(r["pred"] == r["gold"] for r in recs) / len(recs), 4),
        "bias_mean_pred_minus_gold": round(statistics.mean(err), 4) if err else None,
        "mae": round(statistics.mean(abs(e) for e in err), 4) if err else None,
        "off_by_one_share": round(sum(abs(e) == 1 for e in err) / max(sum(e != 0 for e in err), 1), 4),
        "spearman": round(spearman(preds, golds), 4) if len(preds) > 2 else None,
        "accuracy_by_gold": {str(k): round(sum(v) / len(v), 4) for k, v in sorted(by_gold.items())},
        "n_by_gold": {str(k): len(v) for k, v in sorted(by_gold.items())},
        "mean_pred_by_gold": {str(k): round(statistics.mean(
            [r["pred"] for r in recs if r["gold"] == k and r["pred"] is not None] or [float("nan")]), 2)
            for k in sorted(by_gold)},
        "secs_per_question": round(wall / len(recs), 3),
        "thinking": thinking,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"RESULTS_{tag}.json").write_text(json.dumps(res, indent=2))
    (out / f"predictions_{tag}.json").write_text(json.dumps(recs, indent=1))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--questions", required=True)
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    main(a.model, Path(a.questions), Path(a.frames), Path(a.out), a.tag, a.limit, a.thinking)
