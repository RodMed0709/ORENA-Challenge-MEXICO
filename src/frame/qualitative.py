"""Qualitative export: a browsable folder of frame + question + GT + our answer.

Samples ~N cases stratified by answer_format and balanced between correct and
incorrect predictions, writes each frame as a JPEG, plus ``samples.csv`` and a
self-contained ``index.html`` for eyeballing.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from frame.data import FrameProvider

logger = logging.getLogger(__name__)


def _select(results_df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Pick ~n qIDs stratified by answer_format, balanced correct/incorrect."""
    rng = np.random.default_rng(seed)
    formats = list(results_df["answer_format"].unique())
    per_fmt = max(1, n // max(1, len(formats)))
    picked: list[str] = []
    for fmt in formats:
        sub = results_df[results_df["answer_format"] == fmt]
        half = max(1, per_fmt // 2)
        for corr in (True, False):
            pool = sub[sub["correctness"] == corr]["qID"].tolist()
            if not pool:
                continue
            k = min(half, len(pool))
            picked.extend(rng.choice(pool, size=k, replace=False).tolist())
    # dedup, trim, and if short, top up randomly
    picked = list(dict.fromkeys(picked))
    if len(picked) < n:
        rest = [q for q in results_df["qID"].tolist() if q not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def export_qualitative(cfg, items, responses, results_df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    item_by_q = {it.request.qID: it for it in items}
    resp_by_q = {r.qID: r for r in responses}
    res_by_q = {r["qID"]: r for _, r in results_df.iterrows()}

    picked = _select(results_df, cfg.n_qualitative, cfg.seed)
    # read frames grouped by video for reader-cache efficiency
    picked.sort(key=lambda q: (item_by_q[q].dataset, item_by_q[q].video_id, item_by_q[q].frame_index))

    provider = FrameProvider(cfg)
    rows = []
    for q in picked:
        it = item_by_q[q]
        res = res_by_q[q]
        img_rel = f"frames/{q}.jpg"
        try:
            img = provider.get_frame(it)
            img.thumbnail((960, 960))
            img.convert("RGB").save(frames_dir / f"{q}.jpg", quality=88)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qualitative frame %s failed: %s", q, exc)
            img_rel = ""
        rows.append(
            {
                "qID": q,
                "dataset": it.dataset,
                "video": it.video_id,
                "timestamp_s": it.request.start_time,
                "answer_format": res["answer_format"],
                "primary_capability": res["primary"],
                "question": it.request.question,
                "ground_truth": it.reference.answer,
                "our_answer": resp_by_q[q].content,
                "correct": bool(res["correctness"]),
                "latency_s": round(float(resp_by_q[q].latency), 3),
                "image": img_rel,
            }
        )
    provider.close()

    df = pd.DataFrame(rows)
    df.to_csv(out / "samples.csv", index=False)
    _write_html(df, out / "index.html", cfg)
    logger.info("Qualitative export: %d cases → %s", len(df), out)


def _write_html(df: pd.DataFrame, path: Path, cfg) -> None:
    n_ok = int(df["correct"].sum())
    cards = []
    for _, r in df.iterrows():
        badge = "✓ correct" if r["correct"] else "✗ wrong"
        color = "#1a7f37" if r["correct"] else "#cf222e"
        img = (
            f'<img src="{html.escape(r["image"])}" loading="lazy">'
            if r["image"]
            else '<div class="noimg">frame unavailable</div>'
        )
        cards.append(
            f"""
    <div class="card">
      {img}
      <div class="meta">
        <span class="badge" style="background:{color}">{badge}</span>
        <span class="fmt">{html.escape(str(r['answer_format']))}</span>
        <span class="lat">{r['latency_s']:.2f}s</span>
      </div>
      <div class="q"><b>Q:</b> {html.escape(str(r['question']))}</div>
      <div class="gt"><b>GT:</b> {html.escape(str(r['ground_truth']))}</div>
      <div class="ans"><b>Ours:</b> {html.escape(str(r['our_answer']))}</div>
      <div class="src">{html.escape(str(r['dataset']))} · {html.escape(str(r['video']))} · t={r['timestamp_s']:.0f}s · {html.escape(str(r['qID']))}</div>
    </div>"""
        )
    doc = f"""<!doctype html><meta charset="utf-8">
<title>FRAME baseline — qualitative ({cfg.run_name})</title>
<style>
 body{{font:14px/1.4 system-ui,sans-serif;margin:24px;background:#f6f8fa;color:#1f2328}}
 h1{{font-size:20px}} .sub{{color:#57606a;margin-bottom:20px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
 .card{{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
 .card img{{width:100%;border-radius:6px;background:#000}}
 .noimg{{width:100%;aspect-ratio:16/9;display:grid;place-items:center;background:#eee;border-radius:6px;color:#888}}
 .meta{{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}}
 .badge{{color:#fff;padding:2px 8px;border-radius:20px;font-size:12px;font-weight:600}}
 .fmt{{background:#eaeef2;padding:2px 8px;border-radius:20px;font-size:12px}}
 .lat{{color:#57606a;font-size:12px}}
 .q,.gt,.ans{{margin:4px 0}} .gt{{color:#1a7f37}} .ans{{color:#0969da}}
 .src{{color:#8c959f;font-size:11px;margin-top:8px}}
</style>
<h1>FRAME zero-shot baseline — qualitative samples</h1>
<div class="sub">{cfg.run_name} · {len(df)} cases · {n_ok} correct / {len(df) - n_ok} wrong · judge={html.escape(cfg.judge_model)}</div>
<div class="grid">{''.join(cards)}</div>
"""
    path.write_text(doc, encoding="utf-8")
