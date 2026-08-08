"""Rung 31 — where does the model actually look, and what do the visual tokens look like to the LLM?

Folder-private glue. Importable; a notebook cell calls ``main(Config(...))``. Not a launcher.

## The question

In 30 rungs the campaign has varied `learning_rate`, `vit_lr`, `lora_rank`/`alpha`,
`max_grad_norm`, epochs, data and `freeze_vit`. It has never varied `target_modules`
(always ``all-linear``) or `freeze_aligner` (always ``true``), and the G-COV census measured
**720 tensors = 504 LLM + 216 ViT + 0 aligner** — the projector between the two has never been
trained. This probe measures the thing that would justify touching it: **how much of the
model's attention actually reaches the image, and what the projected visual tokens look like in
the LLM's own embedding space.**

Paired across four checkpoints on the SAME frames and the SAME questions, so the comparison is
the checkpoint and nothing else:

| arm | what it is |
|---|---|
| ``base`` | Qwen3-VL-8B, no fine-tuning |
| ``rung02`` | first LoRA SFT — LLM only in practice |
| ``rung06`` | ViT-LoRA, ``freeze_vit=false`` — the campaign's best for 17 days |
| ``a2`` | ``21_lr_2e4_v1/checkpoint-2703`` — shipped as submission 02 |

## 🔴 The trap that makes the first attempt return nothing

`src/frame/engine.py` loads with ``attn_impl="sdpa"``, inherited from the A2 recipe. **SDPA does
not return attention weights**, so ``output_attentions=True`` yields ``None`` with no error —
another silent no-op of the kind this repo keeps paying for. This module loads with
``attn_implementation="eager"`` and asserts the tensors came back.

## What is measured

* **(a) visual attention mass, per layer.** From the readout position — the last prompt token,
  the one that predicts the first answer token — the fraction of attention that lands on image
  tokens rather than text. This is the "how much weight does the LLM give the picture" number,
  as a curve over depth.
* **(b) spatial heatmap.** The same attention, restricted to image tokens and reshaped to the
  patch grid. Saved as ``.npy`` per (arm, question) plus a PNG overlay.
* **(c) the embedding plane.** ``hidden_states[0]`` at image positions IS the post-merger visual
  embedding as it enters the LLM. Reported as: its norm against the text tokens' norm, and the
  nearest vocabulary tokens by cosine — *what words do these patches look like to the LLM?*

⚠️ **Limits, stated up front.** Raw attention is a weak explanation (attention×gradient or
rollout is stronger); the readout-position choice is one defensible convention among several;
and the patch grid is coarse at our ``max_pixels``. This is a description of the model's
internals, **not** a causal claim about the score.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = Path("/workspace/models/qwen3-vl-8b")
REPO = Path("/workspace/repo")

#: epoch 3 of each arm — the comparable point, and A2's is what shipped as submission 02.
ARMS: dict[str, Path | None] = {
    "base": None,
    "rung02": REPO / "experiments/02-lora-sft/runs/02_lora_sft_v1/ckpt/v0-20260714-022142/checkpoint-2580",
    "rung06": REPO / "experiments/06-vit-lora/runs/06_vit_lora_v1/ckpt/v0-20260717-224148/checkpoint-2580",
    "a2": REPO / "experiments/21-recipe-sweep/runs/21_lr_2e4_v1/ckpt/v0-20260729-172404/checkpoint-2703",
}


@dataclass
class Config:
    repo: Path = Path("/workspace/repo_leo")
    data_root: Path = Path("/workspace/orena-data")
    base: Path = BASE
    out: Path = Path("/workspace/repo_leo/experiments/31-attention-probe/runs/31_attn_v1")

    #: questions per arm. Paired: the SAME rows for every arm.
    n_questions: int = 12
    #: same cap the eval uses (`src/frame/config.py:54`), so the token grid is the real one
    max_pixels: int = 1280 * 720
    seed: int = 42
    top_k_tokens: int = 8
    arms: dict = field(default_factory=lambda: dict(ARMS))
    fps: dict = field(default_factory=lambda: {"heico": 25, "lapchole": 30})


def select_questions(cfg: Config):
    """A fixed, seeded, ID/OOD-balanced slice of `number` Clip-count questions."""
    import pandas as pd

    frames = []
    for p in sorted(glob.glob(str(cfg.data_root / "*" / "data" / "frame" / "*.parquet"))):
        ds = p.split("orena-data/")[1].split("/")[0]
        d = pd.read_parquet(p)
        d["ds"] = ds
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["t"] = df.timestamp_start.map(
        lambda s: (lambda h, m, x: int(h) * 3600 + int(m) * 60 + int(x))(*str(s).split(":"))
    )
    df["fi"] = [round(t * cfg.fps[ds]) for t, ds in zip(df.t, df.ds)]
    q = df[
        (df.answer_format == "number")
        & df.question.str.contains("Clip", case=False, na=False)
    ].copy()
    q["gold"] = __import__("pandas").to_numeric(q.answer, errors="coerce")
    q = q.dropna(subset=["gold"])
    half = max(cfg.n_questions // 2, 1)
    ood = q[q.ood]
    idd = q[~q.ood]
    return pd.concat([
        idd.sample(min(half, len(idd)), random_state=cfg.seed),
        ood.sample(min(cfg.n_questions - half, len(ood)), random_state=cfg.seed),
    ]).reset_index(drop=True)


def _frame(cfg: Config, ds: str, video: str, idx: int):
    import decord
    from PIL import Image

    vr = decord.VideoReader(str(cfg.data_root / ds / "videos" / video))
    return Image.fromarray(np.asarray(vr[min(idx, len(vr) - 1)].asnumpy()))


def load_arm(cfg: Config, adapter: Path | None):
    """Base + optional LoRA adapter, forced to EAGER attention."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(cfg.base), max_pixels=cfg.max_pixels)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(cfg.base), dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",  # 🔴 sdpa returns no attention weights, silently
    ).eval()
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).eval()
    return model, processor


def probe_one(cfg: Config, model, processor, image, question: str) -> dict:
    """One forward pass; attention and embeddings read at the readout position."""
    import torch

    from frame.engine import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": question}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    if out.attentions is None or out.attentions[0] is None:
        raise AssertionError(
            "output_attentions returned None. The model is not on eager attention — "
            "sdpa/flash silently drop the weights. Reload with attn_implementation='eager'."
        )

    ids = inputs["input_ids"][0]
    img_tok = getattr(model.config, "image_token_id", None)
    if img_tok is None:
        img_tok = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    is_img = (ids == img_tok)
    n_img = int(is_img.sum())
    if n_img == 0:
        raise AssertionError("no image tokens found in the prompt; image_token_id is wrong")

    # (a) visual mass per layer, from the readout position (last prompt token)
    mass = []
    last_layer_img = None
    for li, att in enumerate(out.attentions):          # [1, heads, seq, seq]
        row = att[0, :, -1, :].float()                 # heads x seq, attention FROM readout
        row = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
        m = row[:, is_img].sum(-1)                     # per head
        mass.append(float(m.mean()))
        if li == len(out.attentions) - 1:
            last_layer_img = row[:, is_img].mean(0).cpu().numpy()

    # (b) spatial map: mean over ALL layers, restricted to image tokens
    per_layer_img = []
    for att in out.attentions:
        row = att[0, :, -1, :].float()
        row = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
        per_layer_img.append(row[:, is_img].mean(0).cpu().numpy())
    heat = np.mean(np.stack(per_layer_img), axis=0)     # [n_img]

    # (c) embedding plane: hidden_states[0] at image positions IS the post-merger embedding
    emb0 = out.hidden_states[0][0].float()              # [seq, hidden]
    vis = emb0[is_img]
    txt = emb0[~is_img]
    W = model.get_input_embeddings().weight.float()     # [vocab, hidden]
    vn = torch.nn.functional.normalize(vis, dim=-1)
    Wn = torch.nn.functional.normalize(W, dim=-1)
    sim = vn @ Wn.T                                     # [n_img, vocab]
    top = sim.mean(0).topk(cfg.top_k_tokens)
    nearest = [processor.tokenizer.decode([i]) for i in top.indices.tolist()]

    return {
        "n_image_tokens": n_img,
        "n_text_tokens": int((~is_img).sum()),
        "visual_mass_per_layer": mass,
        "visual_mass_mean": float(np.mean(mass)),
        "visual_mass_last_layer": mass[-1],
        "heat": heat,
        "heat_last_layer": last_layer_img,
        "emb_norm_visual": float(vis.norm(dim=-1).mean()),
        "emb_norm_text": float(txt.norm(dim=-1).mean()),
        "emb_cos_visual_to_text_mean": float((vn @ torch.nn.functional.normalize(txt, dim=-1).T).mean()),
        "nearest_vocab_tokens": nearest,
        "nearest_vocab_cos": [round(v, 4) for v in top.values.tolist()],
    }


def main(cfg: Config) -> dict:
    import sys

    sys.path.insert(0, str(cfg.repo / "src"))
    import torch

    rows = select_questions(cfg)
    out_dir = Path(cfg.out)
    (out_dir / "heat").mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} questions, paired across {len(cfg.arms)} arms", flush=True)

    images = [_frame(cfg, r.ds, r.video, int(r.fi)) for r in rows.itertuples()]
    summary: dict[str, list] = {}

    for arm, adapter in cfg.arms.items():
        print(f"\n== {arm} == {adapter or cfg.base}", flush=True)
        model, processor = load_arm(cfg, adapter)
        per_q = []
        for i, (r, img) in enumerate(zip(rows.itertuples(), images)):
            res = probe_one(cfg, model, processor, img, r.question)
            np.save(out_dir / "heat" / f"{arm}_q{i:02d}.npy", res.pop("heat"))
            np.save(out_dir / "heat" / f"{arm}_q{i:02d}_last.npy", res.pop("heat_last_layer"))
            res.update({"i": i, "ds": r.ds, "ood": bool(r.ood), "gold": int(r.gold)})
            per_q.append(res)
            print(f"  q{i:02d} {r.ds:<8} ood={bool(r.ood)!s:<5} gold={int(r.gold):>2} | "
                  f"visual_mass mean={res['visual_mass_mean']:.4f} "
                  f"last={res['visual_mass_last_layer']:.4f} | "
                  f"|v|={res['emb_norm_visual']:.2f} |t|={res['emb_norm_text']:.2f}", flush=True)
        summary[arm] = per_q
        del model
        torch.cuda.empty_cache()

    agg = {
        arm: {
            "visual_mass_mean": float(np.mean([q["visual_mass_mean"] for q in qs])),
            "visual_mass_last_layer": float(np.mean([q["visual_mass_last_layer"] for q in qs])),
            "visual_mass_curve": np.mean([q["visual_mass_per_layer"] for q in qs], axis=0).tolist(),
            "emb_norm_visual": float(np.mean([q["emb_norm_visual"] for q in qs])),
            "emb_norm_text": float(np.mean([q["emb_norm_text"] for q in qs])),
            "emb_cos_visual_to_text": float(np.mean([q["emb_cos_visual_to_text_mean"] for q in qs])),
            "nearest_vocab_tokens": qs[0]["nearest_vocab_tokens"],
        }
        for arm, qs in summary.items()
    }
    result = {"config": {"n_questions": cfg.n_questions, "max_pixels": cfg.max_pixels,
                         "seed": cfg.seed}, "per_arm": agg, "per_question": summary}
    (out_dir / "RESULTS_attention.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n" + json.dumps(agg, indent=2)[:2000], flush=True)
    return result
