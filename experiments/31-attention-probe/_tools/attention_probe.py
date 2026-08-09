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
    #: 🔴 NOT the eval's 1280*720 (`src/frame/config.py:54`), and the reason is memory, stated
    #: rather than hidden. `output_attentions=True` on eager attention materialises ALL 36
    #: layers: at ~1,000 sequence positions that is 1000^2 x 32 heads x 36 layers x 4 bytes
    #: ~= 4.6 GB of attention on top of the 16 GB model, and it OOMs a 32 GB card (measured
    #: 2026-08-08, died after 6 questions). At 512x512 the sequence is ~400 and the same
    #: tensors cost ~0.7 GB.
    #: ⚠️ The consequence is a COARSER heatmap and a model that sees fewer visual tokens than
    #: it does at eval. Every arm uses the identical value, so the BETWEEN-ARM comparison —
    #: which is the whole point — stays valid; the absolute visual-mass level does not
    #: transfer to the deployed configuration.
    max_pixels: int = 512 * 512
    #: blocking sanity bound on the resize actually happening — 512x512 measures 231 tokens
    max_image_tokens: int = 400
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
    q["gold"] = pd.to_numeric(q.answer, errors="coerce")
    q = q.dropna(subset=["gold"]).copy()
    # 🔴 OOD is the DATASET, not the `ood` column. `RULES §3`: `heico` is the organizers' own
    # OOD partition. The parquets' `ood` field is False on all 2,071 clip-count rows, so
    # splitting on it silently produced an empty OOD half and the probe ran 6 ID-only
    # questions instead of 12 balanced ones. Measured 2026-08-08 — the run looked healthy,
    # every arm returned rc=0, and only the file count gave it away.
    q["is_ood"] = q.ds.eq("heico")
    half = max(cfg.n_questions // 2, 1)
    idd, ood = q[~q.is_ood], q[q.is_ood]
    if len(ood) < cfg.n_questions - half or len(idd) < half:
        raise AssertionError(
            f"cannot balance: {len(idd)} ID and {len(ood)} OOD rows available for "
            f"{cfg.n_questions} questions"
        )
    return pd.concat([
        idd.sample(half, random_state=cfg.seed),
        ood.sample(cfg.n_questions - half, random_state=cfg.seed),
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
    from qwen_vl_utils import process_vision_info

    from frame.engine import SYSTEM_PROMPT

    # 🔴 Built exactly as `engine.py:157-168` builds it — apply_chat_template(tokenize=False)
    # then process_vision_info then processor(...). A different path would tokenise the image
    # differently and the probe would describe a prompt the model is never asked.
    # 🔴 `max_pixels` MUST go inside the message dict. `AutoProcessor.from_pretrained(...,
    # max_pixels=N)` is a NO-OP on this path: the resize is done by `process_vision_info`,
    # which only reads the per-message key. Measured 2026-08-08 on a 1280x720 frame —
    # processor kwarg 262144 gave 920 image tokens (resized to 1288x728, i.e. UP), the
    # in-message key gave 231 (672x364). Same finding applies to `engine.py:43`.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image,
                                      "max_pixels": cfg.max_pixels},
                                     {"type": "text", "text": question}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
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
    if n_img > cfg.max_image_tokens:
        # 🔴 The guard for the no-op above. A resize that silently does not happen shows up
        # only as an OOM 36 layers later, which reads like a memory problem and is a config
        # problem. Fail here, cheaply, with the number in the message.
        raise AssertionError(
            f"{n_img} image tokens, expected <= {cfg.max_image_tokens} at max_pixels="
            f"{cfg.max_pixels}. The resize did not take effect — check that `max_pixels` is "
            "inside the message dict and not only on the processor."
        )

    # (a) visual mass per layer and (b) the spatial map, in ONE pass over the layers.
    # The readout row is extracted and moved to CPU immediately: a second loop over
    # `out.attentions` would hold all 36 full matrices alive for twice as long.
    mass: list[float] = []
    per_layer_img: list[np.ndarray] = []
    for att in out.attentions:                         # [1, heads, seq, seq]
        row = att[0, :, -1, :].float()                 # heads x seq, attention FROM readout
        row = row / row.sum(-1, keepdim=True).clamp_min(1e-9)
        img = row[:, is_img]
        mass.append(float(img.sum(-1).mean()))
        per_layer_img.append(img.mean(0).cpu().numpy())
        del row, img
    last_layer_img = per_layer_img[-1]
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

    result = {
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
    # free the forward's tensors before the next question; the attention tuple is the
    # single largest allocation in this function and Python will not drop it on its own
    # while `out` is still bound in the caller's frame.
    del out, sim, vn, Wn, emb0, vis, txt, per_layer_img
    torch.cuda.empty_cache()
    return result


def main(cfg: Config, arm: str) -> dict:
    """Run ONE arm and write its own JSON.

    🔴 One arm per PROCESS, not per loop iteration. Four 8B models in one process does not
    fit: ``del model`` + ``empty_cache()`` is not enough because PEFT wraps the base and
    nothing is collected until Python's GC runs, so arm 2 loads its 16 GB beside arm 1's and
    OOMs on a 32 GB card (measured 2026-08-08 — arm 1 finished all 12 questions, arm 2 died
    at load). Process isolation is the only teardown that is actually guaranteed, and it has
    a second benefit: a crash in arm 3 no longer destroys arms 1 and 2.
    """
    import sys

    sys.path.insert(0, str(cfg.repo / "src"))

    adapter = cfg.arms[arm]
    rows = select_questions(cfg)
    out_dir = Path(cfg.out)
    (out_dir / "heat").mkdir(parents=True, exist_ok=True)
    print(f"== {arm} == {adapter or cfg.base} | {len(rows)} questions", flush=True)

    model, processor = load_arm(cfg, adapter)
    per_q = []
    for i, r in enumerate(rows.itertuples()):
        img = _frame(cfg, r.ds, r.video, int(r.fi))
        res = probe_one(cfg, model, processor, img, r.question)
        np.save(out_dir / "heat" / f"{arm}_q{i:02d}.npy", res.pop("heat"))
        np.save(out_dir / "heat" / f"{arm}_q{i:02d}_last.npy", res.pop("heat_last_layer"))
        res.update({"i": i, "ds": r.ds, "ood": bool(r.is_ood), "gold": int(r.gold)})
        per_q.append(res)
        print(f"  q{i:02d} {r.ds:<8} ood={bool(r.is_ood)!s:<5} gold={int(r.gold):>2} | "
              f"img_tok={res['n_image_tokens']:>3} | "
              f"visual_mass mean={res['visual_mass_mean']:.4f} "
              f"last={res['visual_mass_last_layer']:.4f} | "
              f"|v|={res['emb_norm_visual']:.2f} |t|={res['emb_norm_text']:.2f}", flush=True)

    agg = {
        "adapter": str(adapter) if adapter else None,
        "visual_mass_mean": float(np.mean([q["visual_mass_mean"] for q in per_q])),
        "visual_mass_last_layer": float(np.mean([q["visual_mass_last_layer"] for q in per_q])),
        "visual_mass_curve": np.mean([q["visual_mass_per_layer"] for q in per_q], axis=0).tolist(),
        "emb_norm_visual": float(np.mean([q["emb_norm_visual"] for q in per_q])),
        "emb_norm_text": float(np.mean([q["emb_norm_text"] for q in per_q])),
        "emb_cos_visual_to_text": float(np.mean([q["emb_cos_visual_to_text_mean"] for q in per_q])),
        "nearest_vocab_tokens": per_q[0]["nearest_vocab_tokens"],
        "n_image_tokens": per_q[0]["n_image_tokens"],
    }
    result = {"arm": arm, "config": {"n_questions": cfg.n_questions,
                                     "max_pixels": cfg.max_pixels, "seed": cfg.seed},
              "agg": agg, "per_question": per_q}
    (out_dir / f"RESULTS_{arm}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n" + json.dumps(agg, indent=2)[:1200], flush=True)
    return result


def merge(cfg: Config) -> dict:
    """Combine the per-arm JSONs into the one comparable artifact."""
    out_dir = Path(cfg.out)
    per_arm = {}
    for arm in cfg.arms:
        p = out_dir / f"RESULTS_{arm}.json"
        if p.exists():
            per_arm[arm] = json.loads(p.read_text(encoding="utf-8"))["agg"]
        else:
            print(f"⚠️  {arm} missing — not merged", flush=True)
    merged = {"arms_present": list(per_arm), "per_arm": per_arm}
    (out_dir / "RESULTS_attention.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged
