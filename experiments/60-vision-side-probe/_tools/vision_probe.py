"""Rung 60 — is the identity missing from the ENCODER, or lost on the BRIDGE?

Rung 59b probed the last-prompt-token hidden state and found class identity is NOT there:
0.5243 at its best layer against the head's own 0.6752, flat learning curve, losing at all
nineteen layers. That measurement has a blind spot it cannot see past, and this rung is
that blind spot: **every probe this campaign has run reads the LANGUAGE side, at a TEXT
position.** Nobody has read the visual tokens.

So "the identity is not in the representation" is really "the identity is not in the
representation *at the last text token, after the bridge*". Two very different worlds are
compatible with that:

  encoder HAS it, text position does not   ->  the loss is the BRIDGE (merger / DeepStack),
                                               and the bridge is trainable
  encoder does NOT have it either          ->  the loss is the ViT, and that is the
                                               backbone ceiling

The two prescriptions are opposite, which is why this is worth one GPU pass.

## Read points

`encoder`   the vision stack's OWN output — the merger result, before a single LLM layer
            touches it. This is the pure "does the ViT+merger hold it" test.
`visual@L`  the visual token POSITIONS inside the LLM at each kept layer. Gives the decay
            profile: identity present at the encoder and gone by layer 20 is a bridge
            story; absent everywhere is a ViT story.

Both are compared against rung 59b's `last-prompt@L` on the SAME rows, the same split and
the same probe class, or the comparison prices the probe instead of the read point.

## Pooling

🔴 MAX over visual tokens, not mean, is the primary. "Is class X present" is an OR over
patches — one patch showing a needle makes the answer yes, and averaging 1,000 patches
buries that one under the tissue. Mean is reported beside it because it is what the
last-prompt probe implicitly compares against, and because a max/mean gap is itself
informative: identity that survives max and dies under mean is LOCAL, which is what a
foreign object in a laparoscopic frame actually is.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STORE = Path(os.environ.get("RUNG60_STORE", "/mnt/storage/uaq_user"))
REPO = Path(os.environ.get("RUNG60_REPO", str(STORE / "repo_rod")))
_T = str(REPO / "experiments/59-cardinality-decoder/_tools")


@dataclass
class Config:
    out_dir: Path = Path(os.environ.get("RUNG60_OUT", str(STORE / "rung60/runs/60_vision_v1")))
    layer_stride: int = 4          # coarser than 59b: the question is the PROFILE, not the peak
    smoke: bool = True
    smoke_rows: int = 24


def _paths() -> None:
    for p in (str(REPO / "src"), str(REPO / "vendor/orena-focus/src"), _T):
        if p not in sys.path:
            sys.path.insert(0, p)


def _visual_module(model):
    """`model.visual`, wherever PEFT has buried it. Raises rather than guessing wrong."""
    for path in ("visual", "model.visual", "base_model.model.visual",
                 "base_model.model.model.visual"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    raise AttributeError("no `visual` submodule found; the probe would read the wrong tensor")


def dump(cfg: Config) -> Path:
    """One forward per question, capturing the encoder output AND the visual positions."""
    _paths()
    import pandas as pd
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    import identity_probe as IP
    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    icfg = IP.Config(smoke=cfg.smoke, smoke_rows=cfg.smoke_rows)
    df, names = IP.enumeration_slice(icfg)
    idx = {n: i for i, n in enumerate(names)}

    eng = QwenFrameEngine(BaselineConfig(model_path=icfg.base_model, max_pixels=icfg.max_pixels,
                                         max_new_tokens=64))
    eng.load()
    from peft import PeftModel
    eng.model = PeftModel.from_pretrained(eng.model, str(icfg.adapters)).eval()

    visual, where = _visual_module(eng.model)
    print(f"visual module at `{where}`", flush=True)
    grabbed: dict = {}

    def _hook(_m, _i, out):
        # Qwen3-VL's visual stack returns `BaseModelOutputWithDeepstackFeatures`, not a
        # tensor: DeepStack wires extra features into the first LLM layers ALONGSIDE the
        # main output, so "the encoder output" is two things and picking one is a choice.
        # `last_hidden_state` is the merged stream that becomes the image tokens, which is
        # the one the bridge question is about.
        grabbed["enc"] = getattr(out, "last_hidden_state", None)
        if grabbed["enc"] is None:
            grabbed["enc"] = out[0] if isinstance(out, (tuple, list)) else out

    handle = visual.register_forward_hook(_hook)

    # the id of the image-placeholder token, so visual POSITIONS can be found in input_ids
    cfgobj = eng.model.config
    img_id = getattr(cfgobj, "image_token_id", None)
    if img_id is None:
        img_id = eng.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"image token id = {img_id}", flush=True)

    enc_max, enc_mean, vis_max, vis_mean, golds, meta, kept = [], [], [], [], [], [], None
    try:
        for i, r in enumerate(df.itertuples(), 1):
            image = Image.open(r.frame).convert("RGB")
            messages = eng._messages(image, r.question)
            text = eng.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            im_in, vid_in = process_vision_info(messages)
            inputs = eng.processor(text=[text], images=im_in, videos=vid_in, padding=True,
                                   return_tensors="pt").to(eng.model.device)
            grabbed.clear()
            with torch.no_grad():
                out = eng.model(**inputs, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            if kept is None:
                kept = list(range(0, len(hs), cfg.layer_stride))
                if len(hs) - 1 not in kept:
                    kept.append(len(hs) - 1)

            enc = grabbed.get("enc")
            if enc is None:
                raise RuntimeError("the forward hook never fired — no encoder output captured")
            e = enc.reshape(-1, enc.shape[-1]).float()
            enc_max.append(e.max(0).values.cpu().numpy().astype("float16"))
            enc_mean.append(e.mean(0).cpu().numpy().astype("float16"))

            mask = (inputs.input_ids[0] == img_id)
            if not bool(mask.any()):
                raise RuntimeError(f"no image tokens in the sequence for {r.qID}")
            vis_max.append(np.stack([hs[l][0][mask].float().max(0).values.cpu().numpy()
                                     for l in kept]).astype("float16"))
            vis_mean.append(np.stack([hs[l][0][mask].float().mean(0).cpu().numpy()
                                      for l in kept]).astype("float16"))

            y = np.zeros(len(names), dtype="int8")
            for c in r.gold_set:
                if c in idx:
                    y[idx[c]] = 1
            golds.append(y)
            meta.append({"qID": r.qID, "video": r.video, "dataset": r.dataset,
                         "gold": "|".join(sorted(r.gold_set)), "our_answer": r.our_answer,
                         "n_visual_tokens": int(mask.sum())})
            if i % 100 == 0:
                print(f"{i}/{len(df)}", flush=True)
    finally:
        handle.remove()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if cfg.smoke else "full"
    for nm, arr in (("encmax", enc_max), ("encmean", enc_mean),
                    ("vismax", vis_max), ("vismean", vis_mean)):
        np.save(out_dir / f"{nm}_{tag}.npy", np.stack(arr))
    np.save(out_dir / f"gold_{tag}.npy", np.stack(golds))
    pd.DataFrame(meta).to_csv(out_dir / f"meta_{tag}.csv", index=False)
    (out_dir / f"layers_{tag}.json").write_text(json.dumps(kept), encoding="utf-8")
    (out_dir / f"classes_{tag}.json").write_text(json.dumps(list(names)), encoding="utf-8")
    print(f"dumped {len(golds)} rows | encoder dim {enc_max[0].shape[0]} | "
          f"layers {kept} | visual tokens/row median "
          f"{int(np.median([m['n_visual_tokens'] for m in meta]))}")
    return out_dir


def _fit_one(F, Y, fit_i, read_i, names, gold_sets):
    """The rung-59b probe, unchanged, so only the read point differs."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_comp = int(min(128, fit_i.sum() - 1, F.shape[1]))
    pred = np.zeros((len(F), len(names)), dtype="int8")
    for ci in range(len(names)):
        y = Y[:, ci]
        if y[fit_i].sum() == 0:
            continue
        if y[fit_i].sum() == fit_i.sum():
            pred[:, ci] = 1
            continue
        clf = make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=0),
                            LogisticRegression(max_iter=2000, C=1e-4, class_weight="balanced"))
        clf.fit(F[fit_i], y[fit_i])
        pred[:, ci] = clf.predict(F)
    sets = [frozenset(n for n, v in zip(names, pred[i]) if v) for i in range(len(F))]
    return float(np.mean([sets[i] == gold_sets[i] for i in np.where(read_i)[0]]))


def fit(cfg: Config) -> dict:
    _paths()
    import pandas as pd

    from frame.metrics import read_fo_class

    tag = "smoke" if cfg.smoke else "full"
    d = Path(cfg.out_dir)
    Y = np.load(d / f"gold_{tag}.npy")
    meta = pd.read_csv(d / f"meta_{tag}.csv")
    layers = json.loads((d / f"layers_{tag}.json").read_text())
    names = json.loads((d / f"classes_{tag}.json").read_text())
    valid_lower = {n.lower(): n for n in names}

    is_ood = (meta.dataset == "heico").to_numpy()
    fit_i, read_i = ~is_ood, is_ood
    gold_sets = [frozenset(g.split("|")) if isinstance(g, str) and g else frozenset()
                 for g in meta.gold]
    emitted = [read_fo_class(a, valid_lower) or frozenset() for a in meta.our_answer]
    model_exact = float(np.mean([emitted[i] == gold_sets[i] for i in np.where(read_i)[0]]))

    rows = []
    for pool in ("max", "mean"):
        E = np.load(d / f"enc{pool}_{tag}.npy").astype("float32")
        acc = _fit_one(E, Y, fit_i, read_i, names, gold_sets)
        rows.append({"read_point": f"encoder[{pool}]", "layer": None, "probe_exact_OOD": round(acc, 4)})
        print(f"encoder[{pool}]           {acc:.4f}   model {model_exact:.4f}   "
              f"delta {acc - model_exact:+.4f}", flush=True)
        V = np.load(d / f"vis{pool}_{tag}.npy")
        for li, layer in enumerate(layers):
            acc = _fit_one(V[:, li, :].astype("float32"), Y, fit_i, read_i, names, gold_sets)
            rows.append({"read_point": f"visual[{pool}]", "layer": layer,
                         "probe_exact_OOD": round(acc, 4)})
            print(f"visual[{pool}] layer {layer:>3}   {acc:.4f}   model {model_exact:.4f}   "
                  f"delta {acc - model_exact:+.4f}", flush=True)

    res = {"model_exact_OOD": round(model_exact, 4),
           "last_prompt_best_rung59b": 0.5243,
           "n_fit": int(fit_i.sum()), "n_read": int(read_i.sum()), "rows": rows}
    (d / f"RESULTS_vision_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(d / f"RESULTS_vision_{tag}.csv", index=False)
    return res


if __name__ == "__main__":
    c = Config(smoke=os.environ.get("RUNG60_SMOKE", "1") == "1")
    dump(c)
    fit(c)
