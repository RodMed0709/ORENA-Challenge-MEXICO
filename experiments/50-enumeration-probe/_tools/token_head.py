"""Rung 50 -- the token head's OWN confidence, read two different ways for the two tasks.

`number`: rung 33's exact mechanism, reused unchanged in spirit -- proper P(value=1..12),
with the "1" vs "10/11/12" continuation split, not naive first-token logits (which would
silently conflate 1 with 10-12, the trap rung 33's own docstring names).

`fo_class`: rung 33's mechanism does not port -- class names are multi-token, open-ended
free text, not a fixed small single-token vocabulary. Deliberately simpler here: the joint
probability of the model's OWN greedily-generated answer (product of per-step token
probabilities along the sequence it actually produced), a general-purpose "how sure was the
model of what it said" signal that needs no per-class token bookkeeping. This is a real
simplification, not an oversight -- see the module docstring below for what it does and does
not capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: gold support measured on the scored eval (rung 33): 1..12, and 0 never occurs as gold
VALUES = list(range(0, 13))


def _single_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise AssertionError(f"{s!r} encodes to {len(ids)} tokens ({ids})")
    return ids[0]


def dump_number_distribution(model_path, manifest, *, max_pixels: int = 1280 * 720) -> "pd.DataFrame":
    """Per `number` row: P(value=0..12), properly split for the "1" vs "10/11/12" ambiguity.
    `manifest` needs `qID`, `image_path`, `question`. Mirrors rung 33's `logit_dump.run`.
    """
    import sys
    for p in ("/workspace/repo/src", "/workspace/repo/vendor/orena-focus/src"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pandas as pd
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    eng = QwenFrameEngine(BaselineConfig(model_path=model_path, max_pixels=max_pixels,
                                         max_new_tokens=2))
    eng.load()
    tk = eng.processor.tokenizer
    dig = {d: _single_token(tk, str(d)) for d in range(10)}
    dig_ids = torch.tensor([dig[d] for d in range(10)], device=eng.model.device)

    rows = []
    for i, r in enumerate(manifest.itertuples(), 1):
        with Image.open(r.image_path) as im:
            image = im.convert("RGB")
        messages = eng._messages(image, r.question)
        text = eng.processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        im_in, vid_in = process_vision_info(messages)
        inputs = eng.processor(text=[text], images=im_in, videos=vid_in, padding=True,
                               return_tensors="pt").to(eng.model.device)
        with torch.no_grad():
            g = eng.model.generate(**inputs, max_new_tokens=2, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True)
        p1 = torch.softmax(g.scores[0][0].float(), dim=-1)
        first = int(g.sequences[0, inputs.input_ids.shape[1]])
        pv = {d: float(p1[dig[d]]) for d in range(10)}
        p_one = pv[1]
        one_split_exact = first == dig[1]
        if one_split_exact:
            p2 = torch.softmax(g.scores[1][0].float(), dim=-1)
            cont = float(p2[dig_ids].sum())
            pv[1] = p_one * (1.0 - cont)
            for d in (0, 1, 2):
                pv[10 + d] = p_one * float(p2[dig[d]])
        else:
            for d in (0, 1, 2):
                pv[10 + d] = 0.0
        tot = sum(pv[v] for v in VALUES)
        legal = [v for v in VALUES if v != 0]
        ranked_legal = sorted(legal, key=lambda v: -pv[v])
        rows.append({
            "qID": r.qID,
            "token_head_pred": ranked_legal[0],
            "token_head_conf": pv[ranked_legal[0]] / max(tot, 1e-9),
            "one_split_exact": one_split_exact,
            **{f"p{v}": pv[v] for v in VALUES},
        })
        if i % 200 == 0:
            print(f"{i}/{len(manifest)}", flush=True)
    eng.unload()
    return pd.DataFrame(rows)


def dump_foclass_confidence(model_path, manifest, *, max_pixels: int = 1280 * 720,
                            max_new_tokens: int = 32) -> "pd.DataFrame":
    """Per `fo_class` row: the model's own greedy answer, plus the joint probability of the
    exact token sequence it generated (product of per-step P(chosen token)). `manifest` needs
    `qID`, `image_path`, `question`.

    ⚠️ **What this signal is and is not.** A high joint probability means the model was
    confident in EVERY token of its own answer, including formatting tokens (commas, spaces)
    -- it is not a per-class marginal the way the `number` distribution is, so it cannot be
    decomposed into "P(Clip present)" separately from "P(Sponge present)". It is a single
    scalar confidence per row, which is exactly what the merge rule (probe-confidence vs
    token-head-confidence) needs and all it is claimed to provide.
    """
    import sys
    for p in ("/workspace/repo/src", "/workspace/repo/vendor/orena-focus/src"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pandas as pd
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from frame.config import BaselineConfig
    from frame.engine import QwenFrameEngine

    eng = QwenFrameEngine(BaselineConfig(model_path=model_path, max_pixels=max_pixels,
                                         max_new_tokens=max_new_tokens))
    eng.load()

    rows = []
    for i, r in enumerate(manifest.itertuples(), 1):
        with Image.open(r.image_path) as im:
            image = im.convert("RGB")
        messages = eng._messages(image, r.question)
        text = eng.processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        im_in, vid_in = process_vision_info(messages)
        inputs = eng.processor(text=[text], images=im_in, videos=vid_in, padding=True,
                               return_tensors="pt").to(eng.model.device)
        with torch.no_grad():
            g = eng.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True)
        prompt_len = inputs.input_ids.shape[1]
        gen_ids = g.sequences[0, prompt_len:]
        # per-step P(the token actually chosen) -- `g.scores[t]` is the logits for step t,
        # BEFORE any EOS-trim, so this only walks as many steps as were actually generated
        step_probs = []
        for t, tok_id in enumerate(gen_ids.tolist()):
            if t >= len(g.scores):
                break
            p_t = torch.softmax(g.scores[t][0].float(), dim=-1)[tok_id]
            step_probs.append(float(p_t))
            if tok_id == eng.processor.tokenizer.eos_token_id:
                break
        answer = eng.processor.decode(gen_ids, skip_special_tokens=True).strip()
        log_probs = np.log(np.clip(step_probs, 1e-12, 1.0))
        rows.append({
            "qID": r.qID,
            "token_head_answer": answer[: 300],  # same char cap the SDK enforces downstream
            "token_head_joint_logprob": float(log_probs.sum()),
            "token_head_mean_logprob": float(log_probs.mean()) if len(log_probs) else float("nan"),
            "token_head_conf": float(np.exp(log_probs.mean())) if len(log_probs) else float("nan"),
            "n_gen_tokens": len(step_probs),
        })
        if i % 200 == 0:
            print(f"{i}/{len(manifest)}", flush=True)
    eng.unload()
    return pd.DataFrame(rows)


__all__ = ["dump_number_distribution", "dump_foclass_confidence", "VALUES"]
