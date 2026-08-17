"""SEGMENT inference over the 6,254 test rows. The `predict()` the track never had.

WHY IT REUSES `frame.segment.corpus` INSTEAD OF BUILDING ITS OWN PROMPT
The arm was trained on rows produced by `corpus.build_row`. Any difference here — a different
frame grid, a different `<video>` window string, a different system prompt — is an unmeasured
second variable sitting on top of the number. So the eval calls the SAME builder with the SAME
`ExportConfig` and simply drops the assistant turn. `relative_time=True` is NOT a choice: it is
what the arm was trained under, and turning it off here would ask the model for a clock it was
never taught.

THE TIME INVERSION, WHICH IS 38.2 % OF THE TRACK
`corpus.build_row` rewrites a `2a` gold to an offset from the clip start (`00:49:31` with a clip
starting `00:49:28` becomes `00:00:03`) and leaves `2b` elapsed spans alone. The model therefore
emits RELATIVE time, and the reference it is scored against is ABSOLUTE. This module adds the
clip start back before scoring — and only for `2a`. Offsetting a `2b` span would be a category
error, which is why `time_kind` is read from the row rather than inferred here.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


def load_rows(data_root: str, cache: str, limit: int = 0):
    """Build every test row exactly as training built them, minus the answer."""
    from frame.segment import corpus as C

    cfg = C.ExportConfig(data_root=Path(data_root), cache=Path(cache), splits=("test",))
    df = C.load(cfg)
    sysmsg = C.system_prompt()
    rows = []
    for r in df.itertuples():
        rec = C.build_row(cfg, r, sysmsg)
        m = rec["_meta"]
        rows.append({
            "qid": m["qid"], "frames": rec["videos"][0],
            "user": rec["messages"][1]["content"],
            "system": rec["messages"][0]["content"],
            "raw_fps": rec["chat_template_kwargs"].get("raw_fps", 1.0),
            "clip_start_s": m["clip_start_s"], "time_kind": m["time_kind"],
            "answer_format": m["answer_format"], "primary": str(m["primary"]),
            "gold": m["gold_original"], "ds": m["ds"], "video": m["video"], "K": m["K"],
        })
    rows.sort(key=lambda x: x["qid"])          # deterministic; no Date/random anywhere
    if limit:
        # stratified smoke: interleave the two datasets so it is never one prefix (RULES §8)
        out = []
        for ds in ("heico", "lapchole"):
            sub = [x for x in rows if x["ds"] == ds]
            step = max(1, len(sub) // max(1, limit // 2))
            out.extend(sub[::step][: limit // 2])
        rows = sorted(out, key=lambda x: x["qid"])
    return rows


def to_absolute(pred: str, clip_start_s: float, time_kind: str | None) -> str:
    """Undo `corpus`'s relative rewrite. `2b` spans pass through untouched."""
    from frame.segment.corpus import seconds_to_ts, ts_to_seconds

    if time_kind != "2a_relative":
        return pred
    parts = [p.strip() for p in pred.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(seconds_to_ts(ts_to_seconds(p) + clip_start_s))
        except Exception:
            return pred            # unparseable: hand it on untouched and let Time.verify judge
    return ", ".join(out)


class SegEngine:
    """Qwen3-VL over a list of frames, LoRA applied at load. No merge, no 17 GB written."""

    def __init__(self, base: str, adapters: list[str], max_pixels: int):
        """`adapters` are applied IN ORDER, each merged before the next is loaded.

        🔴 The order is not cosmetic and the stack is not optional. Arm A was trained on top of
        the MERGED rung-21 A2 checkpoint, and that merged directory was deleted from the volume
        to reclaim disk — only the two adapters survive. Applying A2, merging it into the
        weights, then applying arm A reproduces exactly what arm A saw. Loading them as two
        live PEFT adapters instead would compose them additively against the BASE, which is a
        different model. Merging in memory also avoids writing the ~17 GB the merged
        checkpoint would cost on a volume that has already been cleaned once.
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(base, max_pixels=max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base, dtype=torch.bfloat16, device_map="cuda").eval()
        from peft import PeftModel
        for i, a in enumerate(adapters):
            self.model = PeftModel.from_pretrained(self.model, a)
            if i < len(adapters) - 1:
                self.model = self.model.merge_and_unload()   # fold in, free the wrapper
        self.model = self.model.eval()
        self.model.generation_config.max_length = None

    @torch.no_grad()
    def answer(self, row: dict, max_new_tokens: int = 32) -> str:
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in row["frames"]]
        # `<video>` in the corpus's user string is ms-swift's tag. For HF the frames go in as
        # a video (a list of images) so the temporal patching matches training.
        text = row["user"].replace("<video>", "")
        msgs = [{"role": "system", "content": [{"type": "text", "text": row["system"]}]},
                {"role": "user", "content": [{"type": "video"},
                                             {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[prompt], videos=[imgs], return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.proc.decode(gen, skip_special_tokens=True).strip()


def run(base: str, adapters: list[str], out_dir: str, data_root: str, cache: str,
        limit: int = 0, max_pixels: int = 921_600, chunk: int = 100) -> dict:
    """Chunked and RESUMABLE: a killed run continues from the last checkpoint, never restarts."""
    from frame.parsing import normalize_answer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(data_root, cache, limit)
    done = {}
    ck = out / "predictions.json"
    if ck.exists():
        done = {d["qid"]: d for d in json.loads(ck.read_text())}
    todo = [r for r in rows if r["qid"] not in done]
    print(f"rows={len(rows)} done={len(done)} todo={len(todo)}", flush=True)

    eng = SegEngine(base, adapters, max_pixels) if todo else None
    t0 = time.perf_counter()
    for i, r in enumerate(todo, 1):
        raw = eng.answer(r)
        # order matters: absolute FIRST (the model speaks relative), then the format repair,
        # because `normalize_answer` is what makes `Time.verify` accept a trailing period.
        pred = normalize_answer(to_absolute(raw, r["clip_start_s"], r["time_kind"]))
        done[r["qid"]] = {"qid": r["qid"], "raw": raw, "pred": pred, "gold": r["gold"],
                          "answer_format": r["answer_format"], "primary": r["primary"],
                          "ds": r["ds"], "video": r["video"], "K": r["K"],
                          "time_kind": r["time_kind"], "clip_start_s": r["clip_start_s"]}
        if i % chunk == 0 or i == len(todo):
            tmp = out / ".predictions.tmp"
            tmp.write_text(json.dumps(list(done.values())))
            tmp.replace(ck)
            el = time.perf_counter() - t0
            print(f"  {i}/{len(todo)}  {el/i:.2f} s/q  eta {(len(todo)-i)*el/i/60:.0f} min",
                  flush=True)
    return {"n": len(done), "out": str(ck)}
