"""Rung 58-D — score the SHIPPED arm end to end through the canonical eval.

58-B measured the arm on `fo_class` alone. A screen of the 58-C answer table then found
that `as_class_set` also fires on 18 of 101 `open_ended` answers ("Sponge.",
"specimen, specimen bag.") -- a format the container CANNOT exclude, because `Request`
carries no answer_format. Those 18 are judged by the LLM judge, so the arm's effect there
was measured by nothing. This runs the real two-model logic through `run_baseline`, which
does the judging, and returns `bucket_mean` for the arm on the same 1,283 questions
58-C scored at 0.6722.

The arbitration function is READ FROM THE CONTAINER's inference.py by AST, not copied --
so what is scored here is literally what ships.
"""
from __future__ import annotations
import ast, json, os, re, sys, time
from pathlib import Path

S = Path(os.environ.get("STORAGE", "/mnt/storage/uaq_user"))
W = S / "rung58"; REPO = S / "repo_rod"
SUB = W / "container"          # inference.py copied up from the build box
sys.path.insert(0, str(REPO / "experiments" / "45-gen36-data-and-reg" / "_tools"))
from eval_arm45 import EvalConfig, held_out_split, build_baseline_config, ensure_paths
ensure_paths(str(REPO))

os.environ.setdefault("HF_HOME", str(S / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── the container's own functions, by AST, from the file that ships ──────────────
_NS = {"re": re}
for _n in ast.parse((SUB / "inference.py").read_text()).body:
    if isinstance(_n, ast.FunctionDef) and _n.name in ("as_class_set", "arbitrate"):
        exec(compile(ast.Module(body=[_n], type_ignores=[]), "inference.py", "exec"), _NS)
assert {"as_class_set", "arbitrate"} <= set(_NS), "container functions not found"
as_class_set, arbitrate = _NS["as_class_set"], _NS["arbitrate"]

from frame.engine import QwenFrameEngine
from frame.metrics import _load_fotype
LEGAL = {n.lower(): n for n in _load_fotype().names()}

FIRED = {"n": 0, "changed": 0}

class PairEngine:
    """model A answers; where the answer parses as a class list, model B is asked and the
    shorter list wins. Same rule, same order, same tie-break as the container."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.a = QwenFrameEngine(cfg)
        cfg_b = build_baseline_config(CFG_B)
        self.b = QwenFrameEngine(cfg_b)
    def load(self):
        self.a.load(); self.b.load()
    def unload(self):
        for e in (self.a, self.b):
            try: e.unload()
            except Exception: pass
    def predict(self, image, question):
        out_a = self.a.predict(image, question)
        if as_class_set(out_a, LEGAL) is None:
            return out_a
        FIRED["n"] += 1
        out_b = self.b.predict(image, question)
        merged, changed = arbitrate(out_a, out_b, LEGAL)
        FIRED["changed"] += int(changed)
        return merged

def mk(arm, merged_dir, run_name):
    return EvalConfig(arm=arm, merged_dir=str(merged_dir), out_dir=str(W / "pair"),
                      run_name=run_name, data_root=str(S / "orena-data"),
                      repo_root=str(REPO), frames_cache=str(S / "frames_cache"),
                      split_json=str(REPO / "experiments" / "42-merged-corpus" / "RESULTS_split_42.json"),
                      use_vllm=False)

CFG_A = mk("pair_arm", W / "merged_r42", "58D_pair")
CFG_B = mk("pair_b", W / "merged_r42_ep2", "58D_pair_b")

t0 = time.time()
split = held_out_split(CFG_A)
cfg_eval = build_baseline_config(CFG_A)
cfg_eval.engine_factory = PairEngine
from frame.run import run_baseline
from frame.metrics import leaderboard_proxy
report = run_baseline(cfg_eval, video_filter=split["held_videos"])
report.update(leaderboard_proxy(report)); report["proxy_leaderboard"] = report.pop("proxy")
out = {"bucket_mean_ARM": report.get("bucket_mean"),
       "bucket_mean_CONTROL_58C": 0.6721777263658064,
       "delta": round(report.get("bucket_mean", 0) - 0.6721777263658064, 4),
       "proxy_leaderboard": report.get("proxy_leaderboard"),
       "rule_fired_on": FIRED["n"], "answers_changed": FIRED["changed"],
       "acc_ID": report.get("acc_ID"), "acc_OOD": report.get("acc_OOD"),
       "object_recognition_ID": report.get("object_recognition_ID"),
       "aggregation_ID": report.get("aggregation_ID"),
       "minutes": round((time.time() - t0) / 60, 1)}
(W / "RESULTS_58D_pair.json").write_text(json.dumps(out, indent=1, default=str))
print("\n=== 58D ARM ===\n" + json.dumps(out, indent=1, default=str), flush=True)
