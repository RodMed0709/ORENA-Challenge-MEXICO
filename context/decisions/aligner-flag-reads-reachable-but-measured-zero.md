---
question: ms-swift 4.4.1 registers the merger as the `aligner` and builds a LoRA target regex from it when `--freeze_aligner false`. So why did rung 32 measure ZERO aligner tensors with exactly that flag?
verdict: RESOLVED THE SAME DAY, AND IT IS VERSION-DEPENDENT. Measured on the UNAM box with ms_swift 4.4.1 + transformers 5.12.1: `--freeze_aligner false` DOES reach the connector — 368 modules matched vs 360 with it true, and the 8 extra are exactly `model.visual.merger.linear_fc{1,2}` plus the three `deepstack_merger_list` blocks. Rung 32 measured 0 on transformers 4.57. Both stand: the likely mechanism is the silent skip below, firing on 4.57 where the module hung at a different path. ⇒ the flag is a REAL route on 5.x and NOT on the stack the ladder was trained on. Original reading kept below.  The source path says the connector SHOULD be reached — `model_arch.py:596` registers `aligner=['model.visual.merger', 'model.visual.deepstack_merger_list']`, `pipelines/train/tuner.py:98` routes `all-linear` on a multimodal model into `get_multimodal_target_regex`, and that builder appends `model_arch.aligner` when `freeze_aligner` is false. Rung 32 ran that exact argv on that exact version and got 720 = 504 llm + 216 vit + 0 aligner. One candidate mechanism is a SILENT SKIP at `transformers_utils.py:231-234`, where a module whose path does not resolve is logged and dropped. Until that is checked, the explicit `target_modules` of rungs 39/42 stays the only PROVEN route
status: SETTLED
kind: source read only, zero GPU — conflicts with an existing measurement
date: 2026-08-18
measured_in: ms_swift 4.4.1 at ~/storage/envs/orena-gen36 on the UNAM box (model_arch.py:592-598, pipelines/train/tuner.py:91-108, utils/transformers_utils.py:178-252) vs experiments/32-aligner-unfreeze/RESULTS_reachability.csv
---

# The aligner flag reads as reachable in the source and measured zero on the machine

- **Status:** 🟢 **SETTLED** · 2026-08-18 · **zero GPU** (meta-device load, seconds).
- 🔑 **The measurement that closed it**, on `orena-train` (ms_swift 4.4.1, transformers 5.12.1),
  `get_multimodal_target_regex(model, freeze_llm=False, freeze_vit=False, freeze_aligner=…)`
  matched against the real module names of Qwen3-VL-8B:

  | `freeze_aligner` | modules matched | of them, connector |
  |---|---|---|
  | `True` | 360 | **0** |
  | `False` | **368** | **8** — `model.visual.merger.linear_fc{1,2}` + `deepstack_merger_list.{0,1,2}.linear_fc{1,2}` |

  ⇒ **the flag works on transformers 5.x.** Rung 32's zero was measured on **4.57**, and the
  silent-skip mechanism described below is the likely cause there: if the connector hung at
  `visual.merger` rather than `model.visual.merger` under that version, `deep_getattr` returned
  None and the module was dropped with a warning.
- ⚠️ **Consequence, and it is not "use the flag":** the 8B ladder is trained on 4.57, where the
  flag is measured NOT to work. Anything that means to train the connector **on the ladder's
  stack** still needs the explicit `target_modules`. The flag becomes available only if the whole
  line moves to 5.x — which rung 47 is the first run to test.
- 🟢 Corroborated independently by rung 47's own smoke on 5.12.1: with `freeze_vit false,
  freeze_aligner true` the resolved LoRA regex carries a negative lookahead
  `(?!(model.visual.merger|model.visual.deepstack_merger_list))`, i.e. the builder is actively
  *excluding* the connector — it can see it.
- **Applies when:** designing any rung that intends to train the merger / aligner / connector, and
  before anyone "simplifies" rung 42's hand-written `target_modules` into a single flag.
- **Extends** [[the-merger-is-unreachable-by-default]]. It does not overturn it.

## What the source says

`swift/model/model_arch.py:592-598` — Qwen3-VL is registered with the connector named explicitly:

```python
language_model = ['model.language_model', 'lm_head']
aligner        = ['model.visual.merger', 'model.visual.deepstack_merger_list']
vision_tower   = 'model.visual'
```

That is **exactly the module set rung 42 wrote by hand** into `target_modules`.

`swift/pipelines/train/tuner.py:91-108` — `get_target_modules` sends `all-linear` on a multimodal
model into `get_multimodal_target_regex(model, freeze_llm=…, freeze_vit=…, freeze_aligner=…)`, and
that builder (`utils/transformers_utils.py:208-252`) does:

```python
if not freeze_aligner:
    modules += model_arch.aligner
```

And the v4.4 parameter reference agrees in prose: *"For multimodal models, tuners attach only to
the LLM component by default; control via freeze flags."* Note also that the leaf matcher
`find_all_linears` (`transformers_utils.py:178-205`) accepts **any** module whose class name
contains `linear`, with **no `attn`/`mlp` requirement** — so the structural cause named in
[[the-merger-is-unreachable-by-default]] for `all-linear` does not, by itself, explain a zero here.

## What the machine said

`experiments/32-aligner-unfreeze/RESULTS_reachability.csv`, both legs, same version:

| leg | `freeze_aligner` | total | llm | vit | **aligner** | passes |
|---|---|---|---|---|---|---|
| unfrozen | `false` | 720 | 504 | 216 | **0** | **False** |
| control_frozen | `true` | 720 | 504 | 216 | 0 | True |

The argv is `--tuner_type lora --target_modules all-linear --freeze_aligner false`
(`_tools/reachability_smoke.py:117-126`). 216 is exactly 108 ViT `Linear` × 2 LoRA tensors, so the
merger is genuinely **absent** — not misfiled into the ViT column.

## The candidate mechanism, and how to settle it

`utils/transformers_utils.py:231-234`, inside the loop over modules:

```python
sub_module = deep_getattr(model, module)
if sub_module is None:
    logger.warning(f'module: {module} is None')
    continue          # the aligner is dropped, silently, and training proceeds
```

If `model.visual.merger` does not resolve on the loaded object — plausible, since where `visual`
is nested moved between `transformers` versions — the connector is skipped with a **warning, not an
error**, and the run looks healthy.

⇒ **Check rung 32's saved log for the literal string `is None`.** Zero GPU. If it fires, the flag
is not broken but *fragile in a version-dependent way*, and the fix is a reachability assertion, not
a new flag. If it does NOT fire, the source reading is wrong somewhere else and the explicit
`target_modules` is the only route, full stop.

## What is safe to act on TODAY

1. **Keep rungs 39/42's explicit `target_modules`.** It is the only route with a measurement behind
   it. Do not swap it for `--freeze_aligner false` on the strength of this note.
2. **Any rung that means to train the connector still needs the blocking reachability gate** that
   [[the-merger-is-unreachable-by-default]] made mandatory. This note strengthens that requirement.
3. ⚠️ **The 27B cannot inherit either route unchanged** — it has no `deepstack_merger_list`, so its
   connector is 2 `Linear`, not 8, and any copied `target_modules` list is wrong for it.

## What this does NOT say

It does not say ms-swift has a bug — nobody has read the log. It does not say the flag works. And
it does not reopen the framework-swap question: [[the-merger-is-unreachable-by-default]] measured
Unsloth independently and the answer there was also zero.
