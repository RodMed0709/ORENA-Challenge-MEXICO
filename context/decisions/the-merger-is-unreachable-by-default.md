---
question: Rung 32 measured that `--freeze_aligner false` adds 0 aligner tensors. Is that an ms-swift defect, and would another trainer reach the ViT→LLM connector?
verdict: Not a defect, and no. The merger is unreachable by DEFAULT in BOTH trainers, measured in each. The cause is structural: `model.visual.merger.linear_fc{1,2}` carries no attention or MLP token in its path, and every generic matcher — ms-swift's `all-linear` and Unsloth's `finetune_vision_layers` regex alike — requires one. Reaching the connector needs an EXPLICIT target, in any framework. This is a mechanism for rung 32's null, and it makes the connector rung's reachability gate mandatory rather than cautious.
status: SETTLED
kind: two runtime measurements + source read, zero training GPU
date: 2026-08-12
measured_in: experiments/32-aligner-unfreeze/RESULTS_reachability.csv (ms-swift) · experiments/38-gen36-ft-screen/RESULTS_smoke_unsloth.json (Unsloth, UNAM) · unsloth_zoo/peft_utils.py:54-144 · module census of Qwen/Qwen3.5-2B on meta device
---

# The ViT→LLM connector is unreachable by default in BOTH trainers, and the reason is structural

- **Status:** SETTLED · 2026-08-12 · **zero training GPU**.
- **Applies when:** designing any rung that intends to train the merger / aligner / connector.
- 🔻 **Amended 2026-08-18 — a source read now CONFLICTS with the ms-swift leg of this note.** In `ms_swift 4.4.1` the connector IS registered as `model_arch.aligner` and `--freeze_aligner false` is supposed to append it to the LoRA target regex, and the leaf matcher carries no `attn`/`mlp` requirement — so the structural cause below explains `all-linear`, but not the zero measured in rung 32. The measurement stands and the explicit `target_modules` stays the only proven route. Open question, mechanism and the zero-GPU check that settles it: [[aligner-flag-reads-reachable-but-measured-zero]]. The **Unsloth** leg is untouched.

## Two independent measurements, same answer

**ms-swift 4.4.1** — `experiments/32-aligner-unfreeze/RESULTS_reachability.csv`, both legs:

```
720 tensors = 504 language_model + 216 vision_tower + 0 aligner, 0 orphans
```

`--freeze_aligner false` changes nothing. The 8 merger `Linear` layers
(`model.visual.merger.linear_fc{1,2}` + `deepstack_merger_list.{0,1,2}.linear_fc{1,2}`) are simply
not selected by `--target_modules all-linear`.

**Unsloth 2026.8.15** — `RESULTS_smoke_unsloth.json`, with `finetune_vision_layers=True`,
`finetune_language_layers=True`, `finetune_attention_modules=True`, `finetune_mlp_modules=True`,
`target_modules="all-linear"`:

```
visual         :  96 LoRA modules   (e.g. visual.blocks.0.attn.proj)
merger         :   0
deepstack      :   0
language_model : 186
```

⚠️ **And the obvious objection was checked before the claim was made.** A module census of
`Qwen/Qwen3.5-2B` on the meta device confirms the connector **exists under the same name**:
`model.visual.merger.linear_fc1` and `linear_fc2`, the only Linears inside `visual` that sit
outside `blocks.N`. So the 0 means *not reached*, not *not present*.

## The mechanism

Unsloth builds its target regex as three chained conditions (`unsloth_zoo/peft_utils.py:126-136`):

```python
regex = r".*?(?:" + vision|image|visual|patch|language|text +
        r").*?(?:" + self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer +
        r").*?"    + <leaf Linear names>
```

PEFT applies it with `re.fullmatch`. Trace `model.visual.merger.linear_fc1`:

| segment | matches |
|---|---|
| `.*?` | `model.` |
| model part | `visual` ✅ |
| **component** | remaining path is `.merger.linear_fc1` — **no `attn`, no `mlp`, no `dense`, no `ffn`** ❌ |

🔑 **`target_modules="all-linear"` is not PEFT's literal all-linear in either framework.** Unsloth
intercepts the string and replaces it with this regex (`unsloth/models/vision.py:1900,1930`);
ms-swift expands its own. Both stop at the same place.

## Consequences

1. **Rung 32's null now has a mechanism**, not just an observation. It was never an ms-swift bug.
2. **A connector rung must pre-register an EXPLICIT `target_modules`** naming
   `model.visual.merger.linear_fc{1,2}` and `deepstack_merger_list.{0,1,2}.linear_fc{1,2}`, plus a
   blocking reachability gate asserting `n_aligner > 0` and `n_orphans == 0` before any GPU is
   spent. Adapt `experiments/32-aligner-unfreeze/_tools/reachability_smoke.py`, which already
   parameterises `target_modules` and `freeze_aligner` — the engine chain does not.
3. **Switching framework does not unlock it.** Anyone proposing Unsloth *in order to* train the
   connector is proposing the wrong reason.
4. ⚠️ **Unsloth constrains explicit `target_modules` too** — `vision.py:1947-1965` intersects a
   user-supplied list with the same flags (*"Explicit target_modules are constrained by the…"*).
   Whether an explicit merger name survives that intersection is **NOT yet measured** in either
   framework, and it is the first thing the connector rung must establish.

## Why this matters beyond plumbing

The merger is not an ordinary projector: DeepStack wires it into the **first 3 LLM layers**
([[viT-swap-nogo]]). It is, structurally, the ViT↔LLM connection — and across 30+ rungs it has
never received a single gradient. That is the largest untouched surface in the campaign, and the
reason it stayed untouched was a regex, not a decision.
