# The connector IS reachable — via `modules_to_save`, not via `target_modules`

**SETTLED 2026-08-13.** Zero GPU, zero pod. Desk research against PEFT/Unsloth source + official
docs, plus a module census read from the model's own weight index.
Supersedes nothing; it **answers the open question** left by
[[the-merger-is-unreachable-by-default]] §4.

---

## Question

The ViT→LLM connector has never received a gradient in 30+ rungs. Is that fixable with the
documented API, and how — on the **Qwen3.6-27B** (`Qwen3_5ForConditionalGeneration`,
`model_type: qwen3_5`) under **Unsloth**?

## What we sought

A path that reaches the connector **without losing coverage of anything else**, backed by official
documentation or library source — not by a guess that costs ~6 h of GPU to disprove.

## What it gave us

### 1. The census — the connector is exactly 2 Linear layers, and they are the only 2 LoRA misses

Read from `model.safetensors.index.json`, no framework needed:

| part | module names | Linear |
|---|---|---|
| vision tower | `model.visual.blocks.N.{attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2}` × 27 | 108 ✅ reached |
| LLM | 16 × `self_attn` (q/k/v/o_proj) + **48 × `linear_attn`** (`in_proj_a/b/qkv/z`, `out_proj`) + 64 × `mlp` | 496 ✅ reached |
| **connector** | **`model.visual.merger.linear_fc1`**, **`model.visual.merger.linear_fc2`** | **2 ❌** |

Independently corroborated: rung 38's own Unsloth census reported `visual 108 · merger 0 ·
language_model 496` (`experiments/38-gen36-ft-screen/RESULTS_arm_27b.json`), and 27×4 = 108 while
192 + 64 + 240 = 496 reproduces it exactly from the index.

🔴 **This model has NO `deepstack_merger_list`.** The 8B does. Rung 39's list of eight merger names
does **not** port — here it is two.

### 2. The cause, confirmed in source (not inferred)

`unsloth_zoo/peft_utils.py` builds the `all-linear` replacement regex from
`attention_tags = ["self_attn","attention","attn","mixer"]` and
`mlp_tags = ["mlp","feed_forward","ffn","dense","mixer"]`, requiring a component tag **between**
the model part and the leaf. `model.visual.merger.linear_fc1` has the model part (`visual`) and no
component tag, so the `fullmatch` fails.

🔑 **In plain PEFT, `all-linear` WOULD include the merger** — *"all linear/Conv1D modules are chosen
(if the model is a PreTrainedModel, the output layer excluded)"*. The exclusion is Unsloth's, not
PEFT's, and `_maybe_include_all_linear_layers` never runs under Unsloth.

### 3. The fix — TWO paths, both live

Two independent research passes (see §Sources) converged on the reachability and diverged on the
route. Both work; they trade different risks.

| | **path A** | **path B** |
|---|---|---|
| mechanism | `target_modules="all-linear"` (string) + `modules_to_save` | explicit **list** of suffixes, no `modules_to_save` |
| connector gets | full weight | LoRA |
| exposed to the Unsloth merge question (§4c) | **yes** | **no** |
| main risk | does Unsloth's merge carry `modules_to_save`? | must enumerate every suffix and miss none |

🔻 **Path B exists because an earlier framing of ours was wrong.** We wrote that suffix-matching
`linear_fc1` would also hit the 27 tower layers and called that "overcoverage we do not want".
**The tower is already trained** (`--freeze_vit false` since rung 06), so matching tower **and**
merger is precisely what we want. A list of `["...","linear_fc1","linear_fc2"]` reaches the
connector by suffix with no `modules_to_save` at all.

⚠️ **Path B carries `out_proj` explicitly** — see the Mamba hazard in §4b.

🔴 **`all-linear` MUST travel ALONE, as a string. VERIFIED first-hand in source.**
`_maybe_include_all_linear_layers` (`peft/tuners/tuners_utils.py:2353`) early-returns unless
`target_modules` is a `str` equal to the shorthand:

```python
    if not (
        isinstance(peft_config.target_modules, str)
        and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND
    ):
        return peft_config
```

⇒ a **list** `["all-linear", "<name>"]` expands **nothing**, and `"all-linear"` then falls to
`key.endswith(".all-linear")`, which matches no module in any model. **Path A is unaffected — it
passes the bare string, and it must keep doing so.**

✅ **The ms-swift half is now answered, and it does not apply there.** Rung 39's gate ran on
2026-08-13 (`3f932db`) and its subject leg returned `736 = 504 LLM + 216 ViT + 16 aligner`, orphans
0, control leg 0 aligner, `grad_norm` 139 → 14. Coverage intact ⇒ **ms-swift resolves `all-linear`
itself before PEFT sees it, and an explicit target survives its intersection.** So the PEFT rule
binds **only where ms-swift is not in the path — which is exactly our lane: Unsloth.**

🔑 **And rung 39 does not cover us.** Its result is ms-swift · 8B · with `deepstack` (16 tensors =
8 layers × 2). Ours is **Unsloth · 27B · no `deepstack` · 2 layers**. Rodrigo's own commit message
says it: *"Unsloth remains unmeasured."* His gate proves the lever exists; it does not measure our
route.

### 3a. Path A in detail, and why it is safe

```python
target_modules  = "all-linear"                          # the 604 stay byte-identical
modules_to_save = ["model.visual.merger.linear_fc1",    # FULL path — see below
                   "model.visual.merger.linear_fc2"]
```

- **Use the FULL path, not the suffix.** The function that actually wraps the module,
  `_set_trainable` in `peft/utils/other.py`, matches with
  `any(key.endswith(target_key) for target_key in module_names)` — **verified first-hand**, so a
  suffix would work. But three independent research passes reported three different matchers for
  the *related* exclusion check in `tuners_utils.py` (`endswith`, a prefixed
  `re.match(rf"(^|.*\.){m}($|\..*)")`, and a bare `re.fullmatch`). The full path satisfies **all
  three readings at once**, at zero cost. Do not spend a run finding out which was right.
- The suffix `visual.merger.linear_fc1` is nonetheless **unique** — the tower's are
  `blocks.N.mlp.linear_fc1` — so overreach is not the risk here; under-matching is.
- 🔴 **Never use the bare suffix `"linear_fc1"`** — it would match the 27 tower layers *and* the
  merger (28 total).
- **No conflict with `all-linear`:** PEFT excludes anything in `modules_to_save` from LoRA matching
  by construction — *"Adapters should never match on modules to save modules as it is a guarantee
  for conflicts of behavior"* (`peft/tuners/tuners_utils.py:2286-2291`).
- **No extra merge step:** `merge_and_unload()` replaces the module via
  `unload_and_optionally_merge_module`.
- `modules_to_save` trains the 2 layers at **full weight**, not LoRA. PEFT supports it for `Linear`,
  which is what these are.

### 4. 🔴 Two hazards this surfaced

**(a) A non-matching target fails SILENTLY.** PEFT raises `ValueError` only if **no** module was
adapted at all. If 604 match and 2 do not, training starts with no error and no warning.
⇒ **Coverage must be asserted before any full run**: `len(model.targeted_module_names) == 604`, and
the two merger layers present as `ModulesToSaveWrapper`.

**(b) `out_proj` is declared incompatible with Mamba-type models, and our guard does not fire.**
🔑 Verified first-hand, not taken from an agent — `_check_lora_target_modules_mamba` read directly
from PEFT `main` on 2026-08-13.
PEFT carries `incompatible_modules = {"out_proj", "conv1d"}` for
`mamba_model_types = {"falcon_h1","mamba","mamba2","falcon_mamba","nemotron_h"}`
(`peft/tuners/tuners_utils.py:144-163`). `qwen3_5` is **not** in that set, so the guard never
raises — while our 48 `linear_attn.out_proj` **are** being adapted today, on every gen-3.6 run.
Unmeasured, but it is a hazard written into PEFT's own source, not a theory.

## Verdict

**GO on the mechanism.** The connector is reachable with documented API, at no cost to existing
coverage. Two conditions bind any rung that uses it:

1. **Coverage assertion before the arm** — 604 LoRA + 2 wrapped, read from
   `targeted_module_names`, because §4a means a miss is silent.
2. 🔴 **A merge-path gate (path A only).** `modules_to_save` is verified in PEFT and **NOT** verified
   in Unsloth's `save_pretrained_merged`. If the merge drops the full-weight modules we would train
   6 h and ship the base model in exactly the 2 layers the rung is about. Cheap test: 20-step smoke
   → merge → compare `visual.merger.*` byte-for-byte against base.
   **Identical ⇒ the merge ate the connector.**

   ### 4c. The merge question, adjudicated

   A second research pass claimed this path is **dead** — Unsloth's merge raising
   `RuntimeError: Extracted keys = {'lm_head.weight'} do not match!` (unsloth issue **#4098**),
   concluding *"invalida completamente el uso de `modules_to_save`"*. **We checked the issue
   ourselves and that generalisation does not hold:** it is titled *"[Bug] `lm_head` is not trained
   using LoRA and merging is broken"*, is specific to **`lm_head` and weight tying**, its
   reproduction passes `lm_head` inside `target_modules` (not `modules_to_save`), and it is
   **CLOSED** with PR #4106. Our two merger layers are neither `lm_head` nor weight-tied.
   ⇒ **Path A is not dead** on that evidence. But see immediately below.

   🔴 **A third pass then supplied a MECHANISM, and it is more worrying than the issue link.** It
   reports that Unsloth's `save_pretrained_merged` **does not call PEFT's `merge_and_unload()` at
   all**: it builds the `state_dict` by hand, iterating
   `for j, layer in enumerate(internal_model.model.layers): for item in LLAMA_WEIGHTS:`
   (`unsloth/save.py`, ~line 1216). Anything outside that loop — the whole vision branch, connector
   included — never reaches the merged checkpoint. That is specific to **our** case, not to
   `lm_head`.

   ⚠️ **Unverified, and there is reason for doubt:** `LLAMA_WEIGHTS` iterating `model.layers` reads
   like the Llama-specific or GGUF export path, while the second pass pointed at
   `unsloth_zoo/saving_utils.py` for the vision route. Unsloth has several save paths and neither
   agent established which one a vision merge takes.

   🔑 **What is now settled is the priority, not the answer.** Two independent passes, by different
   routes, land on the merge as path A's failure point. ⇒ **The merge gate is the highest-priority
   check in this whole line of work, and it must run before any arm, not after.** If it fails,
   path B (§3) becomes the route by default, since it never uses `modules_to_save`.

**NOT settled by this note:** whether training the connector *helps*. This says it is reachable and
how, nothing about the metric.

## Sources

- PEFT source, branch `main` read 2026-08-13: `utils/other.py:1081` (suffix match), `:793-806`
  (unload), `tuners_utils.py:2286-2291` (exclusion), `:1010-1032` (silent-miss), `:144-163`
  (Mamba guard), `:2297-2303` (list vs string matching).
- Unsloth: `models/vision.py:1869` (`modules_to_save=None` default),
  `unsloth_zoo/peft_utils.py:112-136` (the `all-linear` regex).
- PEFT docs: https://huggingface.co/docs/peft/package_reference/lora ·
  https://huggingface.co/docs/peft/en/developer_guides/custom_models
- Our census: `model.safetensors.index.json` of `Qwen/Qwen3.6-27B`, via S3.
- Corroboration: `experiments/38-gen36-ft-screen/RESULTS_arm_27b.json` (`lora_by_part`).

- **Two independent research passes**, same brief, neither shown the other's answers. Raw reports
  and the adjudicated diff: `local/fuentes/investigacion-peft-2026-08-13/` (vault, not versioned).
  Three claims from the second pass were **wrong by mis-attribution** — the Mamba error message
  presented as an `all-linear` conflict, the Mamba guard missed entirely, and issue #4098
  over-generalised — and two were **right and new**: the `all-linear`-alone rule and the tower/
  connector suffix reframing that produced path B.

⚠️ Line numbers are from `main` on 2026-08-13 and will drift. The quoted strings are the durable
anchor, not the numbers.

📌 **Method note.** All three errors in the second pass were of one kind: a real quotation attached
to the wrong question. A verbatim-quote requirement stops invention; it does **not** stop
mis-attribution. Ask for the quote **with its surrounding context** — that is what exposed all
three in under a minute.

Related: [[the-merger-is-unreachable-by-default]], [[unsloth-is-the-route-to-gen35]],
[[gen36-fails-the-8b-recipe-not-the-backbone-test]].
