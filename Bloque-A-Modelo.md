# Bloque A — El Modelo · estrategia, roadmap y pipeline (v3)

> 🔴 **CORRECTION 2026-07-23 — the CoA section of this document is FALSIFIED. Do not cite it.**
> This file's CoA ablation table (below, "El desglose que lo decide todo") is a **second-hand
> summary** and omits the source's `+ Cold Start + SFT` row — the no-RL scaffold arm this document
> claims nobody has run. The source (arXiv:2603.20116, **our exact backbone**) reports that arm at
> **62.0 F1 on EndoVis2018 against bare-gold SFT's 65.7** (62.4 vs 58.7 on CholecT50): scaffold-SFT
> without RL is a **published wash that loses on EndoVis**, and the gain belongs to **RLVR** (which
> beats SFT with no reasoning tags at all, 67.4 vs 65.7). **"El valor está en el FORMATO, no en el
> RL" is backwards.** Primary source now in-repo:
> `literature/vlm-techniques/pdfs/v01_li_2026_chain-of-adaptation.pdf` (Tables 1–3).
> Full note: `context/decisions/coa-sft-published-null.md`. Cite the PDF, never this file.

> **Estado: para revisión (legokna).** Baúl local, no commiteado. Si se aprueba, se decide qué sube a
> `THE_MAP.md` en inglés.
> **v3 — 2026-07-15.** Reescrito de una pasada tras el deep-research y **tu corroboración manual**.
> Sustituye por completo a la v2 (que ya era un borrador con tres capas de correcciones encima).

## §0 — Qué manda

**Alcance:** solo el **Bloque A = el producto** (lo que entra al Docker). El Bloque B
(evaluación/laboratorio) aparece únicamente en sus puntos de conexión. La separación A/B está en
`THE_MAP.md` §"The two blocks" (`2e4b0c9`).

**Precedencia, en este orden:**

1. **`documentacion/overview.md`** (página oficial del reto) manda sobre todo lo demás.
2. **Dato medido del proyecto** > cita.
3. 🆕 **Cita corroborada a mano** > cita de agente. *(Lección cara — ver `documentacion/r-adjudicacion.md`.)*

**Sello de evidencia:** todas las citas de este documento fueron **abiertas y verificadas manualmente
por legokna el 2026-07-15**. Los abstracts verbatim viven en **`documentacion/papers-corroborados.md`**.
**Si una conclusión de aquí choca con un abstract de allá, gana el abstract.**

---

## §1 — El objetivo, en una línea

> **Maximizar la media de accuracy sobre los 4 buckets.**
> **Sujeto a:** < 5 s/pregunta · 48 GB VRAM · offline · una GPU.

Un solo objetivo. Las restricciones **se comprueban, no se maximizan.**

### Los 4 buckets — el volumen NO es el peso

| Bucket | Peso | Qué cae aquí | Nuestro estado |
|---|---|---|---|
| object_recognition × ID | 25% | `fo_class` · `binary` · `MC` · `open_ended` | ≈0.53–0.60 |
| object_recognition × OOD | 25% | ídem | ídem |
| **aggregation × ID** | **25%** | **`number`** | **0.433** ⚠ |
| **aggregation × OOD** | **25%** | **`number`** | **0.433** ⚠ |

**`aggregation` (= conteo) vale el 50% del examen, y es lo peor que tenemos.** `fo_class` **no es un
bucket**: es una porción del otro 50%, compartido con formatos ya fuertes. **Un punto en `number` vale
como dos en `fo_class`.**

**OOD = 2 de 4 = 50%**, sobre **dos ejes** (overview): *"procedure types not represented **and question
phrasings not seen** during training"*. Nuestro proxy (Sigmoid held-out) **solo cubre el primero**.

### La munición

```
  TIEMPO    ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.59 s de 5 s   → 88% SIN GASTAR
  MEMORIA   ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░  18.01 GB de 48  → 63% SIN GASTAR
```

**La velocidad no es una meta: es presupuesto.** No hay premio por contestar en 0.3 s en vez de 2 s.
La pregunta no es *"¿cómo acelero?"* sino ***"¿en qué gasto esto para subir la media por buckets?"***

*(Caveat: el 0.59 s se midió en A100. La L40S es más lenta y **nunca hemos corrido una pregunta en el
hardware real**. El margen es grande, pero su tamaño exacto está sin medir → §12.)*

---

## §2 — Dónde estamos: el diagnóstico

**Lo que tenemos:** Qwen3-VL-8B + LoRA (r=8, α=32) fusionado, bf16. `pre_eval 0.708` · `raw 0.566` ·
`acc_OOD 0.592 > acc_ID 0.521`.

**Lo que el LoRA hizo, mirando dónde ganó:**

| Formato | Antes | Después | Δ |
|---|---|---|---|
| `fo_class` | 0.168 | 0.588 | **+0.42** |
| `number` | 0.141 | 0.433 | **+0.29** |
| `binary` | 0.597 | — | ya estaba bien |

Las dos que explotaron son **las de formato estricto**. Eso no es casualidad: **el LoRA no enseñó a
*ver*, enseñó la *tarea*** — la convención de respuesta y el vocabulario. Congelamos **ViT y aligner**,
así que solo tocó el lado del lenguaje.

**Y ahí está el síntoma:** la loss y `acc_OOD` **se estancan en epoch 1**.

### Las dos hipótesis que compiten

| Hipótesis | Qué predice | Cómo se falsea |
|---|---|---|
| **Techo de percepción** — el ViT congelado, de dominio general, ve la laparoscopia como ruido; el LoRA de lenguaje no puede traducirlo | Tocar el encoder desbloquea la loss | **LoRA sobre el ViT**, pocas iteraciones |
| **Atajo lingüístico** — el modelo aprendió *"si la pregunta se ve así, contesta ~2.7"*, no a mirar | La accuracy **apenas cae sin imagen**, y colapsa con la pregunta reformulada | **Ablación text-only** |

**No son excluyentes** — un encoder inútil *empuja* al modelo hacia el atajo. Pero **la prueba las
separa**, y es barata.

> **Por qué esto es urgente y no académico:** el segundo eje del OOD es **redacción no vista**. Si
> construimos un atajo lingüístico y lo llamamos mejora, **colapsa justo ahí** — y `number`, que es el
> 50%, es donde más fácil se cuela un atajo (adivinar la moda de la distribución).

### 🔴 Nota honesta: no tenemos referencia externa

`2506.06232` (mismo grupo DKFZ, mismos videos HeiCo) **no da un número comparable con nuestro 0.433**.
Usa `Accuracy%(t)` (nivel imagen, con umbral) y MCC; el conteo **no es categoría propia** (vive dentro
de *Detection*); la Figura 4 son barras promediadas sobre modelos, sin tabla. **Detalle en
`papers-corroborados.md`.**

→ **Cualquier afirmación del tipo "nuestro 0.433 está por debajo de lo esperable" NO tiene soporte.**
*(Yo la hice. La retiro.)* Lo único que sostiene el paper es cualitativo: los VLM hacen razonable la
percepción básica y **se hunden cuando hace falta conocimiento médico**.

---

## §3 — El backbone: pregunta cerrada

**Era decreto, no derivación.** `CONSTITUTION.md` fija Qwen3-VL-8B como hecho duro, y el tutorial del
rung 04 verificó que **el SDK no nos ata a él** (`focus` no trae VLM).

**Pero ahora está validado por evidencia**, y eso cierra el debate por el lado correcto:

- **2506.06232:** *"specialized medical VLMs currently **underperform** compared to generalist models
  across both basic and advanced surgical tasks"*.
- **2506.17337:** *"efficiently fine-tuned **generalist** VLMs can achieve comparable or even superior
  performance in most tasks, **particularly when transferring to unseen or rare OOD**"*.

**Un generalista + fine-tuning ligero es la elección correcta, y gana precisamente en OOD** — que es la
mitad de nuestro examen. Ya no es fe en la CONSTITUTION.

### Lo que ata y lo que no

| Restricción | ¿Ata? |
|---|---|
| Licencia — público y liberable, o no hay premio ni co-autoría | **SÍ.** Qwen3-VL es Apache-2.0 |
| Público antes del 15-jul (overview, *Training data policy*) | **SÍ**, y se cumple |
| `ms-swift` + `transformers 4.57` | **SÍ** en la práctica (cambiar = reiniciar) |
| 48 GB VRAM | **NO: usamos 18.01** |
| 5 s/pregunta | **NO: 0.59 medidos (A100)** |

### 🔴 El hallazgo incómodo

**Elegimos el 8B contra la VRAM del POD (24–32 GB), no contra la del EXAMEN (48 GB).** Son
restricciones distintas y nadie las separó. **El pod es donde entrenamos; la L40S es donde nos
califican.**

Corolario: **"el wildcard 32B FP8 no cabe" se concluyó contra el pod.** En FP8 ronda ~32–35 GB →
**cabe en 48**, y la L40S es Ada con FP8 nativo. **La duda del 32B no es memoria, es latencia — y nadie
la ha medido.** *(r1 se la inventó; ver `r-adjudicacion.md`.)*

### Recomendación

**No cambiar de familia.** Sin evidencia de que otro backbone sea mejor en imagen quirúrgica, y el
coste es reiniciar. **El desbloqueo es capacidad dentro de Qwen3-VL**, y cuelga de una sola medición
que nunca hicimos.

---

## §4 — El pipeline del producto

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  BLOQUE A — EL PRODUCTO                             lo único que se envía ║
╚═══════════════════════════════════════════════════════════════════════════╝

  ┌──────────────┐
  │ imagen RGB   │──┐
  └──────────────┘  │   ┌─────────────────────────────────────────────┐
                    ├──►│ [1] ARMADO DEL PROMPT                       │
  ┌──────────────┐  │   │     · system prompt + defs FO (+0.061 med.) │
  │ pregunta     │──┘   │     · la pregunta YA trae: procedimiento,   │
  │ (texto)      │      │       timestamp, formato esperado, clases FO│
  └──────────────┘      │     ⚠ ¿coincide con el formato de train?    │
                        │       → si no, ESO ES el eje 2 del OOD      │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │ [2] VLM   Qwen3-VL-8B + LoRA (fusionado)    │
                        │     bf16 · 18.01 GB de 48 · greedy · ≤32 tok│
                        │     🔴 ViT congelado · aligner congelado    │
                        │        └─ la hipótesis del techo (§2)       │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                        ┌─────────────────────────────────────────────┐
                        │ [3] DECODIFICACIÓN RESTRINGIDA   ✗ NO EXISTE│
                        │     aggregation → SOLO dígitos              │
                        │     fo_class    → SOLO clases del enum      │
                        │     binary      → SOLO yes/no               │
                        │     ⚠ constreñir SOLO la salida final       │
                        │       (2408.02442: restringir daña el CoT)  │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                                   "Sponge"  (≤300 ch)

           SIN video · SIN decodificar · SIN muestrear frame · SIN juez
```

---

## §5 — El roadmap

```
 A0 ── CALIBRAR EL OBJETIVO ──────────────────────────────── bloquea todo
   │   crosstab answer_format × capability_group
   │   → confirmar que aggregation ≈ number · pesos exactos de los 4 buckets
   │   → cambiar la métrica de selección: raw/OOD-mean ─► MEDIA POR BUCKETS
   │   sin GPU · horas
   │   ⇄ B: cambia qué reporta run.py y por qué se elige un checkpoint
   ▼
 A1 ── LAS DOS PRUEBAS ───────────────────────── ⭐ EL ATÓMICO · la bifurcación
   │   sobre el rung 02 (NO el 00: queremos saber si el LoRA creó el atajo)
   │
   │   (a) ABLACIÓN TEXT-ONLY — responder SIN imagen
   │       ¿la accuracy apenas cae?  → ATAJO CONFIRMADO
   │       sin entrenamiento · 1 pasada de eval
   │
   │   (b) LoRA SOBRE EL ViT — unas pocas iteraciones
   │       ¿la loss plana de epoch 1 se desbloquea? → TECHO DE PERCEPCIÓN
   │       ¿no se mueve? → datos/etiquetas, o el atajo ya saturó
   │
   │   ⚠ (b) NO se apoya en la literatura — ver §6. Es MEDICIÓN, no cita.
   │   ⇄ B: reusa el evaluador tal cual; solo cambia el input
   ▼
 A2 ── LA BIFURCACIÓN ─────────────────── la decide A1 con datos, no la opinión
   │
   ├── GENERALIZACIÓN DAÑADA / ATAJO ──► CoA  (§7)
   │      formato de razonamiento estructurado + RL
   │      la mejor evidencia que existe para NUESTRO problema exacto
   │      coste real: fabricar ~13.7k ejemplos con razonamiento
   │
   └── TECHO DE PERCEPCIÓN ──► CAPACIDAD
          · seguir con el ViT descongelado / LoRA-izado
          · un paso de resolución (max_pixels) — ahora sí se puede pagar
          · Qwen3-VL-32B FP8 — cabe en 48 GB
            🚦 GATE DURO: medir latencia en L40S ANTES de entrenar nada
   ▼
 A3 ── HACER IMPOSIBLE EL FORMATO INVÁLIDO ─────── independiente, en paralelo
   │   decodificación restringida (xgrammar / guided_decoding de vLLM)
   │   ⚠ constreñir SOLO el entero final; enumeración libre (2408.02442)
   │   no toca pesos · sube accuracy MEDIDA
   │   ⇄ B: los gates 1–2 dejan de disparar → se vuelven test de regresión
   ▼
 A4 ── EMPAQUETAR ────────────────────── parte ejecutable HOY, parte bloqueada
       · paquete src/frame/serve  (§8)     ─ HOY, sin GPU
       · medir p99 real en L40S            ─ HOY, 1 sesión de pod
       · Docker                            ─ ⛔ la plantilla NO existe
```

### Conexiones con el Bloque B (el lazo)

| Fase A | Qué le pide a B | Qué le devuelve |
|---|---|---|
| **A0** | el parquet + la taxonomía | **la métrica correcta** → B deja de medir lo que no es |
| **A1** | el evaluador, sin cambios | **un hecho**: ¿el 0.708 es percepción o atajo? |
| **A2** | selección por bucket-mean (de A0) | Δ por bucket, una sola variable |
| **A3** | los gates como test de regresión | elimina una clase entera de pérdida |
| **A4** | nada — B se queda en casa | el artefacto |

---

## §6 — Base de evidencia · qué dice REALMENTE cada paper

> **La sección que evita re-litigar.** Todo corroborado a mano; abstracts en
> `documentacion/papers-corroborados.md`. **Tres citas resultaron mal leídas por el research** — están
> marcadas 🔧.

| Paper | Lo que **SÍ** sostiene | Lo que **NO** sostiene |
|---|---|---|
| **2506.06232** — Challenging VLMs with Surgical Data *(DKFZ, mismos videos)* | Los VLM hacen razonable la percepción básica (conteo, localización); **se hunden con conocimiento médico**. **Los VLM médicos especializados rinden PEOR que generalistas.** | 🔧 **NO da accuracy de conteo comparable.** Usa `Accuracy%(t)` (nivel imagen) y MCC. **No sirve para calibrar nuestro 0.433.** |
| **2506.17337** — Generalist vs Specialist Medical VLMs | **Generalista + FT eficiente ≥ especialista**, sobre todo **transfiriendo a OOD**. → valida Qwen. | — |
| **2603.20116** — Chain-of-Adaptation ⭐ | El SFT convencional *"can inadvertently alter a model's pretrained multimodal priors, **leading to reduced generalization**"*. CoA da **más accuracy, mejor generalización y más estabilidad que SFT**, en benchmarks quirúrgicos, **ID y OOD**. **Ablación COMPLETA: SFT 65.7 → formato-sin-RL 62.0 → RLVR-sin-tags 67.4 → RLVR+formato 83.7.** | 🔴 ~~**NO sostiene que el RL sea la palanca.** El RL solo aporta **+1.7**; el **formato** aporta **+16.3**.~~ **FALSIFIED 2026-07-23:** the source's `+ Cold Start + SFT` row (formato sin RL) = **62.0 < 65.7**. The lever IS RLVR. `context/decisions/coa-sft-published-null.md`. |
| **2504.13837** — Does RL Really Incentivize…? | RLVR **afila** lo que el modelo ya puede; **no expande** la frontera del modelo base. | — |
| **2506.07218** — Perception-R1 | El RLVR *ingenuo* (recompensa solo en la respuesta) **no mejora percepción** (test de McNemar). **Con una recompensa dirigida a percepción, sí.** | 🔧 **NO dice que RLVR no sirva.** Esa frase es su **motivación**; el paper **propone el arreglo**. **Y su recompensa NO es verificable:** necesita **anotaciones visuales de trayectorias CoT** + **un LLM juez en el loop**. |
| **2603.17326** — FineViT | *"Their visual encoders frequently remain a **performance bottleneck**"*; supera a **Qwen-ViT** integrado en MLLMs. → sostiene el **diagnóstico** del §2. | 🔧 **NO sostiene "descongelar el ViT".** Es un **encoder nuevo** entrenado desde cero con miles de millones de recaptions. **No hay ablation de congelar-vs-descongelar.** |
| **2406.09246** — OpenVLA | Fine-tuning eficiente de VLA con LoRA en GPUs de consumo. | 🔧 Es **robótica manipulativa**, no VQA. Lo de *"fine-tunear el encoder > congelarlo"* **no está en el abstract**. Transferencia no evidente. |
| **2408.02442** — Let Me Speak Freely? | *"Significant decline in LLMs' reasoning abilities **under format restrictions**"*; más estricto = peor. | **NO condena la decodificación restringida en general.** El daño golpea al **razonamiento con CoT**, no a respuestas cortas atómicas. → constreñir **solo la salida final**. |
| **2607.06420** — HoloCount | El modo *thinking* mejora conteo **+10.6 a +15.4 pts**. | Es un **benchmark**, no un método. La cifra **describe modelos existentes**, no una técnica lista para aplicar. |
| **2501.02385** — MedVP | Prompts visuales (marcas sobre la imagen) mejoran VQA médico. | Requiere **Grounding DINO fine-tuneado en dominio médico** → **bboxes que no tenemos**. |
| **SurgeNetDINO** *(MIDL 2026)* | **El preentrenamiento SSL en dominio quirúrgico mejora** (4.7M frames). Refuerza el diagnóstico del §2. | **NO es un detector open-vocab** *(eso lo inventó r1)*. **Pesos CC-BY-NC-SA** → infectarían nuestro modelo liberado. **ViT-S/B/L @ 224/336 ≠ el ViT de resolución dinámica de Qwen3-VL** → no es intercambiable sin reentrenar toda la alineación. |

### 🔴 De qué NO tenemos evidencia

| Afirmación | Estado |
|---|---|
| **"Descongelar el ViT desbloquea la percepción"** | **Sin cita que lo sostenga.** FineViT es otra cosa; OpenVLA es robótica. → **hay que medirlo (A1b)** |
| **Nuestro 0.433 está por debajo de lo esperable** | **Retirado.** No hay número comparable |
| **Latencia del 32B FP8 en L40S** | **NO ENCONTRADO.** r1 lo fabricó |
| **GRPO en 1×L40S** | Existe en `ms-swift`, pero las recetas oficiales **asumen 6 GPUs + modo server** |
| **Cómo se cronometran los 5 s** | **Sigue sin documentarse.** Presumiblemente llega con la plantilla |

---

## §7 — CoA: la mejor evidencia que existe para nuestro problema

### Por qué encaja como un guante

Su **premisa** es nuestro **diagnóstico**, palabra por palabra:

> *"Conventional fine-tuning on domain-specific datasets can inadvertently alter a model's pretrained
> multimodal priors, **leading to reduced generalization**."*

Y sus resultados son **sobre benchmarks quirúrgicos**, **en ID y OOD**, con ablación que confirma que
*"preserves the model's core visual–language abilities"*. **OOD es el 50% de nuestro examen.**

### El desglose que lo decide todo

| Variante | F1 (EndoVis2018) | Δ |
|---|---|---|
| SFT | 65.7 | — |
| **RLVR sin formato** | **67.4** | **+1.7** |
| **RLVR + formato CoA** | **83.7** | **+16.3** |

~~**El valor está en el FORMATO, no en el RL.**~~ 🔴 **FALSIFIED — this table is incomplete.** The
source also reports `+ Cold Start + SFT` (scaffold in the target, **no RL**) = **62.0 vs SFT's 65.7**
on EndoVis2018 and 62.4 vs 58.7 on CholecT50. The format alone buys ≈0 and *loses* on EndoVis; the
+18.0 is RLVR's. See `context/decisions/coa-sft-published-null.md` and the PDF at
`literature/vlm-techniques/pdfs/v01_li_2026_chain-of-adaptation.pdf`. (Kept for history: this is the
proposal-inversion that was made on the wrong reading.)

### Las dos objeciones que yo tenía, y por qué caen

| Mi objeción | Por qué cae |
|---|---|
| *"Choca con el cap de 300 chars"* | Se genera la CoA **internamente** y se emite **solo `<answer>`**. El cap aplica a lo que enviamos. |
| *"FRAME no puntúa razonamiento"* | **Apuntaba al blanco equivocado.** El beneficio de CoA no son puntos de razonamiento: es **preservar la generalización** → **OOD** → **50%**. |

### El coste real (esto no es un flag)

- **Fabricar los datos:** hay que generar `<general description>`, `<evidence>`, `<thought>` para
  ~13.7k ejemplos. Es un proyecto de datos, probablemente sintetizado con un modelo mayor.
- **Tokens en inferencia:** más generación. Presupuesto: 4.4 s libres — **pero sin medir en L40S**.
- **Infra de RL:** el RL aporta poco (+1.7) pero el paper lo usa; **habría que probar si el formato
  solo, vía SFT, captura la mayor parte**. Eso sería mucho más barato y **nadie lo ha medido**.

### Confianza

**ALTA** en el encaje (su premisa es nuestro diagnóstico; benchmarks quirúrgicos; ganancia OOD).
**MEDIA-BAJA** en la ejecución dentro del calendario (~7 semanas, 3 personas, pod compartido).

> ~~**La pregunta barata que nadie ha hecho:** ¿cuánto del +16.3 sobrevive con **SFT sobre el formato
> CoA**, sin RL?~~ 🔴 **ANSWERED, and by the same paper: none of it** (62.0 vs 65.7 EndoVis2018).
> The question was never unasked — it was in a row this summary dropped.
> `context/decisions/coa-sft-published-null.md`.

---

## §8 — El paquete `src/frame/serve/`

*(Sin cambios: nada del research lo tocó. **Sigue siendo lo único ejecutable hoy** — sin GPU, sin
plantilla, sin bloqueos.)*

`EXPERIMENT_REPO_STRUCTURE_SPEC` (§VIII, vinculante) exige **UN solo paquete en `src/`**. Un paquete
hermano lo violaría. **La solución es un SUBpaquete que haga explícita la frontera A/B.**

### El problema, medido

```
src/frame/__init__.py
    from frame.run import run_baseline      ◄── ESTO
                              │
                              └─► pandas · focus.evaluation.evaluator
                                  · TransformersJudge · focus.enums

  ⇒ `import frame` ARRASTRA EL JUEZ Y PANDAS AL DOCKER.
```

**Pero el producto casi ya está limpio:**

| Módulo | Importa | Bloque |
|---|---|---|
| `config.py` | `dataclasses`, `pathlib` | **A** (puro) |
| `engine.py` | `torch`, `PIL`, `focus.foreign_objects.FO_DEFINITIONS_FILE` | **A** (1 sola dep de `focus`) |
| `data.py` · `run.py` · `delta.py` · `split.py` · `qualitative.py` | `pandas`, `numpy`, `focus.*` | B |

**La frontera ya existe de facto. Lo único que la rompe es `__init__.py`.**

### Propuesta

```
src/frame/
├── serve/                    ◄── BLOQUE A · lo ÚNICO que entra al Docker
│   ├── __init__.py               (sin imports pesados)
│   ├── config.py                 movido tal cual
│   ├── prompt.py                 armado del prompt ── hoy dentro de engine.py
│   ├── engine.py                 load / predict / unload
│   ├── decoding.py               ✗ NUEVO — decodificación restringida (A3)
│   └── fo_defs.py                copia local de FO_DEFINITIONS_FILE
│                                 └─ mata la última dep de `focus` en el producto
│
├── data.py · run.py · delta.py · split.py · qualitative.py   ◄── BLOQUE B
└── __init__.py               ── NO importar run.py desde aquí
```

**La regla que lo hace real, verificable con un `grep`:**

> **`serve/` NUNCA importa del laboratorio. El laboratorio SÍ puede importar de `serve/`.**
> `grep -rE "from frame\.(data|run|delta|split|qualitative)" src/frame/serve/` → **vacío**.

**Qué gana:** el Docker copia `serve/` + pesos, y nada más (menos superficie de fallo offline, imagen
más chica). La frontera pasa de convención a **mecánica** — hoy es una convención que `__init__.py` ya
rompió sin que nadie lo notara. Y **A y B siguen compartiendo el mismo engine**, que es lo que hace que
el número del laboratorio signifique algo.

**Coste:** refactor de infra compartida → **hablarlo con RodMed**. Y `fo_defs.py` **duplica** un recurso
del SDK (justificable como puerta offline, igual que `data.py`) — pero se decide a propósito.
**Riesgo:** que `serve/` se vuelva el cajón de sastre. La regla del `grep` es la defensa.

---

## §9 — Descartado, y por qué

| Propuesta | Por qué no |
|---|---|
| **RLVR con recompensa exact-match** *(era mi S5)* | **+1.7 sobre SFT** (ablación de CoA). El valor está en el formato, no en el RL. |
| **Perception-R1 tal cual** | Su recompensa **necesita anotaciones visuales de trayectorias CoT + un LLM juez en el loop**. No tenemos ninguna de las dos. Y sus experimentos son *math/general*, no percepción quirúrgica. |
| **SurgeNetDINO como encoder** | **CC-BY-NC-SA** infecta nuestro modelo liberado (Qwen es Apache-2.0), **y** ViT-S/B/L @224/336 ≠ ViT de resolución dinámica de Qwen3-VL → **swap = reentrenar toda la alineación** = semanas, probablemente peor. *(Su valor es evidencial: el preentrenamiento en dominio ayuda.)* |
| **YOLO→ROI / scene-graphs (SSG-VQA-Net) / MedVP** | **No hay bboxes** en los datos liberados (verificado). Y no existe detector open-vocab quirúrgico público. Grounding DINO está OOD en laparoscopia; un ROI mal puesto **es peor que ninguno**. |
| **QLoRA 4-bit en el 8B** | 18 GB en 48. Cuantizar añade latencia **sin resolver nada**. Reservado al 32B, donde sí es la única vía. |
| **Unsloth** | Rezagado en Qwen3-VL. `ms-swift` tiene soporte nativo + `freeze_vit`/`freeze_aligner`/`max_pixels` — **las palancas exactas de A1b/A2**. |
| **SGLang por RadixAttention** | Su argumento era *"reusar la imagen para 10 preguntas"*. El contrato es **una imagen + una pregunta**. |
| **PagedAttention** | Resuelve VRAM en **contexto largo y batching**. Nosotros: batch=1, 1 imagen, ≤32 tokens, 18 de 48 GB. |
| **Decodificación especulativa (EAGLE)** | **Acelera** lo que ya sobra (4.4 s libres). Complejidad a cambio de nada. |
| **SurGen-Net** | Su ganancia es **varias QA de una escena en una inferencia**. El contrato es una pregunta. |
| **PitVQA-Net** | **Discriminativo**, conjunto cerrado de respuestas, GPT-2. El contrato es texto libre y hay un bucket OOD de **formulaciones no vistas**. |
| **DPO** | Alinea con **preferencias humanas**. Nuestra métrica es exact-match + juez oculto. |
| **Cambiar de familia de backbone** | Sin evidencia; coste = reiniciar. Y **2506.06232 + 2506.17337 validan al generalista** (§3). |

---

## §10 — Lo que corregimos hoy (para no re-litigarlo)

**Mis errores** — todos reales, todos corregidos arriba:

1. **Vendí RLVR como "el hueco más grande de THE_MAP".** La ablación de CoA dice **+1.7**. Promoví lo
   que no funciona **y rechacé lo que sí** (el formato).
2. **Propuse el currículum de LLaVA-Med (S1).** Con ~14k pares de dominio, la etapa 1 aporta poco.
   **Muerto.**
3. **Afirmé que nuestro 0.433 está por debajo de lo esperable.** Sin soporte: no hay número comparable.
   **Retirado.**
4. **Repetí "descongelar el aligner"** heredándolo de THE_MAP. El diagnóstico apunta al **ViT** — pero
   **sin la cita que creí tener** (§6).

**Errores de r2** *(el research "bueno")*: leyó mal **Perception-R1** (usó la motivación como
conclusión), **FineViT** (no es un ablation de freezing) y **OpenVLA** (robótica).

**Errores de r1:** fabricó papers y datos. **Forense completa en `documentacion/r-adjudicacion.md`.**
La corta: inventó la latencia del 32B en L40S con confianza ALTA — **el número que sostenía su
recomendación de cambiar el backbone** — y citó un paper de **astrofísica** como si fuera de conteo en
VLMs. **No ejecutar nada de su "Plan de Choque".**

---

## §11 — Decisiones (legokna)

- [ ] **¿Se abre A1 (las dos pruebas) como el atómico?** Es lo más barato y lo que más puede doler.
- [ ] ¿Se acepta que el **8B se eligió contra el pod, no contra el examen**, y que el **32B vuelve a la
      mesa** condicionado a medir la L40S?
- [ ] ¿Se aprueba el subpaquete **`src/frame/serve/`**? *(Toca infra compartida → hablarlo con RodMed.)*
- [ ] ~~**CoA:** ¿se explora? Y si sí, ¿se prueba primero la versión barata — **formato CoA vía SFT, sin
      RL** — para ver cuánto del +16.3 sobrevive?~~ 🔴 **Obsolete — the no-RL arm is published at 62.0
      vs 65.7** (`context/decisions/coa-sft-published-null.md`).
- [ ] **¿Qué sube a `THE_MAP.md`?** Hoy dice *"descongelar aligner"* y lo pusheamos a `main` (`2e4b0c9`).
      **RodMed lee eso.**

---

## §12 — Lo que falta medir (nadie lo sabe)

| Incógnita | Por qué bloquea | Coste |
|---|---|---|
| **Latencia real en L40S** | Bloquea **tres** decisiones a la vez: 32B, resolución, y los tokens extra de CoA. **Nunca hemos corrido una pregunta en el hardware real.** | 1 sesión de pod |
| **¿El atajo existe?** | Bifurca todo el roadmap | 1 pasada de eval (A1a) |
| **¿El ViT es el techo?** | Ídem, y **sin respaldo bibliográfico** → hay que medirlo | pocas iteraciones (A1b) |
| ~~**¿Cuánto del +16.3 de CoA es el formato solo, sin RL?**~~ 🔴 **CLOSED — ninguno (62.0 vs 65.7)** | ~~Decide si CoA es viable en 7 semanas~~ → `coa-sft-published-null` | 0 (ya publicado) |
| **¿GRPO cabe en 1×L40S?** | Solo si CoA entra con RL | — |
| **¿Cómo se cronometran los 5 s?** | Sin documentar. Llega con la plantilla | — |
| **¿Existe `Qwen3-VL-32B-Instruct-FP8`? ¿Licencia?** | Puerta del 32B | 5 min |

---

## §13 — Fuentes

- **`documentacion/papers-corroborados.md`** — abstracts verbatim + tus análisis. **Evidencia primaria.**
- **`documentacion/r-adjudicacion.md`** — forense r1 vs r2 y las lecciones.
- **`documentacion/overview.md`** — página oficial. **Manda sobre todo.**
- **`documentacion/notas/`** — reportes generados **sin bibliografía**. Buenos en *técnicas*, **no
  fiables en *reglas***.
- **`literature/INDEX.md`** — corpus del proyecto (30 papers, fichados).
