# ORENA-Challenge-MEXICO 🇲🇽

Equipo mexicano para el **ORENA SAVE FOCUS Challenge — FRAME Track** (MICCAI 2026). VQA sobre video quirúrgico laparoscópico.

**Privado.** Sincronizado entre: local · RunPod · compu del equipo.

## Documentos clave

| Doc | Qué |
|---|---|
| [`CONSTITUTION.md`](CONSTITUTION.md) | 🔒 Hechos duros + reglas no-negociables. **Léelo primero.** Fuente de verdad. |
| 🆕 [`experiments/08-data-card/`](experiments/08-data-card/) | **Qué hay en los datos y qué decidimos sobre ellos. Léelo ANTES de citar cualquier número.** Ver ⬇️ |
| [`THE_MAP.md`](THE_MAP.md) | Estrategia unificada (fases 0–5). |
| [`PLAN.md`](PLAN.md) | Estrategia, técnica, timeline, roles. |
| `doc1.txt` / `doc2.txt` / `*.pdf` | Specs oficiales del challenge (referencia). |

---

# 📊 Cómo medimos (actualizado 2026-07-16) — **léelo antes de citar un número**

> **Los números que este repo citó durante su primer mes eran engañosos.** No por un bug del harness ni
> por mala fe: **nadie había descrito los datos**. Esta sección resume lo que cambió. El detalle vive en
> **[`experiments/08-data-card/README.md`](experiments/08-data-card/README.md)**.

## Lo que decíamos ➜ lo que es

| Citábamos | Es | Por qué |
|---|---|---|
| **`pre_eval = 0.708`** — el número insignia | **`bucket_mean = 0.5486`** | El `pre_eval` promedia buckets `grupo × ood`. En el dato público **`ood` viene siempre en `False`** (por diseño: lo puebla el test privado), así que **colapsa de 10 buckets a 3** — y **uno de esos 3 es UNA sola pregunta** de `temporal_grounding`, un tipo que FRAME no debería tener. **Esa única pregunta vale ⅓ del score y +14.6 puntos.** |
| "el 0.708" | **dos números**: `0.7079` y `0.7087` | Dos corridas distintas (rung 02 `eval_best` y rung 05 `a0_real`, que re-corrió el control). |
| **`raw_acc = 0.5662`** | plano; el estimador del SDK dice **`0.5395`** | El SDK reporta **media de medias por vídeo + bootstrap jerárquico** (`evaluator.py:465`). Nosotros citábamos la media plana. **Ambos son correctos, para preguntas distintas** — plano para el leaderboard, jerárquico para "¿generaliza?". |
| **`acc_number = 0.4331`** | **NO INTERPRETABLE** | `number` **no es una tarea: son 8 plantillas** con suelos triviales de **0.24 a 1.00**, y **4 de ellas son degeneradas** (una sola respuesta posible en val). |
| `number` saca **+8 pts** sobre el suelo | **+4.9** | **Paradoja de Simpson.** El "suelo" usado (*responder siempre "1"*) es más tonto que *responder la moda de cada plantilla*. La señal: **el margen agrupado (+8.1) supera al de TODAS las plantillas individuales (máx +6.1)**. |
| **`acc_OOD 0.5918 > acc_ID 0.5209` → "no hay colapso OOD"** | 🔴 **artefacto** | El slice OOD tiene un **suelo trivial 12 pts más alto** (0.4597 vs 0.3370): sus respuestas están más concentradas. Contra el suelo, **el modelo aporta 5.2 pts MENOS en OOD que en ID**. |
| `number` 37% · `fo_class` 39.1% del examen | **33.5%** · **42.8%** | Nunca se contaron. |
| `heico` = "Sigmoid Resection" | **3 procedimientos** | heico = Proctocolectomy + Rectal Resection (train) + **Sigmoid (solo en test)**. **El proxy OOD es más fuerte de lo que decíamos.** |
| El examen son 5 formatos | **188 plantillas** | La unidad de análisis es la plantilla, no `answer_format`. |

## Por qué esto importa más que cualquier experimento

**El valor central del proyecto es ganarle a AMBOS baselines en el eje OOD** — eso es la co-autoría.
**OOD es el 50% del examen.** Y el titular en el que se apoyaba nuestra confianza (*"OOD > ID, sin
colapso"*) **se invierte al medir contra el suelo**. Un termómetro mal calibrado no te hace perder un
experimento: te hace **elegir mal el siguiente**.

Concretamente: **la accuracy no es interpretable sin su suelo trivial.** `0.9778` en *"How many External
drains?"* parece excelente — hasta que ves que **la respuesta es siempre `1`** y una constante saca
`1.0000`. **El 5.6% del examen no puede medir nada** en nuestro split, y en el **15.2%** el modelo no le
gana a una constante.

## Lo que sí está sólido (medido, no supuesto)

- ✅ **El modelo usa la imagen.** Ablación de 3 brazos sobre las 6252, misma pregunta, solo cambia la
  imagen: real **0.5675** · imagen de otro vídeo **0.3440** · imagen negra **0.2681**. **La imagen
  correcta vale +22.3 puntos.** Es una intervención, no una correlación.
- ✅ **Nuestro parser no derivó del SDK.** `data.py` se declara *"Mirror of `FocusDataset._parse_row`"*
  y nadie lo había comprobado: **coincide en las 20,000 filas**.
- ✅ **El LoRA sirve de verdad.** `raw 0.262 → 0.566`. Lo único inflado era el titular.
- 🔴 **El cuello de botella: el modelo ve el primer objeto y se queda ciego después.** `fo_class` y
  `number` se desploman con **la misma curva** según haya 1/2/3/4 objetos (0.64→0.44→0.07 y
  0.80→0.46→0.19). Dos formatos sin nada en común ⇒ **es percepción, no formato ni conteo.**

## Reglas de lectura (para todo el equipo)

1. **Ningún número sin su suelo trivial**, y el suelo **por plantilla**, nunca por formato.
2. **Ningún número sin decir si es ID, OOD o pooled.** El val es **64% OOD**.
3. **No cites `pre_eval` en local.** Usa `bucket_mean`. *(Y ojo: estampar `ood=True` "para arreglarlo"
   da `0.6389` — parece sano y sigue inflado +9. Los dos defectos están enredados.)*
4. **Las 6252 preguntas no son independientes**: son **4486 frames** en **38 vídeos**. El `n` efectivo
   para generalizar está más cerca de 38.
5. **Lee el SDK antes de reimplementarlo.** `Capability.group` existe; el warning
   `only 3/10 buckets are populated` lleva saliendo en cada corrida desde el principio.

> **El harness nunca falló. Avisaba, y no leíamos el log.**

## Setup (cada máquina)

```bash
git clone https://github.com/RodMed0709/ORENA-Challenge-MEXICO.git
cd ORENA-Challenge-MEXICO

# Entorno
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install orena-focus                              # SDK oficial (módulo: focus)

# Secrets — NO están en el repo. Copiar .secrets.env manualmente (pedirlo al lead).
export HF_TOKEN=...        # o cargar desde .secrets.env
export RUNPOD_API_KEY=...
```

## Flujo de trabajo (SDD + sync)

1. **`git pull`** antes de empezar (siempre).
2. Rama por trabajo: `git checkout -b feat/<algo>` o `exp/<experimento>`.
3. Spec antes que código → `specs/`. Código que corre → merge a `main`.
4. **`git push`** al terminar.
5. **NUNCA commitear:** secrets, datos crudos, pesos, checkpoints, videos (ya bloqueados en `.gitignore`).

Datos y pesos viven en RunPod / HuggingFace, no en git.

## Reglas duras (resumen — full en CONSTITUTION)

- `main` siempre funcional. Siempre existe una submission válida > 0.
- Toda mejora se valida contra **OOD**, no solo ID — **pero la accuracy OOD sola engaña**: su suelo
  trivial es 12 pts más alto que el de ID. **Valida contra el MARGEN sobre el suelo** (ver §Cómo medimos).
- Split por `video`, nunca por frame.
- Docker offline real (`HF_HUB_OFFLINE=1`). Latencia p99 < 5s en L40S.
- NUNCA prompt-injection al juez (= descalificación).
