# ORENA-Challenge-MEXICO 🇲🇽

Equipo mexicano para el **ORENA SAVE FOCUS Challenge — FRAME Track** (MICCAI 2026). VQA sobre video quirúrgico laparoscópico.

**Privado.** Sincronizado entre: local · RunPod · compu del equipo.

## Documentos clave

| Doc | Qué |
|---|---|
| [`CONSTITUTION.md`](CONSTITUTION.md) | 🔒 Hechos duros + reglas no-negociables. **Léelo primero.** Fuente de verdad. |
| [`PLAN.md`](PLAN.md) | Estrategia, técnica, timeline, roles. |
| `doc1.txt` / `doc2.txt` / `*.pdf` | Specs oficiales del challenge (referencia). |

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
- Toda mejora se valida contra **OOD**, no solo ID.
- Split por `video`, nunca por frame.
- Docker offline real (`HF_HUB_OFFLINE=1`). Latencia p99 < 5s en L40S.
- NUNCA prompt-injection al juez (= descalificación).
