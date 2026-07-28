# Clinical-LLM — a domain-adapted language model for clinical informatics

[![CI](https://github.com/asongwe-mutah/clinical-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/asongwe-mutah/clinical-llm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Fine-tunes an open instruction model into a **clinical-informatics assistant**
using **QLoRA**, then ships an **evaluation harness** against public medical-QA
benchmarks and a **FastAPI + Docker inference service** with a browser chat UI.

> ⚠️ **Research / portfolio project — not a medical device.** Outputs can be
> wrong and are **not medical advice**. No patient data (PHI) is used or shipped.
> See [`docs/SAFETY.md`](docs/SAFETY.md).

---

## Why this exists (and why not "train an LLM from scratch")

Pre-training a genuinely *large* model from scratch costs six-to-seven figures of
compute and would underperform an existing open model. The technique that
actually ships in industry — and that demonstrates real ML-engineering skill —
is **parameter-efficient domain adaptation**: take a strong open base model and
QLoRA-tune it on curated in-domain data, then *measure* the lift. That is what
this repo does, end to end, reproducibly.

It is the generative-modeling companion to two other portfolio artifacts: a
multimodal 7-day post-transfusion mortality model (prediction) and a
citation-grounded lab-SOP RAG agent (retrieval).

## What's in the box

```
src/clinical_llm/
├── data/        # dataset adapters + corpus builder (unified chat schema)
│   └── formatting.py   ← single source of truth for prompts (no train/serve skew)
├── train/       # QLoRA SFT trainer (TRL + PEFT) + LoRA→base merge
├── eval/        # MCQ benchmarks + log-likelihood scoring (base vs fine-tuned)
├── serve/       # inference engine + FastAPI app (streaming)
└── utils/       # typed YAML config, logging
configs/         # data.yaml, train_qlora.yaml, train_smoke.yaml, ...
ui/              # self-contained browser chat demo
docker/          # CUDA serving image + compose
docs/            # MODEL_CARD.md, SAFETY.md
tests/           # fast, GPU-free unit tests (22 passing)
```

## Architecture

```
 Public medical corpora                 Unified chat corpus            QLoRA fine-tune            Serving
 ┌─────────────────────┐   prepare.py   ┌──────────────────┐  train_qlora ┌──────────────┐  merge  ┌──────────────┐
 │ PubMedQA (grounded) │──────────────▶ │ {system,user,    │────────────▶ │ 4-bit base + │───────▶ │ FastAPI /chat│
 │ MedMCQA  (MCQ)      │  normalize +   │  assistant} JSONL│   TRL SFT    │ LoRA adapters│  or A+B │ + stream UI  │
 │ MedQuAD  (open QA)  │  safety prompt │  train / val     │              │ (<1% params) │         │ + disclaimer │
 │ MIMIC notes (opt.)  │                └──────────────────┘              └──────┬───────┘         └──────────────┘
 └─────────────────────┘                                                        │ eval (base vs tuned)
                                                                          ┌──────▼───────┐
                                                                          │ MedMCQA/MedQA│
                                                                          │ /PubMedQA acc│
                                                                          └──────────────┘
```

## Quickstart

### 1. Verify the code on your laptop (no GPU, no network)

```bash
pip install -e ".[dev]"
pytest -q          # 22 passing: prompt contract, config loading, data adapters
```

### 2. Smoke-train end-to-end locally (CPU / Apple-Silicon MPS)

Proves the full data → train → adapter path on a tiny model. Not a quality run.

```bash
pip install -r requirements.txt
./scripts/smoke_train.sh
```

### 3. Full QLoRA fine-tune (CUDA GPU — Colab / RunPod / Lambda / local 3090+)

```bash
python -m clinical_llm.data.prepare  --config configs/data.yaml
python -m clinical_llm.train.train_qlora --config configs/train_qlora.yaml
```

Defaults: `Qwen2.5-3B-Instruct`, 4-bit NF4, LoRA r=16, seq-len 2048 — fits a
single 24 GB card. Swap `base_model` to `meta-llama/Llama-3.1-8B-Instruct` on a
40 GB+ GPU.

### 4. Evaluate the lift

```bash
python -m clinical_llm.eval.run_eval \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/clinical-qlora \
  --benchmarks medmcqa pubmedqa medqa \
  --max-items 500 --out reports/eval.json
```

Writes per-benchmark accuracy and the **base→fine-tuned delta** to
`reports/eval.json`. Report these numbers in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

### 5. Serve the demo

```bash
CLINICAL_LLM_ADAPTER=outputs/clinical-qlora ./scripts/serve.sh
# open http://localhost:8000
```

Or containerized (GPU): `docker compose -f docker/docker-compose.yml up --build`.

## Design decisions worth calling out

- **One prompt contract everywhere.** `data/formatting.py` renders every example
  for training, eval, and serving, eliminating the classic train/serve skew bug.
- **The same script runs on a laptop and an A100.** 4-bit quantization is guarded
  behind a CUDA/bitsandbytes check and degrades gracefully to LoRA off-GPU, so the
  pipeline is verifiable without renting a GPU.
- **Log-likelihood MCQ scoring**, not brittle string parsing — deterministic and
  standard (lm-eval-harness style).
- **Safety is learned + enforced:** framed in the training system prompt, attached
  to every API response, and shown in the UI.
- **Honest evaluation:** held-out splits only; base-vs-tuned comparison built in.

## The notes-summarization stretch task

Clinical-note summarization is wired in (`use_notes_summarization`) but ships
**no data**: it reads a local `{ "note", "summary" }` JSONL you produce under
your own PhysioNet MIMIC-IV credentials. The repo provides the *format and code
path*, never PHI.

## Tests

```bash
pytest -q          # fast, offline; runs in CI-friendly < 1s
```

## License

Code: MIT (`LICENSE`). Use of any base model and dataset is bound by their
respective licenses; see the model card.
