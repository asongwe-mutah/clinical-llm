# Clinical-LLM — a domain-adapted language model for clinical informatics

[![CI](https://github.com/asongwe-mutah/clinical-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/asongwe-mutah/clinical-llm/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asongwe-mutah/clinical-llm/blob/main/notebooks/clinical_llm_colab.ipynb)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Fine-tunes an open instruction model into a **clinical-informatics assistant**
using **QLoRA**, then ships an **evaluation harness** against public medical-QA
benchmarks and a **FastAPI + Docker inference service** with a browser chat UI.

> ⚠️ **Research / portfolio project — not a medical device.** Outputs can be
> wrong and are **not medical advice**. No patient data (PHI) is used or shipped.
> See [`docs/SAFETY.md`](docs/SAFETY.md).

---

## Results

QLoRA adapter (r=16, <1% of weights) on `Qwen/Qwen2.5-3B-Instruct`, base vs
fine-tuned on the **same 1,000 items per benchmark**, paired McNemar test.

| Benchmark | Share of training corpus | Base | Fine-tuned | Δ | McNemar *p* | Fix:regress |
|---|---:|---:|---:|---:|---:|---:|
| **MedQA** (USMLE, test) | **0%** | 42.90% | **50.80%** | **+7.90 pp** | 1.5×10⁻⁷ | 150:71 |
| MedMCQA (val) | 38% | 47.10% | **52.70%** | +5.60 pp | 5.3×10⁻⁴ | 154:98 |
| PubMedQA (`pqa_labeled`) | 0%¹ | 64.10% | **73.60%** | +9.50 pp | 1.4×10⁻⁹ | 168:73 |

¹ Training uses PubMedQA's `pqa_artificial` config; evaluation uses
`pqa_labeled`. The two share **0 pubids** (checked across all 211,269
`pqa_artificial` items). Same task format, disjoint articles.

**MedQA is the headline**: it is absent from training entirely and still moves
+7.90 pp. **The delta is the measurement, the absolute is context** — a 3B
model near 51% on USMLE-style questions is not a strong clinical reasoner, and
the fine-tune also *breaks* 71–98 previously-correct answers per benchmark.

**The adapter is published:**
[`mutahfon/clinical-qlora-qwen2.5-3b`](https://huggingface.co/mutahfon/clinical-qlora-qwen2.5-3b)
on Hugging Face — the exact weights behind the table above. Load it on top of
the base model without retraining:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device != "cpu" else torch.float32

tok = AutoTokenizer.from_pretrained("mutahfon/clinical-qlora-qwen2.5-3b")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=dtype)
model = PeftModel.from_pretrained(model, "mutahfon/clinical-qlora-qwen2.5-3b").to(device).eval()
```

A full generation example, including the system prompt the adapter was trained
with, is on the Hugging Face model page.

The eval harness takes the Hub id directly:
`run_eval --adapter mutahfon/clinical-qlora-qwen2.5-3b`.

**An earlier PubMedQA result (+9.90 pp) was withdrawn.** The first adapter was
trained on the same 1,000 `pqa_labeled` items it was then scored on, so that
number was memorisation. The corpus was fixed, a contamination test suite was
added, and the model was retrained; the +9.50 pp above is from the retrained
adapter. The rule for which adapter to report was fixed *before* the retrain
was scored (MedQA ≥ +7.30 pp, the first adapter's figure, or the old adapter
stays the headline). Full history, both adapters' numbers and per-item evidence:
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), [`docs/HANDOFF.md`](docs/HANDOFF.md),
`reports/eval_retrain_pqa_artificial_*.json`.

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
tests/           # fast, GPU-free unit tests (36 passing)
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
pytest -q          # 36 passing: prompt contract, config loading, data adapters
```

### 2. Smoke-train end-to-end locally (CPU / Apple-Silicon MPS)

Proves the full data → train → adapter path on a tiny model. Not a quality run.

```bash
pip install -r requirements.txt
./scripts/smoke_train.sh
```

### 3. Full QLoRA fine-tune (CUDA GPU — Colab / RunPod / Lambda / local 3090+)

**Easiest path — one click:** open the [Colab notebook](notebooks/clinical_llm_colab.ipynb)
([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asongwe-mutah/clinical-llm/blob/main/notebooks/clinical_llm_colab.ipynb)),
set the runtime to GPU, and **Run all** — it installs, builds the corpus, fine-tunes,
evaluates, and lets you chat with the result on a free T4.

**Two configs, picked from the hardware you actually get:**

| GPU | Config | Precision | Attention | Expect |
|---|---|---|---|---|
| T4 (free Colab, sm_75) | `train_colab.yaml` | fp16 | sdpa | ~2–3 h |
| L4 (sm_89) | `train_gpu.yaml` | bf16 | FlashAttn-2 | ~45–70 min |
| A100 (sm_80) | `train_gpu.yaml` | bf16 | FlashAttn-2 | ~20–30 min |

A T4 has neither native bf16 nor FlashAttention-2, which is why it is the slow
case by a wide margin — the trainer detects both and degrades automatically
(see [Design decisions](#design-decisions-worth-calling-out)). Before walking
away, check the startup log says `precision=bf16` (or `fp16` on a T4, **never
`fp32`**), then multiply the progress bar's `s/it` by `max_steps`. If the
projection exceeds your session limit, lower `max_steps` rather than hoping.

Or from a shell on any CUDA box:

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
  --max-items 1000 --out reports/eval_myrun.json
```

Writes per-benchmark accuracy, the **base→fine-tuned delta** and the paired
McNemar test to the `--out` file. Use a fresh filename: the `reports/*.json`
already committed are the evidence behind the [Results](#results) table and
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), and should not be overwritten.

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
- **Precision is chosen from the hardware, not the config.** `bf16: true` means
  "bf16 if this card has it," and `resolve_precision()` guarantees exactly one of
  bf16/fp16 is set on CUDA. The check is compute-capability ≥ 8.0 rather than
  `torch.cuda.is_bf16_supported()`, which returns True on Turing via a software
  emulation path an order of magnitude slower than fp16 — the difference between
  a 3-hour T4 run and a 60-hour one.
- **Packing and attention are chosen together.** `packing: true` is only sound
  with FlashAttention-2, which supplies the block-diagonal mask that keeps packed
  samples from attending across each other. `resolve_attn_implementation()` asks
  for FA2 when the GPU and the package both support it, falls back to `sdpa`
  otherwise, and warns when packing is on without it — so the quality caveat is
  visible in the log rather than silent.
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

## Tests & developer setup

```bash
pytest -q          # fast, offline; runs in CI-friendly < 1s
ruff check src tests

# optional: run the same lint gate as CI automatically before every commit
pip install pre-commit && pre-commit install
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `ruff` + `pytest`
on Python 3.10–3.12 for every push and PR — installing only the light core deps,
so it needs no GPU or network.

## License

Code: MIT (`LICENSE`). Use of any base model and dataset is bound by their
respective licenses; see the model card.
