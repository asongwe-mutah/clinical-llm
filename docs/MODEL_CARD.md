# Model Card — Clinical-LLM (QLoRA adapter)

> This card follows the spirit of the Hugging Face / Mitchell et al. model-card
> framework. Fill in the bracketed metrics after your training run.

## Model details

- **Developed by:** Asongwe Mutah (portfolio project).
- **Model type:** Decoder-only causal LM, domain-adapted by **QLoRA** (4-bit
  base + low-rank adapters) for clinical-informatics instruction following.
- **Base model:** `Qwen/Qwen2.5-3B-Instruct` (configurable; Llama-3.1-8B-Instruct
  supported on larger GPUs).
- **Adapter:** LoRA, rank 16, α 32, applied to attention + MLP projections.
- **Language:** English.
- **License:** Adapter released under MIT; **use is bound by the base model's
  license** (e.g. the Qwen / Llama community license) and by each training
  dataset's terms.

## Intended use

- **In scope:** medical/clinical Q&A, exam-style reasoning, literature-grounded
  answers, and (stretch) clinical-note summarization — for **research,
  education, and demonstration** by informed users.
- **Out of scope / prohibited:** real clinical decision-making, diagnosis, or
  treatment; any use on identifiable patient data; any deployment where a wrong
  answer could reach a patient without expert review.

## Training data

| Source | Task | License / access |
|---|---|---|
| PubMedQA (`pqa_labeled`) | Grounded yes/no/maybe QA | Public (MIT) |
| MedMCQA | Multiple-choice QA (21 subjects) | Public (Apache-2.0) |
| MedQuAD | Consumer-health open QA | Public (CC BY 4.0 terms) |
| MIMIC-IV-Note *(optional stretch)* | Note summarization | **Credentialed (PhysioNet DUA); not redistributed**|

All examples are normalized to a single chat schema with a safety-framed system
prompt (see `src/clinical_llm/data/formatting.py`).

## Evaluation

Accuracy via letter-constrained log-likelihood (lm-eval-harness style), base vs.
fine-tuned, on **held-out** splits (MedMCQA validation, MedQA test, PubMedQA).

| Benchmark | Base | Fine-tuned | Δ |
|---|---:|---:|---:|
| MedMCQA (val) | [ ] | [ ] | [ ] |
| MedQA (test)  | [ ] | [ ] | [ ] |
| PubMedQA      | [ ] | [ ] | [ ] |

> Reproduce with `make eval` (writes `reports/eval.json`).

## Limitations & risks

- **Hallucination:** may produce fluent but incorrect medical claims.
- **Bias:** MedMCQA skews toward the Indian medical curriculum; MedQuAD toward
  US consumer health — coverage and phrasing biases follow.
- **No calibration guarantee:** confident tone ≠ correctness.
- **Not a medical device.** Not FDA-cleared. Not for clinical use.

## Environmental / compute

- Single-GPU QLoRA run (~[X] GPU-hours on [A100-40GB]); no pre-training from
  scratch. Parameter-efficient tuning updates <1% of weights.
