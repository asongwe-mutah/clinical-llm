# Model Card — Clinical-LLM (QLoRA adapter)

> Follows the spirit of the Hugging Face / Mitchell et al. model-card framework.
> All metrics below are measured, not estimated; reproduce with the command in
> [Evaluation](#evaluation).

## Model details

- **Developed by:** Asongwe Mutah (portfolio project).
- **Model type:** Decoder-only causal LM, domain-adapted by **QLoRA** (4-bit
  base + low-rank adapters) for clinical-informatics instruction following.
- **Base model:** `Qwen/Qwen2.5-3B-Instruct` (configurable; Llama-3.1-8B-Instruct
  supported on larger GPUs).
- **Adapter:** LoRA, rank 16, α 32, dropout 0.05, applied to all seven attention
  and MLP projections (`q,k,v,o,gate,up,down`) — ~30M trainable parameters,
  under 1% of the base model.
- **Trained:** 2026-09-26, 300 steps (~4.9M tokens) on a single Colab L4, bf16,
  NF4 double-quantized base, effective batch 16 at sequence length 1024.
- **Language:** English.
- **License:** Adapter released under MIT; **use is bound by the base model's
  license** (Qwen community license) and by each training dataset's terms.

## Intended use

- **In scope:** medical/clinical Q&A, exam-style reasoning, literature-grounded
  answers, and (stretch) clinical-note summarization — for **research,
  education, and demonstration** by informed users.
- **Out of scope / prohibited:** real clinical decision-making, diagnosis, or
  treatment; any use on identifiable patient data; any deployment where a wrong
  answer could reach a patient without expert review.

## Training data

| Source | Examples used | Task | License / access |
|---|---:|---|---|
| MedMCQA | 25,000 | Multiple-choice QA (21 subjects) | Public (Apache-2.0) |
| MedQuAD | 16,407 | Consumer-health open QA | Public (CC BY 4.0 terms) |
| PubMedQA (`pqa_labeled`) | 1,000 | Grounded yes/no/maybe QA | Public (MIT) |
| MIMIC-IV-Note *(optional stretch)* | — | Note summarization | **Credentialed (PhysioNet DUA); not redistributed** |

42,407 examples total → 41,559 train / 848 validation. All normalized to a
single chat schema with a safety-framed system prompt
(`src/clinical_llm/data/formatting.py`), which is the same renderer used at eval
and serve time — there is no train/serve prompt skew.

The 25k per-source cap binds only on MedMCQA: PubMedQA's labeled split contains
just 1,000 items in total, and MedQuAD contributed all 16,407 it has.

## Evaluation

Accuracy by letter-constrained log-likelihood (lm-eval-harness style), base vs.
fine-tuned, on **held-out** splits. Both models score the **same items in the
same order**, so the comparison is paired.

**n = 1,000 per benchmark. Evaluated 2026-09-27.**

| Benchmark | Base | Fine-tuned | Δ | McNemar *p* |
|---|---:|---:|---:|---:|
| MedMCQA (val) | 47.10% | **52.90%** | **+5.80 pp** | 4.3 × 10⁻⁴ |
| PubMedQA | 64.10% | **74.00%** | **+9.90 pp** | 7.8 × 10⁻¹¹ |

Both improvements are statistically significant at α = 0.05.

<details>
<summary>Full statistics, including what the fine-tune breaks</summary>

| Benchmark | Base 95% CI | Fine-tuned 95% CI | Fixed | Regressed | Discordant |
|---|---|---|---:|---:|---:|
| MedMCQA | [0.440, 0.502] | [0.498, 0.560] | 160 | 102 | 262 |
| PubMedQA | [0.611, 0.670] | [0.712, 0.766] | 163 | 64 | 227 |

Intervals are Wilson score intervals. *Fixed* = base wrong, fine-tuned right;
*regressed* = the reverse.

**Fine-tuning is not uniformly positive.** On MedMCQA it corrects 160 items but
breaks 102 — a net gain of 58, a fix-to-regression ratio of 1.57:1, with 26% of
items changing answer in one direction or the other. PubMedQA is cleaner at
2.55:1. Reporting only the net delta would hide that a quarter of MedMCQA
answers moved, and that a non-trivial number moved the wrong way.

McNemar's test is used rather than a two-proportion z-test because the models
score identical items; the unpaired test discards that pairing and is needlessly
conservative. Implementation in `src/clinical_llm/eval/run_eval.py` (exact
binomial below 25 discordant pairs, χ² with Edwards continuity correction above),
validated against published critical values.

</details>

### Why n matters here — a methodology note

An earlier evaluation of a shorter training run at **n = 300** gave MedMCQA
+7.67 pp and PubMedQA +3.33 pp, and neither cleared significance. Both figures
were misleading, in opposite directions:

- The MedMCQA delta **shrank** to +5.80 pp at n = 1,000. The n = 300 base
  accuracy (0.4400) sat ~3 pp below the n = 1,000 estimate (0.4710), inflating
  the apparent gain.
- The PubMedQA delta **grew** from +3.33 pp to +9.90 pp, partly from more
  training and partly because n = 300 could not resolve it at all.

The larger, paired evaluation is the one to trust. Per-item outcomes are stored
in `reports/eval.json`, so these tests can be recomputed without re-running
inference.

### Reproduce

```bash
python -m clinical_llm.eval.run_eval \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/clinical-qlora \
  --benchmarks medmcqa pubmedqa \
  --max-items 1000 \
  --out reports/eval.json
```

MedQA is supported (`--benchmarks medqa`) but was not scored for this card.

## Limitations & risks

- **Hallucination:** may produce fluent but incorrect medical claims.
- **Regressions:** as above, the adapter makes some previously-correct answers
  wrong. Net accuracy improves; per-item behaviour is not monotonic.
- **Bias:** MedMCQA skews toward the Indian medical curriculum; MedQuAD toward
  US consumer health — coverage and phrasing biases follow. MedMCQA is also 59%
  of the training mix, so that skew is weighted heavily.
- **Packing without FlashAttention-2:** training used sequence packing on a GPU
  where FA2 was unavailable, so packed samples were not block-diagonally masked
  and could attend across example boundaries. A known, uncorrected source of
  noise in the training signal.
- **Benchmark scope:** MedMCQA and PubMedQA are multiple-choice. They measure
  answer selection, not generation quality, calibration, or safety of free-text
  output — none of which are evaluated here.
- **No calibration guarantee:** confident tone ≠ correctness.
- **Not a medical device.** Not FDA-cleared. Not for clinical use.

## Environmental / compute

- One QLoRA run on a single Colab **L4**: ~2.6 GPU-hours for 300 steps
  (measured 30.9 s/it at 525 tokens/s), plus ~1 hour of evaluation.
- Evaluation ran on Apple Silicon (MPS), not a datacentre GPU.
- No pre-training from scratch; parameter-efficient tuning updates <1% of
  weights.
