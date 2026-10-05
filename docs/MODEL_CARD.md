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
- **Trained (current adapter):** 2026-10-02, 700 steps = 11,200 samples
  (~4.4M tokens, 0.17 epoch) on a single Colab GPU, NF4 double-quantized base,
  effective batch 16, packing off. Final validation loss 1.105, still falling
  slowly (1.129 at step 100).
- **Previous adapter:** 2026-09-26, 300 steps (~4.9M tokens, packed) on a
  Colab L4, bf16, sequence length 1024. Superseded; its results are kept under
  [Evaluation](#evaluation) as the record of the PubMedQA withdrawal.
- **Language:** English.
- **Weights:** published at
  [`mutahfon/clinical-qlora-qwen2.5-3b`](https://huggingface.co/mutahfon/clinical-qlora-qwen2.5-3b).
- **License:** Code is MIT. The adapter is a derivative of the base model, so
  **its use is bound by the base model's license** (Qwen Research License) and
  by each training dataset's terms.

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
| PubMedQA (`pqa_artificial`) | 25,000 | Grounded yes/no/maybe QA | Public (MIT) |
| MIMIC-IV-Note *(optional stretch)* | — | Note summarization | **Credentialed (PhysioNet DUA); not redistributed** |

66,407 examples total → 65,079 train / 1,328 validation; the current adapter
saw 11,200 of them (700 steps, a fraction of one epoch). All normalized to a
single chat schema with a safety-framed system prompt
(`src/clinical_llm/data/formatting.py`), which is the same renderer used at eval
and serve time — there is no train/serve prompt skew.

The 25k per-source cap binds on MedMCQA (182,822 available) and on
`pqa_artificial` (211,269 available); MedQuAD contributed all 16,407 it has.
`pqa_artificial` shares no pubids with `pqa_labeled`, the 1,000-item config the
PubMedQA benchmark is scored on.

The **previous adapter** was trained on a different, smaller corpus: 42,407
examples (41,559 train / 848 validation), with PubMedQA drawn from the 1,000
`pqa_labeled` items — the same items it was then evaluated on. That is the
contamination described under [Evaluation](#evaluation).

## Evaluation

Accuracy by letter-constrained log-likelihood (lm-eval-harness style), base vs.
fine-tuned, on **held-out** splits. Both models score the **same items in the
same order**, so the comparison is paired.

### Current adapter (retrained on the corrected corpus, scored 2026-10-04)

**n = 1,000 per benchmark**, one adapter for all three. Trained with PubMedQA
drawn from `pqa_artificial` (corpus: MedMCQA 25,000 + MedQuAD 16,407 +
`pqa_artificial` 25,000 = 66,407), packing off, 700 steps = 11,200 samples.

| Benchmark | Share of training corpus | Base | Fine-tuned | Δ | McNemar *p* | Fixed | Regressed |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MedQA (test)** | **0%** | 42.90% | **50.80%** | **+7.90 pp** | 1.5 × 10⁻⁷ | 150 | 71 |
| MedMCQA (val) | 38% | 47.10% | **52.70%** | +5.60 pp | 5.3 × 10⁻⁴ | 154 | 98 |
| PubMedQA (`pqa_labeled`) | 0% of scored items | 64.10% | **73.60%** | +9.50 pp | 1.4 × 10⁻⁹ | 168 | 73 |

This adapter is the headline under a rule fixed before it was scored: it had to
match or beat the previous adapter's +7.30 pp on MedQA, otherwise the previous
adapter stayed the headline and this one would have been reported only as the
clean PubMedQA datapoint. PubMedQA is reinstated because `pqa_artificial` and
`pqa_labeled` share 0 pubids (checked across all 211,269 `pqa_artificial`
items); it remains the same task format, so it is held-out items, not an
out-of-distribution test. MedMCQA is unchanged within noise against the previous
adapter. Per-item outcomes: `reports/eval_retrain_pqa_artificial_*.json`.

**Second MedQA sample.** On the MedQA `dev` split (n = 1,272, disjoint from
test) the same adapter scores 47.01% → 50.39%, **+3.38 pp** (p = 0.012; 160
fixed, 117 regressed; `reports/eval_v1_medqa_dev.json`). Fine-tuned accuracy
matches the test split; the base model is stronger on dev, so the lift is less
than half as large. Quote the two together.

The remainder of this section describes the **previous adapter** (saved
2026-09-26) and is kept as the record of how the PubMedQA contamination was
found and withdrawn.

### Previous adapter

**n = 1,000 per benchmark.** MedMCQA and PubMedQA evaluated 2026-09-27;
MedQA 2026-10-02. Same adapter throughout.

| Benchmark | Share of training corpus | Base | Fine-tuned | Δ | McNemar *p* |
|---|---:|---:|---:|---:|---:|
| MedMCQA (val) | 59% | 47.10% | **52.90%** | **+5.80 pp** | 4.3 × 10⁻⁴ |
| **MedQA (test)** | **0%** | 42.80% | **50.10%** | **+7.30 pp** | 1.0 × 10⁻⁶ |
| ~~PubMedQA~~ | — | ~~64.10%~~ | ~~74.00%~~ | ~~+9.90 pp~~ | **withdrawn — see below** |

**The MedQA result is the headline.** MedQA (USMLE-style clinical vignettes) is
**absent from the training corpus entirely** — a different task format, a
different source, never seen during fine-tuning. It nonetheless shows a *larger*
lift than MedMCQA, which supplies 59% of the training data, and a cleaner
fix-to-regression ratio (2.01:1 vs 1.57:1).

That ordering is the substantive finding: the adaptation transferred rather
than memorising MedMCQA's house style. A gain confined to the in-distribution
benchmark would have been the weaker, more common outcome.

Absolute accuracy is unremarkable — a 3B model near 50% on USMLE-style
questions is not a strong clinical reasoner, and nothing here should be read as
one. The *delta* is the measurement; the absolute number is context.

> ### ⚠️ The PubMedQA result was contaminated and is withdrawn
>
> Training read `qiaojin/PubMedQA` config `pqa_labeled` split `train`. So did
> the evaluation. `pqa_labeled` contains exactly 1,000 items, and `prepare.py`
> holds out only 2% for validation — so roughly **980 of the 1,000 evaluated
> items had been trained on**. The +9.90 pp was memorisation, not
> generalisation, and no conclusion should be drawn from it.
>
> Fixed in `data/datasets.py`: training now reads the disjoint `pqa_artificial`
> split. `tests/test_no_contamination.py` asserts that no training source and
> evaluation benchmark share a (dataset, config, split) triple, and fails on
> the exact configuration that caused this.
>
> PubMedQA was re-scored on 2026-10-04 after a retrain on the corrected corpus:
> 64.10% → 73.60% (+9.50 pp). See *Current adapter* above. The +9.90 pp stays
> withdrawn.

<details>
<summary>Full statistics, including what the fine-tune breaks</summary>

| Benchmark | Base 95% CI | Fine-tuned 95% CI | Fixed | Regressed | Ratio | Discordant |
|---|---|---|---:|---:|---:|---:|
| MedMCQA | [0.440, 0.502] | [0.498, 0.560] | 160 | 102 | 1.57:1 | 262 (26.2%) |
| MedQA | [0.398, 0.459] | [0.470, 0.532] | 145 | 72 | 2.01:1 | 217 (21.7%) |
| ~~PubMedQA~~ *(contaminated)* | ~~[0.611, 0.670]~~ | ~~[0.712, 0.766]~~ | ~~163~~ | ~~64~~ | — | ~~227~~ |

Intervals are Wilson score intervals. *Fixed* = base wrong, fine-tuned right;
*regressed* = the reverse.

**Fine-tuning is not uniformly positive.** On MedMCQA it corrects 160 items but
breaks 102 — a net gain of 58, a ratio of 1.57:1, with 26% of items changing
answer in one direction or the other. MedQA is cleaner at 2.01:1 (145 fixed,
72 broken). Reporting only the net delta would hide that roughly a quarter of
answers moved on each benchmark, and that a non-trivial number moved the wrong
way. Per-item outcomes for both are stored, so this is checkable.

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
- The PubMedQA delta appeared to **grow** from +3.33 pp to +9.90 pp — but that
  benchmark was contaminated in both evaluations, so neither figure means
  anything.

The larger, paired evaluation is the one to trust. Per-item outcomes are stored
in `reports/eval.json`, so these tests can be recomputed without re-running
inference.

### Reproduce

```bash
# MedMCQA (reports/eval.json); the PubMedQA column there is the withdrawn one
python -m clinical_llm.eval.run_eval \
  --base Qwen/Qwen2.5-3B-Instruct --adapter outputs/clinical-qlora \
  --benchmarks medmcqa pubmedqa --max-items 1000 --out reports/eval.json

# MedQA (reports/eval_medqa.json)
python -m clinical_llm.eval.run_eval \
  --base Qwen/Qwen2.5-3B-Instruct --adapter outputs/clinical-qlora \
  --benchmarks medqa --max-items 1000 --out reports/eval_medqa.json
```

## Limitations & risks

- **Hallucination:** may produce fluent but incorrect medical claims.
- **Regressions:** as above, the adapter makes some previously-correct answers
  wrong. Net accuracy improves; per-item behaviour is not monotonic.
- **Bias:** MedMCQA skews toward the Indian medical curriculum; MedQuAD toward
  US consumer health — coverage and phrasing biases follow. MedMCQA is 38% of
  the training mix (59% for the previous adapter), so that skew carries weight.
- **Packing without FlashAttention-2 (previous adapter only):** training used sequence packing on a GPU
  where FA2 was unavailable, so packed samples were not block-diagonally masked
  and could attend across example boundaries. A known, uncorrected source of
  noise in the training signal.
- **Three benchmarks, only one out-of-distribution.** MedQA is the independent
  check and the strongest evidence. MedMCQA supplies 38% of the training mix,
  and PubMedQA is scored on held-out articles in a format the model trained on,
  so those two results are the less surprising ones. The contaminated PubMedQA
  figure from the previous adapter stays withdrawn.
- **MedQA is no longer untouched.** It has been scored once per adapter, and the
  second score decided which adapter to report. Further recipe changes should be
  selected on a separate dev split, not on these 1,000 items.
- **Benchmark scope:** MedMCQA, MedQA and PubMedQA are all multiple-choice. They measure
  answer selection, not generation quality, calibration, or safety of free-text
  output — none of which are evaluated here.
- **No calibration guarantee:** confident tone ≠ correctness.
- **Not a medical device.** Not FDA-cleared. Not for clinical use.

## Environmental / compute

- Current adapter: one QLoRA run of 700 steps on a single Colab GPU (budgeted
  at ~3 GPU-hours), plus three n = 1,000 evaluations on Colab (CUDA).
- Previous adapter: one run on a Colab **L4**, ~2.6 GPU-hours for 300 steps
  (measured 30.9 s/it at 525 tokens/s), plus ~1 hour of evaluation on Apple
  Silicon (MPS).
- No pre-training from scratch; parameter-efficient tuning updates <1% of
  weights.
