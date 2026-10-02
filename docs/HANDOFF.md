# Handoff — Clinical-LLM

**As of 2026-10-02, HEAD `f1ad0db`.** Written so anyone (including future-you)
can pick this up without reading the conversation that produced it.

---

## 1. What this is

A **domain-adapted language model** for clinical informatics: a QLoRA adapter
(~30M trainable params, <1% of weights) on `Qwen/Qwen2.5-3B-Instruct`, plus an
evaluation harness with paired significance testing and a FastAPI + browser chat
demo.

Deliberate about what it is **not**:

- **Not a biomedical LLM.** No pretraining on biomedical corpora. This is
  supervised fine-tuning (~4.9M tokens) on top of a general model — four to five
  orders of magnitude less training than BioGPT/Meditron-style models. The
  medical knowledge was already in Qwen; the adapter made it easier to elicit.
- **Not an agent.** No tools, no retrieval, no planning loop. One prompt in,
  one answer out. (The lab-SOP RAG project is the retrieval/agent piece.)
- **Not a chatbot product.** `ui/index.html` + FastAPI `/chat` is a demo
  surface, not the substance.

Honest one-liner:

> QLoRA domain adaptation of Qwen2.5-3B-Instruct for clinical informatics, with
> a paired-statistics evaluation harness. +7.30 pp on MedQA (held out from
> training entirely) and +5.80 pp on MedMCQA, both significant by McNemar at
> n=1000.

---

## 2. Current results

Adapter: `outputs/clinical-qlora/`, saved **2026-09-26 23:09**, r=16, α=32.
All numbers n=1000, paired, McNemar (χ² with Edwards continuity correction).

| Benchmark | Share of training corpus | Base | Fine-tuned | Δ | McNemar *p* | Fix:regress |
|---|---:|---:|---:|---:|---:|---:|
| **MedQA** (USMLE) | **0%** | 42.80% | **50.10%** | **+7.30 pp** | 1.0×10⁻⁶ | 2.01:1 |
| MedMCQA (val) | 59% | 47.10% | **52.90%** | +5.80 pp | 4.3×10⁻⁴ | 1.57:1 |
| ~~PubMedQA~~ | — | ~~64.10%~~ | ~~74.00%~~ | ~~+9.90 pp~~ | **WITHDRAWN** | — |

**MedQA is the headline.** It is absent from the training corpus entirely, yet
shows a *larger* lift and a cleaner fix-to-regression ratio than MedMCQA, which
supplies 59% of the training data. The adaptation transferred rather than
learning MedMCQA's house style — the opposite of the more common outcome.

Absolute accuracy is unremarkable; a 3B model near 50% on USMLE questions is not
a strong clinical reasoner. **The delta is the measurement, the absolute is
context.** Say both.

Evidence is committed: `reports/eval.json`, `reports/eval_medqa.json` carry
per-item outcomes, so every test can be recomputed without re-running inference.

---

## 3. The single most important thing to know

**PubMedQA was contaminated and the result is withdrawn.**

`iter_pubmedqa` trained on `qiaojin/PubMedQA` config `pqa_labeled` split
`train`. `load_pubmedqa_test` evaluates the *same* dataset, config and split.
`pqa_labeled` holds exactly 1,000 items and `prepare.py` holds out only 2% — so
**~980 of the 1,000 scored items had been trained on**. The +9.90 pp was
memorisation.

Fixed in `7a79fe2`: training now reads `pqa_artificial` (disjoint). Guarded by
`tests/test_no_contamination.py` — five offline tests that assert no training
source and benchmark share a `(dataset, config, split)` triple, verified to fail
by name on the exact configuration that caused it.

**But there is still no valid PubMedQA number.** The only run on the corrected
corpus crashed. Getting one requires a retrain (§6).

---

## 4. Bug history — read before changing training code

Each of these was a *silent* failure that produced plausible output. They are
the reason the configs carry long comment headers; do not strip them.

| Bug | Symptom | Fix |
|---|---|---|
| **Precision fallback** | 59-hour projection on T4, 438 s/it | `fp16` was keyed off *requested* `cfg.bf16`, not effective — bf16 on non-Ampere silently gave fp32. `resolve_precision()` now guarantees exactly one flag. `2868f70` |
| **`is_bf16_supported()`** | reports True on Turing via slow emulation | check compute capability ≥ 8.0 directly. `2868f70` |
| **Packing without FA2** | 402 tok/s, 8% of L4 peak | packed attention mask forces sdpa onto its MATH backend, materialising seq×seq per layer. Also let packed samples attend across each other. `packing: false`. `cf65022` |
| **Dead `eval_steps`** | no `eval_loss` ever, for months | declared in config, never passed to `SFTConfig`, no eval strategy set. `5fd7307` |
| **Train/eval contamination** | +9.90 pp that meant nothing | §3. `7a79fe2` |
| **Step budget vs packing** | 600 steps was ~5.8 epochs, not ~1 | packing collapses samples into sequences; scale `max_steps` with the corpus. `ca59284` |
| **Warnings that didn't stop anything** | 3.5 h lost, Drive unmounted, 74% reclaim | notebook now *refuses* to train without writable Drive; `--max-hours` *aborts* at the probe. `998b761` |

Pattern worth internalising: **every one of these produced believable output.**
Nothing crashed. The tests exist because the failures were invisible.

---

## 5. How to run things

### Evaluate (local, Mac, no GPU needed)

```bash
cd ~/clinical-llm && source .venv/bin/activate
python -m clinical_llm.eval.run_eval \
  --base Qwen/Qwen2.5-3B-Instruct --adapter outputs/clinical-qlora \
  --benchmarks medqa --max-items 1000 --out reports/eval_medqa.json
```

~35 min per benchmark on MPS. Prints McNemar directly. **Write each benchmark to
its own `--out`** — do not overwrite committed evidence.

### Install a downloaded adapter

```bash
python scripts/install_adapter.py --dry-run   # shows what it would pick
python scripts/install_adapter.py
```

Handles Drive's multi-part zips *and* browser-auto-extracted folders, ranks
candidates by adapter save time, backs up the existing one, skips checkpoints.
Check the `incoming:` vs `current:` dates before confirming.

### Train (Colab only — the one step needing a GPU)

1. New runtime, **L4**. Upload `clinical-llm.zip`.
2. Open the notebook, tick `FRESH_START` if old checkpoints exist in Drive.
3. Run all. Watch three things:
   - cell 8: `using configs/data_gpu.yaml`, **66,407 examples**
   - step 3: throughput probe (~15 s/it ⇒ ~3 h; aborts itself above the budget)
   - ~13 min in: `checkpoint-50` appears in Drive

### Test

```bash
pytest -q          # 5 files; contamination + precision suites are the load-bearing ones
ruff check src tests   # the exact CI gate
```

---

## 6. Open items, ranked

1. **Push to GitHub.** 20 commits, local only, branch `master`, **no remote
   configured**. `github.com/asongwe-mutah/clinical-llm` 404s, so the README's
   CI badge, Colab badge and the notebook's clone fallback all point at nothing.
   Zero GPU, highest visibility-per-effort. Rename `master` → `main` first.
2. **PubMedQA retrain** (~3 h Colab + ~1.75 h local eval). Gives three valid
   benchmarks from one adapter and answers whether `packing: false` reduces the
   102 MedMCQA regressions. Config is ready and guarded.
3. **`docs/BRIEF.md`.** The originating prompt (Claude conversation, 2026-07-27)
   is not saved anywhere. Everything else is documented — why packing is off,
   why `pqa_artificial`, why McNemar — but not the requirements that shaped it.
4. **Delete `_to_delete/`** — stale git lock files, untracked.

### If you retrain: decide this *before* seeing numbers

The new adapter may be worse (11,200 samples vs the current 13,600, different
recipe). If MedQA drops below +7.30 pp, report the current adapter as the
headline and the new one as the clean-PubMedQA datapoint. **Do not pick whichever
wins per benchmark** — that is the same failure mode as the contamination, just
slower.

---

## 7. Operating hazards

- **Colab reclaims at ~4.2–4.6 h** in practice. Budget ≤3 h.
- **Drive mount fails silently and often.** The notebook now hard-blocks on it;
  do not route around that with `ALLOW_NO_DRIVE` for a long run.
- **`flash-attn` install OOM-kills the runtime.** Default off. At seq-len 1024
  with packing off, sdpa is fine.
- **A fresh Colab runtime wipes `/content`** — code, corpus, checkpoints. The
  execution counter resetting to `[1]` is the tell.
- **`device_bash` is a Linux VM, not macOS.** The repo's `.venv` is macOS-arm64
  and cannot execute there. Evals run in the Mac terminal.
- **Repeated tuning against MedMCQA validation will overfit to that split.**
  MedQA has now been scored once; treat further use of it the same way.

---

## 8. Environment

Local venv (Mac, MPS): torch 2.13.0, transformers 5.14.1, peft 0.19.1,
trl 1.9.1, datasets 5.0.0. `bitsandbytes` absent — irrelevant, eval runs fp16
unquantized.

Colab trains under PEFT 0.20.0, which writes config keys 0.19.1 ignores
(`monteclora_config`, `velora_config`). Both are null; verified harmless.

Corpus (`data_gpu.yaml`, 25k cap per source): MedMCQA 25,000 + MedQuAD 16,407 +
PubMedQA `pqa_artificial` 25,000 = **66,407** → 65,079 train / 1,328 val.
The cap binds only on MedMCQA (182,822 available).
