# Handoff — Clinical-LLM

**As of 2026-10-04** (retrained adapter installed and scored; previous
revision was 2026-10-02 at `f1ad0db`). Written so anyone (including future-you)
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
> a paired-statistics evaluation harness. +7.90 pp on MedQA (42.9% → 50.8%,
> held out from training entirely), +5.60 pp on MedMCQA (47.1% → 52.7%) and
> +9.50 pp on PubMedQA (64.1% → 73.6%), all significant by McNemar at n=1000.

---

## 2. Current results

### 2a. Current adapter — the retrain (headline)

Adapter: `outputs/clinical-qlora/`, saved **2026-10-03 00:16 UTC**, r=16, α=32.
Trained on the corrected corpus (`data_gpu.yaml`, PubMedQA from
`pqa_artificial`), packing off, 700 steps = 11,200 samples (0.17 epoch), final
`eval_loss` 1.105. Scored on Colab (CUDA) 2026-10-04.
All numbers n=1000, paired, McNemar (χ² with Edwards continuity correction).

| Benchmark | Share of training corpus | Base | Fine-tuned | Δ | McNemar *p* | Fix:regress |
|---|---:|---:|---:|---:|---:|---:|
| **MedQA** (USMLE) | **0%** | 42.90% | **50.80%** | **+7.90 pp** | 1.5×10⁻⁷ | 150:71 (2.11:1) |
| MedMCQA (val) | 38% | 47.10% | **52.70%** | +5.60 pp | 5.3×10⁻⁴ | 154:98 (1.57:1) |
| PubMedQA (`pqa_labeled`) | 0% of scored items | 64.10% | **73.60%** | +9.50 pp | 1.4×10⁻⁹ | 168:73 (2.30:1) |

**MedQA is the headline, and this adapter is the one to report.** The rule was
set before the retrain was scored (§6): if MedQA came in below +7.30 pp, the
old adapter stayed the headline. It came in at +7.90 pp, so the retrain takes
over for all three benchmarks. No per-benchmark picking between adapters.

MedQA is absent from the training corpus entirely, yet shows a *larger* lift
and a cleaner fix-to-regression ratio than MedMCQA, which is in the training
mix. The adaptation transferred rather than learning MedMCQA's house style —
the opposite of the more common outcome.

**PubMedQA is reinstated.** Training now reads `pqa_artificial`; evaluation
reads `pqa_labeled`. They share **0 pubids**, checked across all 211,269
`pqa_artificial` items. It is still the same task format from the same source,
so read it as in-format, held-out-items — not out-of-distribution like MedQA.

Absolute accuracy is unremarkable; a 3B model near 51% on USMLE questions is not
a strong clinical reasoner. **The delta is the measurement, the absolute is
context.** Say both.

What did not change: MedMCQA is flat against the old adapter (+5.60 vs
+5.80 pp, 98 regressions vs 102). Turning packing off did **not** meaningfully
reduce MedMCQA regressions; compared directly on the same 1,000 MedQA items
(`scripts/compare_runs.py`), the retrain is +0.70 pp over the old adapter with
32 fixed and 25 regressed, p = 0.43 — within noise. (Cross-hardware caveat
applies: one was scored on MPS, the other on CUDA.)

Evidence is committed: `reports/eval_retrain_pqa_artificial_{medqa,medmcqa,pubmedqa}.json`
carry per-item outcomes, so every test can be recomputed without re-running
inference. The eval manifest's adapter fingerprint (`20767b65b6e30aaa`) matches
the installed files' sizes and save times.

**The MedQA lift is smaller on a second sample — read this before quoting
+7.90.** Scored on MedQA `dev` (all 1,272 items, disjoint from test, MPS,
2026-10-04; `reports/eval_v1_medqa_dev.json`):

| Split | n | Base | Fine-tuned | Δ | McNemar *p* | Fix:regress |
|---|---:|---:|---:|---:|---:|---:|
| MedQA test (headline) | 1,000 | 42.90% | 50.80% | +7.90 pp | 1.5×10⁻⁷ | 150:71 |
| MedQA dev | 1,272 | 47.01% | 50.39% | +3.38 pp | 0.012 | 160:117 |

Fine-tuned accuracy agrees across the two (50.8% vs 50.4%). What differs is
the base model, which is 4 points stronger on dev, and the regression count.
The two deltas differ by 4.5 pp against a standard error of about 2 pp
(roughly p ≈ 0.02), so this is probably not all sampling noise. Both are
legitimate held-out measurements; neither was trained on. The honest summary
is a lift of roughly +3 to +8 pp, significant on both samples. The test figure
stays the headline because that was the rule, but it should not be quoted
without the dev figure next to it.

Caveat on the base column: MedQA base is 429/1000 here and 428/1000 in §2b.
Same items, different hardware (Colab CUDA vs Mac MPS); one item flipped.

### 2b. Previous adapter — kept for the record

Adapter saved **2026-09-26 23:09 UTC**, now at
`outputs/clinical-qlora-backup-20260926/`. Scored locally on MPS.

| Benchmark | Share of training corpus | Base | Fine-tuned | Δ | McNemar *p* | Fix:regress |
|---|---:|---:|---:|---:|---:|---:|
| MedQA (USMLE) | 0% | 42.80% | 50.10% | +7.30 pp | 1.0×10⁻⁶ | 2.01:1 |
| MedMCQA (val) | 59% | 47.10% | 52.90% | +5.80 pp | 4.3×10⁻⁴ | 1.57:1 |
| ~~PubMedQA~~ | — | ~~64.10%~~ | ~~74.00%~~ | ~~+9.90 pp~~ | **WITHDRAWN** | — |

Evidence: `reports/eval.json`, `reports/eval_medqa.json`. The PubMedQA column
in `reports/eval.json` is the withdrawn one (§3).

---

## 3. The single most important thing to know

**The first PubMedQA result was contaminated and is withdrawn. It stays
withdrawn; the valid number comes from the retrain (§2a).**

`iter_pubmedqa` trained on `qiaojin/PubMedQA` config `pqa_labeled` split
`train`. `load_pubmedqa_test` evaluates the *same* dataset, config and split.
`pqa_labeled` holds exactly 1,000 items and `prepare.py` holds out only 2% — so
**~980 of the 1,000 scored items had been trained on**. The +9.90 pp was
memorisation.

Fixed in `7a79fe2`: training now reads `pqa_artificial` (disjoint). Guarded by
`tests/test_no_contamination.py` — five offline tests that assert no training
source and benchmark share a `(dataset, config, split)` triple, verified to fail
by name on the exact configuration that caused it.

**Resolved 2026-10-04.** The retrain on the corrected corpus completed and
scores 64.1% → 73.6% (+9.50 pp, p = 1.4×10⁻⁹). Beyond the config-level guard,
the two configs were checked item-by-item: 0 of the 211,269 `pqa_artificial`
pubids appear in `pqa_labeled`. The clean number landing close to the
contaminated +9.90 pp is a coincidence of magnitude, not a vindication — the
old figure measured recall of trained items and remains struck through.

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

1. ~~**Push to GitHub.**~~ Done 2026-10-04: branch renamed `master` → `main`,
   public at `github.com/asongwe-mutah/clinical-llm`.
2. ~~**PubMedQA retrain.**~~ Done 2026-10-04, see §2a. Three valid benchmarks
   from one adapter. Answer on packing: MedMCQA regressions went 102 → 98,
   i.e. no real change.
   Adapter weights published 2026-10-04 at
   `huggingface.co/mutahfon/clinical-qlora-qwen2.5-3b` (SHA-256 matches
   `outputs/clinical-qlora/adapter_model.safetensors`). Verified end to end
   from the Hub: the model-card snippet generates, and the first 100 MedQA
   items score identically to the local adapter (100/100 per-item).
3. **`docs/BRIEF.md`.** The originating prompt (Claude conversation, 2026-07-27)
   is not saved anywhere. Everything else is documented — why packing is off,
   why `pqa_artificial`, why McNemar — but not the requirements that shaped it.
4. **Delete `_to_delete/`** — stale git lock files, untracked.

5. **v2: a longer run.** Prepared 2026-10-04, not yet trained. See §9.

### If you retrain: decide this *before* seeing numbers

*(Written 2026-10-02, before the retrain was scored. Kept verbatim because it is
the rule §2a was judged by. Outcome: MedQA +7.90 pp ≥ +7.30 pp, so the retrain
is the headline.)*

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
  MedQA has now been scored **twice** (once per adapter); every further look
  spends it. Do not iterate recipes against it.

---

## 8. Environment

Local venv (Mac, MPS): torch 2.13.0, transformers 5.14.1, peft 0.19.1,
trl 1.9.1, datasets 5.0.0. `bitsandbytes` absent — irrelevant, eval runs fp16
unquantized.

Colab trains under a newer PEFT (0.20.0 for the first adapter, 0.21.0 for the
retrain), which writes config keys 0.19.1 ignores (`monteclora_config`,
`velora_config`, and from 0.21.0 `kasa_config`). All are null; the retrained
adapter's config loads under 0.19.1 with a warning. Its scores were produced on
Colab. Local spot-check on MPS (2026-10-04, first 100 MedQA items): fine-tuned
agrees with the Colab per-item outcomes on 99/100, base on 100/100 — the
installed adapter is the scored one and behaves the same here. The full
n=1000 runs have not been repeated locally.

Corpus (`data_gpu.yaml`, 25k cap per source): MedMCQA 25,000 + MedQuAD 16,407 +
PubMedQA `pqa_artificial` 25,000 = **66,407** → 65,079 train / 1,328 val.
The cap binds only on MedMCQA (182,822 available).

---

## 9. v2 — a longer run of the same recipe (prepared, not yet trained)

**Hypothesis.** v1 saw 11,200 of 65,079 training examples (0.17 epoch) and
`eval_loss` was still falling when its schedule ended (1.129 → 1.105). More of
the same training may help. v2 changes **one thing**: 2,100 steps instead of
700 (33,600 samples, 0.52 epoch). Same data, LoRA shape, LR, batch, seed;
packing off. `tests/test_sessions.py` asserts the two configs differ only there.

**Honest prior.** This may do nothing. The retrain moved MedQA by +0.70 pp over
the adapter before it (p = 0.43). A 3B model has a ceiling.

### How to run it

Open `notebooks/clinical_llm_v2_train.ipynb` in Colab on an **L4 or A100**, Run
all, walk away. About 9 h on an L4, so it spans ~3 sessions: each trains for
3 h, checkpoints to `Drive/clinical-qlora-v2`, and stops itself
(`--session-hours`). Run all again in a fresh session to resume. The last cell
says whether it is finished. Nothing is written as a final adapter until step
2,100, so a partial run cannot be mistaken for the result.

When finished, download `clinical-qlora-v2/milestones/` (three ~70 MB
adapters: steps 700, 1400, 2100) to the Mac. **Do not run
`install_adapter.py` yet** — it would pick the newest `clinical-qlora*`
download and replace v1 before the rule below has been applied.

### The rule — fixed 2026-10-04, before any v2 number exists

MedQA **test** has been scored once per adapter and decided the last headline.
It is not used for any choice here.

1. **Selection set:** `medqa_dev` — the MedQA `dev` split, all 1,272 items, no
   question shared with `test` (checked) and never trained on (tested). All
   scoring for selection runs on the Mac (MPS), so every adapter is compared
   on the same hardware. v1's score there is the incumbent:
   `reports/eval_v1_medqa_dev.json` — **50.39% (641/1272)**, base 47.01%,
   scored 2026-10-04 before any v2 training.
2. **Candidates:** the v2 milestones at steps 700, 1400 and 2100 — or whichever
   exist if the run is cut short. Pick the one with the highest dev accuracy;
   ties go to the later step.
3. **Bar:** the pick replaces v1 only if it beats v1 on dev in a direct paired
   McNemar test (`scripts/compare_runs.py`) at **p < 0.0167** — 0.05 divided by
   the three candidates, because taking the best of three is itself a way to
   find a difference that is not there.
4. **If it clears the bar:** score that one adapter, once, on MedQA test,
   MedMCQA and PubMedQA (n = 1,000, same items as v1). Those numbers become
   the headline **whatever they are** — including if MedQA test comes out
   below v1's +7.90 pp. No reverting per benchmark.
5. **If it does not:** v1 stays the headline. v2 is written up as a null
   result with its dev numbers. v2 is **not** scored on MedQA test.
6. Whatever happens, no further recipe is chosen by looking at MedQA test.

Why step-700 of v2 is not a rerun of v1: v2's cosine schedule is stretched over
2,100 steps, so at step 700 it is still at ~78% of peak LR where v1 had
annealed to zero.

