#!/usr/bin/env bash
# End-to-end smoke test of the whole pipeline on a laptop (CPU or Apple MPS).
# Builds a tiny corpus, runs a few QLoRA/LoRA steps, and confirms an adapter is
# written. Proves the code path without a GPU. NOT a quality run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/3] building tiny corpus"
python -m clinical_llm.data.prepare --config configs/data_smoke.yaml

echo "==> [2/3] running smoke train (downloads ~1GB base model on first run)"
python -m clinical_llm.train.train_qlora --config configs/train_smoke.yaml

echo "==> [3/3] verifying adapter output"
test -d outputs/smoke && ls -1 outputs/smoke && echo "SMOKE TRAIN OK"
