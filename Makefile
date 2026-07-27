.PHONY: help install install-dev test lint data data-smoke train train-smoke eval serve merge clean

help:
	@echo "Clinical-LLM — common targets"
	@echo "  make install-dev   install core + dev deps (laptop; runs unit tests)"
	@echo "  make install       install full train+serve stack (GPU box)"
	@echo "  make test          run unit tests (no GPU / network needed)"
	@echo "  make data          build the full instruction corpus"
	@echo "  make data-smoke    build a tiny corpus for the smoke train"
	@echo "  make train         full QLoRA fine-tune (CUDA GPU)"
	@echo "  make train-smoke   end-to-end smoke train (CPU/MPS)"
	@echo "  make eval          evaluate base vs fine-tuned on MCQ benchmarks"
	@echo "  make serve         launch the FastAPI demo on :8000"

install:
	pip install -r requirements.txt && pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

data:
	python -m clinical_llm.data.prepare --config configs/data.yaml

data-smoke:
	python -m clinical_llm.data.prepare --config configs/data_smoke.yaml

train:
	python -m clinical_llm.train.train_qlora --config configs/train_qlora.yaml

train-smoke: data-smoke
	python -m clinical_llm.train.train_qlora --config configs/train_smoke.yaml

eval:
	python -m clinical_llm.eval.run_eval \
		--base Qwen/Qwen2.5-3B-Instruct \
		--adapter outputs/clinical-qlora \
		--benchmarks medmcqa pubmedqa \
		--max-items 500 \
		--out reports/eval.json

merge:
	python -m clinical_llm.train.merge_lora \
		--base Qwen/Qwen2.5-3B-Instruct \
		--adapter outputs/clinical-qlora \
		--out outputs/clinical-merged

serve:
	CLINICAL_LLM_ADAPTER=outputs/clinical-qlora \
	uvicorn clinical_llm.serve.app:app --host 0.0.0.0 --port 8000

clean:
	rm -rf outputs reports data/processed data/processed_smoke
