# Serving image for the Clinical-LLM FastAPI demo.
# CUDA runtime base so the container can use a GPU when one is present; the app
# also runs CPU-only (slowly) if you drop the --gpus flag.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/.hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY ui ./ui

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install ".[serve]"

# Model is provided at run time via a mounted volume + env vars, so the image
# stays small and ships no weights:
#   docker run --gpus all -p 8000:8000 \
#     -e CLINICAL_LLM_BASE=Qwen/Qwen2.5-3B-Instruct \
#     -e CLINICAL_LLM_ADAPTER=/models/clinical-qlora \
#     -v $PWD/outputs:/models clinical-llm
EXPOSE 8000
CMD ["uvicorn", "clinical_llm.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
