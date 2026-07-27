#!/usr/bin/env bash
# Launch the FastAPI demo. Point it at your trained adapter (or a merged model).
#   ./scripts/serve.sh                       # base model only
#   CLINICAL_LLM_ADAPTER=outputs/clinical-qlora ./scripts/serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export CLINICAL_LLM_BASE="${CLINICAL_LLM_BASE:-Qwen/Qwen2.5-3B-Instruct}"
echo "serving base=$CLINICAL_LLM_BASE adapter=${CLINICAL_LLM_ADAPTER:-<none>} on http://localhost:8000"
uvicorn clinical_llm.serve.app:app --host 0.0.0.0 --port "${PORT:-8000}"
