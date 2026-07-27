"""Clinical-LLM: a domain-adapted language model for clinical informatics.

This package contains a full, config-driven pipeline for adapting an open
instruction-tuned base model to the medical / clinical-informatics domain via
parameter-efficient fine-tuning (QLoRA), together with an evaluation harness
against public medical-QA benchmarks and a FastAPI inference service.

The code is organised so that the *data formatting* logic (``clinical_llm.data``)
is dependency-light and unit-testable without a GPU or the heavy ML stack,
while the *training* / *serving* modules pull in ``torch`` / ``transformers``
lazily.
"""

__version__ = "0.1.0"
