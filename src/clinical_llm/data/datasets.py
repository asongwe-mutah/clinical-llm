"""Adapters that turn public medical datasets into :class:`ChatExample`s.

Each adapter is a generator that yields normalized examples. They pull from the
Hugging Face Hub via ``datasets.load_dataset`` (imported lazily so this module
can be imported without ``datasets`` installed).

Sources (all publicly downloadable, no credentialing required):

* **PubMedQA** (`qiaojin/PubMedQA`, ``pqa_artificial``) — yes/no/maybe research
  QA over PubMed abstracts. Great for grounded, evidence-style answers.
  **Deliberately NOT ``pqa_labeled``**: that 1,000-item split is the evaluation
  benchmark, and training on it leaked ~98% of the eval set into the corpus.
  See ``TRAIN_SPECS`` below and ``tests/test_no_contamination.py``.
* **MedMCQA** (`openlifescienceai/medmcqa`) — Indian medical-entrance MCQs
  across 21 subjects. Large and broad.
* **MedQuAD** (`lavita/MedQuAD`) — consumer-health Q&A pairs curated from NIH
  websites. Good for natural free-text answers.

The notes-summarization stretch task reads a *local* JSONL (see
:func:`iter_notes_summarization`) because the strong clinical-note corpora
(e.g. MIMIC-IV-Note) require PhysioNet credentialing and must not be
redistributed. The repo therefore ships the *format*, not the data.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from clinical_llm.data.formatting import (
    ChatExample,
    build_chat_example,
    format_mcq_answer,
    format_mcq_question,
)


def _load(hf_id: str, name: str | None = None, split: str = "train"):
    from datasets import load_dataset

    return load_dataset(hf_id, name) if name else load_dataset(hf_id, split=split)


def iter_pubmedqa(max_examples: int | None = None) -> Iterator[ChatExample]:
    from datasets import load_dataset

    # TRAIN ON pqa_artificial, NOT pqa_labeled.
    #
    # pqa_labeled holds exactly 1,000 expert-annotated items and is what
    # eval/benchmarks.py scores. Training on it leaked ~98% of the evaluation
    # set into the training corpus (prepare.py holds out only 2%), which
    # produced a +9.90 pp "improvement" that was memorisation. pqa_artificial
    # is the large auto-labelled split intended for training and is disjoint
    # from pqa_labeled.
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        question = row["question"]
        contexts = row.get("context", {})
        ctx_texts = contexts.get("contexts", []) if isinstance(contexts, dict) else []
        context_block = "\n".join(ctx_texts)
        long_answer = row.get("long_answer", "").strip()
        decision = str(row.get("final_decision", "")).strip()

        user = (
            "Based on the following research abstract excerpts, answer the "
            "question.\n\n"
            f"Excerpts:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Give a yes/no/maybe verdict and then justify it in 2-3 sentences "
            "grounded in the excerpts."
        )
        assistant = f"Verdict: {decision}.\n{long_answer}" if decision else long_answer
        if not assistant.strip():
            continue
        yield build_chat_example(
            user,
            assistant,
            meta={"source": "pubmedqa", "task": "grounded_qa", "id": str(row.get("pubid", i))},
        )


def iter_medmcqa(max_examples: int | None = None) -> Iterator[ChatExample]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        # medmcqa stores the correct option as 0-indexed ``cop``.
        cop = int(row["cop"])
        if cop < 0 or cop > 3:
            continue
        rationale = (row.get("exp") or "").strip()
        user = format_mcq_question(row["question"], options)
        assistant = format_mcq_answer(cop, options, rationale=rationale)
        yield build_chat_example(
            user,
            assistant,
            meta={"source": "medmcqa", "task": "mcq", "subject": row.get("subject_name", "")},
        )


def iter_medquad(max_examples: int | None = None) -> Iterator[ChatExample]:
    from datasets import load_dataset

    ds = load_dataset("lavita/MedQuAD", split="train")
    count = 0
    for row in ds:
        if max_examples is not None and count >= max_examples:
            break
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        yield build_chat_example(
            question,
            answer,
            meta={"source": "medquad", "task": "open_qa", "focus": row.get("focus", "")},
        )
        count += 1


def iter_notes_summarization(
    jsonl_path: str, max_examples: int | None = None
) -> Iterator[ChatExample]:
    """Notes-summarization stretch task from a *local* JSONL.

    Expected line schema::

        {"note": "<clinical note text>", "summary": "<reference summary>"}

    This is where a credentialed corpus such as MIMIC-IV-Note (BHC
    summarization) plugs in without ever committing PHI to the repo.
    """
    instruction = (
        "Summarize the following clinical note for a busy clinician. Produce a "
        "concise summary covering the active problems, key findings, and plan. "
        "Do not invent information not present in the note.\n\nNote:\n"
    )
    with open(jsonl_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            note = (row.get("note") or "").strip()
            summary = (row.get("summary") or "").strip()
            if not note or not summary:
                continue
            yield build_chat_example(
                instruction + note,
                summary,
                meta={"source": "notes", "task": "summarization"},
            )


# Registry consumed by prepare.py. Notes is handled separately (needs a path).
QA_SOURCES = {
    "pubmedqa": iter_pubmedqa,
    "medmcqa": iter_medmcqa,
    "medquad": iter_medquad,
}

# (hf_dataset, config, split) each training source reads. Declared so
# tests/test_no_contamination.py can assert, offline, that no training source
# overlaps an evaluation benchmark. Keep in sync with the loaders above --
# the test is the thing that catches drift.
TRAIN_SPECS: dict[str, tuple[str, str | None, str]] = {
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_artificial", "train"),
    "medmcqa": ("openlifescienceai/medmcqa", None, "train"),
    "medquad": ("lavita/MedQuAD", None, "train"),
}
