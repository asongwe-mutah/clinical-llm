"""Loaders for held-out medical multiple-choice benchmarks.

These are the *evaluation* datasets. They are kept separate from the training
adapters in :mod:`clinical_llm.data.datasets` so there is a clean train/test
boundary — MedMCQA's *validation* split is never used for SFT.

Each loader yields ``MCQItem``s with a question stem, options, and the
0-indexed gold answer, so the eval loop is dataset-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class MCQItem:
    question: str
    options: list[str]
    gold_index: int
    source: str
    subject: str = ""


def load_medmcqa_val(max_items: int | None = None) -> Iterator[MCQItem]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    for i, row in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        cop = int(row["cop"])
        if cop < 0 or cop > 3:
            continue
        yield MCQItem(
            question=row["question"],
            options=options,
            gold_index=cop,
            source="medmcqa",
            subject=row.get("subject_name", ""),
        )


def _load_medqa(split: str, source: str, max_items: int | None) -> Iterator[MCQItem]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medqa", split=split)
    for i, row in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        # openlifescienceai/medqa exposes a nested ``data`` dict.
        data = row.get("data", row)
        question = data.get("Question") or data.get("question")
        opts_field = data.get("Options") or data.get("options")
        if isinstance(opts_field, dict):
            keys = sorted(opts_field.keys())
            options = [opts_field[k] for k in keys]
            gold_letter = (data.get("Correct Option") or "").strip().upper()
            gold_index = keys.index(gold_letter) if gold_letter in keys else -1
        else:
            options = list(opts_field or [])
            gold_index = int(data.get("answer_idx", -1))
        if not question or gold_index < 0 or gold_index >= len(options):
            continue
        yield MCQItem(question=question, options=options, gold_index=gold_index, source=source)


def load_medqa_test(max_items: int | None = None) -> Iterator[MCQItem]:
    """MedQA (USMLE) 4-option English test split. The reported benchmark."""
    return _load_medqa("test", "medqa", max_items)


def load_medqa_dev(max_items: int | None = None) -> Iterator[MCQItem]:
    """MedQA ``dev`` split (1,272 items, no question shared with ``test``).

    This is the split to make decisions on: which checkpoint, which recipe,
    whether a new adapter beats the current one. ``medqa`` (test) has been
    scored once per adapter and each further look spends it, so selection
    happens here and test is scored once, at the end, on whatever was chosen.
    """
    return _load_medqa("dev", "medqa_dev", max_items)


def load_pubmedqa_test(max_items: int | None = None) -> Iterator[MCQItem]:
    """PubMedQA as a 3-way (yes/no/maybe) classification task."""
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    choices = ["yes", "no", "maybe"]
    for i, row in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        decision = str(row.get("final_decision", "")).strip().lower()
        if decision not in choices:
            continue
        contexts = row.get("context", {})
        ctx = "\n".join(contexts.get("contexts", [])) if isinstance(contexts, dict) else ""
        stem = f"{ctx}\n\nQuestion: {row['question']}"
        yield MCQItem(
            question=stem,
            options=choices,
            gold_index=choices.index(decision),
            source="pubmedqa",
        )


BENCHMARKS = {
    "medmcqa": load_medmcqa_val,
    "medqa": load_medqa_test,
    "medqa_dev": load_medqa_dev,
    "pubmedqa": load_pubmedqa_test,
}

# (hf_dataset, config, split) each benchmark reads. Must stay disjoint from
# clinical_llm.data.datasets.TRAIN_SPECS -- see tests/test_no_contamination.py.
EVAL_SPECS: dict[str, tuple[str, str | None, str]] = {
    "medmcqa": ("openlifescienceai/medmcqa", None, "validation"),
    "medqa": ("openlifescienceai/medqa", None, "test"),
    "medqa_dev": ("openlifescienceai/medqa", None, "dev"),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled", "train"),
}
