"""Tests for the local (network-free) data adapters."""

import json

from clinical_llm.data.datasets import iter_notes_summarization


def test_notes_summarization_adapter(tmp_path):
    path = tmp_path / "notes.jsonl"
    rows = [
        {"note": "72M with CHF exacerbation, started on IV furosemide.", "summary": "CHF exac; diuresed."},
        {"note": "", "summary": "skip me"},          # missing note -> skipped
        {"note": "has note", "summary": ""},           # missing summary -> skipped
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    examples = list(iter_notes_summarization(str(path)))
    assert len(examples) == 1
    ex = examples[0]
    assert ex.meta["task"] == "summarization"
    assert "Summarize the following clinical note" in ex.user
    assert "furosemide" in ex.user
    assert ex.assistant == "CHF exac; diuresed."


def test_notes_summarization_respects_max(tmp_path):
    path = tmp_path / "notes.jsonl"
    rows = [{"note": f"note {i}", "summary": f"sum {i}"} for i in range(10)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    examples = list(iter_notes_summarization(str(path), max_examples=3))
    assert len(examples) == 3
