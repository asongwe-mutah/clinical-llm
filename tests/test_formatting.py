"""Unit tests for the prompt/format contract. No GPU or network required."""

import pytest

from clinical_llm.data.formatting import (
    SYSTEM_PROMPT,
    build_chat_example,
    extract_mcq_letter,
    format_mcq_answer,
    format_mcq_question,
    render_prompt,
)


def test_build_chat_example_roundtrip():
    ex = build_chat_example("What is sepsis?", "Sepsis is life-threatening organ dysfunction.")
    msgs = ex.to_messages()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    rec = ex.to_record()
    assert rec["messages"] == msgs
    assert "meta" in rec


@pytest.mark.parametrize("bad_user,bad_asst", [("", "a"), ("q", ""), ("  ", "a")])
def test_build_chat_example_rejects_empty(bad_user, bad_asst):
    with pytest.raises(ValueError):
        build_chat_example(bad_user, bad_asst)


def test_format_mcq_question_letters_and_instruction():
    q = format_mcq_question("Which is a beta-blocker?", ["Metoprolol", "Lisinopril", "Amlodipine"])
    assert "A. Metoprolol" in q
    assert "B. Lisinopril" in q
    assert "C. Amlodipine" in q
    assert "single letter" in q.lower()


def test_format_mcq_question_rejects_too_many_options():
    with pytest.raises(ValueError):
        format_mcq_question("q", ["o"] * 9)


def test_format_mcq_answer_starts_with_answer_letter():
    ans = format_mcq_answer(0, ["Metoprolol", "Lisinopril"], rationale="It blocks beta receptors.")
    assert ans.startswith("Answer: A. Metoprolol")
    assert "beta receptors" in ans


def test_format_mcq_answer_out_of_range():
    with pytest.raises(ValueError):
        format_mcq_answer(5, ["a", "b"])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: C. Amlodipine", "C"),
        ("answer c", "C"),
        ("B) because ...", "B"),
        ("The best choice is D here", "D"),
        ("(A) metoprolol", "A"),
        ("no letters here at all", None),
        ("", None),
    ],
)
def test_extract_mcq_letter(text, expected):
    assert extract_mcq_letter(text) == expected


def test_render_prompt_chatml_shape():
    ex = build_chat_example("Q?", "A.")
    text = render_prompt(ex.to_messages(), add_generation_prompt=True)
    assert text.count("<|im_start|>") == 4  # system, user, assistant, generation
    assert text.rstrip().endswith("<|im_start|>assistant")


def test_extract_letter_matches_generated_answer():
    """The eval parser must recover the letter that format_mcq_answer emits."""
    options = ["Metoprolol", "Lisinopril", "Amlodipine", "Furosemide"]
    for idx in range(len(options)):
        ans = format_mcq_answer(idx, options)
        assert extract_mcq_letter(ans) == chr(ord("A") + idx)
