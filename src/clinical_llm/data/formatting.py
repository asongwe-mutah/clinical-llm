"""Prompt / chat formatting shared across training, evaluation and serving.

Keeping a *single* source of truth for how examples are turned into prompts is
what stops the classic train/serve skew bug: if the fine-tune sees one prompt
shape and the inference server sends another, quality silently collapses. Every
entry point in this project routes through :func:`build_chat_example` and
:func:`render_prompt`.

This module is deliberately dependency-light (standard library only) so it can
be unit-tested on any machine without ``torch`` / ``transformers`` installed.
The tokenizer-aware rendering (applying a model's real chat template) lives in
:func:`render_with_tokenizer`, which imports lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A conservative clinical system prompt. It sets the assistant's persona and,
# crucially, bakes the safety framing into every training example so the model
# learns to hedge and defer rather than fabricate clinical certainty.
SYSTEM_PROMPT = (
    "You are a clinical informatics assistant for use by healthcare "
    "professionals and researchers. Answer using accurate medical knowledge. "
    "Be concise and evidence-oriented, define abbreviations on first use, and "
    "state when evidence is uncertain or a question is outside your scope. You "
    "do not provide individualized medical advice, diagnosis, or treatment "
    "decisions; defer those to a licensed clinician."
)

# Letter labels used to render multiple-choice questions consistently.
_MCQ_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class ChatExample:
    """A single supervised example in a normalized chat schema.

    Attributes
    ----------
    system:
        System prompt for the turn. Defaults to :data:`SYSTEM_PROMPT`.
    user:
        The user / instruction content.
    assistant:
        The target completion the model should learn to produce.
    meta:
        Free-form provenance (source dataset, split, original id, task type).
    """

    user: str
    assistant: str
    system: str = SYSTEM_PROMPT
    meta: dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        """Return the OpenAI-style ``messages`` list for this example."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable record for writing to a ``.jsonl`` file."""
        return {"messages": self.to_messages(), "meta": self.meta}


def format_mcq_question(
    question: str,
    options: list[str],
    *,
    include_instruction: bool = True,
) -> str:
    """Render a multiple-choice question as a stable prompt string.

    Parameters
    ----------
    question:
        The stem of the question.
    options:
        Answer choices, in order. Up to 8 are supported (A-H).
    include_instruction:
        If ``True`` (default) append an instruction asking the model to answer
        with the single best option letter, which is what the eval harness
        parses.
    """
    if not options:
        raise ValueError("MCQ requires at least one option")
    if len(options) > len(_MCQ_LETTERS):
        raise ValueError(f"at most {len(_MCQ_LETTERS)} options supported")

    lines = [question.strip(), ""]
    # options may be shorter than the letter list; zip stops at options (intended).
    for letter, opt in zip(_MCQ_LETTERS, options, strict=False):
        lines.append(f"{letter}. {opt.strip()}")
    if include_instruction:
        lines.append("")
        lines.append(
            "Respond with the single letter of the best answer, then a brief "
            "one-sentence rationale."
        )
    return "\n".join(lines)


def format_mcq_answer(correct_index: int, options: list[str], rationale: str = "") -> str:
    """Render the target answer for an MCQ example.

    The label always *starts* with ``"Answer: <letter>"`` so that both the
    training target and the eval parser agree on where the letter lives.
    """
    if correct_index < 0 or correct_index >= len(options):
        raise ValueError("correct_index out of range for options")
    letter = _MCQ_LETTERS[correct_index]
    text = f"Answer: {letter}. {options[correct_index].strip()}"
    if rationale:
        text += f"\n{rationale.strip()}"
    return text


def build_chat_example(
    user: str,
    assistant: str,
    *,
    system: str | None = None,
    meta: dict[str, Any] | None = None,
) -> ChatExample:
    """Construct a :class:`ChatExample`, validating that both sides are present."""
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if not user:
        raise ValueError("user content is empty")
    if not assistant:
        raise ValueError("assistant content is empty")
    return ChatExample(
        user=user,
        assistant=assistant,
        system=(system or SYSTEM_PROMPT),
        meta=meta or {},
    )


def render_prompt(messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> str:
    """A minimal, tokenizer-free ChatML-style renderer.

    This mirrors the widely used ``<|im_start|>role ... <|im_end|>`` layout
    (Qwen / many instruct models). It exists so tests and quick local sanity
    checks do not need a tokenizer. At real train / inference time we instead
    call :func:`render_with_tokenizer`, which applies the *model's own* chat
    template — the authoritative format.
    """
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    text = "\n".join(parts)
    if add_generation_prompt:
        text += "\n<|im_start|>assistant\n"
    return text


def render_with_tokenizer(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render ``messages`` with a HF tokenizer's chat template.

    Imported lazily elsewhere; kept here so the format contract stays in one
    file. Falls back to :func:`render_prompt` if the tokenizer has no chat
    template configured.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return render_prompt(messages, add_generation_prompt=add_generation_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def extract_mcq_letter(text: str) -> str | None:
    """Best-effort parse of a predicted option letter from model output.

    Used by the eval harness. Looks for ``Answer: X`` first, then a leading
    standalone letter, then any first standalone A-H token. Returns ``None`` if
    nothing parseable is found.
    """
    import re

    if not text:
        return None
    t = text.strip()

    m = re.search(r"answer\s*[:\-]?\s*\(?([A-Ha-h])\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.match(r"^\(?([A-Ha-h])[\).\:\s]", t)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-H])\b", t)
    if m:
        return m.group(1).upper()

    return None
