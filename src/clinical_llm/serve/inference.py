"""Inference engine: load a model (base, base+adapter, or merged) and generate.

Routes every prompt through the same :func:`render_with_tokenizer` contract as
training, so there is no train/serve skew. Supports streaming via a background
generation thread.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Thread
from typing import Iterator, Optional

from clinical_llm.data.formatting import SYSTEM_PROMPT, render_with_tokenizer


@dataclass
class GenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True


class ClinicalLLM:
    """Thin wrapper around a causal LM for chat-style clinical generation."""

    def __init__(
        self,
        base_model: str,
        adapter: Optional[str] = None,
        merged_model: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = merged_model or base_model
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter or model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
        if adapter and not merged_model:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, user: str, history: Optional[list[dict]], system: str) -> str:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        return render_with_tokenizer(self.tokenizer, messages, add_generation_prompt=True)

    def _gen_kwargs(self, settings: GenerationSettings) -> dict:
        do_sample = settings.do_sample and settings.temperature > 0
        kwargs = dict(
            max_new_tokens=settings.max_new_tokens,
            repetition_penalty=settings.repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Only pass sampling params when actually sampling; otherwise recent
        # transformers warns that temperature/top_p are ignored for greedy.
        if do_sample:
            kwargs["temperature"] = settings.temperature
            kwargs["top_p"] = settings.top_p
        return kwargs

    def generate(
        self,
        user: str,
        history: Optional[list[dict]] = None,
        system: str = SYSTEM_PROMPT,
        settings: Optional[GenerationSettings] = None,
    ) -> str:
        import torch

        settings = settings or GenerationSettings()
        prompt = self._build_prompt(user, history, system)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(settings))
        new_tokens = out[0, inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(
        self,
        user: str,
        history: Optional[list[dict]] = None,
        system: str = SYSTEM_PROMPT,
        settings: Optional[GenerationSettings] = None,
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        settings = settings or GenerationSettings()
        prompt = self._build_prompt(user, history, system)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(inputs, streamer=streamer, **self._gen_kwargs(settings))
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()


def load_from_env() -> ClinicalLLM:
    """Construct a :class:`ClinicalLLM` from environment variables.

    ``CLINICAL_LLM_BASE``      base model id (default Qwen2.5-3B-Instruct)
    ``CLINICAL_LLM_ADAPTER``   optional LoRA adapter dir
    ``CLINICAL_LLM_MERGED``    optional merged-model dir (takes precedence)
    """
    return ClinicalLLM(
        base_model=os.environ.get("CLINICAL_LLM_BASE", "Qwen/Qwen2.5-3B-Instruct"),
        adapter=os.environ.get("CLINICAL_LLM_ADAPTER") or None,
        merged_model=os.environ.get("CLINICAL_LLM_MERGED") or None,
        trust_remote_code=os.environ.get("CLINICAL_LLM_TRUST_REMOTE_CODE", "0") == "1",
    )
