"""QLoRA supervised fine-tuning of a clinical instruction model.

Usage::

    # full run on a CUDA GPU
    python -m clinical_llm.train.train_qlora --config configs/train_qlora.yaml

    # tiny local smoke run (CPU / Apple-Silicon MPS) to verify the code path
    python -m clinical_llm.train.train_qlora --config configs/train_smoke.yaml

Design notes
------------
* 4-bit quantization (the "Q" in QLoRA) is only enabled when a CUDA device and
  ``bitsandbytes`` are actually available. On Apple Silicon / CPU we fall back
  to a plain LoRA run so the *exact same script* is exercised locally — this is
  what makes the pipeline verifiable without a GPU.
* We render each example with the base model's own chat template so the
  fine-tune sees precisely the format the inference server will send.
* Training is delegated to TRL's ``SFTTrainer``; we pin behaviour through
  ``SFTConfig`` and pass a pre-rendered ``text`` column, which is the most
  version-stable way to drive it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clinical_llm.data.formatting import render_with_tokenizer
from clinical_llm.utils.config import TrainConfig, load_train_config
from clinical_llm.utils.logging import get_logger

log = get_logger("train")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch may be absent
        return False


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def build_quant_config(cfg: TrainConfig):
    """Return a BitsAndBytesConfig for 4-bit, or None when unsupported."""
    want_4bit = cfg.model.load_in_4bit
    if not want_4bit:
        return None
    if not (_cuda_available() and _bitsandbytes_available()):
        log.warning(
            "load_in_4bit requested but CUDA/bitsandbytes unavailable; "
            "falling back to non-quantized LoRA."
        )
        return None
    import torch
    from transformers import BitsAndBytesConfig

    dtype = getattr(torch, cfg.model.bnb_4bit_compute_dtype, torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = build_quant_config(cfg)
    on_cuda = _cuda_available()
    # Choose a compute dtype that is valid for the device.
    if on_cuda and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif on_cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32  # CPU/MPS smoke run

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        quantization_config=quant_config,
        torch_dtype=dtype,
        device_map="auto" if on_cuda else None,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.config.use_cache = False

    if quant_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    return model, tokenizer


def build_peft_config(cfg: TrainConfig):
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_and_render_dataset(cfg: TrainConfig, tokenizer):
    """Load the JSONL splits and add a rendered ``text`` column."""
    from datasets import load_dataset

    data_dir = Path(cfg.data_dir)
    files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "val.jsonl"),
    }
    missing = [k for k, v in files.items() if not Path(v).exists()]
    if "train" in missing:
        raise FileNotFoundError(
            f"{files['train']} not found; run `python -m clinical_llm.data.prepare` first"
        )
    if "validation" in missing:
        files.pop("validation")

    ds = load_dataset("json", data_files=files)

    def _render(example):
        text = render_with_tokenizer(
            tokenizer, example["messages"], add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(_render, remove_columns=[c for c in ds["train"].column_names if c != "text"])
    return ds


def train(cfg: TrainConfig) -> str:
    import torch  # noqa: F401  (ensures torch present / clear error otherwise)
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_model_and_tokenizer(cfg)
    peft_config = build_peft_config(cfg)
    ds = load_and_render_dataset(cfg, tokenizer)

    on_cuda = _cuda_available()
    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=cfg.bf16 and on_cuda and torch.cuda.is_bf16_supported(),
        fp16=(not cfg.bf16) and on_cuda,
        gradient_checkpointing=cfg.gradient_checkpointing and on_cuda,
        packing=cfg.packing,
        dataset_text_field="text",
        seed=cfg.seed,
        report_to=[],
    )
    # The max-sequence-length arg was renamed `max_seq_length` -> `max_length`
    # across TRL versions; set whichever the installed SFTConfig accepts.
    import inspect

    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    seq_arg = "max_length" if "max_length" in accepted else "max_seq_length"
    sft_kwargs[seq_arg] = cfg.model.max_seq_len
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in accepted}
    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info(
        "starting training: base=%s | 4bit=%s | cuda=%s | steps=%s | epochs=%s",
        cfg.model.base_model,
        build_quant_config(cfg) is not None,
        on_cuda,
        cfg.max_steps,
        cfg.epochs,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    log.info("saved LoRA adapter + tokenizer to %s", cfg.output_dir)
    return cfg.output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_train_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
