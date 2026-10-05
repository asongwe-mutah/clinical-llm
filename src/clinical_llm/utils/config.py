"""Typed configuration objects loaded from YAML.

Every runnable entry point (data prep, training, eval, serving) is driven by a
small YAML file under ``configs/``. Using dataclasses gives us validation and
editor autocompletion while keeping the config files human-readable and
diffable in git — which matters for a portfolio project where reviewers read
the configs to understand what you actually ran.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Which datasets to build the instruction corpus from, and how to mix them."""

    output_dir: str = "data/processed"
    # Per-source cap on examples (None = use all available). Keeps smoke runs fast.
    max_per_source: int | None = None
    # QA sources (Hugging Face dataset ids resolved in data/datasets.py).
    use_pubmedqa: bool = True
    use_medmcqa: bool = True
    use_medquad: bool = True
    # Notes-summarization stretch task. Disabled by default because it needs a
    # credentialed dataset (e.g. MIMIC-IV-Note) placed at ``notes_jsonl``.
    use_notes_summarization: bool = False
    notes_jsonl: str | None = None
    seed: int = 13
    val_fraction: float = 0.02


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    # 4-bit QLoRA. Automatically disabled when CUDA / bitsandbytes is unavailable
    # (e.g. on Apple Silicon), falling back to a bf16/fp32 LoRA smoke run.
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    max_seq_len: int = 2048
    trust_remote_code: bool = False
    # Attention kernel: "auto" picks flash_attention_2 when the GPU and the
    # installed packages both support it, else sdpa. This matters with
    # packing=True: only FA2 gives packed samples block-diagonal masking, so
    # without it examples in the same packed sequence attend across each other.
    # Override with an explicit "flash_attention_2" | "sdpa" | "eager".
    attn_implementation: str = "auto"


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Attention + MLP projections; the union works across Llama/Qwen/Phi.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    output_dir: str = "outputs/clinical-qlora"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    epochs: float = 1.0
    max_steps: int = -1  # >0 overrides epochs; used by the smoke config.
    per_device_batch_size: int = 8
    grad_accum: int = 4
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    # In-loop eval cost is easy to underestimate. A 1,328-example validation
    # set took 5.8 min per eval on an L4 -- 18% of a 6-hour run. Subsample it;
    # the in-loop eval is an overfitting signal, not the headline metric.
    eval_max_samples: int | None = 200
    # How many checkpoints to retain. 2 is too few when a run may be killed:
    # a crash at 74% left exactly two candidates and no eval curve to pick from.
    save_total_limit: int = 6
    # Keep an adapter-only snapshot every N steps under output_dir/milestones/.
    # Rolling checkpoints are pruned by save_total_limit, so without this a
    # multi-session run leaves nothing from its early stages to compare on the
    # dev split. ~70 MB each (no optimizer state). 0 disables.
    milestone_steps: int = 0
    bf16: bool = True
    gradient_checkpointing: bool = True
    packing: bool = True
    seed: int = 13


def _from_dict(cls, data: dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a plain dict.

    ``from __future__ import annotations`` turns field annotations into
    strings, so we resolve the real types with ``typing.get_type_hints`` before
    deciding whether to recurse into a nested dataclass.
    """
    import typing

    if not dataclasses.is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    valid = {f.name for f in dataclasses.fields(cls)}
    resolved = typing.get_type_hints(cls)
    for key, value in (data or {}).items():
        if key not in valid:
            raise ValueError(f"unknown config key '{key}' for {cls.__name__}")
        field_type = resolved.get(key)
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _from_dict(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_train_config(path: str | Path) -> TrainConfig:
    return _from_dict(TrainConfig, load_yaml(path))


def load_data_config(path: str | Path) -> DataConfig:
    return _from_dict(DataConfig, load_yaml(path))
