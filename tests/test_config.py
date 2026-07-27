"""Tests for YAML config loading into typed dataclasses."""

import textwrap

import pytest

from clinical_llm.utils.config import (
    DataConfig,
    TrainConfig,
    load_data_config,
    load_train_config,
)


def test_load_train_config_defaults_and_nested(tmp_path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            output_dir: outputs/test
            epochs: 2.0
            model:
              base_model: Qwen/Qwen2.5-0.5B-Instruct
              load_in_4bit: false
            lora:
              r: 8
            """
        )
    )
    cfg = load_train_config(cfg_path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.output_dir == "outputs/test"
    assert cfg.epochs == 2.0
    assert cfg.model.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.model.load_in_4bit is False
    assert cfg.lora.r == 8
    # Untouched fields keep their defaults.
    assert cfg.lora.alpha == 32
    assert cfg.per_device_batch_size == 8


def test_load_data_config(tmp_path):
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text("max_per_source: 16\nuse_medquad: false\n")
    cfg = load_data_config(cfg_path)
    assert isinstance(cfg, DataConfig)
    assert cfg.max_per_source == 16
    assert cfg.use_medquad is False
    assert cfg.use_pubmedqa is True  # default preserved


def test_unknown_key_raises(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("not_a_real_key: 1\n")
    with pytest.raises(ValueError):
        load_train_config(cfg_path)
