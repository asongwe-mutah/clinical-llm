"""Precision-selection regression tests (GPU-free).

These pin the bug that made a Colab T4 run project 59 hours: `fp16` was keyed
off the *requested* `cfg.bf16` rather than the *effective* one, so requesting
bf16 on a card without it selected neither flag and trained in fp32.
"""

from __future__ import annotations

import pytest

from clinical_llm.train import train_qlora
from clinical_llm.utils.config import load_train_config


@pytest.fixture
def colab_cfg():
    return load_train_config("configs/train_colab.yaml")


def _patch(monkeypatch, *, cuda: bool, bf16: bool) -> None:
    monkeypatch.setattr(train_qlora, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(train_qlora, "bf16_supported", lambda: bf16)


def test_ampere_uses_bf16(monkeypatch, colab_cfg):
    _patch(monkeypatch, cuda=True, bf16=True)
    assert train_qlora.resolve_precision(colab_cfg) == (True, False)


def test_turing_falls_back_to_fp16_not_fp32(monkeypatch, colab_cfg):
    """The regression: bf16 requested, hardware lacks it -> fp16, never fp32."""
    _patch(monkeypatch, cuda=True, bf16=False)
    assert train_qlora.resolve_precision(colab_cfg) == (False, True)


def test_explicit_fp16_request_is_honoured(monkeypatch, colab_cfg):
    _patch(monkeypatch, cuda=True, bf16=True)
    colab_cfg.bf16 = False
    assert train_qlora.resolve_precision(colab_cfg) == (False, True)


def test_cpu_sets_no_half_precision_flag(monkeypatch, colab_cfg):
    _patch(monkeypatch, cuda=False, bf16=False)
    assert train_qlora.resolve_precision(colab_cfg) == (False, False)


@pytest.mark.parametrize("bf16_hw", [True, False])
@pytest.mark.parametrize("requested", [True, False])
def test_exactly_one_flag_on_cuda(monkeypatch, colab_cfg, bf16_hw, requested):
    """Never both, never neither -- fp32 on a GPU is always a bug here."""
    _patch(monkeypatch, cuda=True, bf16=bf16_hw)
    colab_cfg.bf16 = requested
    use_bf16, use_fp16 = train_qlora.resolve_precision(colab_cfg)
    assert use_bf16 + use_fp16 == 1


def test_bf16_supported_is_false_without_cuda(monkeypatch):
    monkeypatch.setattr(train_qlora, "_cuda_available", lambda: False)
    assert train_qlora.bf16_supported() is False
