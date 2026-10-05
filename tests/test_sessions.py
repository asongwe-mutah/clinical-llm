"""Multi-session training: the decisions that must not be wrong.

A run longer than one Colab session stops on a time budget and resumes later.
Two ways that goes silently bad, both tested here without a GPU:

* a partial run gets saved as the final adapter, and the eval tooling -- which
  treats ``output_dir/adapter_config.json`` as "training finished" -- scores a
  half-trained model as if it were the result;
* a milestone snapshot lands somewhere the resume logic mistakes for a
  checkpoint.
"""

from __future__ import annotations

from clinical_llm.train.train_qlora import (
    milestone_dir,
    run_is_complete,
    session_budget_spent,
)
from clinical_llm.utils.config import load_train_config


def test_no_budget_never_stops():
    assert not session_budget_spent(10**9, 0.0)


def test_budget_stops_at_and_after_the_limit():
    assert not session_budget_spent(3 * 3600 - 1, 3.0)
    assert session_budget_spent(3 * 3600, 3.0)
    assert session_budget_spent(4 * 3600, 3.0)


def test_partial_run_is_not_complete():
    assert not run_is_complete(700, 2100)
    assert not run_is_complete(2099, 2100)
    assert run_is_complete(2100, 2100)


def test_epoch_bounded_run_is_complete_when_train_returns():
    assert run_is_complete(123, -1)


def test_milestones_are_invisible_to_checkpoint_resume(tmp_path):
    d = milestone_dir(str(tmp_path), 700)
    d.mkdir(parents=True)
    assert d.name == "step-00700"
    assert not list(tmp_path.glob("checkpoint-*"))


def test_v2_config_is_a_longer_run_of_the_same_recipe():
    """v2 changes the step budget and nothing else, so a difference in score
    can be attributed to it."""
    v1 = load_train_config("configs/train_gpu.yaml")
    v2 = load_train_config("configs/train_gpu_v2.yaml")
    assert v2.max_steps > v1.max_steps
    assert v2.milestone_steps > 0 and v2.max_steps % v2.milestone_steps == 0
    assert v1.max_steps % v2.milestone_steps == 0, "need a milestone at v1's step count"
    assert v2.output_dir != v1.output_dir, "v2 must not resume from v1's checkpoints"
    same = ("learning_rate", "per_device_batch_size", "grad_accum", "warmup_ratio",
            "weight_decay", "lr_scheduler_type", "packing", "seed", "bf16")
    for k in same:
        assert getattr(v1, k) == getattr(v2, k), k
    assert v1.model == v2.model
    assert v1.lora == v2.lora
