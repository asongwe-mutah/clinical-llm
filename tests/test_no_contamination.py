"""Guard against train/eval contamination.

This exists because it already happened. ``iter_pubmedqa`` trained on
``qiaojin/PubMedQA`` config ``pqa_labeled`` split ``train`` -- the exact 1,000
items ``load_pubmedqa_test`` scores. ``prepare.py`` holds out 2%, so ~980 of
the evaluation items were in the training corpus, and the resulting +9.90 pp
"improvement" was memorisation.

Nothing in the pipeline objected. These tests are that objection, and they run
offline against declared specs rather than downloading anything.
"""

from __future__ import annotations

from clinical_llm.data.datasets import QA_SOURCES, TRAIN_SPECS
from clinical_llm.eval.benchmarks import BENCHMARKS, EVAL_SPECS


def test_no_training_source_shares_a_split_with_a_benchmark():
    """The regression: identical (dataset, config, split) on both sides."""
    overlaps = []
    for tname, tspec in TRAIN_SPECS.items():
        for ename, espec in EVAL_SPECS.items():
            if tspec == espec:
                overlaps.append(f"train:{tname} and eval:{ename} both read {tspec}")
    assert not overlaps, "train/eval contamination:\n  " + "\n  ".join(overlaps)


def test_pubmedqa_trains_on_artificial_not_labeled():
    """pqa_labeled is the 1,000-item evaluation set. Never train on it."""
    dataset, config, _ = TRAIN_SPECS["pubmedqa"]
    assert dataset == "qiaojin/PubMedQA"
    assert config != "pqa_labeled", (
        "pqa_labeled IS the benchmark -- train on pqa_artificial instead"
    )
    assert EVAL_SPECS["pubmedqa"][1] == "pqa_labeled"


def test_medmcqa_uses_disjoint_splits():
    assert TRAIN_SPECS["medmcqa"][2] == "train"
    assert EVAL_SPECS["medmcqa"][2] == "validation"


def test_medqa_is_never_trained_on():
    """MedQA is the clean held-out benchmark; keep it out of the corpus.

    Covers every MedQA split, including the dev split used for selection:
    training on the selection set would bias the choice it exists to make.
    """
    for ename in ("medqa", "medqa_dev"):
        medqa_eval = EVAL_SPECS[ename]
        for tname, tspec in TRAIN_SPECS.items():
            assert tspec[0] != medqa_eval[0], f"train:{tname} reads the MedQA dataset"


def test_selection_split_is_not_the_reported_split():
    """Choosing a checkpoint on the split you then report is contamination by
    selection rather than by training -- slower, same failure."""
    assert EVAL_SPECS["medqa_dev"][0] == EVAL_SPECS["medqa"][0]
    assert EVAL_SPECS["medqa_dev"][2] == "dev"
    assert EVAL_SPECS["medqa"][2] == "test"
    assert EVAL_SPECS["medqa_dev"] != EVAL_SPECS["medqa"]


def test_no_two_benchmarks_read_the_same_split():
    specs = list(EVAL_SPECS.values())
    assert len(specs) == len(set(specs)), f"duplicate eval specs: {specs}"


def test_specs_cover_every_loader():
    """A loader with no declared spec is invisible to the checks above."""
    assert set(TRAIN_SPECS) == set(QA_SOURCES), (
        f"TRAIN_SPECS {set(TRAIN_SPECS)} != QA_SOURCES {set(QA_SOURCES)}"
    )
    assert set(EVAL_SPECS) == set(BENCHMARKS), (
        f"EVAL_SPECS {set(EVAL_SPECS)} != BENCHMARKS {set(BENCHMARKS)}"
    )
