"""Pre-download the benchmark/training datasets into the HF cache.

Handy on a fresh GPU box so the first training/eval run does not stall on
network. Purely optional — the pipeline downloads on demand otherwise.

    python scripts/download_data.py
"""

from __future__ import annotations


def main() -> None:
    from datasets import load_dataset

    specs = [
        ("qiaojin/PubMedQA", "pqa_labeled", "train"),
        ("openlifescienceai/medmcqa", None, "train"),
        ("openlifescienceai/medmcqa", None, "validation"),
        ("lavita/MedQuAD", None, "train"),
        ("openlifescienceai/medqa", None, "test"),
    ]
    for hf_id, config, split in specs:
        print(f"downloading {hf_id} [{config or 'default'}:{split}] ...")
        try:
            if config:
                load_dataset(hf_id, config, split=split)
            else:
                load_dataset(hf_id, split=split)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed ({exc}); you may need `huggingface-cli login` for gated sets")
    print("done.")


if __name__ == "__main__":
    main()
