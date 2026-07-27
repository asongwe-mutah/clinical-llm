"""Build the unified instruction corpus as train/val JSONL files.

Usage::

    python -m clinical_llm.data.prepare --config configs/data.yaml

Output: ``<output_dir>/train.jsonl`` and ``<output_dir>/val.jsonl``, each line a
record of shape ``{"messages": [...], "meta": {...}}`` as produced by
:meth:`ChatExample.to_record`. Every downstream stage consumes this schema, so
the corpus is fully decoupled from the specific datasets that built it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from clinical_llm.data.datasets import QA_SOURCES, iter_notes_summarization
from clinical_llm.utils.config import DataConfig, load_data_config
from clinical_llm.utils.logging import get_logger

log = get_logger("prepare")


def build_corpus(cfg: DataConfig) -> list[dict]:
    records: list[dict] = []
    enabled = {
        "pubmedqa": cfg.use_pubmedqa,
        "medmcqa": cfg.use_medmcqa,
        "medquad": cfg.use_medquad,
    }
    for name, adapter in QA_SOURCES.items():
        if not enabled.get(name):
            continue
        log.info("loading %s (cap=%s) ...", name, cfg.max_per_source)
        n = 0
        for ex in adapter(max_examples=cfg.max_per_source):
            records.append(ex.to_record())
            n += 1
        log.info("  -> %d examples from %s", n, name)

    if cfg.use_notes_summarization:
        if not cfg.notes_jsonl or not Path(cfg.notes_jsonl).exists():
            log.warning(
                "notes summarization enabled but notes_jsonl missing (%s); skipping",
                cfg.notes_jsonl,
            )
        else:
            log.info("loading notes summarization from %s ...", cfg.notes_jsonl)
            n = 0
            for ex in iter_notes_summarization(cfg.notes_jsonl, max_examples=cfg.max_per_source):
                records.append(ex.to_record())
                n += 1
            log.info("  -> %d note-summary examples", n)

    return records


def write_splits(records: list[dict], cfg: DataConfig) -> tuple[Path, Path]:
    rng = random.Random(cfg.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * cfg.val_fraction)) if records else 0
    val, train = records[:n_val], records[n_val:]

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out / "train.jsonl", out / "val.jsonl"
    for path, rows in ((train_path, train), (val_path, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("wrote %d train / %d val examples to %s", len(train), len(val), out)
    return train_path, val_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()

    cfg = load_data_config(args.config)
    records = build_corpus(cfg)
    if not records:
        raise SystemExit("no records produced; check your data config / network access")
    write_splits(records, cfg)


if __name__ == "__main__":
    main()
