"""Merge a trained LoRA adapter into the base model weights.

A merged model is a single standalone checkpoint that any standard
``transformers`` / vLLM / TGI runtime can serve without PEFT. Useful for
deployment; the serving code can also load base+adapter directly.

Usage::

    python -m clinical_llm.train.merge_lora \
        --base Qwen/Qwen2.5-3B-Instruct \
        --adapter outputs/clinical-qlora \
        --out outputs/clinical-merged
"""

from __future__ import annotations

import argparse

from clinical_llm.utils.logging import get_logger

log = get_logger("merge")


def merge(base_model: str, adapter_dir: str, out_dir: str, trust_remote_code: bool = False) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading base model %s ...", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    log.info("attaching adapter %s ...", adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    log.info("merging weights ...")
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(out_dir)
    log.info("merged model written to %s", out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    merge(args.base, args.adapter, args.out, args.trust_remote_code)


if __name__ == "__main__":
    main()
