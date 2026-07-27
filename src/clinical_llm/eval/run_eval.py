"""Evaluate a model on medical MCQ benchmarks and write a JSON report.

Scores accuracy by letter-constrained *log-likelihood* rather than free
generation: for each option we compute the model's average token log-prob of
the answer continuation and pick the argmax. This is standard practice for
MCQ eval (used by lm-eval-harness) — it is deterministic, needs no parsing,
and does not penalise a correct-but-verbose model.

Usage::

    # evaluate the fine-tuned adapter and compare against the base model
    python -m clinical_llm.eval.run_eval \
        --base Qwen/Qwen2.5-3B-Instruct \
        --adapter outputs/clinical-qlora \
        --benchmarks medmcqa pubmedqa \
        --max-items 500 \
        --out reports/eval.json

Pass ``--no-base`` to skip the base-model comparison, or omit ``--adapter`` to
score the base model alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from clinical_llm.data.formatting import SYSTEM_PROMPT, format_mcq_question, render_with_tokenizer
from clinical_llm.eval.benchmarks import BENCHMARKS, MCQItem
from clinical_llm.utils.logging import get_logger

log = get_logger("eval")


def _load_model(base_model: str, adapter: Optional[str], trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _option_logprob(model, tokenizer, device, prompt: str, continuation: str) -> float:
    """Average per-token log-prob of ``continuation`` given ``prompt``."""
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids.to(device)
    cont_len = full_ids.shape[1] - prompt_ids.shape[1]
    if cont_len <= 0:
        return float("-inf")

    with torch.no_grad():
        logits = model(full_ids).logits
    # logits[:, t] predicts token t+1; align to the continuation region.
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = full_ids[0, 1:]
    cont_logprobs = logprobs[-cont_len:].gather(1, targets[-cont_len:].unsqueeze(1)).squeeze(1)
    return float(cont_logprobs.mean().item())


def evaluate_item(model, tokenizer, device, item: MCQItem) -> bool:
    letters = [chr(ord("A") + i) for i in range(len(item.options))]
    user = format_mcq_question(item.question, item.options, include_instruction=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt = render_with_tokenizer(tokenizer, messages, add_generation_prompt=True)

    scores = []
    for letter, opt in zip(letters, item.options):
        continuation = f"Answer: {letter}. {opt}"
        scores.append(_option_logprob(model, tokenizer, device, prompt, continuation))
    pred = int(max(range(len(scores)), key=lambda i: scores[i]))
    return pred == item.gold_index


def run(
    base_model: str,
    adapter: Optional[str],
    benchmarks: list[str],
    max_items: Optional[int],
    trust_remote_code: bool,
) -> dict:
    model, tokenizer, device = _load_model(base_model, adapter, trust_remote_code)
    tag = "fine-tuned" if adapter else "base"
    log.info("evaluating %s model on %s (device=%s)", tag, device, benchmarks)

    results = {}
    for name in benchmarks:
        if name not in BENCHMARKS:
            log.warning("unknown benchmark '%s'; skipping", name)
            continue
        correct = total = 0
        for item in BENCHMARKS[name](max_items=max_items):
            correct += int(evaluate_item(model, tokenizer, device, item))
            total += 1
            if total % 50 == 0:
                log.info("  %s: %d/%d (%.1f%%)", name, correct, total, 100 * correct / total)
        acc = correct / total if total else 0.0
        results[name] = {"accuracy": acc, "n": total, "correct": correct}
        log.info("%s [%s]: accuracy=%.4f (n=%d)", name, tag, acc, total)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base model id or path")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
    ap.add_argument("--benchmarks", nargs="+", default=["medmcqa", "pubmedqa"])
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--no-base", action="store_true", help="skip base-model comparison")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out", default="reports/eval.json")
    args = ap.parse_args()

    report: dict = {"base_model": args.base, "adapter": args.adapter, "runs": {}}

    if args.adapter:
        report["runs"]["fine-tuned"] = run(
            args.base, args.adapter, args.benchmarks, args.max_items, args.trust_remote_code
        )
    if not args.no_base:
        report["runs"]["base"] = run(
            args.base, None, args.benchmarks, args.max_items, args.trust_remote_code
        )

    # Convenience: compute deltas when both runs exist.
    if "base" in report["runs"] and "fine-tuned" in report["runs"]:
        deltas = {}
        for name in report["runs"]["base"]:
            if name in report["runs"]["fine-tuned"]:
                b = report["runs"]["base"][name]["accuracy"]
                f = report["runs"]["fine-tuned"][name]["accuracy"]
                deltas[name] = {"base": b, "fine_tuned": f, "delta": f - b}
        report["deltas"] = deltas

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote report to %s", out)
    print(json.dumps(report.get("deltas", report["runs"]), indent=2))


if __name__ == "__main__":
    main()
