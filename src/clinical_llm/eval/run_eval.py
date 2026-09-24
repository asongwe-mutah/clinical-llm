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
import math
from pathlib import Path

from clinical_llm.data.formatting import SYSTEM_PROMPT, format_mcq_question, render_with_tokenizer
from clinical_llm.eval.benchmarks import BENCHMARKS, MCQItem
from clinical_llm.utils.logging import get_logger

log = get_logger("eval")


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 dof. No scipy dependency."""
    return math.erfc(math.sqrt(x / 2.0))


def mcnemar(base_items: list[int], tuned_items: list[int]) -> dict:
    """McNemar's paired test on per-item correctness.

    The base and fine-tuned models score the *same* items, so an unpaired
    two-proportion test throws away the pairing and is needlessly
    conservative. McNemar looks only at the discordant pairs -- items where
    exactly one model was right -- which is where the evidence actually lives.

    ``b`` = base right / tuned wrong (regressions), ``c`` = base wrong /
    tuned right (fixes). Under the null they are equal.
    """
    if len(base_items) != len(tuned_items):
        return {"error": f"length mismatch: {len(base_items)} vs {len(tuned_items)}"}

    pairs = list(zip(base_items, tuned_items, strict=True))
    b = sum(1 for x, y in pairs if x == 1 and y == 0)
    c = sum(1 for x, y in pairs if x == 0 and y == 1)
    n_disc = b + c

    if n_disc == 0:
        return {"b_regressions": b, "c_fixes": c, "n_discordant": 0,
                "p_value": 1.0, "method": "degenerate (no discordant pairs)"}

    if n_disc < 25:
        # Exact two-sided binomial test against p=0.5.
        k = min(b, c)
        tail = sum(math.comb(n_disc, i) for i in range(k + 1)) * (0.5 ** n_disc)
        p = min(1.0, 2.0 * tail)
        method = "exact binomial (two-sided)"
        stat = None
    else:
        stat = (abs(b - c) - 1) ** 2 / n_disc   # Edwards continuity correction
        p = _chi2_sf_1df(stat)
        method = "chi-square with continuity correction, 1 dof"

    return {"b_regressions": b, "c_fixes": c, "n_discordant": n_disc,
            "statistic": stat, "p_value": p, "method": method}


def wilson_interval(correct: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval -- better than normal approximation near 0/1."""
    if n == 0:
        return [0.0, 0.0]
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _describe_outputs(p: Path) -> str:
    """Best-effort hint about what actually is on disk near ``p``."""
    parent = p.parent
    if parent.exists():
        found = sorted(c.name for c in parent.iterdir() if c.is_dir())
        return f"{parent}/ contains: {found or '(no subdirectories)'}"
    return f"{parent}/ does not exist either"


def validate_adapter(adapter: str) -> None:
    """Fail fast, and legibly, on a local adapter path that isn't one.

    Without this, ``PeftModel.from_pretrained`` treats a non-existent local path
    as a Hugging Face Hub repo id, so the user gets a 404 for
    ``huggingface.co/outputs/clinical-qlora`` followed by "Invalid username or
    password" -- which reads like an auth problem when the real cause is that
    training never wrote an adapter.
    """
    p = Path(adapter)
    if p.exists():
        if not p.is_dir():
            raise SystemExit(f"--adapter {adapter} is a file; expected a directory.")
        if not (p / "adapter_config.json").exists():
            raise SystemExit(
                f"--adapter {adapter} exists but contains no adapter_config.json, so it "
                f"is not a saved LoRA adapter.\n  {_describe_outputs(p)}\n"
                "Let the training step finish -- it writes the adapter at the very end, "
                "via trainer.save_model()."
            )
        return
    # Not on disk. If its parent is a real local directory the user plainly meant
    # a path, so say so instead of silently querying the Hub.
    if p.parent.exists():
        raise SystemExit(
            f"--adapter {adapter} does not exist.\n  {_describe_outputs(p)}\n"
            "Train first, or point --adapter at an existing adapter directory."
        )


def _load_model(base_model: str, adapter: str | None, trust_remote_code: bool):
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
    for letter, opt in zip(letters, item.options, strict=True):
        continuation = f"Answer: {letter}. {opt}"
        scores.append(_option_logprob(model, tokenizer, device, prompt, continuation))
    pred = int(max(range(len(scores)), key=lambda i: scores[i]))
    return pred == item.gold_index


def run(
    base_model: str,
    adapter: str | None,
    benchmarks: list[str],
    max_items: int | None,
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
        per_item: list[int] = []
        for item in BENCHMARKS[name](max_items=max_items):
            hit = int(evaluate_item(model, tokenizer, device, item))
            per_item.append(hit)
            correct += hit
            total += 1
            if total % 50 == 0:
                log.info("  %s: %d/%d (%.1f%%)", name, correct, total, 100 * correct / total)
        acc = correct / total if total else 0.0
        lo, hi = wilson_interval(correct, total)
        results[name] = {
            "accuracy": acc,
            "n": total,
            "correct": correct,
            "ci95": [lo, hi],
            # Per-item correctness, in benchmark order. Enables the paired
            # McNemar test; the two runs iterate the same items in the same
            # order, so index i refers to the same question in both.
            "per_item": per_item,
        }
        log.info(
            "%s [%s]: accuracy=%.4f (n=%d, 95%% CI %.3f-%.3f)", name, tag, acc, total, lo, hi
        )
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
        validate_adapter(args.adapter)
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
                base_run = report["runs"]["base"][name]
                tuned_run = report["runs"]["fine-tuned"][name]
                entry = {
                    "base": base_run["accuracy"],
                    "fine_tuned": tuned_run["accuracy"],
                    "delta": tuned_run["accuracy"] - base_run["accuracy"],
                    "n": base_run.get("n"),
                    "base_ci95": base_run.get("ci95"),
                    "fine_tuned_ci95": tuned_run.get("ci95"),
                }
                bi, ti = base_run.get("per_item"), tuned_run.get("per_item")
                if bi and ti:
                    entry["mcnemar"] = mcnemar(bi, ti)
                deltas[name] = entry
        report["deltas"] = deltas

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("wrote report to %s", out)

    # Print a readable summary; the raw per-item arrays stay in the JSON.
    if report.get("deltas"):
        print("\n=== base vs fine-tuned ===")
        for name, d in report["deltas"].items():
            print(f"\n{name}  (n={d.get('n')})")
            print(f"  base       {d['base']:.4f}")
            print(f"  fine-tuned {d['fine_tuned']:.4f}")
            print(f"  delta      {d['delta']:+.4f}  ({d['delta']*100:+.2f} pp)")
            m = d.get("mcnemar")
            if m and "error" not in m:
                verdict = "significant" if m["p_value"] < 0.05 else "NOT significant"
                print(f"  McNemar    p = {m['p_value']:.4f}  ({verdict} at alpha=0.05)")
                print(f"             {m['c_fixes']} fixed, {m['b_regressions']} regressed, "
                      f"{m['n_discordant']} discordant")
                print(f"             {m['method']}")
    else:
        print(json.dumps(report["runs"], indent=2))


if __name__ == "__main__":
    main()
