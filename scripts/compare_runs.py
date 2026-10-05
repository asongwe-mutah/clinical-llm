#!/usr/bin/env python3
"""Paired comparison of two fine-tuned adapters from their eval reports.

``run_eval`` compares an adapter against the base model. Choosing between two
adapters needs the adapters compared against *each other*, on the same items:

    python scripts/compare_runs.py reports/eval_v1_medqa_dev.json \\
        reports/eval_v2_step2100_medqa_dev.json --benchmark medqa_dev

Reads the stored per-item outcomes; no inference is run. The first report is
the incumbent (A), the second the challenger (B). "Fixed" means A wrong and B
right; "regressed" the reverse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clinical_llm.eval.run_eval import mcnemar  # noqa: E402


def per_item(path: str, benchmark: str) -> list[int]:
    report = json.loads(Path(path).read_text())
    try:
        return report["runs"]["fine-tuned"][benchmark]["per_item"]
    except KeyError:
        sys.exit(f"{path} has no fine-tuned per-item results for '{benchmark}'.")


def compare(a: list[int], b: list[int]) -> dict:
    if len(a) != len(b):
        sys.exit(f"item counts differ ({len(a)} vs {len(b)}): not the same benchmark slice.")
    n = len(a)
    out = {"n": n, "a_accuracy": sum(a) / n, "b_accuracy": sum(b) / n}
    out["delta"] = out["b_accuracy"] - out["a_accuracy"]
    out["mcnemar"] = mcnemar(a, b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("incumbent")
    ap.add_argument("challenger")
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    r = compare(per_item(args.incumbent, args.benchmark), per_item(args.challenger, args.benchmark))
    m = r["mcnemar"]
    print(f"{args.benchmark}  (n={r['n']})")
    print(f"  A (incumbent)  {r['a_accuracy']:.4f}   {args.incumbent}")
    print(f"  B (challenger) {r['b_accuracy']:.4f}   {args.challenger}")
    print(f"  delta B-A      {r['delta']*100:+.2f} pp")
    verdict = "significant" if m["p_value"] < 0.05 else "NOT significant"
    print(f"  McNemar        p = {m['p_value']:.4g}  ({verdict} at alpha=0.05)")
    print(f"                 {m['c_fixes']} fixed, {m['b_regressions']} regressed, "
          f"{m['n_discordant']} discordant; {m['method']}")


if __name__ == "__main__":
    main()
