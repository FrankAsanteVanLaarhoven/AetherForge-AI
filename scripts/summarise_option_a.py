#!/usr/bin/env python3
"""summarise_option_a.py — Compare Option A result against the 75.0% clean baseline.

Decision rule:
  > 75.0%  = Option A improves clean held-out performance. Promote.
  = 75.0%  = No improvement, no damage. Champion unchanged.
  < 75.0%  = Regression. Reject Option A; champion remains 300-step LoRA.
"""
import argparse
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)


def summarise(df: pd.DataFrame, label: str):
    n = len(df)
    passed = int(df["passed"].fillna(False).sum())
    rate = 100 * passed / n
    print(f"  {label}")
    print(f"    Pass rate : {passed}/{n} = {rate:.1f}%")
    if "category" in df.columns:
        by_cat = df.groupby("category")["passed"].agg(["sum", "count"])
        by_cat["pct"] = (100 * by_cat["sum"] / by_cat["count"]).round(1)
        for cat, row in by_cat.iterrows():
            print(f"    {cat:10s}: {int(row['sum'])}/{int(row['count'])} ({row.pct:.0f}%)")
    failed = df.loc[~df["passed"].fillna(False)]
    if "id" in df.columns and not failed.empty:
        print(f"    Failed    : {list(failed['id'])}")
    elif "task" in df.columns and not failed.empty:
        print(f"    Failed    : {list(failed['task'])}")
    return passed, n, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   help="CSV for the clean 75.0% baseline")
    p.add_argument("--option-a", required=True,
                   help="CSV for the Option A result")
    args = p.parse_args()

    baseline_df = load(Path(args.baseline))
    option_a_df = load(Path(args.option_a))

    print()
    print("=" * 72)
    print("Option A — clean training improvement vs 75.0% frozen held-out baseline")
    print("=" * 72)

    b_pass, b_n, b_rate = summarise(baseline_df, "Baseline (300-step LoRA, original memory/index)")
    print()
    a_pass, a_n, a_rate = summarise(option_a_df, "Option A  (fresh LoRA from merged base, blended data)")

    delta = a_rate - b_rate
    print()
    print(f"  Delta: {delta:+.1f} pp  ({b_rate:.1f}% → {a_rate:.1f}%)")
    print()

    if a_rate > b_rate:
        print("  DECISION: PROMOTE Option A")
        print(f"  Option A improves clean frozen held-out from {b_rate:.1f}% to {a_rate:.1f}%.")
        print(f"  New clean champion: outputs/qwen15b_fresh_blended_350/final")
    elif a_rate == b_rate:
        print("  DECISION: NEUTRAL — no improvement, no damage.")
        print(f"  Option A matches baseline ({a_rate:.1f}%). Champion unchanged.")
        print(f"  Champion remains: outputs/qwen15b_memory_300steps/final")
    else:
        print("  DECISION: REJECT Option A")
        print(f"  Option A regresses from {b_rate:.1f}% to {a_rate:.1f}% ({delta:.1f} pp).")
        print(f"  Champion remains: outputs/qwen15b_memory_300steps/final")

    print()
    print("Claim boundaries:")
    print("  - This result uses memory/index (original, clean).")
    print("  - Do not mix with adapted-memory results (82.1% using memory/index_adapted).")
    print("=" * 72)


if __name__ == "__main__":
    main()
