#!/usr/bin/env python3
"""build_option_a_blended_data.py — Build blended training file for Option A.

Combines general agent-only data with targeted failure traces.
Output is one JSONL file suitable for --training-file in finetune_qwen_code_agent.py.

This file is local-only (in .git/info/exclude). Do not commit it.
"""
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--general-file", default="data/agent_only_data.jsonl",
                   help="General agent-only training data")
    p.add_argument("--targeted-file", default="data/dev_set_failing_5.jsonl",
                   help="Primary targeted failure traces")
    p.add_argument("--fallback-targeted", default="data/dev_set_data.jsonl",
                   help="Fallback targeted file if --targeted-file is missing")
    p.add_argument("--output", default="data/dev_set_blended.jsonl",
                   help="Output blended JSONL file")
    args = p.parse_args()

    general_path = Path(args.general_file)
    targeted_path = Path(args.targeted_file)
    fallback_path = Path(args.fallback_targeted)
    out_path = Path(args.output)

    if not general_path.exists() or general_path.stat().st_size == 0:
        print(f"ERROR: general file missing or empty: {general_path}")
        print("       Run: make data-agent-only")
        sys.exit(1)

    if not targeted_path.exists() or targeted_path.stat().st_size == 0:
        if fallback_path.exists() and fallback_path.stat().st_size > 0:
            print(f"[blend] {targeted_path} not found — using fallback: {fallback_path}")
            targeted_path = fallback_path
        else:
            print(f"ERROR: targeted file missing: {targeted_path}")
            print(f"       Also checked fallback: {fallback_path}")
            print("       Run: make generate-targeted-dev-traces")
            sys.exit(1)

    general_lines = [l for l in general_path.read_text().splitlines() if l.strip()]
    targeted_lines = [l for l in targeted_path.read_text().splitlines() if l.strip()]
    blended = general_lines + targeted_lines

    out_path.write_text("\n".join(blended) + "\n")

    print(f"[blend] general  : {len(general_lines):5d} lines  ({general_path})")
    print(f"[blend] targeted : {len(targeted_lines):5d} lines  ({targeted_path})")
    print(f"[blend] total    : {len(blended):5d} lines  → {out_path}")


if __name__ == "__main__":
    main()
