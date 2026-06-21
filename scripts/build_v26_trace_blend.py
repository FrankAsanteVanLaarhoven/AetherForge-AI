"""
Build v2.6 trace-gated blended training datasets.

Generates one JSONL per trace ratio from the fixed base pool:
  base pool = general agent examples + failure cases + memory records
  traces    = execution_traces.jsonl, subsampled to the requested ratio

Ratio is expressed as fraction of the *total* dataset that should be traces.
Example: --trace-ratio 0.25 means traces make up 25% of the final dataset.

Usage:
  python scripts/build_v26_trace_blend.py --trace-ratio 0.0
  python scripts/build_v26_trace_blend.py --trace-ratio 0.10
  python scripts/build_v26_trace_blend.py --trace-ratio 0.25
  python scripts/build_v26_trace_blend.py --trace-ratio 0.50
  python scripts/build_v26_trace_blend.py --trace-ratio 1.0   # = v2.5 full blend
"""

import argparse
import json
import math
import pathlib
import random
import sys


def load_jsonl(path: pathlib.Path) -> list[dict]:
    records = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skip malformed line {i} in {path}: {e}", file=sys.stderr)
    return records


def as_training_example(record: dict) -> dict | None:
    if "instruction" in record and "response" in record:
        return {"instruction": record["instruction"], "response": record["response"]}
    if "task" in record and "corrected_tool_call" in record:
        return {"instruction": record["task"], "response": record["corrected_tool_call"]}
    return None


def load_examples(path: pathlib.Path, label: str) -> list[dict]:
    if not path.exists():
        print(f"  SKIP  {label}: {path} not found", file=sys.stderr)
        return []
    raw = load_jsonl(path)
    examples = [ex for r in raw if (ex := as_training_example(r)) is not None]
    skipped = len(raw) - len(examples)
    if skipped:
        print(f"  NOTE  {label}: {skipped} records skipped (format mismatch)", file=sys.stderr)
    print(f"  OK    {label}: {len(examples)} examples")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2.6 trace-gated blended dataset")
    parser.add_argument("--trace-ratio", type=float, required=True,
                        help="Fraction of final dataset that should be execution traces (0.0–1.0)")
    parser.add_argument("--general-file",   default="data/agent_only_data.jsonl")
    parser.add_argument("--traces-file",    default="data/execution_traces.jsonl")
    parser.add_argument("--failure-file",   default="data/dev_set_data.jsonl")
    parser.add_argument("--memory-records", default="memory/index/records.jsonl")
    parser.add_argument("--output-dir",     default="data")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.trace_ratio <= 1.0:
        print("ERROR: --trace-ratio must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    root = pathlib.Path(__file__).parent.parent
    random.seed(args.seed)

    ratio_tag = f"{int(args.trace_ratio * 100):03d}"
    out_name = f"v26_blend_traces{ratio_tag}pct.jsonl"
    out_path = root / args.output_dir / out_name

    print(f"Building v2.6 blend — trace ratio {args.trace_ratio:.0%} → {out_name}")

    base_pool = (
        load_examples(root / args.general_file,   "general-agent")
        + load_examples(root / args.failure_file, "failure-cases")
        + load_examples(root / args.memory_records, "memory-records")
    )
    all_traces = load_examples(root / args.traces_file, "execution-traces (full pool)")

    if args.trace_ratio == 0.0:
        traces_used = []
        final = base_pool
    elif args.trace_ratio >= 1.0:
        traces_used = all_traces
        final = base_pool + all_traces
    else:
        # n_traces / (n_base + n_traces) = ratio  =>  n_traces = ratio * n_base / (1 - ratio)
        n_traces = int(math.ceil(args.trace_ratio * len(base_pool) / (1.0 - args.trace_ratio)))
        n_traces = min(n_traces, len(all_traces))
        traces_used = random.sample(all_traces, n_traces)
        final = base_pool + traces_used

    random.shuffle(final)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    trace_pct = len(traces_used) / len(final) * 100 if final else 0
    print(f"\nDataset composition ({out_name}):")
    print(f"  base pool (general + failure + memory) : {len(base_pool)}")
    print(f"  execution traces used                  : {len(traces_used)}")
    print(f"  trace fraction (actual)                : {trace_pct:.1f}%")
    print(f"  TOTAL                                  : {len(final)}")
    print(f"\nWrote {len(final)} examples → {out_path}")


if __name__ == "__main__":
    main()
