"""
Build v2.5 blended training dataset from four source categories:
  1. general agent examples   (agent_only_data.jsonl)
  2. verified tool-use traces (execution_traces.jsonl)
  3. failure cases            (dev_set_failing_5.jsonl or dev_set_data.jsonl)
  4. memory adaptation records (memory/index/records.jsonl)

All sources are converted to {"instruction": ..., "response": ...} format.
Output is written to data/v25_blended.jsonl.
"""

import argparse
import json
import pathlib
import sys


def load_jsonl(path: pathlib.Path) -> list[dict]:
    lines = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skip malformed line {i} in {path}: {e}", file=sys.stderr)
    return lines


def as_training_example(record: dict) -> dict | None:
    """Normalise a record from any source into {"instruction", "response"}."""
    if "instruction" in record and "response" in record:
        return {"instruction": record["instruction"], "response": record["response"]}
    # memory/index/records.jsonl format
    if "task" in record and "corrected_tool_call" in record:
        return {"instruction": record["task"], "response": record["corrected_tool_call"]}
    return None


def load_source(path: pathlib.Path, label: str) -> list[dict]:
    if not path.exists():
        print(f"  SKIP  {label}: {path} not found", file=sys.stderr)
        return []
    raw = load_jsonl(path)
    examples = [ex for r in raw if (ex := as_training_example(r)) is not None]
    skipped = len(raw) - len(examples)
    if skipped:
        print(f"  NOTE  {label}: {skipped} records skipped (unrecognised format)", file=sys.stderr)
    print(f"  OK    {label}: {len(examples)} examples from {path}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2.5 blended training dataset")
    parser.add_argument("--general-file",     default="data/agent_only_data.jsonl")
    parser.add_argument("--traces-file",      default="data/execution_traces.jsonl")
    parser.add_argument("--failure-file",   default="data/dev_set_data.jsonl",
                        help="Failure-case training pairs (instruction/response format)")
    parser.add_argument("--memory-records", default="memory/index/records.jsonl")
    parser.add_argument("--output",           default="data/v25_blended.jsonl")
    args = parser.parse_args()

    root = pathlib.Path(__file__).parent.parent

    print("Building v2.5 blended training dataset …")

    buckets: dict[str, list[dict]] = {
        "general-agent":    load_source(root / args.general_file,   "general-agent"),
        "execution-traces": load_source(root / args.traces_file,    "execution-traces"),
        "failure-cases":    load_source(root / args.failure_file,   "failure-cases"),
        "memory-records":   load_source(root / args.memory_records, "memory-records"),
    }

    all_examples: list[dict] = []
    for examples in buckets.values():
        all_examples.extend(examples)

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDataset composition:")
    for label, examples in buckets.items():
        print(f"  {label:<20}: {len(examples)}")
    print(f"  {'TOTAL':<20}: {len(all_examples)}")
    print(f"\nWrote {len(all_examples)} examples → {out_path}")


if __name__ == "__main__":
    main()
