"""
scripts/retrieve_memory.py — retrieve top-k verified memories for a query.

CLI wrapper around memory.core.retrieve().  Fully offline.

Usage:
    conda run -n ml-torch python scripts/retrieve_memory.py \\
        --query "Write word_count returning frequency dict" \\
        --top-k 4

    conda run -n ml-torch python scripts/retrieve_memory.py \\
        --query "LRU cache eviction" \\
        --top-k 2 \\
        --index-dir memory/index \\
        --format json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.core import retrieve, format_memory_block


def main():
    p = argparse.ArgumentParser(
        description="Retrieve verified memory records relevant to a task"
    )
    p.add_argument("--query",     required=True,          help="Task text to search for")
    p.add_argument("--top-k",     type=int, default=4,    help="Number of results to return")
    p.add_argument("--min-score", type=float, default=0.0,help="Minimum cosine similarity threshold")
    p.add_argument("--index-dir", default="memory/index", help="Index directory")
    p.add_argument("--format",    choices=["text", "json", "prompt"],
                   default="text", help="Output format")
    args = p.parse_args()

    index_dir = Path(args.index_dir)

    print(f"Query: {args.query!r}", file=sys.stderr)
    results = retrieve(
        task_text=args.query,
        index_dir=index_dir,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if not results:
        print("No relevant memories found.", file=sys.stderr)
        sys.exit(0)

    if args.format == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif args.format == "prompt":
        print(format_memory_block(results))
    else:
        print(f"\nFound {len(results)} result(s):\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r.get('score', 0):.4f}  "
                  f"category={r.get('category')}  "
                  f"failure_type={r.get('failure_type', 'n/a')}")
            print(f"     Task: {r.get('task', '')}")
            print(f"     Fix:  {r.get('corrected_tool_call', '')[:120]}...")
            print()


if __name__ == "__main__":
    main()
