"""
scripts/build_dense_memory_index.py — build a dense vector index from a memory source.

Reads records from an existing TF-IDF index (via memory.store.load_index)
or from a raw JSONL file, then embeds them with a SentenceTransformer model.

Output is written to a separate directory and does NOT overwrite the TF-IDF index.

Usage (from repo root):
    conda run -n aetherforge-train python scripts/build_dense_memory_index.py \\
        --source-index memory/index_adapted \\
        --output-dir memory/dense_index_adapted \\
        --dense-model sentence-transformers/all-MiniLM-L6-v2 \\
        --batch-size 32 \\
        --device auto

    # From raw JSONL instead of existing index:
    conda run -n aetherforge-train python scripts/build_dense_memory_index.py \\
        --memory-jsonl memory/raw_adapted/records.jsonl \\
        --output-dir memory/dense_index_adapted \\
        --dense-model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_records_from_index(index_dir: Path) -> list[dict]:
    """Load verified records from an existing TF-IDF index via SQLite."""
    from memory.store import load_index
    state = load_index(index_dir)
    records = state.get("records", [])
    verified = [r for r in records if r.get("verified", False)]
    print(f"[build] Loaded {len(verified)} verified records from {index_dir}")
    return verified


def load_records_from_jsonl(jsonl_path: Path) -> list[dict]:
    """Load records from a raw JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
                records.append(r)
            except json.JSONDecodeError as e:
                print(f"  WARN {jsonl_path.name}:{lineno} — JSON error: {e}")
    verified = [r for r in records if r.get("verified", False)]
    print(f"[build] Loaded {len(verified)} verified records from {jsonl_path}")
    return verified


def main():
    p = argparse.ArgumentParser(description="Build dense memory index")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--source-index",
        help="Path to existing TF-IDF index directory (e.g. memory/index_adapted)",
    )
    src.add_argument(
        "--memory-jsonl",
        help="Path to raw JSONL file with memory records",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write dense index (e.g. memory/dense_index_adapted)",
    )
    p.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local directory path",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32)",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for encoding (default: auto)",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    # Guard: never overwrite TF-IDF indexes
    protected = {
        "index", "index_adapted", "index_adapted_v2", "index_adapted_v29",
        "index_adapted_v3", "index_v28_filtered", "index_v29_repair",
    }
    if output_dir.name in protected:
        print(
            f"ERROR: output-dir '{output_dir}' matches a protected TF-IDF index name. "
            f"Use a name prefixed with 'dense_' to make the intent clear.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load source records
    if args.source_index:
        records = load_records_from_index(Path(args.source_index))
    else:
        records = load_records_from_jsonl(Path(args.memory_jsonl))

    if not records:
        print("ERROR: No verified records found. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Build index
    from memory.dense_index import build_dense_index
    result = build_dense_index(
        records=records,
        output_dir=output_dir,
        model_name=args.dense_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\n[build] Done.")
    print(f"  Records:    {result['n_records']}")
    print(f"  Dimensions: {result['dim']}")
    print(f"  Model:      {result['model_name']}")
    print(f"  Device:     {result['device']}")
    print(f"  Output:     {result['output_dir']}")


if __name__ == "__main__":
    main()
