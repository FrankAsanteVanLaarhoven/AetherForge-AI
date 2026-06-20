"""
scripts/build_vector_memory.py — build the offline vector memory index.

Loads all JSONL records from memory/raw/, validates them, embeds query_text
for each, and saves the FAISS index (or NumPy fallback) + SQLite metadata to
memory/index/.

Fully offline: no network calls, no remote API, no auto-downloads.
Fails closed if any record fails validation.

Usage:
    conda run -n ml-torch python scripts/build_vector_memory.py
    conda run -n ml-torch python scripts/build_vector_memory.py \\
        --raw-dir memory/raw \\
        --index-dir memory/index
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.core import build_index
from memory.validate import validate_record


def main():
    p = argparse.ArgumentParser(
        description="Build offline vector memory index from JSONL records"
    )
    p.add_argument("--raw-dir",   default="memory/raw",   help="Directory containing *.jsonl raw records")
    p.add_argument("--index-dir", default="memory/index", help="Output directory for index + metadata")
    p.add_argument("--dry-run",   action="store_true",    help="Validate records only, do not write index")
    args = p.parse_args()

    raw_dir   = Path(args.raw_dir)
    index_dir = Path(args.index_dir)

    if not raw_dir.exists():
        print(f"ERROR: raw_dir does not exist: {raw_dir}")
        sys.exit(1)

    # ── Dry run: validate only ─────────────────────────────────────────────
    if args.dry_run:
        print(f"[dry-run] Validating records in {raw_dir} ...")
        total = 0
        failed = 0
        for path in sorted(raw_dir.glob("*.jsonl")):
            with open(path) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  FAIL {path.name}:{lineno} JSON error: {e}")
                        failed += 1
                        continue
                    errs = validate_record(rec)
                    total += 1
                    if errs:
                        failed += 1
                        print(f"  FAIL {path.name}:{lineno} id={rec.get('id')!r}")
                        for e in errs:
                            print(f"       {e}")
                    else:
                        print(f"  OK   {path.name}:{lineno} id={rec.get('id')!r} [{rec.get('category')}]")
        print(f"\n{'OK' if not failed else 'FAIL'}  {total - failed}/{total} records valid")
        sys.exit(0 if not failed else 1)

    # ── Full build ─────────────────────────────────────────────────────────
    print(f"Building vector memory index ...")
    print(f"  raw_dir   : {raw_dir}")
    print(f"  index_dir : {index_dir}")
    print()

    n = build_index(raw_dir=raw_dir, index_dir=index_dir)

    # Write build manifest
    manifests_dir = ROOT / "memory" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_records": n,
        "raw_dir": str(raw_dir),
        "index_dir": str(index_dir),
        "backend": (index_dir / "backend.txt").read_text().strip()
                   if (index_dir / "backend.txt").exists() else "unknown",
    }
    manifest_path = manifests_dir / f"build_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")
    print(f"\nDone. {n} records indexed in {index_dir}")


if __name__ == "__main__":
    main()
