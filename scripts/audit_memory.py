"""
scripts/audit_memory.py — audit raw memory records and index consistency.

Checks:
  1. All raw JSONL records pass schema validation
  2. Index exists and record count matches raw count
  3. No unverified records in the index
  4. No banned markers (VALID:, DIFFERENT_CORRECTED_TOOL_CALL, triple-quotes, fences)
  5. All corrected_tool_calls have valid JSON args with a 'code' key
  6. Spot-checks that the code in each record is valid Python

Usage:
    conda run -n ml-torch python scripts/audit_memory.py
    conda run -n ml-torch python scripts/audit_memory.py --raw-dir memory/raw --index-dir memory/index
"""

import ast
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.validate import validate_record


def _check_code_parseable(ctc: str) -> str | None:
    """Return error string if code in corrected_tool_call is not parseable Python."""
    prefix = "TOOL_CALL: execute_code("
    if not ctc.strip().startswith(prefix):
        return None
    body = ctc.strip()[len(prefix):]
    if not body.endswith(")"):
        return "unmatched paren"
    try:
        args = json.loads(body[:-1])
    except json.JSONDecodeError:
        return None  # already caught by validate_record
    code = args.get("code", "")
    if not code:
        return None
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError in code: {e}"
    return None


def audit_raw(raw_dir: Path) -> tuple[int, int]:
    """Validate all raw JSONL records.  Returns (total, failed)."""
    total = 0
    failed = 0
    for path in sorted(raw_dir.glob("*.jsonl")):
        print(f"\n  {path.name}")
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"    FAIL line {lineno}: JSON error: {e}")
                    failed += 1
                    continue

                errs = validate_record(rec)
                code_err = _check_code_parseable(rec.get("corrected_tool_call", ""))
                if code_err:
                    errs.append(code_err)

                if errs:
                    failed += 1
                    print(f"    FAIL line {lineno} id={rec.get('id')!r}")
                    for e in errs:
                        print(f"         {e}")
                else:
                    print(f"    OK   line {lineno} [{rec.get('category')}] {rec.get('id')!r}")
    return total, failed


def audit_index(index_dir: Path, raw_total: int) -> tuple[int, int]:
    """Audit the index for consistency.  Returns (checks, failed)."""
    checks = 0
    failed = 0

    db_path = index_dir / "metadata.sqlite"
    if not db_path.exists():
        print(f"\n  WARNING: index not built yet ({db_path} missing)")
        print("  Run: python scripts/build_vector_memory.py")
        return 0, 0

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM memories").fetchall()
    conn.close()

    checks += 1
    if len(rows) != raw_total:
        print(f"  WARN: index has {len(rows)} records; raw has {raw_total} — rebuild index")
    else:
        print(f"  OK   index count matches raw: {len(rows)}")

    for row in rows:
        rec = dict(row)
        checks += 1
        if not rec.get("verified"):
            failed += 1
            print(f"  FAIL id={rec.get('id')!r}: verified=False in index")

    backend_file = index_dir / "backend.txt"
    if backend_file.exists():
        backend = backend_file.read_text().strip()
        checks += 1
        if backend == "faiss":
            faiss_file = index_dir / "faiss.index"
            if not faiss_file.exists():
                failed += 1
                print(f"  FAIL: backend=faiss but faiss.index missing")
            else:
                print(f"  OK   faiss.index present")
        else:
            vec_file = index_dir / "vectors.npy"
            if not vec_file.exists():
                failed += 1
                print(f"  FAIL: backend=numpy but vectors.npy missing")
            else:
                print(f"  OK   vectors.npy present")

    return checks, failed


def main():
    p = argparse.ArgumentParser(description="Audit offline vector memory records and index")
    p.add_argument("--raw-dir",   default="memory/raw",   help="Raw JSONL directory")
    p.add_argument("--index-dir", default="memory/index", help="Index directory")
    p.add_argument("--raw-only",  action="store_true",    help="Audit raw records only (skip index)")
    args = p.parse_args()

    raw_dir   = Path(args.raw_dir)
    index_dir = Path(args.index_dir)

    print("=" * 60)
    print("Memory audit")
    print("=" * 60)

    print(f"\n[1] Raw records ({raw_dir})")
    raw_total, raw_failed = audit_raw(raw_dir)

    if not args.raw_only:
        print(f"\n[2] Index consistency ({index_dir})")
        idx_checks, idx_failed = audit_index(index_dir, raw_total)
    else:
        idx_checks = idx_failed = 0

    total_failed = raw_failed + idx_failed
    print(f"\n{'=' * 60}")
    print(f"Raw records:   {raw_total - raw_failed}/{raw_total} OK")
    if not args.raw_only:
        print(f"Index checks:  {idx_checks - idx_failed}/{idx_checks} OK")
    status = "PASS" if total_failed == 0 else "FAIL"
    print(f"Overall: {status}  ({total_failed} issue(s))")
    print("=" * 60)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
