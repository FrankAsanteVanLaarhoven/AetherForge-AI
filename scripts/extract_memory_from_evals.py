"""
scripts/extract_memory_from_evals.py — extract verified memory records from eval outputs.

Scans outputs/**/single.csv, best_of_3.csv, failure_analysis.csv for rows that
passed with a real tool call, validates each candidate record, deduplicates by
content hash, and writes to memory/raw/extracted_memories.jsonl.

Safety guarantees (same as seed_memories.jsonl):
  - Only rows where passed==True AND passed_via_tool==True
  - used_fallback_extraction must be False
  - Extracted TOOL_CALL must not contain markdown fences, triple-quoted JSON,
    VALID:, DIFFERENT_CORRECTED_TOOL_CALL, or Revised TOOL_CALL:
  - All records validated through memory.validate.validate_record before write
  - Deduplicated by sha256 content_hash

Usage:
    conda run -n ml-torch python scripts/extract_memory_from_evals.py
    conda run -n ml-torch python scripts/extract_memory_from_evals.py \\
        --outputs-dir outputs \\
        --out memory/raw/extracted_memories.jsonl \\
        --max-records 500 \\
        --min-quality verified_tool
"""

import argparse
import csv
import glob
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.validate import validate_record, content_hash as _content_hash


# ── Constants ─────────────────────────────────────────────────────────────────

PRIORITY_TASKS = frozenset({
    "word_count", "lru_cache", "graph_bfs", "roman_numerals",
    "merge_sorted", "fizzbuzz", "factorial", "unique_sorted",
})

# Markers that disqualify a candidate TOOL_CALL from being stored
_BANNED_IN_TOOL_CALL = [
    "VALID:",
    "DIFFERENT_CORRECTED_TOOL_CALL",
    'Revised TOOL_CALL:',
    '"""',
    "```",
]


# ── Task description map ──────────────────────────────────────────────────────

def _load_task_map() -> dict[str, dict]:
    """Return {task_id: task_dict} from the benchmark TASKS list."""
    try:
        from scripts.evaluate_code_agent import TASKS
        return {t["id"]: t for t in TASKS}
    except Exception as exc:
        print(f"[extract] Warning: could not load TASKS from evaluate_code_agent: {exc}")
        return {}


# ── Transcript parsing ────────────────────────────────────────────────────────

def _extract_passing_tool_call(transcript: str) -> str | None:
    """Extract the TOOL_CALL: execute_code(...) that immediately preceded OBSERVATION: PASS.

    Returns the full 'TOOL_CALL: execute_code({...})' string, or None if not found.
    """
    prefix = "TOOL_CALL: execute_code("

    # Find the position of the first OBSERVATION: PASS in the transcript
    obs_marker = "OBSERVATION: PASS"
    obs_pos = transcript.find(obs_marker)
    if obs_pos == -1:
        return None

    # Look only in the section before the first OBSERVATION: PASS
    section = transcript[:obs_pos]

    # Find the last TOOL_CALL: execute_code( before that marker
    tc_pos = section.rfind(prefix)
    if tc_pos == -1:
        return None

    # The JSON object starts right after the opening paren
    json_start = tc_pos + len(prefix)
    rest = transcript[json_start:]

    if not rest or rest[0] != "{":
        return None

    # Use JSONDecoder to parse exactly the JSON object
    try:
        decoder = json.JSONDecoder()
        obj, end_idx = decoder.raw_decode(rest)
    except json.JSONDecodeError:
        return None

    # After the JSON object must come the closing paren
    after_json = rest[end_idx:]
    close_paren_pos = after_json.find(")")
    if close_paren_pos == -1:
        return None

    # Reconstruct the full TOOL_CALL string
    json_str = rest[:end_idx]
    return f"TOOL_CALL: execute_code({json_str})"


def _clean_tool_call(ctc: str) -> str:
    """Normalise a tool call string extracted from a transcript.

    The transcript may contain CSV double-quote escaping. After csv.DictReader
    parses the row, the string is already unescaped, so no extra work is needed
    here — we just strip trailing whitespace.
    """
    return ctc.strip()


def _tool_call_is_clean(ctc: str) -> list[str]:
    """Return list of reasons the tool call is disqualified (empty = clean)."""
    reasons: list[str] = []
    for marker in _BANNED_IN_TOOL_CALL:
        if marker in ctc:
            reasons.append(f"contains banned marker {marker!r}")
    return reasons


# ── Failure type / category inference ────────────────────────────────────────

def _infer_failure_type(row: dict) -> str:
    """Determine the failure type from CSV columns."""
    if row.get("has_invalid_json") == "True":
        return "invalid_json"
    no_out = _int(row.get("no_output_count", 0))
    if no_out > 0:
        return "no_output"
    if row.get("has_indentation_error") == "True":
        return "indentation_error"
    exc_type = row.get("first_exception_type", "")
    if exc_type == "SyntaxError":
        return "syntax_error"
    if _int(row.get("n_errors", 0)) > 0:
        return "assertion_error"
    return "none"


def _infer_category(row: dict, task_id: str, failure_type: str) -> str:
    """Determine the memory extraction category."""
    if task_id in PRIORITY_TASKS:
        return "task_specific_success"
    if failure_type == "no_output":
        return "no_output_to_pass"
    if failure_type == "invalid_json":
        return "invalid_json_to_valid_json"
    if failure_type in ("assertion_error",):
        return "assertion_error_to_correct_assert"
    if _int(row.get("n_errors", 0)) > 0:
        return "failure_to_fix"
    return "verified_success"


def _int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ── Record construction ───────────────────────────────────────────────────────

def _build_record(
    row: dict,
    corrected_tool_call: str,
    task_map: dict,
    source: str,
) -> dict | None:
    """Build a candidate memory record from a passing CSV row.

    Returns None if the row is disqualified before validation.
    """
    task_id = row.get("id", "")

    # Get the full task description from the benchmark if available
    task_desc = ""
    if task_id in task_map:
        task_desc = task_map[task_id].get("task", task_id)
    else:
        task_desc = task_id  # fallback: use id as description

    failure_type = _infer_failure_type(row)
    category = _infer_category(row, task_id, failure_type)

    # Build a searchable query_text from task id + category + key words
    bench_cat = row.get("category", "")
    query_text = (
        f"{task_id} {bench_cat} "
        + " ".join(task_desc.split()[:20])
    ).strip()

    # Extract final_answer from transcript if present
    transcript = row.get("full_transcript", "")
    final_answer = _extract_final_answer(transcript)

    now = datetime.now(timezone.utc).isoformat()
    rec = {
        "id":                  str(uuid.uuid4()),
        "task":                task_desc,
        "category":            category,
        "failure_type":        failure_type,
        "query_text":          query_text,
        "corrected_tool_call": corrected_tool_call,
        "observation":         "PASS",
        "final_answer":        final_answer,
        "source":              source,
        "verified":            True,
        "created_at":          now,
        "content_hash":        "",   # filled below
        "sensitivity":         "internal",
    }
    rec["content_hash"] = _content_hash(rec)
    return rec


def _extract_final_answer(transcript: str) -> str:
    """Extract the FINAL_ANSWER: line from a transcript, if present."""
    for line in transcript.splitlines():
        stripped = line.strip()
        if stripped.startswith("FINAL_ANSWER:"):
            return stripped
    return ""


# ── CSV scanning ──────────────────────────────────────────────────────────────

def _row_qualifies(row: dict, min_quality: str) -> tuple[bool, str]:
    """Check whether a CSV row meets the minimum quality bar.

    Returns (qualifies: bool, reason: str).
    """
    if row.get("passed") != "True":
        return False, "passed!=True"
    if row.get("passed_via_tool") != "True":
        return False, "passed_via_tool!=True"
    if row.get("used_fallback_extraction") == "True":
        return False, "used_fallback_extraction==True"
    if row.get("scoring_mode") and "verified" not in row.get("scoring_mode", ""):
        if min_quality == "verified_tool":
            return False, f"scoring_mode={row.get('scoring_mode')!r} not verified"

    transcript = row.get("full_transcript", "")
    if not transcript:
        return False, "empty full_transcript"

    return True, ""


def scan_csv(
    csv_path: Path,
    task_map: dict,
    seen_hashes: set,
    min_quality: str,
    verbose: bool = False,
) -> list[dict]:
    """Scan a single CSV file and return candidate memory records."""
    records: list[dict] = []
    try:
        source = str(csv_path.relative_to(ROOT))
    except ValueError:
        source = str(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qualifies, reason = _row_qualifies(row, min_quality)
            if not qualifies:
                if verbose:
                    print(f"    SKIP {row.get('id')}: {reason}")
                continue

            transcript = row.get("full_transcript", "")
            ctc = _extract_passing_tool_call(transcript)
            if ctc is None:
                if verbose:
                    print(f"    SKIP {row.get('id')}: no TOOL_CALL before OBSERVATION: PASS")
                continue

            ctc = _clean_tool_call(ctc)
            ban_reasons = _tool_call_is_clean(ctc)
            if ban_reasons:
                if verbose:
                    print(f"    SKIP {row.get('id')}: {'; '.join(ban_reasons)}")
                continue

            rec = _build_record(row, ctc, task_map, source)
            if rec is None:
                continue

            errs = validate_record(rec)
            if errs:
                if verbose:
                    print(f"    SKIP {row.get('id')}: validation failed: {errs}")
                continue

            h = rec["content_hash"]
            if h in seen_hashes:
                if verbose:
                    print(f"    DEDUP {row.get('id')}: duplicate content_hash")
                continue

            seen_hashes.add(h)
            records.append(rec)
            if verbose:
                print(f"    ADD  {row.get('id')} [{rec['category']}]")

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def extract(
    outputs_dir: Path,
    out_path: Path,
    min_quality: str = "verified_tool",
    max_records: int = 500,
    verbose: bool = False,
) -> int:
    """Run the full extraction pipeline.  Returns number of records written."""
    task_map = _load_task_map()
    seen_hashes: set[str] = set()
    candidates: list[dict] = []

    # Collect CSV files: single.csv and best_of_3.csv.
    # failure_analysis.csv uses a different schema (no full_transcript) — skip it.
    patterns = [
        str(outputs_dir / "**" / "single.csv"),
        str(outputs_dir / "**" / "best_of_3.csv"),
    ]
    csv_files: list[Path] = []
    for pattern in patterns:
        csv_files.extend(sorted(Path(p) for p in glob.glob(pattern, recursive=True)))

    if not csv_files:
        print(f"[extract] No CSV files found under {outputs_dir}")
        return 0

    print(f"[extract] Scanning {len(csv_files)} CSV file(s) ...")

    # Single pass: collect all qualifying records without dedup
    for csv_path in csv_files:
        if verbose:
            rel = str(csv_path.relative_to(ROOT)) if csv_path.is_absolute() else str(csv_path)
            print(f"  {rel}")
        # Use a temporary hash set so scan_csv deduplicates within a single file,
        # but we deduplicate across files below.
        file_hashes: set[str] = set(seen_hashes)
        recs = scan_csv(csv_path, task_map, file_hashes, min_quality, verbose)
        for rec in recs:
            h = rec["content_hash"]
            if h not in seen_hashes:
                seen_hashes.add(h)
                candidates.append(rec)

    if not candidates:
        print("[extract] No qualifying records found.")
        return 0

    # Sort: priority tasks first, then by category, then by id
    def _sort_key(r: dict) -> tuple:
        task_id = _task_id_from_query(r)
        is_priority = 0 if task_id in PRIORITY_TASKS else 1
        return (is_priority, r.get("category", ""), task_id)

    candidates.sort(key=_sort_key)
    final = candidates[:max_records]

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    task_counts: dict[str, int] = {}
    for rec in final:
        tid = _task_id_from_query(rec)
        task_counts[tid] = task_counts.get(tid, 0) + 1
    cat_counts: dict[str, int] = {}
    for rec in final:
        c = rec.get("category", "unknown")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"[extract] Wrote {len(final)} records to {out_path}")
    print(f"[extract] By category: {sorted(cat_counts.items())}")
    print(f"[extract] By task: {sorted(task_counts.items(), key=lambda x: -x[1])[:10]}")
    return len(final)


def _task_id_from_query(rec: dict) -> str:
    """Recover the task id from query_text (first token)."""
    return rec.get("query_text", "").split()[0] if rec.get("query_text") else ""


def main():
    p = argparse.ArgumentParser(
        description="Extract verified memory records from eval output CSVs"
    )
    p.add_argument("--outputs-dir", default="outputs",
                   help="Root outputs directory to scan (default: outputs)")
    p.add_argument("--out",         default="memory/raw/extracted_memories.jsonl",
                   help="Output JSONL file")
    p.add_argument("--min-quality", default="verified_tool",
                   choices=["verified_tool"],
                   help="Minimum quality gate (only 'verified_tool' currently)")
    p.add_argument("--max-records", type=int, default=500,
                   help="Maximum number of records to write")
    p.add_argument("--verbose",     action="store_true",
                   help="Print per-row decisions")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_path    = Path(args.out)

    if not outputs_dir.exists():
        print(f"ERROR: outputs-dir does not exist: {outputs_dir}")
        sys.exit(1)

    n = extract(
        outputs_dir=outputs_dir,
        out_path=out_path,
        min_quality=args.min_quality,
        max_records=args.max_records,
        verbose=args.verbose,
    )
    sys.exit(0 if n > 0 else 1)


if __name__ == "__main__":
    main()
