"""
memory/core.py — high-level API for the AetherForge offline vector memory.

Public functions:
  build_index(raw_dir, index_dir)
      Load JSONL records from raw_dir, validate, embed, and save the index.

  retrieve(task_text, index_dir, top_k, min_score)
      Embed task_text, search the index, return top-k verified records.

  write_memory(record, index_dir)
      Validate + audit a new record and append it to the index.
      Only call this after a real OBSERVATION: PASS from the runtime.
      Disabled by default (write_back_enabled=False at module level).

  format_memory_block(records)
      Format retrieved records as a RETRIEVED_VERIFIED_MEMORY: block for
      insertion into the agent system prompt.  Guidance only — never an
      OBSERVATION.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory.validate import validate_record, content_hash
from memory.embed import embed_texts, embed_query
from memory.store import load_index, save_index, search, append_record

RAW_DIR   = Path("memory/raw")
INDEX_DIR = Path("memory/index")

# Safety gate: write-back is disabled by default.
# Only enable after confirming verified_agent scoring passes.
write_back_enabled: bool = False


# ── Build ─────────────────────────────────────────────────────────────────────

def build_index(
    raw_dir: Path = RAW_DIR,
    index_dir: Path = INDEX_DIR,
) -> int:
    """Load, validate, embed, and save all JSONL records from raw_dir.

    Returns number of records indexed.
    Raises SystemExit if any record fails validation.
    """
    records: list[dict] = []
    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {raw_dir}")

    print(f"[memory/core] Loading raw records from {raw_dir} ...")
    errors_found = 0
    for path in jsonl_files:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  FAIL {path.name}:{lineno} — JSON parse error: {e}")
                    errors_found += 1
                    continue
                errs = validate_record(rec)
                if errs:
                    print(f"  FAIL {path.name}:{lineno} id={rec.get('id')!r}")
                    for e in errs:
                        print(f"       {e}")
                    errors_found += 1
                else:
                    records.append(rec)

    if errors_found:
        raise SystemExit(
            f"ABORT: {errors_found} record(s) failed validation. Fix before building index."
        )

    if not records:
        raise SystemExit("ABORT: no valid records found.")

    print(f"[memory/core] {len(records)} valid records.")

    texts = [r["query_text"] for r in records]
    print("[memory/core] Embedding ...")
    vectors, vocab = embed_texts(texts)

    save_index(records, vectors, vocab, index_dir)
    return len(records)


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve(
    task_text: str,
    index_dir: Path = INDEX_DIR,
    top_k: int = 4,
    min_score: float = 0.05,
) -> list[dict]:
    """Return top-k verified memory records relevant to task_text.

    Fails closed if the index is missing (returns [] with a warning,
    so the agent can still run without memory).
    """
    try:
        state = load_index(index_dir)
    except FileNotFoundError as exc:
        print(f"[memory/core] Warning: {exc}")
        return []

    stored_texts = state["query_texts"]
    vocab        = state["vocab"]

    query_vec = embed_query(task_text, stored_texts, vocab)
    hits = search(state, query_vec, top_k)

    # Filter by minimum relevance score and verified flag
    results = [
        h for h in hits
        if h.get("score", 0.0) >= min_score and h.get("verified", False)
    ]
    return results


# ── Write-back ────────────────────────────────────────────────────────────────

def write_memory(
    record: dict,
    index_dir: Path = INDEX_DIR,
) -> bool:
    """Validate and append a new verified memory record.

    Must only be called after:
      - A real execute_code ran
      - OBSERVATION contains PASS
      - verified_agent scoring passed
      - corrected_tool_call passes audit

    Returns True if written, False if rejected.
    Raises RuntimeError if write_back_enabled is False.
    """
    if not write_back_enabled:
        raise RuntimeError(
            "write_memory() called but write_back_enabled is False. "
            "Enable it explicitly after verifying your scoring pipeline."
        )

    if not record.get("id"):
        record["id"] = str(uuid.uuid4())
    if not record.get("created_at"):
        record["created_at"] = datetime.now(timezone.utc).isoformat()
    if not record.get("content_hash"):
        record["content_hash"] = content_hash(record)
    if "verified" not in record:
        record["verified"] = True

    errs = validate_record(record)
    if errs:
        print(f"[memory/core] write_memory rejected (validation failed):")
        for e in errs:
            print(f"  {e}")
        return False

    try:
        state = load_index(index_dir)
    except FileNotFoundError:
        print("[memory/core] write_memory: no index found — build first")
        return False

    vec = embed_query(
        record["query_text"],
        state["query_texts"],
        state["vocab"],
    )
    append_record(record, vec, index_dir)
    return True


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_memory_block(records: list[dict]) -> str:
    """Format retrieved records as a RETRIEVED_VERIFIED_MEMORY: prompt block.

    This is guidance only.  It must not appear as OBSERVATION.
    The model must still produce a real TOOL_CALL; the runtime still executes it.
    """
    if not records:
        return ""

    lines = [
        "RETRIEVED_VERIFIED_MEMORY:",
        "The following verified examples may be relevant to this task.",
        "They are guidance only. You must still produce a real TOOL_CALL.",
        "The runtime will execute your call and verify the result.",
        "",
    ]
    for i, rec in enumerate(records, 1):
        cat    = rec.get("category", "unknown")
        fail   = rec.get("failure_type", "")
        task   = rec.get("task", "")
        ctc    = rec.get("corrected_tool_call", "")
        obs    = rec.get("observation", "PASS")
        fa     = rec.get("final_answer", "")
        score  = rec.get("score", 0.0)

        lines.append(f"[{i}] category={cat}  failure_type={fail or 'n/a'}"
                     f"  relevance={score:.2f}")
        lines.append(f"    Task: {task}")
        lines.append(f"    Verified fix:")
        lines.append(f"    {ctc}")
        lines.append(f"    [verified: {obs}]")
        if fa:
            lines.append(f"    {fa.replace('FINAL_ANSWER:', '[result:')}")
        lines.append("")

    return "\n".join(lines).rstrip()
