"""
scripts/build_structured_memory_records.py — v2.19 structured memory transform.

Reads the protected champion index (memory/index_adapted, 99 verified records) and
emits structured repair records. The transform is deterministic and offline: it uses
fields already present in each record (task, category, failure_type, corrected_tool_call,
observation) plus the shared helpers in memory.structured_common. No model weights, no
network, no randomness.

Each output record keeps the ORIGINAL injection fields (task, corrected_tool_call,
observation, ...) so the prompt block is byte-for-byte comparable to the baseline; the
experiment varies only how records are *embedded, retrieved, and reranked*. The composed
structured text is stored in `query_text` so the existing dense-index builder embeds it.

The protected source index is never modified. Output goes to a gitignored directory.

Usage:
    python scripts/build_structured_memory_records.py \
        --source-index memory/index_adapted \
        --output memory/structured_v219/records.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from memory.structured_common import (  # noqa: E402
    parse_execute_code, extract_signature, primary_function_name,
    extract_minimal_test, infer_family, extract_cues,
)

PROTECTED = {"index", "index_adapted", "index_adapted_v2", "index_adapted_v29",
             "index_adapted_v3", "index_v28_filtered", "index_v29_repair"}


def build_structured_record(rec: dict) -> dict:
    task = rec.get("task", "") or rec.get("query_text", "")
    ctc = rec.get("corrected_tool_call", "")
    code = parse_execute_code(ctc)
    func = primary_function_name(code, task)
    family = infer_family(task, func)
    signature = extract_signature(code) or func
    failure_mode = (rec.get("failure_type") or "none").strip() or "none"
    cues = extract_cues(task, code, func)
    minimal_test = extract_minimal_test(code)
    tool_trace = f"execute_code -> OBSERVATION: {rec.get('observation', 'PASS') or 'PASS'}"
    why = (
        f"Verified {family} example defining `{signature or func or 'a function'}`"
        + (f"; recovers from failure mode '{failure_mode}'" if failure_mode != "none" else "")
        + (f"; relevant when the task needs {', '.join(cues[:5])}." if cues else ".")
    )
    embed_text = (
        f"family: {family}\n"
        f"signature: {signature}\n"
        f"failure_mode: {failure_mode}\n"
        f"cues: {', '.join(cues)}\n"
        f"task: {task}"
    )
    return {
        # identity
        "id": rec.get("id", ""),
        "record_id": rec.get("id", ""),
        "source_record": rec.get("source", "") or rec.get("id", ""),
        # structured fields
        "task_family": family,
        "task_signature": signature,
        "failure_mode": failure_mode,
        "retrieval_cues": cues,
        "verified_solution": code,
        "minimal_test": minimal_test,
        "tool_trace_summary": tool_trace,
        "why_this_memory_helps": why,
        # text used for embedding (the dense builder embeds query_text)
        "query_text": embed_text,
        "original_query_text": rec.get("query_text", ""),
        # ORIGINAL injection fields (kept verbatim for prompt parity)
        "task": task,
        "category": rec.get("category", ""),
        "failure_type": rec.get("failure_type", ""),
        "corrected_tool_call": ctc,
        "observation": rec.get("observation", "PASS"),
        "final_answer": rec.get("final_answer", ""),
        "verified": True,
    }


def main():
    ap = argparse.ArgumentParser(description="Build v2.19 structured memory records")
    ap.add_argument("--source-index", default="memory/index_adapted",
                    help="Protected source index directory (read-only)")
    ap.add_argument("--output", default="memory/structured_v219/records.jsonl",
                    help="Output JSONL path (gitignored)")
    ap.add_argument("--schema-doc", default="results/v219_structured_memory/structured_record_schema.md",
                    help="Small committed schema/sample evidence file")
    args = ap.parse_args()

    out_path = Path(args.output)
    if out_path.parent.name in PROTECTED or out_path.name in PROTECTED:
        print(f"ERROR: refusing to write into a protected index location: {out_path}", file=sys.stderr)
        sys.exit(1)

    from memory.store import load_index
    state = load_index(Path(args.source_index))
    records = [r for r in state.get("records", []) if r.get("verified")]
    if not records:
        print(f"ERROR: no verified records in {args.source_index}", file=sys.stderr)
        sys.exit(1)
    print(f"[structured] Loaded {len(records)} verified records from {args.source_index}")

    structured = [build_structured_record(r) for r in records]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in structured:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[structured] Wrote {len(structured)} structured records -> {out_path}")

    # Family distribution (for the run log / sanity).
    from collections import Counter
    fam = Counter(r["task_family"] for r in structured)
    fmode = Counter(r["failure_mode"] for r in structured)
    print(f"[structured] families: {dict(fam)}")
    print(f"[structured] failure_modes: {dict(fmode)}")

    # Small committed evidence: schema + one redacted sample.
    sample = structured[0]
    doc_path = Path(args.schema_doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["record_id", "task_family", "task_signature", "failure_mode",
              "retrieval_cues", "verified_solution", "minimal_test",
              "tool_trace_summary", "why_this_memory_helps", "source_record"]
    lines = [
        "# v2.19 Structured Memory Record Schema",
        "",
        "Deterministically derived from the protected champion index "
        "(`memory/index_adapted`, 99 verified records). No model weights, no network.",
        "Original injection fields (`task`, `corrected_tool_call`, `observation`, ...) are",
        "preserved verbatim so the prompt block matches the baseline; only embedding,",
        "retrieval, and reranking change.",
        "",
        "## Fields",
        "",
        "| Field | Meaning |",
        "|---|---|",
        "| record_id | Source record id |",
        "| task_family | Algorithm family (deterministic classifier) |",
        "| task_signature | Primary `name(args)` from the verified solution |",
        "| failure_mode | Recovered failure type (from the source record) |",
        "| retrieval_cues | Deterministic keyword cues (task + identifiers) |",
        "| verified_solution | Parsed execute_code payload |",
        "| minimal_test | Assert/print lines extracted from the solution |",
        "| tool_trace_summary | Compact tool-trace summary |",
        "| why_this_memory_helps | Templated relevance rationale |",
        "| source_record | Provenance pointer |",
        "",
        f"## Family distribution ({len(structured)} records)",
        "",
        "| Family | Count |",
        "|---|---:|",
    ]
    for k, v in sorted(fam.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Example (one record, fields abbreviated)",
        "",
        "```json",
    ]
    abbreviated = {k: (sample.get(k) if not isinstance(sample.get(k), str)
                       else (sample.get(k)[:200] + ("…" if len(sample.get(k, "")) > 200 else "")))
                   for k in fields}
    lines.append(json.dumps(abbreviated, indent=2, ensure_ascii=False))
    lines.append("```")
    doc_path.write_text("\n".join(lines) + "\n")
    print(f"[structured] Wrote schema evidence -> {doc_path}")


if __name__ == "__main__":
    main()
