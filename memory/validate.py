"""
memory/validate.py — schema validation for memory records.

A record is valid only if:
  - All required fields are present
  - verified is True
  - observation contains 'PASS'
  - corrected_tool_call starts with 'TOOL_CALL: execute_code('
  - No markdown code fences (```)
  - No banned markers: VALID:, DIFFERENT_CORRECTED_TOOL_CALL, triple-quote strings
  - No fake model-generated OBSERVATION inside corrected_tool_call
"""

import json
import re

REQUIRED_FIELDS = [
    "id", "task", "category", "query_text",
    "corrected_tool_call", "observation",
    "verified", "created_at", "content_hash",
]

OPTIONAL_FIELDS = [
    "failure_type", "final_answer", "source", "sensitivity",
]

# Patterns that must never appear in supervised targets
_BANNED_MARKERS = [
    "VALID:",
    "DIFFERENT_CORRECTED_TOOL_CALL",
    '"""',          # triple-quoted strings invalid in JSON code values
]


def validate_record(record: dict) -> list[str]:
    """Return list of validation errors.  Empty list = record is valid."""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing required field: '{field}'")

    if not record.get("verified", False):
        errors.append("verified must be True")

    obs = record.get("observation", "")
    if "PASS" not in obs:
        errors.append("observation must contain 'PASS'")

    ctc = record.get("corrected_tool_call", "")
    if not ctc.strip().startswith("TOOL_CALL: execute_code("):
        errors.append(
            "corrected_tool_call must start with 'TOOL_CALL: execute_code('"
        )

    for field in ("corrected_tool_call", "final_answer"):
        val = record.get(field, "")
        if not val:
            continue
        if "```" in val:
            errors.append(f"{field}: markdown code fences (```) not allowed")
        for marker in _BANNED_MARKERS:
            if marker in val:
                errors.append(f"{field}: banned marker {marker!r}")

    # Validate that the corrected_tool_call contains valid JSON args
    ctc = record.get("corrected_tool_call", "")
    _ctc_args_errors = _validate_ctc_json(ctc)
    errors.extend(_ctc_args_errors)

    return errors


def _validate_ctc_json(ctc: str) -> list[str]:
    """Check that the execute_code args are valid JSON with a 'code' key."""
    errors: list[str] = []
    prefix = "TOOL_CALL: execute_code("
    if not ctc.strip().startswith(prefix):
        return errors
    body = ctc.strip()[len(prefix):]
    if not body.endswith(")"):
        errors.append("corrected_tool_call: unmatched parenthesis")
        return errors
    json_str = body[:-1]
    try:
        args = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"corrected_tool_call: invalid JSON in args: {e}")
        return errors
    if "code" not in args:
        errors.append("corrected_tool_call: args missing 'code' key")
    return errors


def content_hash(record: dict) -> str:
    """Compute a stable content hash from task + corrected_tool_call."""
    import hashlib
    key = (record.get("task", "") + "\n" + record.get("corrected_tool_call", ""))
    return "sha256:" + hashlib.sha256(key.encode()).hexdigest()[:20]
