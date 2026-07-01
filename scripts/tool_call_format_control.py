"""
scripts/tool_call_format_control.py — v2.34 deterministic tool-call format controller.

v2.31–v2.33 all failed primarily through `no_tool_call` collapse: the model writes `TOOL_CALL:`
followed by RAW code/asserts but never the `execute_code({...})` schema the agent parser needs, so a
genuine tool-use intent is scored as `no_tool_call`. This module is a pure, deterministic,
inference-time controller that detects tool-call format failures and, when UNAMBIGUOUS, re-wraps the
model's already-emitted code into the canonical `execute_code({"code": ...})` schema. It NEVER calls a
model and NEVER invents code — it only re-frames code the model already produced, so it cannot
fabricate an unsafe tool call.

Public API (all deterministic, side-effect free):
    detect_tool_call(text)        -> dict(has_tool_call, kind, valid_json, args, raw)
    is_no_tool_call(text)         -> bool
    has_invalid_tool_json(text)   -> bool
    extract_intended_code(text)   -> str | None
    repair_to_execute_code(text)  -> dict(status, action, code, call, reason)
"""

import json
import re

EXECUTE_CODE_RE = re.compile(r"execute_code\(\s*(\{.*\})\s*\)", re.DOTALL)
# a TOOL_CALL / CHANGED_TOOL_CALL block stops at the next UPPERCASE label, a code fence, or end
TOOL_CALL_BLOCK_RE = re.compile(
    r"(?:CHANGED_TOOL_CALL|TOOL_CALL)\s*:\s*(.*?)(?=\n[A-Z][A-Z_]{2,}\s*:|\n```|\Z)", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
CODEISH_RE = re.compile(r"(^|\n)\s*(def |class |import |from |assert |return |print\(|@)", re.MULTILINE)
REASON_NO_CODE = "no_code"
REASON_AMBIGUOUS = "ambiguous"
REASON_NOT_CODE = "not_code"


def detect_tool_call(text):
    """Detect a literal execute_code(...) call and whether its JSON args are valid (have a 'code' key)."""
    text = text or ""
    m = EXECUTE_CODE_RE.search(text)
    if not m:
        return {"has_tool_call": False, "kind": "none", "valid_json": False, "args": None, "raw": None}
    raw = m.group(1)
    try:
        obj = json.loads(raw)
        valid = isinstance(obj, dict) and "code" in obj and isinstance(obj["code"], str)
    except Exception:
        obj, valid = None, False
    return {"has_tool_call": True, "kind": "execute_code", "valid_json": valid,
            "args": obj if valid else None, "raw": raw}


def is_no_tool_call(text):
    """True when no execute_code(...) call is present at all (the v2.31–v2.33 collapse mode)."""
    return not detect_tool_call(text)["has_tool_call"]


def has_invalid_tool_json(text):
    """True when an execute_code(...) call is present but its JSON args are malformed."""
    d = detect_tool_call(text)
    return d["has_tool_call"] and not d["valid_json"]


def _looks_like_code(s):
    return bool(s and s.strip() and CODEISH_RE.search(s))


def _candidates(text):
    """Ordered, de-duplicated (kind, code) candidates the model already emitted (latest intent last).

    kind in {'invalid_json', 'tool_call', 'fence'}. Same-kind blocks are iterative refinement (latest
    wins); conflicting candidates of DIFFERENT kinds signal genuine ambiguity.
    """
    text = text or ""
    out = []
    d = detect_tool_call(text)
    if d["has_tool_call"] and not d["valid_json"] and d["raw"]:
        m = re.search(r'"code"\s*:\s*(?:"""(.*?)"""|"((?:\\.|[^"\\])*)")', d["raw"], re.DOTALL)
        if m:
            out.append(("invalid_json", m.group(1) if m.group(1) is not None else m.group(2)))
    for blk in TOOL_CALL_BLOCK_RE.findall(text):
        b = blk.strip()
        if b and "execute_code" not in b:
            out.append(("tool_call", b))
    for blk in FENCE_RE.findall(text):
        out.append(("fence", blk.strip()))
    seen, uniq = set(), []
    for kind, c in out:
        c = (c or "").strip()
        if c and c not in seen:
            seen.add(c); uniq.append((kind, c))
    return uniq


def extract_intended_code(text):
    """Return the model's latest intended code, or None. Latest candidate wins (model's final attempt)."""
    cands = _candidates(text)
    return cands[-1][1] if cands else None


def repair_to_execute_code(text):
    """Deterministically repair a tool-call format failure into a canonical execute_code call.

    Returns dict(status, action, code, call, reason):
      - status 'ok', action 'passthrough'  : already a valid execute_code call (unchanged).
      - status 'ok', action 'wrapped'       : the model's emitted code wrapped into the schema.
      - status 'rejected', reason ...        : no_code / ambiguous / not_code (no unsafe fabrication).
    """
    d = detect_tool_call(text)
    if d["has_tool_call"] and d["valid_json"]:
        code = d["args"]["code"]
        return {"status": "ok", "action": "passthrough", "code": code,
                "call": "execute_code(" + json.dumps({"code": code}) + ")", "reason": ""}

    cands = _candidates(text)
    if not cands:
        return {"status": "rejected", "action": None, "code": None, "call": None, "reason": REASON_NO_CODE}
    code_like = [(k, c) for k, c in cands if _looks_like_code(c)]
    if not code_like:
        return {"status": "rejected", "action": None, "code": None, "call": None, "reason": REASON_NOT_CODE}
    chosen_kind, chosen = code_like[-1]
    # ambiguous only when a DIFFERENT-kind candidate conflicts with the latest (not a sub/superset).
    # same-kind earlier blocks are iterative refinement and are subsumed by "latest wins".
    conflict = [c for k, c in code_like[:-1]
                if k != chosen_kind and c not in chosen and chosen not in c]
    if conflict:
        return {"status": "rejected", "action": None, "code": None, "call": None, "reason": REASON_AMBIGUOUS}
    return {"status": "ok", "action": "wrapped", "code": chosen,
            "call": "execute_code(" + json.dumps({"code": chosen}) + ")", "reason": ""}


if __name__ == "__main__":
    valid = 'execute_code({"code": "def f():\\n    return 1\\nassert f()==1\\nprint(\'PASS\')"})'
    near = "PLAN: x\nTOOL_CALL: \nassert f()==1\nprint('PASS')\n"
    for name, t in (("valid", valid), ("near_miss", near), ("empty", "no code here")):
        r = repair_to_execute_code(t)
        print(f"{name:10} no_tool_call={is_no_tool_call(t)} -> {r['status']}/{r.get('action') or r['reason']}")
