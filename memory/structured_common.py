"""
memory/structured_common.py — deterministic helpers for v2.19 structured memory.

Shared by scripts/build_structured_memory_records.py (record construction) and
memory/structured_retriever.py (query-view construction + reranking) so the family
classifier, signature extractor, and cue tokenizer cannot drift between build and
retrieval. Everything here is pure-Python and deterministic — no model weights,
no randomness, no network.
"""

import json
import re

# ── Code / tool-call parsing ────────────────────────────────────────────────────

def parse_execute_code(call_str: str) -> str:
    """Extract the 'code' string from a TOOL_CALL: execute_code({...}) wrapper.

    Mirrors the brace/string-aware parser in scripts/evaluate_code_agent.py.
    Returns "" if no execute_code payload is found.
    """
    if not call_str:
        return ""
    m = re.search(r"execute_code\(", call_str)
    if not m:
        # Some records may store raw code directly.
        return call_str if "def " in call_str or "assert " in call_str else ""
    paren_start = m.end()
    depth = 0
    in_str = False
    qch = ""
    esc = False
    end_pos = -1
    for i, ch in enumerate(call_str[paren_start:], paren_start):
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == qch:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            qch = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break
    if end_pos == -1:
        return ""
    try:
        args = json.loads(call_str[paren_start:end_pos])
        return args.get("code", "") if isinstance(args, dict) else ""
    except (json.JSONDecodeError, AttributeError):
        return ""


_DEF_RE = re.compile(r"^\s*def\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)", re.M)
_CLASS_RE = re.compile(r"^\s*class\s+([a-zA-Z_]\w*)", re.M)
_IMPORT_CALL_RE = re.compile(r"from\s+\w[\w.]*\s+import\s+([a-zA-Z_]\w*)")
# A backtick-quoted callable in a task prompt, e.g. `interval_union(a, b)`.
_BACKTICK_CALL_RE = re.compile(r"`([a-zA-Z_]\w*)\s*\(([^`)]*)\)?`")


def extract_signature(code: str) -> str:
    """Return the primary 'name(args)' signature defined in code (or "" )."""
    m = _DEF_RE.search(code or "")
    if m:
        return f"{m.group(1)}({m.group(2).strip()})"
    m = _CLASS_RE.search(code or "")
    if m:
        return f"{m.group(1)} (class)"
    m = _IMPORT_CALL_RE.search(code or "")
    if m:
        return f"{m.group(1)}()"
    return ""


def primary_function_name(code: str = "", task: str = "") -> str:
    """Best-effort function name from code (preferred) or task backtick callable."""
    m = _DEF_RE.search(code or "")
    if m:
        return m.group(1)
    m = _CLASS_RE.search(code or "")
    if m:
        return m.group(1)
    m = _BACKTICK_CALL_RE.search(task or "")
    if m:
        return m.group(1)
    m = _IMPORT_CALL_RE.search(code or "")
    if m:
        return m.group(1)
    return ""


def task_signature_from_prompt(task: str) -> str:
    """Extract 'name(args)' from a task prompt's backtick callable, else "" ."""
    m = _BACKTICK_CALL_RE.search(task or "")
    if m:
        args = (m.group(2) or "").strip()
        return f"{m.group(1)}({args})"
    return ""


def extract_minimal_test(code: str) -> str:
    """Collect assert / equality-check lines plus a trailing print as a minimal test."""
    lines = []
    for raw in (code or "").splitlines():
        s = raw.strip()
        if s.startswith("assert ") or s.startswith("print(") or (" == " in s and not s.startswith("def ")):
            lines.append(s)
    return "\n".join(lines)


# ── Family classification ───────────────────────────────────────────────────────
# Ordered (family, keyword-set). First family with a keyword hit wins, so more
# specific families must precede generic ones.

_FAMILY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("rle",       ("rle", "run_length", "run-length", "run length", "delta_encode")),
    ("graph",     ("graph", "bfs", "dfs", "adjacency", "shortest_path", "island")),
    ("tree",      ("tree", "binary_tree", "subtree", "leaves", "ancestor", "bst")),
    ("interval",  ("interval", "meeting_room", "meeting room", "overlap", "merge_interval",
                   "non_overlapping", "insert_interval", "range_summary")),
    ("dict",      ("deep_", "flatten", "unflatten", "nested", "dict", "deep merge",
                   "deep get", "deep delete")),
    ("matrix",    ("matrix", "grid", "kth_smallest")),
    ("search",    ("binary_search", "search_rotated", "rotated", "find_peak", "peak_element",
                   "median", "count_smaller", "wiggle")),
    ("sort",      ("sort", "merge_sorted", "sorted")),
    ("string",    ("palindrome", "anagram", "roman", "word_count", "reverse", "substring",
                   "vowel", "uppercase", "lowercase")),
    ("cache",     ("lru", "cache", "lfu")),
    ("math",      ("factorial", "fibonacci", "prime", "gcd", "lcm", "fizzbuzz", "clamp",
                   "divide", "sum_list", "sum_to_n")),
]


def infer_family(text: str, func_name: str = "") -> str:
    """Deterministically classify a record/task into an algorithm family.

    The function name is the strongest, least-noisy family signal, so it is matched
    first; free-text prompt keywords (which often mention an unrelated technique,
    e.g. "use BFS" inside a tree task) are only a fallback.
    """
    name = (func_name or "").lower()
    for family, keys in _FAMILY_RULES:
        if any(k in name for k in keys):
            return family
    hay = f"{func_name} {text}".lower()
    for family, keys in _FAMILY_RULES:
        if any(k in hay for k in keys):
            return family
    return "misc"


# ── Cue extraction ──────────────────────────────────────────────────────────────

_STOPWORDS = frozenset("""
a an the of to in on for and or is are be that this it with as by from your you
write python function returns return given list value values verify check test
should takes take into one two each its their than then else if not no using use
""".split())

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def extract_cues(task: str, code: str = "", func_name: str = "", limit: int = 12) -> list[str]:
    """Deterministic, de-duplicated cue keywords from task + function name + identifiers."""
    cues: list[str] = []
    seen: set[str] = set()

    def _add(tok: str):
        t = tok.lower()
        if len(t) >= 3 and t not in _STOPWORDS and t not in seen:
            seen.add(t)
            cues.append(t)

    if func_name:
        for part in func_name.split("_"):
            _add(part)
        _add(func_name)
    for tok in tokenize(task):
        _add(tok)
    # A few salient identifiers from the code (def name + first-line params).
    sig = extract_signature(code)
    for tok in tokenize(sig):
        _add(tok)

    return cues[:limit]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
