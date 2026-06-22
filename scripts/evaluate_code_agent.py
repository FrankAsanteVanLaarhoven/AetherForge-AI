"""
scripts/evaluate_code_agent.py
Evaluate AetherForge Code Agent on a fixed benchmark of coding tasks.

Metrics reported per run:
  pass_rate        — % tasks where execution output matches expected OR tests pass
  critique_rate    — % trajectories that included a CRITIQUE step
  fix_loop_rate    — % where CRITIQUE said "Fix needed" and model continued to fix
  avg_steps        — average generate/observe cycles per task
  avg_tool_calls   — average tool dispatches per task
  error_rate       — % trajectories that ended with an ERROR observation

Modes:
  single   — one trajectory per task
  best_of_n — N trajectories, pick best by execution score (default n=3)
  compare  — runs both modes and prints a side-by-side table

Usage:
    # Evaluate a fine-tuned Qwen LoRA adapter (single-pass)
    conda run -n ml-torch python scripts/evaluate_code_agent.py \\
        --hf-model outputs/qwen_code_agent/final

    # Best-of-3 evaluation
    conda run -n ml-torch python scripts/evaluate_code_agent.py \\
        --hf-model outputs/qwen_code_agent/final --mode best_of_n --n 3

    # Compare base Qwen vs fine-tuned
    conda run -n ml-torch python scripts/evaluate_code_agent.py \\
        --hf-model outputs/qwen_code_agent/final --compare-base

    # Quick dry-run (no model, prints task list)
    python scripts/evaluate_code_agent.py --list-tasks
"""

import argparse
import csv
import json
import re
import sys
import textwrap
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Benchmark task definitions
# ---------------------------------------------------------------------------
# Each task has:
#   task        — natural-language prompt for the agent
#   check       — callable(obs_list: list[str]) -> bool  (pass/fail)
#   category    — "basic" | "medium" | "hard" | "bugfix" | "edge-case"
#   critique_helpful — True if CRITIQUE is especially important for correctness

TASKS = [
    # ── basic ────────────────────────────────────────────────────────────
    {
        "id": "fizzbuzz",
        "category": "basic",
        "critique_helpful": False,
        "task": (
            "Write a Python function `fizzbuzz(n)` that returns a list of strings: "
            "'FizzBuzz' for multiples of 15, 'Fizz' for multiples of 3, 'Buzz' for "
            "multiples of 5, else the number as a string. Verify that fizzbuzz(15) "
            "returns a list of 15 elements ending in 'FizzBuzz'."
        ),
        "check": lambda obs: any("FizzBuzz" in o and len(o.split()) >= 3 for o in obs),
    },
    {
        "id": "factorial",
        "category": "basic",
        "critique_helpful": False,
        "task": (
            "Write a Python function `factorial(n)` (iterative, no recursion). "
            "Verify factorial(0)=1, factorial(5)=120, factorial(10)=3628800."
        ),
        "check": lambda obs: any("3628800" in o for o in obs),
    },
    {
        "id": "palindrome",
        "category": "basic",
        "critique_helpful": False,
        "task": (
            "Write `is_palindrome(s)` returning True if s is a palindrome. "
            "Test on: 'racecar' (True), 'hello' (False), '' (True), 'a' (True)."
        ),
        "check": lambda obs: any("True" in o and "False" in o for o in obs),
    },
    {
        "id": "sum_list",
        "category": "basic",
        "critique_helpful": False,
        "task": (
            "Write `sum_list(lst)` that returns the sum of all numbers in a list. "
            "Verify sum_list([1,2,3,4,5]) == 15 and sum_list([]) == 0."
        ),
        "check": lambda obs: any("15" in o for o in obs),
    },
    # ── medium ───────────────────────────────────────────────────────────
    {
        "id": "binary_search",
        "category": "medium",
        "critique_helpful": False,
        "task": (
            "Write `binary_search(arr, target)` that returns the index of target "
            "in sorted arr, or -1 if not found. Verify it finds 7 at index 3 in "
            "[1, 3, 5, 7, 9, 11], and returns -1 for 4."
        ),
        "check": lambda obs: any("3" in o for o in obs),
    },
    {
        "id": "flatten",
        "category": "medium",
        "critique_helpful": True,
        "task": (
            "Write `flatten(lst)` that takes a nested list (arbitrary depth) and "
            "returns a flat list. Verify flatten([[1,[2,3]],[4,[5,[6]]]]) == [1,2,3,4,5,6]."
        ),
        "check": lambda obs: any("[1, 2, 3, 4, 5, 6]" in o or "1, 2, 3, 4, 5, 6" in o for o in obs),
    },
    {
        "id": "word_count",
        "category": "medium",
        "critique_helpful": True,
        "task": (
            "Write `word_count(text)` that returns a dict mapping each word (lowercase) "
            "to its count. Verify on 'the cat sat on the mat' — 'the' should map to 2."
        ),
        "check": lambda obs: any("2" in o and ("the" in o.lower() or "cat" in o.lower()) for o in obs),
    },
    {
        "id": "merge_sorted",
        "category": "medium",
        "critique_helpful": False,
        "task": (
            "Write `merge_sorted(a, b)` that merges two sorted lists into one sorted list "
            "without using sort(). Verify merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]."
        ),
        "check": lambda obs: any("[1, 2, 3, 4, 5, 6]" in o or "1, 2, 3, 4, 5, 6" in o for o in obs),
    },
    # ── hard ─────────────────────────────────────────────────────────────
    {
        "id": "lru_cache",
        "category": "hard",
        "critique_helpful": True,
        "task": (
            "Implement an LRU cache class `LRUCache(capacity)` with `get(key)` "
            "(returns -1 if not found) and `put(key, value)`. Demonstrate: capacity=2, "
            "put(1,1), put(2,2), get(1)=1, put(3,3) evicts key 2, get(2)=-1."
        ),
        "check": lambda obs: any("-1" in o for o in obs),
    },
    {
        "id": "graph_bfs",
        "category": "hard",
        "critique_helpful": True,
        "task": (
            "Implement BFS on a graph represented as an adjacency dict. "
            "Write `bfs(graph, start)` returning nodes in BFS order. "
            "Verify on graph={'A':['B','C'],'B':['D'],'C':['D'],'D':[]}, start='A'. "
            "Expected order starts with A, then B and C."
        ),
        "check": lambda obs: any("A" in o and "B" in o and "D" in o for o in obs),
    },
    {
        "id": "roman_numerals",
        "category": "hard",
        "critique_helpful": True,
        "task": (
            "Write `to_roman(n)` converting an integer (1–3999) to a Roman numeral string. "
            "Verify: to_roman(1)='I', to_roman(4)='IV', to_roman(9)='IX', "
            "to_roman(58)='LVIII', to_roman(1994)='MCMXCIV'."
        ),
        "check": lambda obs: any("MCMXCIV" in o or "LVIII" in o for o in obs),
    },
    # ── bugfix ───────────────────────────────────────────────────────────
    {
        "id": "bugfix_off_by_one",
        "category": "bugfix",
        "critique_helpful": True,
        "task": textwrap.dedent("""\
            Fix the bug in this code and verify it produces the correct sum 1+2+...+10 = 55:
            ```python
            def sum_to_n(n):
                total = 0
                for i in range(n):   # BUG: should include n
                    total += i
                return total
            print(sum_to_n(10))
            ```"""),
        "check": lambda obs: any("55" in o for o in obs),
    },
    {
        "id": "bugfix_index_error",
        "category": "bugfix",
        "critique_helpful": True,
        "task": textwrap.dedent("""\
            Fix the bug in this code and verify it prints 'last':
            ```python
            def last_element(lst):
                return lst[len(lst)]   # BUG: off-by-one in index
            print(last_element(['first', 'middle', 'last']))
            ```"""),
        "check": lambda obs: any("last" in o.lower() and "error" not in o.lower() for o in obs),
    },
    # ── edge-case heavy (CRITIQUE especially valuable here) ───────────────
    {
        "id": "safe_divide",
        "category": "edge-case",
        "critique_helpful": True,
        "task": (
            "Write `safe_divide(a, b)` returning a/b or 0 if b is zero. "
            "Verify: safe_divide(10,2)=5.0, safe_divide(7,0)=0, "
            "safe_divide(0,0)=0, safe_divide(-6,2)=-3.0."
        ),
        "check": lambda obs: any("5.0" in o or "5" in o for o in obs),
    },
    {
        "id": "clamp",
        "category": "edge-case",
        "critique_helpful": True,
        "task": (
            "Write `clamp(x, lo, hi)` constraining x to [lo, hi]. "
            "Verify on: (-5,0,10)=0, (5,0,10)=5, (15,0,10)=10, (0,0,10)=0, (10,0,10)=10."
        ),
        "check": lambda obs: any(
            all(v in o for v in ["0", "5", "10"]) for o in obs
        ),
    },
    {
        "id": "unique_sorted",
        "category": "edge-case",
        "critique_helpful": True,
        "task": (
            "Write `unique_sorted(lst)` returning unique elements in sorted order. "
            "Verify: unique_sorted([3,1,2,1,3])=[1,2,3], unique_sorted([])=[], "
            "unique_sorted([1])=[1]."
        ),
        "check": lambda obs: any("[1, 2, 3]" in o or "1, 2, 3" in o for o in obs),
    },
]

# IDs of the built-in benchmark tasks (used for overlap checks).
BUILTIN_TASK_IDS = frozenset(t["id"] for t in TASKS)


# ---------------------------------------------------------------------------
# External task file loader
# ---------------------------------------------------------------------------

def _default_check(obs: list) -> bool:
    """Default pass-check for JSONL-loaded tasks: any PASS observation counts."""
    return any("PASS" in o and "Traceback" not in o and "Error" not in o for o in obs)


def load_tasks_from_jsonl(path: Path) -> list[dict]:
    """Load tasks from a JSONL file.

    Each record must have ``id``, ``category``, and either ``prompt`` or ``task``.
    Optional fields: ``expected_properties``, ``critique_helpful``.
    A default pass-check (OBSERVATION: PASS) is attached automatically.
    """
    tasks = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {e}") from e
            if "id" not in rec:
                raise ValueError(f"{path}:{lineno}: record missing 'id'")
            task_text = rec.get("prompt") or rec.get("task") or ""
            if not task_text:
                raise ValueError(f"{path}:{lineno}: record {rec['id']!r} has no 'prompt'/'task'")
            tasks.append({
                "id":                 rec["id"],
                "category":           rec.get("category", "external"),
                "critique_helpful":   rec.get("critique_helpful", False),
                "task":               task_text,
                "expected_properties": rec.get("expected_properties", []),
                "check":              _default_check,
            })
    return tasks


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_PROMPT_MARKERS = ("<|im_start|>system", "<|im_start|>user", "Rules:", "Protocol:")
_AGENT_MARKERS  = ("TOOL_CALL:", "OBSERVATION:", "FIXED:", "CRITIQUE:", "FINAL_ANSWER:")

_SCORING_MODES = ("answer_quality", "verified_agent")

# Tool names the agent is allowed to call.  Any other name → direct task-tool call.
_ALLOWED_TOOL_NAMES = frozenset({
    "execute_code", "run_script", "run_tests", "read_file", "write_file",
    "install_package", "check_syntax",
    "git_commit", "git_push", "git_log", "git_diff", "git_status",
})

_TOOL_CALL_NAME_RE = re.compile(r"(?m)^TOOL_CALL:\s*(\w+)\s*\(")
_EXC_TYPE_RE       = re.compile(r"(\w+Error|\w+Exception):")


def _extract_observations(tool_calls: list[dict]) -> list[str]:
    return [c["obs"] for c in tool_calls]


def _get_assistant_text(result) -> str:
    """Return only the assistant-generated portion of the trajectory.

    Prefers result.assistant_text (set by agent_loop.run_agent since the
    fix that records initial_len).  Falls back to result.answer only when
    it does not contain system-prompt markers.
    """
    at = getattr(result, "assistant_text", "")
    if at:
        return at
    # Fallback: result.answer (text after FINAL_ANSWER:) is usually safe.
    ans = str(getattr(result, "answer", "") or "")
    for m in _PROMPT_MARKERS:
        if m in ans:
            return ""   # polluted — return nothing rather than poisoning extraction
    return ans


def _assert_no_prompt_pollution(text: str, label: str = "") -> None:
    """Raise loudly if text still contains system-prompt fragments."""
    for m in _PROMPT_MARKERS:
        if m in text:
            tag = f" (task={label})" if label else ""
            raise ValueError(
                f"[EVALUATOR BUG]{tag} generated_text contains prompt marker {m!r}. "
                "Check AgentResult.assistant_text / _get_assistant_text."
            )


def _strip_non_code_blocks(text: str) -> str:
    """Remove OBSERVATION and runtime-injected blocks from assistant text.

    Keeps model-written narrative and code blocks; removes:
      - OBSERVATION: <runtime output>
      - [SYSTEM HINT: ...]
      - [DEBUG] lines
      - RUNTIME-INJECTED lines
    """
    # Remove OBSERVATION: blocks (everything until next agent marker or EOF)
    text = re.sub(
        r"(?m)^OBSERVATION:.*?(?=^(?:TOOL_CALL|FIXED|CRITIQUE|FINAL_ANSWER|OBSERVATION):|\Z)",
        "",
        text,
        flags=re.S,
    )
    # Remove runtime-injected markers
    text = re.sub(r"\[SYSTEM HINT:[^\]]*\]", "", text)
    text = re.sub(r"\[DEBUG\][^\n]*", "", text)
    text = re.sub(r"^RUNTIME-INJECTED[^\n]*", "", text, flags=re.M)
    return text.strip()


def _extract_python_code(text: str) -> str:
    """Extract the best executable Python from clean assistant output.

    Priority:
    1. Last fenced ```python block after the last FIXED: or FINAL_ANSWER: marker
    2. Last fenced ```python block anywhere
    3. Last generic ``` block after FIXED: / FINAL_ANSWER:
    4. Last generic ``` block anywhere
    5. RUN_CODE: block
    6. Line-by-line heuristic (skipping OBSERVATION / TOOL_CALL lines)
    """
    if not text:
        return ""

    # Work on cleaned text without OBSERVATION / runtime blocks
    clean = _strip_non_code_blocks(text)

    # Build a "preferred segment": last FIXED: or FINAL_ANSWER: section
    pref = ""
    for marker in ("FINAL_ANSWER:", "FIXED:"):
        if marker in clean:
            pref = clean.split(marker)[-1]
            break

    def _last_python(src: str) -> str:
        blocks = re.findall(r"```python\s*(.*?)```", src, flags=re.S | re.I)
        return blocks[-1].strip() if blocks else ""

    def _last_generic(src: str) -> str:
        blocks = re.findall(r"```\s*(.*?)```", src, flags=re.S)
        return blocks[-1].strip() if blocks else ""

    # 1. Preferred segment — python block
    if pref:
        c = _last_python(pref)
        if c:
            return c

    # 2. Full clean text — python block (last, not first — prefers corrections)
    c = _last_python(clean)
    if c:
        return c

    # 3. Preferred segment — generic block
    if pref:
        c = _last_generic(pref)
        if c:
            return c

    # 4. Full clean text — generic block
    c = _last_generic(clean)
    if c:
        return c

    # 5. RUN_CODE: block
    m = re.search(
        r"RUN_CODE:\s*(.*?)(?:\n(?:OBSERVATION|CRITIQUE|FINAL_ANSWER|TOOL_CALL):|\Z)",
        clean, flags=re.S,
    )
    if m:
        return m.group(1).strip()

    # 6. Line-by-line heuristic (skip TOOL_CALL / OBSERVATION / agent-marker lines)
    skip_prefixes = ("TOOL_CALL:", "OBSERVATION:", "FIXED:", "CRITIQUE:",
                     "FINAL_ANSWER:", "[SYSTEM", "[DEBUG", "RUNTIME-INJECTED")
    lines = clean.splitlines()
    keep = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in skip_prefixes):
            capture = False
            continue
        if stripped.startswith(("def ", "class ", "import ", "from ",
                                 "assert ", "print(")):
            capture = True
            keep.append(line)
        elif capture and (line.startswith((" ", "\t")) or stripped == ""):
            keep.append(line)
        elif capture and stripped.startswith("#"):
            keep.append(line)
        elif capture:
            capture = False
    return "\n".join(keep).strip()


def _verification_code(task_id: str) -> str:
    """Task-specific tests appended after extracted code."""
    tests = {
        "fizzbuzz": """
assert fizzbuzz(15)[-1] == "FizzBuzz"
assert len(fizzbuzz(15)) == 15
print("PASS fizzbuzz [1, 2, Fizz, 4, Buzz, ..., FizzBuzz]")
""",
        "factorial": """
assert factorial(0) == 1
assert factorial(5) == 120
assert factorial(10) == 3628800
print("PASS factorial 3628800")
""",
        "palindrome": """
assert is_palindrome("racecar") is True
assert is_palindrome("hello") is False
assert is_palindrome("") is True
assert is_palindrome("a") is True
print("PASS palindrome True False")
""",
        "sum_list": """
assert sum_list([1,2,3,4,5]) == 15
assert sum_list([]) == 0
print("PASS sum_list 15 0")
""",
        "binary_search": """
arr = [1,3,5,7,9,11]
assert binary_search(arr, 7) == 3
assert binary_search(arr, 4) == -1
print("PASS binary_search 3 -1")
""",
        "flatten": """
assert flatten([[1,[2,3]],[4,[5,[6]]]]) == [1,2,3,4,5,6]
print("PASS flatten [1, 2, 3, 4, 5, 6]")
""",
        "word_count": """
d = word_count("the cat sat on the mat")
assert d["the"] == 2
print("PASS word_count the 2")
""",
        "merge_sorted": """
assert merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
print("PASS merge_sorted [1, 2, 3, 4, 5, 6]")
""",
        "lru_cache": """
cache = LRUCache(2)
cache.put(1,1)
cache.put(2,2)
assert cache.get(1) == 1
cache.put(3,3)
assert cache.get(2) == -1
print("PASS lru_cache 1 -1")
""",
        "graph_bfs": """
graph={'A':['B','C'],'B':['D'],'C':['D'],'D':[]}
order = bfs(graph, 'A')
assert order[0] == 'A'
assert set(order[:3]) == {'A','B','C'}
print("PASS bfs A B C")
""",
        "roman_numerals": """
assert to_roman(1) == 'I'
assert to_roman(4) == 'IV'
assert to_roman(9) == 'IX'
assert to_roman(58) == 'LVIII'
assert to_roman(1994) == 'MCMXCIV'
print("PASS roman MCMXCIV LVIII")
""",
        "bugfix_off_by_one": """
assert sum_to_n(10) == 55
print("PASS sum_to_n 55")
""",
        "bugfix_index_error": """
assert last_element(['first','middle','last']) == 'last'
print("PASS last_element last")
""",
        "safe_divide": """
assert safe_divide(10,2) == 5.0
assert safe_divide(7,0) == 0
assert safe_divide(0,0) == 0
assert safe_divide(-6,2) == -3.0
print("PASS safe_divide 5.0 0 -3.0")
""",
        "clamp": """
assert clamp(-5,0,10) == 0
assert clamp(5,0,10) == 5
assert clamp(15,0,10) == 10
assert clamp(0,0,10) == 0
assert clamp(10,0,10) == 10
print("PASS clamp 0 5 10")
""",
        "unique_sorted": """
assert unique_sorted([3,1,2,1,3]) == [1,2,3]
assert unique_sorted([]) == []
assert unique_sorted([1]) == [1]
print("PASS unique_sorted [1, 2, 3]")
""",
    }
    return tests.get(task_id, "")


def _first_exception_type(obs_list: list[str]) -> str:
    """Return the first exception class name found in any ERROR observation."""
    for o in obs_list:
        if "ERROR" in o:
            m = _EXC_TYPE_RE.search(o)
            if m:
                return m.group(1)
    return ""


def _has_direct_task_tool_call(text: str) -> bool:
    """True if assistant_text contains a TOOL_CALL: with a non-allowed tool name."""
    for m in _TOOL_CALL_NAME_RE.finditer(text):
        if m.group(1) not in _ALLOWED_TOOL_NAMES:
            return True
    return False


def _repeated_identical_tool_call_count(text: str) -> int:
    """Count how many TOOL_CALL: lines are duplicates of a previously seen call."""
    from collections import Counter
    calls = [m.group(0).strip() for m in _TOOL_CALL_NAME_RE.finditer(text)]
    if len(calls) <= 1:
        return 0
    c = Counter(calls)
    return sum(v - 1 for v in c.values() if v > 1)


def _execute_extracted_code(code: str, task: dict) -> tuple[list[str], str]:
    """Run extracted code plus task tests in a subprocess."""
    import subprocess
    import tempfile
    import textwrap
    from pathlib import Path

    if not code.strip():
        return [], "no_extracted_code"

    verify = _verification_code(task["id"])
    if not verify:
        return [], "no_verification_for_task"

    program = code + "\n\n# --- evaluator verification ---\n" + textwrap.dedent(verify)

    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "candidate.py"
        fp.write_text(program)
        try:
            cp = subprocess.run(
                ["python", str(fp)],
                text=True,
                capture_output=True,
                timeout=8,
            )
        except subprocess.TimeoutExpired:
            return ["ERROR: timeout"], "timeout"

    out = (cp.stdout or "").strip()
    err = (cp.stderr or "").strip()
    obs = []
    if out:
        obs.append(out)
    if err:
        obs.append("ERROR: " + err)
    if cp.returncode != 0:
        return obs, "execution_failed"
    return obs, ""



def _parse_execute_code_call(call_str: str) -> Optional[str]:
    """Extract the 'code' string from a TOOL_CALL: execute_code({...}) string."""
    m = re.search(r"execute_code\(", call_str)
    if not m:
        return None
    paren_start = m.end()
    depth = 0; in_str = False; qch = ""; esc = False; end_pos = -1
    for i, ch in enumerate(call_str[paren_start:], paren_start):
        if esc: esc = False; continue
        if in_str:
            if ch == "\\": esc = True
            elif ch == qch: in_str = False
            continue
        if ch in ('"', "'"): in_str = True; qch = ch
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end_pos = i + 1; break
    if end_pos == -1:
        return None
    try:
        args = json.loads(call_str[paren_start:end_pos])
        return args.get("code", "")
    except (json.JSONDecodeError, AttributeError):
        return None


def _has_critique(result) -> bool:
    return bool(result.critique.strip())


def _critique_found_fix(result) -> bool:
    return "fix needed" in result.critique.lower()


def _critique_ok(result) -> bool:
    return "→ solution ok" in result.critique.lower()


def _score_task(result, task: dict, scoring_mode: str = "answer_quality") -> dict:
    # ── Extract clean assistant text (no system/user prompt) ──────────────
    assistant_text = _get_assistant_text(result)
    final_answer   = str(getattr(result, "answer", "") or "")
    _assert_no_prompt_pollution(assistant_text, task["id"])

    # ── Code extraction from clean text only ──────────────────────────────
    extracted = _extract_python_code(assistant_text)

    # ── Gather real tool observations ─────────────────────────────────────
    obs = _extract_observations(result.tool_calls)

    unknown_tool_count = sum(
        1 for c in result.tool_calls if c["obs"].startswith("ERROR: unknown tool")
    )
    invalid_json_count = sum(
        1 for c in result.tool_calls if c["obs"].startswith("ERROR: invalid JSON")
    )
    first_tool_call = result.tool_calls[0]["call"] if result.tool_calls else ""

    # ── 1. Real-tool scoring ───────────────────────────────────────────────
    passed_via_tool = task["check"](obs) if obs else False
    if not passed_via_tool and obs:
        passed_via_tool = any(
            "PASS" in o and "Traceback" not in o and "Error" not in o
            for o in obs
        )

    # ── 2. Diagnostic fields needed before the scoring branch ─────────────
    error_obs = [c["obs"] for c in result.tool_calls if c["obs"].startswith("ERROR")]
    n_errors  = len(error_obs)
    # "Solution OK" after an execution error is a false-positive CRITIQUE
    has_solution_ok_after_error = (
        bool(error_obs) and not passed_via_tool and _critique_ok(result)
    )

    # ── 3. Preliminary failure_reason ─────────────────────────────────────
    failure_reason = ""
    if not result.tool_calls:
        failure_reason = "no_tool_call"
    elif unknown_tool_count and unknown_tool_count == len(result.tool_calls):
        failure_reason = "unknown_tool"
    elif invalid_json_count and invalid_json_count == len(result.tool_calls):
        failure_reason = "invalid_tool_json"

    # ── 4. Fallback computation (answer_quality only) ──────────────────────
    passed_via_fallback      = False
    used_fallback_extraction = False
    executed_code            = False  # fallback extraction ran (answer_quality only)
    used_execute_code        = any(  # any real execute_code tool call was made
        "execute_code" in tc.get("call", "") for tc in result.tool_calls
    )
    _fb_exec_obs: list[str]  = []
    _fb_reason               = ""

    if not passed_via_tool and scoring_mode == "answer_quality" and extracted.strip():
        _fb_exec_obs, _fb_reason = _execute_extracted_code(extracted, task)
        if _fb_exec_obs:
            executed_code  = True
            combined       = obs + _fb_exec_obs
            passed_via_fallback = (
                task["check"](combined)
                or any("PASS" in o and "Traceback" not in o for o in _fb_exec_obs)
            )
            used_fallback_extraction = True

    # ── 5. Apply scoring mode ──────────────────────────────────────────────
    if scoring_mode == "verified_agent":
        passed = passed_via_tool
        if not passed:
            if has_solution_ok_after_error:
                failure_reason = failure_reason or "solution_ok_after_error"
            elif result.tool_calls:
                failure_reason = failure_reason or "tool_call_no_pass"
            # else: "no_tool_call" already set in step 3
    else:  # answer_quality
        passed = passed_via_tool or passed_via_fallback
        if not passed:
            if used_fallback_extraction and _fb_exec_obs:
                failure_reason = (
                    "assertion_failed"
                    if any("AssertionError" in o for o in _fb_exec_obs)
                    else (_fb_reason or "execution_failed")
                )
            elif not failure_reason:
                failure_reason = _fb_reason or "no_extracted_code"

    # ── 6. Invariant ──────────────────────────────────────────────────────
    if passed:
        failure_reason = ""

    # ── 7. Remaining diagnostic fields ────────────────────────────────────
    first_exc = _first_exception_type(error_obs)
    has_indentation_error          = any("IndentationError" in o for o in error_obs)
    has_invalid_json               = invalid_json_count > 0
    has_unknown_tool               = unknown_tool_count > 0
    has_direct_task_tool_call      = _has_direct_task_tool_call(assistant_text)
    repeated_identical_tool_call_count = _repeated_identical_tool_call_count(assistant_text)

    # ── 8. Inference PASS-discipline failure analysis ─────────────────────
    no_pass_print_count    = 0  # execute_code ran, obs is not PASS/ERROR, no print('PASS') in code
    no_assert_count        = 0  # execute_code call whose code lacks assert
    no_output_count        = 0  # execute_code obs is "(no output)" or empty non-error
    invalid_json_retry_count = 0  # invalid JSON errors that are NOT the first tool call

    exec_calls = [tc for tc in result.tool_calls if "execute_code" in tc.get("call", "")]

    for idx, tc in enumerate(result.tool_calls):
        call_s = tc.get("call", "")
        obs_s  = tc.get("obs", "")
        if "execute_code" not in call_s:
            continue
        code_s = _parse_execute_code_call(call_s) or ""
        is_error = obs_s.startswith("ERROR")
        is_pass  = obs_s.strip().startswith("PASS")
        is_no_output = (not obs_s.strip() or obs_s.strip() == "(no output)")
        if is_no_output and not is_error:
            no_output_count += 1
        if not is_error and not is_pass and not is_no_output:
            # ran and produced output, but not PASS — check if print('PASS') was missing
            if "print('PASS')" not in code_s and 'print("PASS")' not in code_s:
                no_pass_print_count += 1
        if code_s and "assert " not in code_s:
            no_assert_count += 1
        if obs_s.startswith("ERROR: invalid JSON") and idx > 0:
            invalid_json_retry_count += 1

    # ── 9. C-lite recovery metrics ────────────────────────────────────────
    _ec_obs = [tc.get("obs", "") for tc in exec_calls]

    first_call_passed = bool(_ec_obs) and _ec_obs[0].strip().startswith("PASS")
    eventually_passed = any(o.strip().startswith("PASS") for o in _ec_obs)

    _had_error   = any(o.startswith("ERROR") for o in _ec_obs)
    _had_no_out  = any(
        (not o.strip() or o.strip() == "(no output)") and not o.startswith("ERROR")
        for o in _ec_obs
    )
    recovered_after_error     = _had_error  and eventually_passed
    recovered_after_no_output = _had_no_out and eventually_passed

    repeated_same_error_count = 0
    for _i in range(1, len(_ec_obs)):
        if _ec_obs[_i].startswith("ERROR") and _ec_obs[_i] == _ec_obs[_i - 1]:
            repeated_same_error_count += 1

    _had_syntax = any("SyntaxError" in o or "IndentationError" in o for o in _ec_obs)
    _had_indent = any("IndentationError" in o for o in _ec_obs)
    pass_after_syntax_fix = _had_syntax and eventually_passed
    pass_after_indent_fix = _had_indent and eventually_passed

    _first_pass_idx = next(
        (i for i, o in enumerate(_ec_obs) if o.strip().startswith("PASS")), None
    )
    unnecessary_retry_after_pass = (
        _first_pass_idx is not None and _first_pass_idx < len(_ec_obs) - 1
    )

    return {
        "id":                 task["id"],
        "category":           task["category"],
        "passed":             passed,
        "has_critique":       _has_critique(result),
        "critique_ok":        _critique_ok(result),
        "fix_loop":           _critique_found_fix(result) and passed,
        "steps":              result.steps,
        "tool_calls":         len(result.tool_calls),
        "executed_code":      executed_code,
        "n_errors":           n_errors,
        "first_tool_call":    first_tool_call[:200],
        "unknown_tool_count": unknown_tool_count,
        "invalid_json_count": invalid_json_count,
        "full_transcript":    assistant_text[:6000],
        "assistant_text":     assistant_text[:4000],
        "generated_text":     assistant_text[:4000],
        "final_answer":       final_answer[:1000],
        "extracted_code":     extracted[:4000],
        "observations":       "\n---\n".join(obs)[:4000],
        "failure_reason":     failure_reason,
        # scoring split — both always populated
        "scoring_mode":                       scoring_mode,
        "passed_via_tool":                    passed_via_tool,
        "passed_via_fallback":                passed_via_fallback,
        "used_fallback_extraction":           used_fallback_extraction,
        # failure_analysis columns
        "first_exception_type":               first_exc,
        "has_indentation_error":              has_indentation_error,
        "has_invalid_json":                   has_invalid_json,
        "has_unknown_tool":                   has_unknown_tool,
        "has_direct_task_tool_call":          has_direct_task_tool_call,
        "repeated_identical_tool_call_count": repeated_identical_tool_call_count,
        "has_solution_ok_after_error":        has_solution_ok_after_error,
        "used_execute_code":                  used_execute_code,
        # PASS-discipline inference analysis
        "no_pass_print_count":               no_pass_print_count,
        "no_assert_count":                   no_assert_count,
        "no_output_count":                   no_output_count,
        "invalid_json_retry_count":          invalid_json_retry_count,
        # C-lite recovery metrics
        "first_call_passed":                 first_call_passed,
        "eventually_passed":                 eventually_passed,
        "recovered_after_error":             recovered_after_error,
        "recovered_after_no_output":         recovered_after_no_output,
        "repeated_same_error_count":         repeated_same_error_count,
        "pass_after_syntax_fix":             pass_after_syntax_fix,
        "pass_after_indent_fix":             pass_after_indent_fix,
        "unnecessary_retry_after_pass":      unnecessary_retry_after_pass,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, tasks: list[dict], mode: str, n: int,
             verbose: bool, is_hf: bool,
             scoring_mode: str = "answer_quality",
             system_prompt: str = None,
             stop_after_pass: bool = False,
             memory_state: dict = None,
             memory_top_k: int = 4,
             retriever=None) -> list[dict]:
    from scripts.agent_loop import run_agent, run_agent_best_of_n

    _mem_kwargs = {
        "memory_state": memory_state,
        "memory_top_k": memory_top_k,
        "retriever": retriever,
    }

    results = []
    for i, task in enumerate(tasks):
        print(f"\n  [{i+1:02d}/{len(tasks)}] {task['id']} ({task['category']}) …", flush=True)
        t0 = time.time()

        if mode == "best_of_n":
            result = run_agent_best_of_n(
                model, tokenizer, task["task"],
                n=n, verbose=False, is_hf=is_hf,
                system_prompt=system_prompt,
                stop_after_pass=stop_after_pass,
                **_mem_kwargs,
            )
        else:
            result = run_agent(
                model, tokenizer, task["task"],
                verbose=verbose, is_hf=is_hf,
                system_prompt=system_prompt,
                stop_after_pass=stop_after_pass,
                **_mem_kwargs,
            )

        elapsed = time.time() - t0
        row = _score_task(result, task, scoring_mode=scoring_mode)
        row["elapsed_s"] = round(elapsed, 1)
        results.append(row)

        status = "PASS" if row["passed"] else "FAIL"
        crit   = "C✓" if row["has_critique"] else "  "
        fallback_tag = " [fallback]" if row["used_fallback_extraction"] else ""
        print(f"    {status}{fallback_tag}  {crit}  steps={row['steps']}  "
              f"tools={row['tool_calls']}  errs={row['n_errors']}  "
              f"{elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _summary(results: list[dict], label: str) -> dict:
    from collections import Counter
    n      = len(results)
    passed = sum(1 for r in results if r["passed"])
    crits  = sum(1 for r in results if r["has_critique"])
    fix_ok = sum(1 for r in results if r["fix_loop"])
    exec_c = sum(1 for r in results if r.get("used_execute_code"))
    real_tools      = sum(1 for r in results if r.get("tool_calls", 0) > 0)
    tool_passed     = sum(1 for r in results if r.get("passed_via_tool"))
    fallback_passed = sum(1 for r in results if r.get("passed_via_fallback"))
    sol_ok_err      = sum(1 for r in results if r.get("has_solution_ok_after_error"))
    failures        = [r["failure_reason"] for r in results if r.get("failure_reason")]
    return {
        "label":                    label,
        "n":                        n,
        "pass_rate":                round(100 * passed        / n, 1),
        "verified_tool_pass":       round(100 * tool_passed   / n, 1),
        "fallback_pass_rate":       round(100 * fallback_passed / n, 1),
        "critique_rate":            round(100 * crits          / n, 1),
        "fix_loop_rate":            round(100 * fix_ok         / n, 1),
        "avg_steps":                round(sum(r["steps"]      for r in results) / n, 1),
        "avg_tools":                round(sum(r["tool_calls"] for r in results) / n, 1),
        "error_rate":               round(100 * sum(r["n_errors"] > 0 for r in results) / n, 1),
        "executed_code_rate":       round(100 * exec_c         / n, 1),
        "real_tool_rate":           round(100 * real_tools     / n, 1),
        "unknown_tool_total":       sum(r.get("unknown_tool_count", 0) for r in results),
        "invalid_json_total":       sum(r.get("invalid_json_count", 0) for r in results),
        "indent_error_count":       sum(1 for r in results if r.get("has_indentation_error")),
        "repeated_call_total":      sum(r.get("repeated_identical_tool_call_count", 0) for r in results),
        "sol_ok_after_error_count":   sol_ok_err,
        "no_pass_print_total":        sum(r.get("no_pass_print_count", 0) for r in results),
        "no_assert_total":            sum(r.get("no_assert_count", 0) for r in results),
        "no_output_total":            sum(r.get("no_output_count", 0) for r in results),
        "invalid_json_retry_total":   sum(r.get("invalid_json_retry_count", 0) for r in results),
        "failure_counts":             dict(Counter(failures)),
        # C-lite recovery metrics
        "first_call_passed_count":        sum(1 for r in results if r.get("first_call_passed")),
        "eventually_passed_count":        sum(1 for r in results if r.get("eventually_passed")),
        "recovered_after_error_count":    sum(1 for r in results if r.get("recovered_after_error")),
        "recovered_after_no_output_count":sum(1 for r in results if r.get("recovered_after_no_output")),
        "repeated_same_error_total":      sum(r.get("repeated_same_error_count", 0) for r in results),
        "pass_after_syntax_fix_count":    sum(1 for r in results if r.get("pass_after_syntax_fix")),
        "pass_after_indent_fix_count":    sum(1 for r in results if r.get("pass_after_indent_fix")),
        "unnecessary_retry_after_pass_count": sum(1 for r in results if r.get("unnecessary_retry_after_pass")),
    }


def _print_summary(summaries: list[dict]) -> None:
    col_w = 20
    sep   = "─" * (col_w * (len(summaries) + 1) + 2)

    keys = [
        ("pass_rate",                "Pass rate (%)"),
        ("verified_tool_pass",       "Verified-tool pass (%)"),
        ("fallback_pass_rate",       "Fallback pass (%)"),
        ("critique_rate",            "Critique rate (%)"),
        ("fix_loop_rate",            "Fix-loop rate (%)"),
        ("avg_steps",                "Avg steps"),
        ("avg_tools",                "Avg tool calls"),
        ("real_tool_rate",           "Real tool calls (%)"),
        ("executed_code_rate",       "Executed code (%)"),
        ("error_rate",               "Had errors (%)"),
        ("unknown_tool_total",       "Unknown tool calls"),
        ("invalid_json_total",       "Invalid JSON calls"),
        ("indent_error_count",       "IndentationError tasks"),
        ("repeated_call_total",      "Repeated identical calls"),
        ("sol_ok_after_error_count", "Solution-OK-after-error"),
        ("no_pass_print_total",      "No print('PASS') calls"),
        ("no_assert_total",          "No assert calls"),
        ("no_output_total",          "No-output observations"),
        ("invalid_json_retry_total", "Invalid JSON on retry"),
        # C-lite recovery metrics
        ("first_call_passed_count",         "1st call → PASS"),
        ("eventually_passed_count",         "Eventually PASS"),
        ("recovered_after_error_count",     "Recovered after ERROR"),
        ("recovered_after_no_output_count", "Recovered after no-output"),
        ("repeated_same_error_total",       "Repeated same error"),
        ("pass_after_syntax_fix_count",     "PASS after SyntaxError fix"),
        ("pass_after_indent_fix_count",     "PASS after IndentError fix"),
        ("unnecessary_retry_after_pass_count", "Retry after PASS (bad)"),
    ]

    print(f"\n{sep}")
    header = f"{'Metric':{col_w}}" + "".join(f"{s['label']:>{col_w}}" for s in summaries)
    print(header)
    print(sep)
    for key, label in keys:
        row = f"{label:{col_w}}" + "".join(f"{s.get(key, '—'):>{col_w}}" for s in summaries)
        print(row)
    print(sep)

    # Failure reason breakdown per run
    any_failures = any(s["failure_counts"] for s in summaries)
    if any_failures:
        print("\nFailure reason breakdown:")
        all_reasons = sorted({r for s in summaries for r in s["failure_counts"]})
        reason_w = max(len(r) for r in all_reasons) + 2 if all_reasons else 30
        hdr = f"  {'Reason':{reason_w}}" + "".join(f"{s['label']:>{col_w}}" for s in summaries)
        print(hdr)
        print("  " + "─" * (reason_w + col_w * len(summaries)))
        for reason in all_reasons:
            counts = [str(s["failure_counts"].get(reason, 0)) for s in summaries]
            row = f"  {reason:{reason_w}}" + "".join(f"{c:>{col_w}}" for c in counts)
            print(row)


def _print_per_task(all_results: list[tuple[str, list[dict]]]) -> None:
    """Per-task pass/fail table for all run configurations."""
    print("\nPer-task results:")
    labels = [label for label, _ in all_results]
    rows   = {r["id"]: [] for _, res in all_results for r in res}

    for label, res in all_results:
        for r in res:
            rows[r["id"]].append(("✓" if r["passed"] else "✗", r["category"]))

    id_w = max(len(k) for k in rows) + 2
    print(f"  {'Task':{id_w}} {'Category':10}" + "".join(f"  {l:8}" for l in labels))
    print(f"  {'─'*id_w} {'─'*10}" + "".join(f"  {'─'*8}" for _ in labels))
    for task_id, cols in rows.items():
        cat  = cols[0][1] if cols else ""
        vals = "".join(f"  {v:>8}" for v, _ in cols)
        print(f"  {task_id:{id_w}} {cat:10}{vals}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_csv(results: list[dict], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) + ["run_label", "timestamp"]
    ts = datetime.now().isoformat(timespec="seconds")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({**r, "run_label": label, "timestamp": ts})
    print(f"\n  Results saved → {path}")


_FA_COLS = [
    "id", "category", "mode", "passed", "passed_via_tool", "passed_via_fallback",
    "tool_calls", "n_errors", "failure_reason",
    "first_exception_type", "has_indentation_error",
    "has_invalid_json", "has_unknown_tool", "has_direct_task_tool_call",
    "repeated_identical_tool_call_count", "has_solution_ok_after_error",
    "used_fallback_extraction",
]


def _save_failure_analysis_csv(all_run_results: list[tuple[str, list[dict]]],
                                output_dir: Path) -> None:
    rows = []
    for label, results in all_run_results:
        for r in results:
            row = {"mode": label}
            for col in _FA_COLS:
                if col == "mode":
                    continue
                row[col] = r.get(col, "")
            rows.append(row)

    path = output_dir / "failure_analysis.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FA_COLS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n  Failure analysis saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate AetherForge Code Agent")
    ap.add_argument("--hf-model",     default="",  help="HF model ID or path")
    ap.add_argument("--hf-lora",      default="",  help="PEFT adapter path (optional)")
    ap.add_argument("--mode",         default="single",
                    choices=["single", "best_of_n", "compare"],
                    help="single | best_of_n | compare (runs both and shows diff)")
    ap.add_argument("--n",            type=int, default=3,
                    help="N for best_of_n (default 3)")
    ap.add_argument("--compare-base", action="store_true",
                    help="Also run base model (no LoRA) for comparison")
    ap.add_argument("--categories",   nargs="+",
                    choices=["basic", "medium", "hard", "bugfix", "edge-case"],
                    default=None,
                    help="Only run these task categories (default: all)")
    ap.add_argument("--task-ids",     nargs="+", default=None,
                    help="Run only specific task IDs")
    ap.add_argument("--output",       default="outputs/eval_code_agent",
                    help="Directory for CSV results")
    ap.add_argument("--scoring-mode",    default="answer_quality",
                    choices=list(_SCORING_MODES),
                    help="answer_quality (fallback extraction allowed) | "
                         "verified_agent (real tool obs only, no fallback)")
    ap.add_argument("--agent-contract",  choices=["standard", "strict"], default="standard",
                    help="standard: default system prompt | "
                         "strict: tool-first enforced, no markdown preamble allowed")
    ap.add_argument("--stop-after-pass", action="store_true",
                    help="Terminate the agent loop immediately when OBSERVATION: PASS "
                         "is received (normal verified termination, not repair)")
    ap.add_argument("--memory-enabled", action="store_true",
                    help="Enable offline vector memory retrieval")
    ap.add_argument("--memory-index",   default="memory/index",
                    help="Directory containing the pre-built TF-IDF memory index")
    ap.add_argument("--memory-top-k",   type=int, default=4,
                    help="Number of memory records to retrieve per task")
    ap.add_argument("--retrieval-mode", default="tfidf",
                    choices=["tfidf", "dense", "hybrid", "structured", "structured-hybrid"],
                    help="Retrieval backend: tfidf (default) | dense | hybrid | "
                         "structured (v2.19 multi-view + rerank) | structured-hybrid "
                         "(structured rerank gated by the baseline shortlist)")
    ap.add_argument("--dense-index",    default="memory/dense_index_adapted",
                    help="Directory containing the pre-built dense vector index")
    ap.add_argument("--structured-index", default="memory/dense_index_v219_structured",
                    help="Directory containing the v2.19 structured dense index")
    ap.add_argument("--dense-model",    default=None,
                    help="SentenceTransformer model name or path (overrides stored model name)")
    ap.add_argument("--rerank-top-n",   type=int, default=20,
                    help="Shortlist candidate count for hybrid reranking (default: 20)")
    ap.add_argument("--embedding-backend", default="sentence-transformer",
                    choices=["sentence-transformer"],
                    help="Embedding backend for dense/hybrid retrieval. Only "
                         "'sentence-transformer' is supported; reserved for future "
                         "backends (e.g. a 4-bit HF encoder).")
    ap.add_argument("--tasks-file",   default="",
                    help="JSONL file with tasks to evaluate instead of the built-in benchmark. "
                         "Each record needs 'id', 'category', and 'prompt'.")
    ap.add_argument("--verbose",      action="store_true")
    ap.add_argument("--list-tasks",   action="store_true",
                    help="Print task list and exit (no model needed)")
    ap.add_argument("--prompt-variant", default="default",
                    choices=["default", "direct_answer"],
                    help="System prompt variant: 'default' uses the contract, "
                         "'direct_answer' uses a more concise direct-code prompt.")
    args = ap.parse_args()

    # Resolve task source: external file or built-in benchmark
    if args.tasks_file:
        tasks_source = load_tasks_from_jsonl(Path(args.tasks_file))
        source_label = Path(args.tasks_file).name
    else:
        tasks_source = TASKS
        source_label = "built-in benchmark"

    # List-only mode — useful for reviewing tasks without a model
    if args.list_tasks:
        print(f"\n{'ID':28} {'Category':14} {'Critique?':10} Task  [{source_label}]")
        print("─" * 100)
        for t in tasks_source:
            helpful = "yes" if t["critique_helpful"] else "—"
            snippet = t["task"][:52].replace("\n", " ") + "…"
            print(f"  {t['id']:26} {t['category']:14} {helpful:10} {snippet}")
        print(f"\n{len(tasks_source)} tasks total")
        return

    if not args.hf_model:
        print("ERROR: --hf-model required (or --list-tasks to see benchmark)")
        sys.exit(1)

    # Filter tasks
    tasks = tasks_source
    if args.categories:
        tasks = [t for t in tasks if t["category"] in args.categories]
    if args.task_ids:
        tasks = [t for t in tasks if t["id"] in args.task_ids]
    if not tasks:
        print("ERROR: no tasks match the given filters")
        sys.exit(1)
    sap_tag = " +stop-after-pass" if args.stop_after_pass else ""
    print(f"\nRunning {len(tasks)} tasks  "
          f"(mode={args.mode}, n={args.n}, scoring={args.scoring_mode}, "
          f"contract={args.agent_contract}{sap_tag})")

    # Resolve system prompt: contract takes effect first, then variant overrides.
    eval_system_prompt: Optional[str] = None
    if args.agent_contract == "strict":
        from scripts.agent_loop import STRICT_SYSTEM as _STRICT_SYS
        eval_system_prompt = _STRICT_SYS
    if args.prompt_variant == "direct_answer":
        from scripts.agent_loop import DIRECT_ANSWER_SYSTEM as _DIRECT_SYS
        eval_system_prompt = _DIRECT_SYS

    # ── Memory loading ─────────────────────────────────────────────────────
    memory_state = None
    retriever = None
    if args.memory_enabled:
        mode_tag = args.retrieval_mode
        if mode_tag == "tfidf":
            try:
                from memory.store import load_index as _load_mem
                memory_state = _load_mem(Path(args.memory_index))
                n_mem = len(memory_state.get("records", []))
                print(f"[memory] Loaded {n_mem} verified records from {args.memory_index} [tfidf]")
            except FileNotFoundError as exc:
                print(f"[memory] ERROR: {exc}")
                print("[memory] Build the index: python scripts/build_vector_memory.py")
                sys.exit(1)
        elif mode_tag == "dense":
            try:
                from memory.dense_retriever import DenseRetriever
                dr = DenseRetriever(
                    index_dir=Path(args.dense_index),
                    model_name=args.dense_model,
                )
                retriever = dr.retrieve
                print(f"[memory] Dense retriever loaded from {args.dense_index} "
                      f"[backend={args.embedding_backend}]")
            except FileNotFoundError as exc:
                print(f"[memory] ERROR: {exc}")
                print("[memory] Build dense index: python scripts/build_dense_memory_index.py")
                sys.exit(1)
        elif mode_tag == "hybrid":
            try:
                from memory.hybrid_retriever import HybridRetriever
                hr = HybridRetriever(
                    tfidf_index_dir=Path(args.memory_index),
                    dense_index_dir=Path(args.dense_index),
                    model_name=args.dense_model,
                    rerank_top_n=args.rerank_top_n,
                )
                retriever = hr.retrieve
                print(
                    f"[memory] Hybrid retriever: shortlist={args.memory_index} "
                    f"dense={args.dense_index} rerank_top_n={args.rerank_top_n} "
                    f"[backend={args.embedding_backend}]"
                )
            except FileNotFoundError as exc:
                print(f"[memory] ERROR: {exc}")
                print(
                    "[memory] Build TF-IDF index and dense index first:\n"
                    "  python scripts/build_vector_memory.py\n"
                    "  python scripts/build_dense_memory_index.py"
                )
                sys.exit(1)
        elif mode_tag in ("structured", "structured-hybrid"):
            try:
                from memory.structured_retriever import StructuredReranker
                is_hybrid = mode_tag == "structured-hybrid"
                sr = StructuredReranker(
                    index_dir=Path(args.structured_index),
                    model_name=args.dense_model,
                    rerank_top_n=args.rerank_top_n,
                    mode="hybrid" if is_hybrid else "dense",
                    shortlist_index=Path(args.memory_index) if is_hybrid else None,
                )
                retriever = sr.retrieve
                print(
                    f"[memory] Structured retriever ({mode_tag}): "
                    f"index={args.structured_index} rerank_top_n={args.rerank_top_n}"
                    + (f" shortlist={args.memory_index}" if is_hybrid else "")
                    + f" [backend={args.embedding_backend}]"
                )
            except FileNotFoundError as exc:
                print(f"[memory] ERROR: {exc}")
                print("[memory] Build the structured index: "
                      "make build-v219-structured-memory-index")
                sys.exit(1)

    # Load model
    from scripts.agent_loop import load_hf_model
    print(f"\nLoading {args.hf_model} …")
    model, tokenizer = load_hf_model(
        args.hf_model,
        lora_path=args.hf_lora or None,
    )

    all_run_results: list[tuple[str, list[dict]]] = []

    sm = args.scoring_mode

    # ── compare-base: run the bare base model first ───────────────────
    if args.compare_base:
        # Find base model ID from adapter config if available
        adapter_config = Path(args.hf_model) / "adapter_config.json"
        if adapter_config.exists():
            import json as _json
            base_id = _json.loads(adapter_config.read_text()).get("base_model_name_or_path", "")
        else:
            base_id = args.hf_model
        print(f"\n── Baseline: {base_id} (no adapter) ──")
        base_model, base_tok = load_hf_model(base_id, lora_path=None)
        base_results = evaluate(base_model, base_tok, tasks, "single", 1,
                                args.verbose, is_hf=True, scoring_mode=sm,
                                system_prompt=eval_system_prompt,
                                stop_after_pass=args.stop_after_pass,
                                memory_state=memory_state,
                                memory_top_k=args.memory_top_k,
                                retriever=retriever)
        all_run_results.append(("base", base_results))
        _save_csv(base_results, Path(args.output) / "base.csv", "base")
        del base_model  # free VRAM

    # ── main model run(s) ──────────────────────────────────────────────
    modes_to_run = (["single", "best_of_n"] if args.mode == "compare"
                    else [args.mode])

    for run_mode in modes_to_run:
        label = run_mode if run_mode != "best_of_n" else f"best_of_{args.n}"
        print(f"\n── {args.hf_model}  mode={run_mode} ──")
        results = evaluate(model, tokenizer, tasks, run_mode, args.n,
                           args.verbose, is_hf=True, scoring_mode=sm,
                           system_prompt=eval_system_prompt,
                           stop_after_pass=args.stop_after_pass,
                           memory_state=memory_state,
                           memory_top_k=args.memory_top_k,
                           retriever=retriever)
        all_run_results.append((label, results))
        _save_csv(results, Path(args.output) / f"{label}.csv", label)

    # ── print tables ──────────────────────────────────────────────────
    summaries = [_summary(res, label) for label, res in all_run_results]
    _print_summary(summaries)
    _print_per_task(all_run_results)

    # ── failure analysis CSV ──────────────────────────────────────────
    _save_failure_analysis_csv(all_run_results, Path(args.output))

    # ── print critique-specific breakdown ─────────────────────────────
    for label, results in all_run_results:
        crit_results    = [r for r in results if r["has_critique"]]
        nocrit_results  = [r for r in results if not r["has_critique"]]
        if crit_results:
            crit_pass  = 100 * sum(r["passed"] for r in crit_results)  / len(crit_results)
            ncrit_pass = 100 * sum(r["passed"] for r in nocrit_results) / max(len(nocrit_results), 1)
            print(f"\n  [{label}] With CRITIQUE:    {crit_pass:.0f}% pass rate  ({len(crit_results)} tasks)")
            print(f"  [{label}] Without CRITIQUE:  {ncrit_pass:.0f}% pass rate  ({len(nocrit_results)} tasks)")
            if crit_results:
                print(f"  [{label}] CRITIQUE lift:     {crit_pass - ncrit_pass:+.0f} pp")


if __name__ == "__main__":
    main()
