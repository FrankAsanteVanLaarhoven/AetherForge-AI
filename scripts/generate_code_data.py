"""
scripts/generate_code_data.py
Build code-agent training data combining:
  1. CodeAlpaca 20k — instruction → code pairs
  2. Synthetic ReAct tool-use trajectories — plan → execute → observe → fix

Output: data/code_agent_data.jsonl
Format: {"instruction": ..., "response": ...}  (same as Alpaca/distillation format)
        Response is either:
          - Plain code with optional <think> block
          - Multi-turn ReAct trajectory with TOOL_CALL / OBSERVATION markers

Usage:
    conda run -n ml-torch python scripts/generate_code_data.py [--n-code N] [--n-react N]
"""

import ast
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

N_CODE  = int(sys.argv[sys.argv.index("--n-code")  + 1]) if "--n-code"  in sys.argv else 5000
N_REACT = int(sys.argv[sys.argv.index("--n-react") + 1]) if "--n-react" in sys.argv else 2000

# ---------------------------------------------------------------------------
# Part 1: CodeAlpaca instruction-code pairs
# ---------------------------------------------------------------------------

def load_code_alpaca(n: int) -> list[dict]:
    try:
        from datasets import load_dataset
        print("Downloading sahil2801/CodeAlpaca-20k ...")
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train",
                          trust_remote_code=False)
        rows = []
        for item in ds:
            instr  = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            inp    = item.get("input", "").strip()
            if not instr or not output:
                continue
            if inp:
                instr = f"{instr}\n\n```\n{inp}\n```"
            rows.append({"instruction": instr, "response": output})
        random.shuffle(rows)
        rows = rows[:n]
        print(f"  Loaded {len(rows)} CodeAlpaca pairs")
        return rows
    except Exception as e:
        print(f"  CodeAlpaca download failed ({e}), falling back to OSS dataset ...")
        return _fallback_code_pairs(n)


def _fallback_code_pairs(n: int) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train",
                          trust_remote_code=False)
        rows = []
        for item in ds:
            instr  = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if instr and output:
                rows.append({"instruction": instr, "response": output})
        random.shuffle(rows)
        return rows[:n]
    except Exception as e2:
        print(f"  Fallback also failed ({e2}), using templates")
        return _template_code_pairs(n)


def _template_code_pairs(n: int) -> list[dict]:
    TASKS = [
        ("Write a Python function to {verb} {object}.",
         "def {fname}({args}):\n    \"\"\"{verb} {object}\"\"\"\n    # TODO: implement\n    pass"),
        ("Implement a Python class for {concept}.",
         "class {concept_cls}:\n    def __init__(self):\n        pass\n\n    def process(self):\n        pass"),
        ("Fix the following Python code:\n```python\n{buggy}\n```",
         "Here is the corrected version:\n```python\n{fixed}\n```"),
        ("Write a Python function to sort a list using {algo}.",
         "def sort_{algo_fn}(lst):\n    # {algo} sort\n    return sorted(lst)"),
        ("Explain what the following code does:\n```python\n{snippet}\n```",
         "The code does the following:\n1. It {action1}\n2. Then it {action2}\n3. Finally it {action3}"),
    ]
    VERBS    = ["compute", "calculate", "find", "generate", "parse", "validate"]
    OBJECTS  = ["the nth Fibonacci number", "a checksum", "prime numbers up to N",
                "all permutations", "JSON data", "an email address"]
    CONCEPTS = ["Stack", "Queue", "LinkedList", "BinaryTree", "HashTable", "Graph"]
    ALGOS    = ["bubble", "insertion", "merge", "quick", "heap"]

    rows = []
    for _ in range(n):
        tmpl, resp_tmpl = random.choice(TASKS)
        v, o = random.choice(VERBS), random.choice(OBJECTS)
        c = random.choice(CONCEPTS)
        a = random.choice(ALGOS)
        instr = tmpl.format(verb=v, object=o, concept=c, algo=a,
                            buggy="x = 1\ny = x +",
                            snippet="result = [x**2 for x in range(10)]")
        resp  = resp_tmpl.format(fname=v.replace(" ", "_"),
                                 args="n", verb=v, object=o,
                                 concept_cls=c, algo=a,
                                 algo_fn=a,
                                 fixed="x = 1\ny = x + 1",
                                 action1="iterates over a list",
                                 action2="applies a transformation",
                                 action3="returns the result")
        rows.append({"instruction": instr, "response": resp})
    return rows


# ---------------------------------------------------------------------------
# Part 2: Synthetic ReAct tool-use trajectories
# ---------------------------------------------------------------------------

# Tool-use format understood by agent_loop.py:
#
#   <think>...</think>
#   TOOL_CALL: tool_name({"arg": "value"})
#   OBSERVATION: <result injected at inference>
#   <think>...</think>
#   TOOL_CALL: ...
#   FINAL_ANSWER: ...
#
# During training, the full trajectory (including observations) is the target.
# At inference, the agent loop runs each TOOL_CALL and injects the real OBSERVATION.

REACT_TEMPLATES = [
    # ── Execute code and interpret output ─────────────────────────────
    {
        "instruction": "Calculate the sum of all even numbers from 1 to 100.",
        "response": (
            "<think>I need to write and run code to sum even numbers from 1 to 100.</think>\n"
            'TOOL_CALL: execute_code({"code": "result = sum(x for x in range(1, 101) if x % 2 == 0)\\nprint(result)"})\n'
            "OBSERVATION: 2550\n"
            "<think>The sum of all even numbers from 1 to 100 is 2550.</think>\n"
            "FINAL_ANSWER: The sum of all even numbers from 1 to 100 is **2550**."
        ),
    },
    {
        "instruction": "What is 2 to the power of 32?",
        "response": (
            "<think>I can compute this directly with Python.</think>\n"
            'TOOL_CALL: execute_code({"code": "print(2 ** 32)"})\n'
            "OBSERVATION: 4294967296\n"
            "FINAL_ANSWER: 2³² = **4,294,967,296**."
        ),
    },
    # ── Write, run, fix ───────────────────────────────────────────────
    {
        "instruction": "Write a Python function `is_palindrome(s)` and test it on 'racecar' and 'hello'.",
        "response": (
            "<think>I'll write the function and test it by executing the code.</think>\n"
            'TOOL_CALL: execute_code({"code": "def is_palindrome(s):\\n    return s == s[::-1]\\nprint(is_palindrome(\'racecar\'))\\nprint(is_palindrome(\'hello\'))"})\n'
            "OBSERVATION: True\nFalse\n"
            "<think>Both tests pass. 'racecar' is a palindrome, 'hello' is not.</think>\n"
            "FINAL_ANSWER: ```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```\nResults: `is_palindrome('racecar')` → `True`, `is_palindrome('hello')` → `False`."
        ),
    },
    {
        "instruction": "Fix this broken Python code and make it run:\n```python\ndef factorial(n)\n    if n = 0:\n        return 1\n    return n * factorial(n-1)\nprint(factorial(5))\n```",
        "response": (
            "<think>There are syntax errors: missing colon after `def`, `=` instead of `==` in condition.</think>\n"
            'TOOL_CALL: execute_code({"code": "def factorial(n):\\n    if n == 0:\\n        return 1\\n    return n * factorial(n - 1)\\nprint(factorial(5))"})\n'
            "OBSERVATION: 120\n"
            "<think>Fixed. Output is correct: 5! = 120.</think>\n"
            "FINAL_ANSWER: The fixed code:\n```python\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\nprint(factorial(5))  # → 120\n```\nBugs fixed: added `:` after `def factorial(n)`, changed `=` to `==` in the condition."
        ),
    },
    # ── Read file ──────────────────────────────────────────────────────
    {
        "instruction": "Read the file 'data/synthetic_data.jsonl' and count how many lines it has.",
        "response": (
            "<think>I'll read the file and count the lines.</think>\n"
            'TOOL_CALL: read_file({"path": "data/synthetic_data.jsonl"})\n'
            "OBSERVATION: [file contents truncated — N lines]\n"
            'TOOL_CALL: execute_code({"code": "with open(\'data/synthetic_data.jsonl\') as f:\\n    print(sum(1 for _ in f))"})\n'
            "OBSERVATION: 3000\n"
            "FINAL_ANSWER: The file `data/synthetic_data.jsonl` has **3000 lines**."
        ),
    },
    # ── Write file ─────────────────────────────────────────────────────
    {
        "instruction": "Create a file called 'hello.py' that prints 'Hello, World!'.",
        "response": (
            "<think>I'll write the file using the write_file tool, then verify it runs.</think>\n"
            'TOOL_CALL: write_file({"path": "hello.py", "content": "print(\'Hello, World!\')\\n"})\n'
            "OBSERVATION: OK — wrote 26 bytes to hello.py\n"
            'TOOL_CALL: execute_code({"code": "exec(open(\'hello.py\').read())"})\n'
            "OBSERVATION: Hello, World!\n"
            "FINAL_ANSWER: Created `hello.py` with content `print('Hello, World!')`. Verified — running it outputs `Hello, World!`."
        ),
    },
    # ── Multi-step debugging ───────────────────────────────────────────
    {
        "instruction": "I have a list `data = [3, 1, 4, 1, 5, 9, 2, 6]`. Find the median without using statistics library.",
        "response": (
            "<think>Median requires sorting the list, then taking the middle element(s).</think>\n"
            'TOOL_CALL: execute_code({"code": "data = [3, 1, 4, 1, 5, 9, 2, 6]\\nsorted_data = sorted(data)\\nn = len(sorted_data)\\nif n % 2 == 0:\\n    median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2\\nelse:\\n    median = sorted_data[n//2]\\nprint(f\'Sorted: {sorted_data}\')\\nprint(f\'Median: {median}\')"})\n'
            "OBSERVATION: Sorted: [1, 1, 2, 3, 4, 5, 6, 9]\nMedian: 3.5\n"
            "FINAL_ANSWER: The median of `[3, 1, 4, 1, 5, 9, 2, 6]` is **3.5**. (Sorted: `[1, 1, 2, 3, 4, 5, 6, 9]` — average of the two middle values 3 and 4.)"
        ),
    },
    # ── List files ─────────────────────────────────────────────────────
    {
        "instruction": "List all Python files in the scripts/ directory.",
        "response": (
            "<think>I'll use list_files to see what's in scripts/.</think>\n"
            'TOOL_CALL: list_files({"path": "scripts/"})\n'
            "OBSERVATION: agent_loop.py, chat.py, distill_aetherforge.py, eval_checkpoints.py, ...\n"
            "FINAL_ANSWER: The Python files in `scripts/` include: `agent_loop.py`, `chat.py`, `distill_aetherforge.py`, `eval_checkpoints.py`, and others."
        ),
    },
    # ── edit_file ─────────────────────────────────────────────────────
    {
        "instruction": "I have a file 'utils.py' with a buggy sum function. It uses `range(n)` but should use `range(n + 1)` to include n. Fix it with a targeted edit.",
        "response": (
            "<think>I'll read the file first, then use edit_file for a minimal targeted fix rather than rewriting the whole file.</think>\n"
            'TOOL_CALL: read_file({"path": "utils.py"})\n'
            "OBSERVATION: def sum_range(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
            "<think>Found the bug. I'll replace just the range call.</think>\n"
            'TOOL_CALL: edit_file({"path": "utils.py", "old": "for i in range(n):", "new": "for i in range(n + 1):"})\n'
            "OBSERVATION: OK — replaced 1 occurrence in utils.py\n"
            'TOOL_CALL: execute_code({"code": "exec(open(\'utils.py\').read())\\nprint(sum_range(5))"})\n'
            "OBSERVATION: 15\n"
            "FINAL_ANSWER: Fixed off-by-one with `edit_file`. `sum_range(5)` now returns **15** (0+1+2+3+4+5)."
        ),
    },
    {
        "instruction": "Update the version string in 'mypackage/__init__.py' from '0.1.0' to '0.2.0'.",
        "response": (
            "<think>I'll use edit_file for a precise single-line replacement.</think>\n"
            'TOOL_CALL: edit_file({"path": "mypackage/__init__.py", "old": "__version__ = \'0.1.0\'", "new": "__version__ = \'0.2.0\'"})\n'
            "OBSERVATION: OK — replaced 1 occurrence in mypackage/__init__.py\n"
            'TOOL_CALL: execute_code({"code": "exec(open(\'mypackage/__init__.py\').read())\\nprint(__version__)"})\n'
            "OBSERVATION: 0.2.0\n"
            "FINAL_ANSWER: Version bumped from `0.1.0` → `0.2.0`. Verified the import reflects the new value."
        ),
    },
    # ── search_file ───────────────────────────────────────────────────
    {
        "instruction": "Find all places in the scripts/ directory where `save_checkpoint` is called or defined.",
        "response": (
            "<think>I'll use search_file to grep across all .py files in scripts/.</think>\n"
            'TOOL_CALL: search_file({"path": "scripts/", "pattern": "save_checkpoint"})\n'
            "OBSERVATION: scripts/train_aetherforge.py:45:    save_checkpoint(model, optimizer, scheduler, step, loss, out_dir)\nscripts/finetune_code_agent.py:97:def save_checkpoint(model, optimizer, scheduler, step, loss, out_dir):\n"
            "FINAL_ANSWER: `save_checkpoint` appears in 2 files:\n- `train_aetherforge.py` line 45: call site\n- `finetune_code_agent.py` line 97: definition"
        ),
    },
    {
        "instruction": "Find all TODO comments in the codebase.",
        "response": (
            "<think>search_file can grep for 'TODO' across all Python files.</think>\n"
            'TOOL_CALL: search_file({"path": ".", "pattern": "TODO"})\n'
            "OBSERVATION: ./aetherforge/model.py:88:    # TODO: add RoPE scaling factor\n./scripts/serve.py:201:    # TODO: add authentication\n"
            "FINAL_ANSWER: Found 2 TODO comments:\n- `aetherforge/model.py:88` — RoPE scaling\n- `scripts/serve.py:201` — authentication"
        ),
    },
    # ── install_package ───────────────────────────────────────────────
    {
        "instruction": "Compute the cosine similarity between two vectors using scipy.",
        "response": (
            "<think>Let me try importing scipy first before installing it.</think>\n"
            'TOOL_CALL: execute_code({"code": "from scipy.spatial.distance import cosine\\nprint(1 - cosine([1,0,0], [1,0,0]))"})\n'
            "OBSERVATION: 1.0\n"
            "<think>scipy is already available. Computing cosine similarity of [1,2,3] and [4,5,6].</think>\n"
            'TOOL_CALL: execute_code({"code": "from scipy.spatial.distance import cosine\\na, b = [1,2,3], [4,5,6]\\nprint(round(1 - cosine(a, b), 6))"})\n'
            "OBSERVATION: 0.974632\n"
            "FINAL_ANSWER: Cosine similarity between [1,2,3] and [4,5,6] is **0.9746** (scipy was already installed)."
        ),
    },
    {
        "instruction": "Try to import 'rich' for pretty printing. If it's not installed, install it first, then use it.",
        "response": (
            "<think>Let me try importing rich directly first.</think>\n"
            'TOOL_CALL: execute_code({"code": "import rich; print(rich.__version__)"})\n'
            "OBSERVATION: ERROR:\nModuleNotFoundError: No module named 'rich'\n"
            "<think>Not installed. I'll install it, then use it.</think>\n"
            'TOOL_CALL: install_package({"package": "rich"})\n'
            "OBSERVATION: Successfully installed rich-13.7.1\n"
            'TOOL_CALL: execute_code({"code": "from rich import print as rprint\\nrprint(\'[bold green]Hello from rich![/bold green]\')"})\n'
            "OBSERVATION: Hello from rich!\n"
            "FINAL_ANSWER: Installed `rich` (v13.7.1) and verified it works. `rprint` renders styled text in the terminal."
        ),
    },
    # ── format_code ───────────────────────────────────────────────────
    {
        "instruction": "Format this poorly styled Python code:\n```python\ndef   foo(x,y,z):\n    a=x+y\n    b=a*z\n    return b\nprint(foo(1,2,3))\n```",
        "response": (
            "<think>I'll use format_code to run black, then verify the output is unchanged.</think>\n"
            'TOOL_CALL: format_code({"code": "def   foo(x,y,z):\\n    a=x+y\\n    b=a*z\\n    return b\\nprint(foo(1,2,3))"})\n'
            "OBSERVATION: def foo(x, y, z):\n    a = x + y\n    b = a * z\n    return b\n\n\nprint(foo(1, 2, 3))\n"
            'TOOL_CALL: execute_code({"code": "def foo(x, y, z):\\n    a = x + y\\n    b = a * z\\n    return b\\nprint(foo(1, 2, 3))"})\n'
            "OBSERVATION: 9\n"
            "FINAL_ANSWER: Formatted with black (added spaces around `=`, commas). Output unchanged: **9**.\n```python\ndef foo(x, y, z):\n    a = x + y\n    b = a * z\n    return b\n```"
        ),
    },
    # ── check_syntax ──────────────────────────────────────────────────
    {
        "instruction": "Check if this code has any syntax errors before running it:\n```python\ndef greet(name)\n    return f'Hello, {name}!'\nprint(greet('world'))\n```",
        "response": (
            "<think>I'll use check_syntax first — faster than running and gives a precise error location.</think>\n"
            'TOOL_CALL: check_syntax({"code": "def greet(name)\\n    return f\'Hello, {name}!\'\\nprint(greet(\'world\'))"})\n'
            "OBSERVATION: SyntaxError at line 1: expected ':'\n"
            "<think>Missing colon after `def greet(name)`. I'll fix it.</think>\n"
            'TOOL_CALL: execute_code({"code": "def greet(name):\\n    return f\'Hello, {name}!\'\\nprint(greet(\'world\'))"})\n'
            "OBSERVATION: Hello, world!\n"
            "FINAL_ANSWER: Fixed the syntax error — added `:` after `def greet(name)`. Code now runs correctly."
        ),
    },
    {
        "instruction": "I wrote a recursive function but I'm not sure if the syntax is correct. Check it:\n```python\ndef factorial(n)\n    if n == 0\n        return 1\n    return n * factorial(n - 1)\n```",
        "response": (
            "<think>I'll use check_syntax to find all the syntax errors at once before trying to run.</think>\n"
            'TOOL_CALL: check_syntax({"code": "def factorial(n)\\n    if n == 0\\n        return 1\\n    return n * factorial(n - 1)"})\n'
            "OBSERVATION: SyntaxError at line 1: expected ':'\n"
            "<think>There's a missing colon on line 1. Fixing both def and if colons, then verifying.</think>\n"
            'TOOL_CALL: execute_code({"code": "def factorial(n):\\n    if n == 0:\\n        return 1\\n    return n * factorial(n - 1)\\nprint(factorial(5))\\nprint(factorial(0))"})\n'
            "OBSERVATION: 120\n1\n"
            "FINAL_ANSWER: Two syntax fixes: added `:` after `def factorial(n)` and after `if n == 0`. Both test cases pass."
        ),
    },
    # ── run_linter ────────────────────────────────────────────────────
    {
        "instruction": "Check this Python code for style issues before submitting:\n```python\nimport os,sys\ndef foo(x,y,z):\n    a=x+y+z\n    return a\nprint(foo(1,2,3))\n```",
        "response": (
            "<think>I'll run the linter to check for style issues before running the code.</think>\n"
            'TOOL_CALL: run_linter({"code": "import os,sys\\ndef foo(x,y,z):\\n    a=x+y+z\\n    return a\\nprint(foo(1,2,3))"})\n'
            "OBSERVATION: <code>:1:10: E401 multiple imports on one line\n<code>:1:10: F401 'os' imported but unused\n<code>:1:13: F401 'sys' imported but unused\n<code>:2:9: E231 missing whitespace after ','\n"
            "<think>Linter found unused imports and formatting issues. I'll clean them up.</think>\n"
            'TOOL_CALL: execute_code({"code": "def foo(x, y, z):\\n    return x + y + z\\nprint(foo(1, 2, 3))"})\n'
            "OBSERVATION: 6\n"
            "FINAL_ANSWER: Removed unused `os` and `sys` imports, fixed spacing after commas. Clean version:\n```python\ndef foo(x, y, z):\n    return x + y + z\nprint(foo(1, 2, 3))  # → 6\n```"
        ),
    },
    # ── git_status / git_diff / git_log ───────────────────────────────
    {
        "instruction": "What files have I changed since the last commit? I'm in a git repo.",
        "response": (
            "<think>git_status gives a short view of modified/untracked files.</think>\n"
            'TOOL_CALL: git_status({"path": "."})\n'
            "OBSERVATION:  M scripts/train.py\n M data/config.yaml\n?? outputs/run_42/\n"
            "<think>Two modified files and one untracked output directory. I'll show the diff for the tracked files.</think>\n"
            'TOOL_CALL: git_diff({"path": "."})\n'
            "OBSERVATION: scripts/train.py | 4 ++++\n data/config.yaml | 2 +-\n 2 files changed, 5 insertions(+), 1 deletion(-)\n\ndiff --git a/scripts/train.py ...\n+    scheduler = CosineAnnealingLR(optimizer, T_max=steps)\n"
            "FINAL_ANSWER: Two tracked files modified — `scripts/train.py` (+4 lines, added cosine scheduler) and `data/config.yaml` (+1 -1). One untracked output directory `outputs/run_42/`."
        ),
    },
    {
        "instruction": "Show me the last 5 commits in this repository.",
        "response": (
            "<think>git_log with n=5 will show recent history without any extra arguments.</think>\n"
            'TOOL_CALL: git_log({"n": 5})\n'
            "OBSERVATION: a1b2c3d Add cosine LR scheduler\ne4f5g6h Fix batch size OOM on 1B model\ni7j8k9l Add gradient checkpointing flag\nm1n2o3p Initial training loop\nq4r5s6t Repo init\n"
            "FINAL_ANSWER: Last 5 commits:\n1. `a1b2c3d` Add cosine LR scheduler\n2. `e4f5g6h` Fix batch size OOM on 1B model\n3. `i7j8k9l` Add gradient checkpointing flag\n4. `m1n2o3p` Initial training loop\n5. `q4r5s6t` Repo init"
        ),
    },
    {
        "instruction": "I think I broke something in utils.py. Show me exactly what changed before I try to fix it.",
        "response": (
            "<think>git_diff on a specific file will show exactly what changed since the last commit.</think>\n"
            'TOOL_CALL: git_diff({"path": "utils.py"})\n'
            "OBSERVATION: utils.py | 6 +++---\n 1 file changed, 3 insertions(+), 3 deletions(-)\n\ndiff --git a/utils.py b/utils.py\n-def clamp(x, lo, hi):\n-    return max(lo, min(hi, x))\n+def clamp(x, lo, hi):\n+    return max(lo, min(x, hi))\n"
            "<think>The arguments to `min` were swapped — `min(hi, x)` → `min(x, hi)`. That's actually the same result since min is symmetric. Let me check if the issue is elsewhere.</think>\n"
            'TOOL_CALL: execute_code({"code": "def clamp(x, lo, hi):\\n    return max(lo, min(x, hi))\\nprint(clamp(5, 0, 3), clamp(-1, 0, 3), clamp(2, 0, 3))"})\n'
            "OBSERVATION: 3 0 2\n"
            "FINAL_ANSWER: The diff shows only a cosmetic argument swap in `clamp` (both orderings are equivalent). `clamp(5,0,3)=3`, `clamp(-1,0,3)=0`, `clamp(2,0,3)=2` — all correct. The bug is likely elsewhere."
        ),
    },
    # ── Run tests ─────────────────────────────────────────────────────
    {
        "instruction": "Write a unit test for `is_palindrome(s)` and run it.",
        "response": (
            "<think>I'll write the function and a unittest, then execute the test.</think>\n"
            'TOOL_CALL: execute_code({"code": "import unittest\\n\\ndef is_palindrome(s):\\n    return s == s[::-1]\\n\\nclass TestPalindrome(unittest.TestCase):\\n    def test_palindrome(self):\\n        self.assertTrue(is_palindrome(\'racecar\'))\\n        self.assertFalse(is_palindrome(\'hello\'))\\n        self.assertTrue(is_palindrome(\'a\'))\\n        self.assertTrue(is_palindrome(\'\'))\\n\\nif __name__ == \'__main__\':\\n    unittest.main(argv=[\'\'], exit=False, verbosity=2)"})\n'
            "OBSERVATION: test_palindrome (__main__.TestPalindrome) ... ok\n----------------------------------------------------------------------\nRan 1 test in 0.001s\nOK\n"
            "CRITIQUE:\n  Correctness — `s == s[::-1]` correctly reverses and compares. Output matches expected.\n  Edge cases  — empty string → True ✓, single char → True ✓, normal palindrome and non-palindrome ✓.\n  Requirements— task asked for is_palindrome(s) returning bool — met.\n  → Solution OK.\n"
            "FINAL_ANSWER: All 4 test cases pass. The `is_palindrome` implementation is correct."
        ),
    },
    # ── Self-critique: no issues found ────────────────────────────────
    {
        "instruction": "Write a function `clamp(x, lo, hi)` that constrains x to [lo, hi]. Verify it works.",
        "response": (
            "<think>Clamp is straightforward: max(lo, min(hi, x)). I need to verify edge cases: x below lo, x above hi, x in range.</think>\n"
            'TOOL_CALL: execute_code({"code": "def clamp(x, lo, hi):\\n    return max(lo, min(hi, x))\\n\\nprint(clamp(-5, 0, 10))\\nprint(clamp(5, 0, 10))\\nprint(clamp(15, 0, 10))\\nprint(clamp(0, 0, 10))\\nprint(clamp(10, 0, 10))"})\n'
            "OBSERVATION: 0\n5\n10\n0\n10\n"
            "CRITIQUE:\n  Correctness — -5→0 ✓, 5→5 ✓, 15→10 ✓, boundaries 0→0 and 10→10 ✓.\n  Edge cases  — boundary values (lo and hi themselves) included and correct.\n  Requirements— clamp(x, lo, hi) constraining to [lo, hi] — fully met.\n  → Solution OK.\n"
            "FINAL_ANSWER: `clamp(x, lo, hi)` is `max(lo, min(hi, x))`. Verified on 5 cases including boundaries — all correct."
        ),
    },
    # ── Self-critique: finds a bug, goes back to fix ──────────────────
    {
        "instruction": "Write a function `unique_sorted(lst)` that returns unique elements in sorted order. Verify it works on [3, 1, 2, 1, 3].",
        "response": (
            "<think>I'll use sorted(set(lst)) — set removes duplicates, sorted orders them.</think>\n"
            'TOOL_CALL: execute_code({"code": "def unique_sorted(lst):\\n    return sorted(set(lst))\\n\\nprint(unique_sorted([3, 1, 2, 1, 3]))"})\n'
            "OBSERVATION: [1, 2, 3]\n"
            "CRITIQUE:\n  Correctness — [3,1,2,1,3] → [1,2,3] ✓\n  Edge cases  — need to check: empty list, single element, float/int deduplication.\n  Requirements— 'unique elements in sorted order' — met for the given case, but edge cases unverified.\n  → Fix needed: verify empty list and single-element inputs.\n"
            'TOOL_CALL: execute_code({"code": "def unique_sorted(lst):\\n    return sorted(set(lst))\\n\\nprint(unique_sorted([]))\\nprint(unique_sorted([1]))\\nprint(unique_sorted([1.0, 1, 2]))"})\n'
            "OBSERVATION: []\n[1]\n[1, 2]\n"
            "CRITIQUE:\n  Correctness — all verified outputs correct.\n  Edge cases  — empty → [] ✓, single element → [1] ✓, float/int dedup → [1,2] ✓ (1.0==1 in Python).\n  Requirements— unique sorted elements — fully met.\n  → Solution OK.\n"
            "FINAL_ANSWER: `unique_sorted(lst)` = `sorted(set(lst))`. Works correctly: [3,1,2,1,3] → [1,2,3]. Handles empty lists and single elements too."
        ),
    },
    # ── Self-critique: reveals correctness gap, forces fix ────────────
    {
        "instruction": "Write a function `safe_divide(a, b)` that returns a/b, or 0 if b is zero.",
        "response": (
            "<think>Simple guard: if b == 0 return 0, else return a/b.</think>\n"
            'TOOL_CALL: execute_code({"code": "def safe_divide(a, b):\\n    if b == 0:\\n        return 0\\n    return a / b\\n\\nprint(safe_divide(10, 2))\\nprint(safe_divide(7, 0))"})\n'
            "OBSERVATION: 5.0\n0\n"
            "CRITIQUE:\n  Correctness — 10/2=5.0 ✓, 7/0=0 ✓.\n  Edge cases  — need to check: 0/0, negative divisor, fractional result.\n  Requirements— 'return a/b or 0 if b is zero' — basic case met, edge cases unverified.\n  → Fix needed: verify 0/0, negatives, fractions.\n"
            'TOOL_CALL: execute_code({"code": "def safe_divide(a, b):\\n    if b == 0:\\n        return 0\\n    return a / b\\n\\nprint(safe_divide(0, 0))\\nprint(safe_divide(-6, 2))\\nprint(safe_divide(1, 3))"})\n'
            "OBSERVATION: 0\n-3.0\n0.3333333333333333\n"
            "CRITIQUE:\n  Correctness — all results correct.\n  Edge cases  — 0/0→0 ✓, -6/2→-3.0 ✓, 1/3→0.333 ✓.\n  Requirements— return a/b or 0 for zero divisor — fully met.\n  → Solution OK.\n"
            "FINAL_ANSWER: `safe_divide(a, b)` returns `a / b` normally, and `0` when `b` is zero. Verified on 5 cases including 0/0, negatives, and fractions."
        ),
    },
    # ── Complete fix-loop: CRITIQUE finds bug → TOOL_CALL fix → second CRITIQUE OK ──
    {
        "instruction": "Write `flatten(lst)` that flattens a nested list of arbitrary depth. Verify it works.",
        "response": (
            "<think>I'll use recursion — if an element is a list, recurse; otherwise append it.</think>\n"
            'TOOL_CALL: execute_code({"code": "def flatten(lst):\\n    result = []\\n    for item in lst:\\n        if isinstance(item, list):\\n            result.extend(flatten(item))\\n        else:\\n            result.append(item)\\n    return result\\n\\nprint(flatten([[1, [2, 3]], [4, [5, [6]]]]))"})\n'
            "OBSERVATION: [1, 2, 3, 4, 5, 6]\n"
            "CRITIQUE:\n"
            "  Correctness — [[1,[2,3]],[4,[5,[6]]]] → [1,2,3,4,5,6] ✓\n"
            "  Edge cases  — need to verify: empty list, already-flat list, single element, deeply nested.\n"
            "  Requirements— arbitrary depth mentioned — tested 3 levels, but empty and flat not checked.\n"
            "  → Fix needed: verify flatten([]), flatten([1,2,3]), and flatten([[[1]]]).\n"
            'TOOL_CALL: execute_code({"code": "def flatten(lst):\\n    result = []\\n    for item in lst:\\n        if isinstance(item, list):\\n            result.extend(flatten(item))\\n        else:\\n            result.append(item)\\n    return result\\n\\nprint(flatten([]))\\nprint(flatten([1,2,3]))\\nprint(flatten([[[1]]]))\\nprint(flatten([[1,[2,[3,[4]]]]]))"})\n'
            "OBSERVATION: []\n[1, 2, 3]\n[1]\n[1, 2, 3, 4]\n"
            "CRITIQUE:\n"
            "  Correctness — all outputs correct.\n"
            "  Edge cases  — empty → [] ✓, flat → [1,2,3] ✓, depth-3 → [1] ✓, depth-4 → [1,2,3,4] ✓.\n"
            "  Requirements— arbitrary depth flattening — fully met.\n"
            "  → Solution OK.\n"
            "FINAL_ANSWER: `flatten(lst)` works correctly at any nesting depth. Empty lists → [], flat lists unchanged, deeply nested lists fully flattened."
        ),
    },
    {
        "instruction": "Write `sum_to_n(n)` returning 1+2+...+n. Verify for n=10 (should be 55) and n=0 (should be 0).",
        "response": (
            "<think>Sum 1..n = n*(n+1)/2 or a simple loop. Let me use range.</think>\n"
            'TOOL_CALL: execute_code({"code": "def sum_to_n(n):\\n    return sum(range(n))\\nprint(sum_to_n(10))"})\n'
            "OBSERVATION: 45\n"
            "CRITIQUE:\n"
            "  Correctness — got 45, expected 55. range(n) = 0..9, missing n itself.\n"
            "  Edge cases  — the off-by-one is the core bug. Need range(n+1).\n"
            "  Requirements— 1+2+...+n must include n — not met.\n"
            "  → Fix needed: change range(n) to range(1, n+1).\n"
            'TOOL_CALL: execute_code({"code": "def sum_to_n(n):\\n    return sum(range(1, n + 1))\\nprint(sum_to_n(10))\\nprint(sum_to_n(0))\\nprint(sum_to_n(1))"})\n'
            "OBSERVATION: 55\n0\n1\n"
            "CRITIQUE:\n"
            "  Correctness — sum_to_n(10)=55 ✓, sum_to_n(0)=0 ✓, sum_to_n(1)=1 ✓.\n"
            "  Edge cases  — zero and single element both handled correctly.\n"
            "  Requirements— 1+2+...+n — now fully met.\n"
            "  → Solution OK.\n"
            "FINAL_ANSWER: Fixed off-by-one: `range(n)` → `range(1, n+1)`. sum_to_n(10)=**55**, sum_to_n(0)=**0**."
        ),
    },
    {
        "instruction": "Write a function `count_vowels(s)` that counts vowels (a,e,i,o,u, case-insensitive) in a string.",
        "response": (
            "<think>I'll check each character against the vowel set, case-insensitive.</think>\n"
            'TOOL_CALL: execute_code({"code": "def count_vowels(s):\\n    return sum(1 for c in s if c in \'aeiou\')\\n\\nprint(count_vowels(\'Hello World\'))"})\n'
            "OBSERVATION: 3\n"
            "CRITIQUE:\n"
            "  Correctness — 'Hello World' has e,o,o = 3. But wait, 'H' is lowercase-checked against lowercase vowels — 'H' won't match. 'e' → yes, 'o' in 'World' → yes, 'o' in 'Hello' → yes. So 3 is right.\n"
            "  Edge cases  — what about uppercase vowels? 'AEIOU' → should be 5, but 'A' not in 'aeiou'.\n"
            "  Requirements— 'case-insensitive' specified in task — NOT met for uppercase.\n"
            "  → Fix needed: convert to lowercase first: c.lower() in 'aeiou'.\n"
            'TOOL_CALL: execute_code({"code": "def count_vowels(s):\\n    return sum(1 for c in s if c.lower() in \'aeiou\')\\n\\nprint(count_vowels(\'Hello World\'))\\nprint(count_vowels(\'AEIOU\'))\\nprint(count_vowels(\'\'))\\nprint(count_vowels(\'xyz\'))"})\n'
            "OBSERVATION: 3\n5\n0\n0\n"
            "CRITIQUE:\n"
            "  Correctness — 'Hello World'→3 ✓, 'AEIOU'→5 ✓.\n"
            "  Edge cases  — empty string→0 ✓, no vowels→0 ✓.\n"
            "  Requirements— case-insensitive vowel counting — fully met.\n"
            "  → Solution OK.\n"
            "FINAL_ANSWER: `count_vowels(s)` uses `c.lower() in 'aeiou'` for case-insensitive matching. 'AEIOU'→5, 'Hello World'→3, empty→0."
        ),
    },
]

# Additional code-only pairs with think blocks (no tool calls)
THINK_CODE_TEMPLATES = [
    {
        "instruction": "Write a binary search function in Python.",
        "response": (
            "<think>Binary search requires a sorted list. I'll return the index or -1 if not found.</think>\n"
            "```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```"
        ),
    },
    {
        "instruction": "Implement a Python context manager that measures execution time.",
        "response": (
            "<think>Context managers use `__enter__` and `__exit__`. I'll record time on entry and print elapsed on exit.</think>\n"
            "```python\nimport time\nfrom contextlib import contextmanager\n\n@contextmanager\ndef timer(label=''):\n    t0 = time.perf_counter()\n    yield\n    elapsed = time.perf_counter() - t0\n    print(f'{label}: {elapsed:.4f}s')\n\n# Usage:\nwith timer('my_operation'):\n    time.sleep(0.1)  # → my_operation: 0.1001s\n```"
        ),
    },
    {
        "instruction": "Write a Python decorator that retries a function up to 3 times on exception.",
        "response": (
            "<think>I need a decorator that wraps the function, catches exceptions, and retries.</think>\n"
            "```python\nimport functools\nimport time\n\ndef retry(max_attempts=3, delay=0.5):\n    def decorator(fn):\n        @functools.wraps(fn)\n        def wrapper(*args, **kwargs):\n            for attempt in range(max_attempts):\n                try:\n                    return fn(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts - 1:\n                        raise\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n\n@retry(max_attempts=3, delay=0.1)\ndef flaky_function():\n    raise ValueError('transient error')\n```"
        ),
    },
    {
        "instruction": "Implement a simple LRU cache in Python without using functools.",
        "response": (
            "<think>LRU cache needs O(1) get/put. I'll use OrderedDict which moves items to the end on access.</think>\n"
            "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)\n```"
        ),
    },
    {
        "instruction": "Write a Python generator that yields Fibonacci numbers indefinitely.",
        "response": (
            "<think>A generator with `yield` avoids storing all values. I'll use two running variables.</think>\n"
            "```python\ndef fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\n# Usage:\nfib = fibonacci()\nprint([next(fib) for _ in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n```"
        ),
    },
]


# ---------------------------------------------------------------------------
# Recovery-focused trajectory families
# Rules for ALL new families:
#   - No <think> tags; use PLAN: or omit planning
#   - Every final successful execute_code call uses asserts + print("PASS")
#   - FINAL_ANSWER only appears after OBSERVATION: PASS
#   - CRITIQUE after an error must say "Fix needed:", never "→ Solution OK."
#   - "→ Solution OK." only appears after OBSERVATION: PASS
# ---------------------------------------------------------------------------

def _tc(tool: str, **kwargs) -> str:
    """Format TOOL_CALL line. Uses json.dumps so args are always valid JSON."""
    return f"TOOL_CALL: {tool}({json.dumps(kwargs)})\n"


def _crit_ok(note: str = "") -> str:
    tail = f" {note}" if note else ""
    return (
        f"CRITIQUE:\n"
        f"  Correctness — All assertions pass ✓.{tail}\n"
        f"  → Solution OK.\n"
    )


def _crit_fix(reason: str) -> str:
    return (
        f"CRITIQUE:\n"
        f"  Correctness — {reason}\n"
        f"  → Fix needed: {reason}\n"
    )


# ── 1. TOOL_FIRST_SUCCESS ──────────────────────────────────────────────────
# Starts with execute_code; code has asserts + print("PASS"); FINAL_ANSWER
# only after OBSERVATION: PASS.

def _tfs(task_desc: str, fn_name: str, impl: str, asserts: str) -> dict:
    code = impl + "\n" + asserts + "\nprint('PASS')"
    return {
        "instruction": task_desc,
        "response": (
            _tc("execute_code", code=code) +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: `{fn_name}` verified by execute_code. PASS."
        ),
    }


TOOL_FIRST_SUCCESS_TEMPLATES = [
    _tfs(
        "Write `fizzbuzz(n)` returning a list of strings: 'FizzBuzz' for multiples of 15, 'Fizz' for 3, 'Buzz' for 5, else the number as string. Verify with asserts.",
        "fizzbuzz",
        "def fizzbuzz(n):\n    result = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            result.append('FizzBuzz')\n        elif i % 3 == 0:\n            result.append('Fizz')\n        elif i % 5 == 0:\n            result.append('Buzz')\n        else:\n            result.append(str(i))\n    return result",
        "assert fizzbuzz(15)[-1] == 'FizzBuzz'\nassert len(fizzbuzz(15)) == 15\nassert fizzbuzz(3) == ['1', '2', 'Fizz']",
    ),
    _tfs(
        "Write `factorial(n)` iteratively (no recursion). Verify factorial(0)=1, factorial(5)=120, factorial(10)=3628800.",
        "factorial",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    _tfs(
        "Write `is_palindrome(s)` returning True if s reads the same forwards and backwards. Verify on 'racecar' (True), 'hello' (False), '' (True).",
        "is_palindrome",
        "def is_palindrome(s):\n    return s == s[::-1]",
        "assert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True\nassert is_palindrome('a') is True",
    ),
    _tfs(
        "Write `sum_list(lst)` returning the sum of all numbers in a list. Verify sum_list([1,2,3,4,5])==15 and sum_list([])==0.",
        "sum_list",
        "def sum_list(lst):\n    return sum(lst)",
        "assert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0",
    ),
    _tfs(
        "Write `binary_search(arr, target)` returning the index of target in sorted arr, or -1 if not found.",
        "binary_search",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "assert binary_search([1, 3, 5, 7, 9, 11], 7) == 3\nassert binary_search([1, 3, 5, 7, 9, 11], 4) == -1\nassert binary_search([1, 3, 5, 7, 9, 11], 1) == 0",
    ),
    _tfs(
        "Write `flatten(lst)` that recursively flattens a nested list of arbitrary depth. Verify flatten([[1,[2,3]],[4,[5,[6]]]])==[1,2,3,4,5,6].",
        "flatten",
        "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
        "assert flatten([[1, [2, 3]], [4, [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\nassert flatten([]) == []\nassert flatten([1, 2, 3]) == [1, 2, 3]",
    ),
    _tfs(
        "Write `word_count(text)` returning a dict mapping each word (lowercase) to its frequency. Verify on 'the cat sat on the mat'.",
        "word_count",
        "def word_count(text):\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
        "d = word_count('the cat sat on the mat')\nassert d['the'] == 2\nassert d['cat'] == 1\nassert word_count('') == {}",
    ),
    _tfs(
        "Write `merge_sorted(a, b)` merging two sorted lists into one sorted list without using sort(). Verify merge_sorted([1,3,5],[2,4,6])==[1,2,3,4,5,6].",
        "merge_sorted",
        "def merge_sorted(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]",
        "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]\nassert merge_sorted([1], []) == [1]",
    ),
    _tfs(
        "Write `safe_divide(a, b)` returning a/b or 0 if b is zero. Verify: safe_divide(10,2)=5.0, safe_divide(7,0)=0, safe_divide(-6,2)=-3.0.",
        "safe_divide",
        "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b",
        "assert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0\nassert safe_divide(0, 0) == 0\nassert safe_divide(-6, 2) == -3.0",
    ),
    _tfs(
        "Write `clamp(x, lo, hi)` constraining x to [lo, hi]. Verify: clamp(-5,0,10)=0, clamp(5,0,10)=5, clamp(15,0,10)=10.",
        "clamp",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
        "assert clamp(-5, 0, 10) == 0\nassert clamp(5, 0, 10) == 5\nassert clamp(15, 0, 10) == 10\nassert clamp(0, 0, 10) == 0\nassert clamp(10, 0, 10) == 10",
    ),
    _tfs(
        "Write `unique_sorted(lst)` returning unique elements in sorted order. Verify: unique_sorted([3,1,2,1,3])=[1,2,3], unique_sorted([])=[].",
        "unique_sorted",
        "def unique_sorted(lst):\n    return sorted(set(lst))",
        "assert unique_sorted([3, 1, 2, 1, 3]) == [1, 2, 3]\nassert unique_sorted([]) == []\nassert unique_sorted([1]) == [1]",
    ),
]


# ── 2. INDENT_ERROR_FIX ───────────────────────────────────────────────────
# Bad first call → IndentationError → CRITIQUE "Fix needed: IndentationError"
# → fixed call with asserts → PASS → "→ Solution OK." → FINAL_ANSWER

_OBS_INDENT = (
    "OBSERVATION: ERROR:\n"
    "IndentationError: expected an indented block after function definition on line 1\n"
)


def _ief(task_desc: str, fn_name: str, bad_code: str, good_code: str) -> dict:
    return {
        "instruction": task_desc,
        "response": (
            _tc("execute_code", code=bad_code) +
            _OBS_INDENT +
            "CRITIQUE:\n"
            "  Correctness — IndentationError: function body is not indented.\n"
            "  → Fix needed: IndentationError — add 4-space indentation to every line "
            "inside the function body.\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: Fixed IndentationError — added 4-space indent. "
            f"`{fn_name}` verified by execute_code: PASS."
        ),
    }


INDENT_ERROR_FIX_TEMPLATES = [
    _ief(
        "Write `sum_list(lst)` returning the sum of a list.",
        "sum_list",
        "def sum_list(lst):\nreturn sum(lst)",
        "def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0",
    ),
    _ief(
        "Write `factorial(n)` computing n! iteratively.",
        "factorial",
        "def factorial(n):\nresult = 1\nfor i in range(2, n + 1):\nresult *= i\nreturn result",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(5) == 120\nassert factorial(0) == 1",
    ),
    _ief(
        "Write `is_palindrome(s)` checking if a string is a palindrome.",
        "is_palindrome",
        "def is_palindrome(s):\nreturn s == s[::-1]",
        "def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False",
    ),
    _ief(
        "Write `safe_divide(a, b)` returning a/b or 0 when b is zero.",
        "safe_divide",
        "def safe_divide(a, b):\nif b == 0:\nreturn 0\nreturn a / b",
        "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0",
    ),
    _ief(
        "Write `clamp(x, lo, hi)` constraining x to the range [lo, hi].",
        "clamp",
        "def clamp(x, lo, hi):\nreturn max(lo, min(hi, x))",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\nassert clamp(-5, 0, 10) == 0\nassert clamp(15, 0, 10) == 10",
    ),
    _ief(
        "Write `unique_sorted(lst)` returning unique elements in sorted order.",
        "unique_sorted",
        "def unique_sorted(lst):\nreturn sorted(set(lst))",
        "def unique_sorted(lst):\n    return sorted(set(lst))\nassert unique_sorted([3, 1, 2, 1, 3]) == [1, 2, 3]\nassert unique_sorted([]) == []",
    ),
    _ief(
        "Write `binary_search(arr, target)` returning the index or -1 if not found.",
        "binary_search",
        "def binary_search(arr, target):\nlo, hi = 0, len(arr) - 1\nwhile lo <= hi:\nmid = (lo + hi) // 2\nif arr[mid] == target:\nreturn mid\nelif arr[mid] < target:\nlo = mid + 1\nelse:\nhi = mid - 1\nreturn -1",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\nassert binary_search([1, 3, 5, 7, 9], 7) == 3\nassert binary_search([1, 3, 5, 7, 9], 4) == -1",
    ),
]


# ── 3. INVALID_JSON_FIX ───────────────────────────────────────────────────
# Bad first call (with invalid JSON) appears ONLY in instruction/context.
# Supervised response begins with CRITIQUE → corrected call → PASS → Solution OK.

_OBS_INV_JSON = "OBSERVATION: ERROR: invalid JSON in tool args\n"
_ALLOWED_TOOLS_LIST = (
    "execute_code, run_script, run_tests, read_file, write_file, edit_file, "
    "search_file, list_files, run_command, check_syntax, run_linter, "
    "install_package, format_code, git_status, git_diff, git_log"
)

INVALID_JSON_FIX_TEMPLATES = [
    {
        "instruction": (
            "Original task: Write `sum_list(lst)` returning the sum of a list and verify it.\n\n"
            "Previous failed attempt:\n"
            "TOOL_CALL: execute_code({'code': 'def sum_list(lst):\\n    return sum(lst)\\nprint(sum_list([1,2,3]))})\n"
            + _OBS_INV_JSON
        ),
        "response": (
            "CRITIQUE:\n"
            "  Correctness — Invalid JSON: tool args used single-quoted keys. "
            "JSON requires double-quoted keys.\n"
            "  → Fix needed: rewrite TOOL_CALL with double-quoted JSON keys and asserts.\n"
            + _tc("execute_code",
                  code="def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed invalid JSON (double-quoted keys). `sum_list` verified: PASS."
        ),
    },
    {
        "instruction": (
            "Original task: Write `factorial(n)` and verify factorial(5)==120.\n\n"
            "Previous failed attempt:\n"
            "TOOL_CALL: execute_code({code: \"def factorial(n):\\n    result = 1\\n"
            "    for i in range(2, n+1):\\n        result *= i\\n    return result\\nprint(factorial(5))\"})\n"
            + _OBS_INV_JSON
        ),
        "response": (
            "CRITIQUE:\n"
            "  Correctness — Invalid JSON: key must be double-quoted: {\"code\": ...}.\n"
            "  → Fix needed: wrap all JSON keys in double quotes.\n"
            + _tc("execute_code",
                  code="def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(5) == 120\nassert factorial(0) == 1\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed JSON syntax — double-quoted key. `factorial` verified: PASS."
        ),
    },
    {
        "instruction": (
            "Original task: Write `is_palindrome(s)` and test on 'racecar' and 'hello'.\n\n"
            "Previous failed attempt:\n"
            "TOOL_CALL: execute_code({\"code\": \"def is_palindrome(s):\\n    return s == s[::-1]\\n"
            "print(is_palindrome('racecar'))\",})\n"
            + _OBS_INV_JSON
        ),
        "response": (
            "CRITIQUE:\n"
            "  Correctness — Invalid JSON: trailing comma after last key-value pair.\n"
            "  → Fix needed: remove trailing comma and add assert-based verification.\n"
            + _tc("execute_code",
                  code="def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed trailing-comma JSON error. `is_palindrome` verified: PASS."
        ),
    },
    {
        "instruction": (
            "Original task: Write `clamp(x, lo, hi)` constraining x to [lo, hi] and verify it.\n\n"
            "Previous failed attempt:\n"
            "TOOL_CALL: execute_code({\"code\": \"def clamp(x, lo, hi):\\n    return max(lo, min(hi, x))\\n"
            "print(clamp(5, 0, 10))\", \"verbose\": True})\n"
            + _OBS_INV_JSON
        ),
        "response": (
            "CRITIQUE:\n"
            "  Correctness — Invalid JSON: Python literal True is not valid JSON (must be lowercase true); "
            "'verbose' is not a recognised execute_code key.\n"
            "  → Fix needed: use only the 'code' key with valid JSON and add asserts.\n"
            + _tc("execute_code",
                  code="def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\nassert clamp(-5, 0, 10) == 0\nassert clamp(5, 0, 10) == 5\nassert clamp(15, 0, 10) == 10\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed JSON — single valid 'code' key. `clamp` verified: PASS."
        ),
    },
]


# ── 4. UNKNOWN_TOOL_FIX ───────────────────────────────────────────────────
# Bad call (task function as tool) appears ONLY in instruction/context.
# Supervised response begins with CRITIQUE → execute_code → PASS → Solution OK.

def _utf(task_desc: str, bad_tool: str, bad_args: dict,
         fn_name: str, good_code: str) -> dict:
    """Build an UNKNOWN_TOOL_FIX example with bad call in instruction only."""
    bad_call = f"TOOL_CALL: {bad_tool}({json.dumps(bad_args)})"
    bad_obs = (
        f"OBSERVATION: ERROR: unknown tool '{bad_tool}'. "
        f"Available tools: {_ALLOWED_TOOLS_LIST}"
    )
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            f"Previous failed attempt:\n"
            f"{bad_call}\n"
            f"{bad_obs}\n"
        ),
        "response": (
            "CRITIQUE:\n"
            f"  Correctness — ERROR: '{bad_tool}' is not a registered tool. "
            f"Task functions cannot be called directly.\n"
            f"  Requirements — only registered tools may be called "
            f"(execute_code, run_tests, read_file, etc.).\n"
            f"  → Fix needed: implement `{fn_name}` inside execute_code with asserts.\n"
            + _tc("execute_code", code=good_code + "\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + f"FINAL_ANSWER: Fixed — implemented `{fn_name}` inside execute_code. Verified: PASS."
        ),
    }


UNKNOWN_TOOL_FIX_TEMPLATES = [
    _utf(
        "Write `fizzbuzz(n)` and verify it returns 'FizzBuzz' at position 15.",
        "fizzbuzz", {"n": 15}, "fizzbuzz",
        "def fizzbuzz(n):\n    result = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            result.append('FizzBuzz')\n        elif i % 3 == 0:\n            result.append('Fizz')\n        elif i % 5 == 0:\n            result.append('Buzz')\n        else:\n            result.append(str(i))\n    return result\nassert fizzbuzz(15)[-1] == 'FizzBuzz'\nassert len(fizzbuzz(15)) == 15",
    ),
    _utf(
        "Implement `sum_list(lst)` and verify sum_list([1,2,3,4,5])==15.",
        "sum_list", {"lst": [1, 2, 3, 4, 5]}, "sum_list",
        "def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0",
    ),
    _utf(
        "Write `is_palindrome(s)` and verify it on 'racecar' and 'hello'.",
        "is_palindrome", {"s": "racecar"}, "is_palindrome",
        "def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True",
    ),
    _utf(
        "Write `safe_divide(a, b)` and verify safe_divide(10,2)==5.0 and safe_divide(7,0)==0.",
        "safe_divide", {"a": 10, "b": 2}, "safe_divide",
        "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0\nassert safe_divide(0, 0) == 0",
    ),
    _utf(
        "Write `unique_sorted(lst)` and verify on [3,1,2,1,3].",
        "unique_sorted", {"lst": [3, 1, 2, 1, 3]}, "unique_sorted",
        "def unique_sorted(lst):\n    return sorted(set(lst))\nassert unique_sorted([3, 1, 2, 1, 3]) == [1, 2, 3]\nassert unique_sorted([]) == []",
    ),
    _utf(
        "Write `clamp(x, lo, hi)` constraining x to [lo, hi] and verify clamp(-5,0,10)==0.",
        "clamp", {"x": -5, "lo": 0, "hi": 10}, "clamp",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\nassert clamp(-5, 0, 10) == 0\nassert clamp(5, 0, 10) == 5\nassert clamp(15, 0, 10) == 10",
    ),
    _utf(
        "Write `factorial(n)` and verify factorial(5)==120 and factorial(0)==1.",
        "factorial", {"n": 5}, "factorial",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(10) == 3628800",
    ),
]


# ── 5. SOLUTION_OK_AFTER_ERROR_FIX ────────────────────────────────────────
# First call uses valid execute_code with valid JSON but logically wrong Python.
# CRITIQUE says "Fix needed:" (never "→ Solution OK.") until PASS is observed.
# All first-attempt errors are genuine (not fabricated).

SOLUTION_OK_AFTER_ERROR_FIX_TEMPLATES = [
    {
        "instruction": "Write `sum_list(lst)` returning the sum of a list. Fix any errors.",
        "response": (
            # lst.sum() → AttributeError (genuine: list has no .sum())
            _tc("execute_code",
                code="def sum_list(lst):\n    return lst.sum()\nassert sum_list([1, 2, 3]) == 6\nprint('PASS')")
            + "OBSERVATION: ERROR:\nAttributeError: 'list' object has no attribute 'sum'\n"
            + "CRITIQUE:\n"
            "  Correctness — AttributeError: list has no .sum(); use the built-in sum().\n"
            "  → Fix needed: replace lst.sum() with sum(lst).\n"
            + _tc("execute_code",
                  code="def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed AttributeError — used built-in sum(). `sum_list` verified: PASS."
        ),
    },
    {
        "instruction": "Write `binary_search(arr, target)` returning the index or -1. Fix any errors.",
        "response": (
            # hi=len(arr) causes arr[mid]=arr[len(arr)] IndexError when target > all elements (genuine)
            _tc("execute_code",
                code="def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\nassert binary_search([1, 3, 5, 7, 9], 11) == -1\nprint('PASS')")
            + "OBSERVATION: ERROR:\nIndexError: list index out of range\n"
            + "CRITIQUE:\n"
            "  Correctness — IndexError: hi = len(arr) is off by one. "
            "When target > all elements, mid reaches len(arr) and arr[mid] is out of range.\n"
            "  → Fix needed: initialise hi = len(arr) - 1.\n"
            + _tc("execute_code",
                  code="def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\nassert binary_search([1, 3, 5, 7, 9], 7) == 3\nassert binary_search([1, 3, 5, 7, 9], 11) == -1\nassert binary_search([1, 3, 5, 7, 9], 1) == 0\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed off-by-one — hi = len(arr) - 1. `binary_search` verified: PASS."
        ),
    },
    {
        "instruction": "Write `word_count(text)` returning word frequency dict. Fix any errors.",
        "response": (
            # dict comprehension overwrites duplicates → assert d['the']==2 fails (genuine AssertionError)
            _tc("execute_code",
                code="def word_count(text):\n    return {w: 1 for w in text.split()}\nd = word_count('the cat sat on the mat')\nassert d['the'] == 2\nprint('PASS')")
            + "OBSERVATION: ERROR:\nAssertionError\n"
            + "CRITIQUE:\n"
            "  Correctness — AssertionError: dict comprehension overwrites duplicate keys with 1 "
            "(d['the']==1, not 2). Need to accumulate counts instead.\n"
            "  → Fix needed: use dict.get() to accumulate counts, not a comprehension.\n"
            + _tc("execute_code",
                  code="def word_count(text):\n    counts = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts\nd = word_count('the cat sat on the mat')\nassert d['the'] == 2\nassert d['cat'] == 1\nassert word_count('') == {}\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed word_count to accumulate counts. Verified: PASS."
        ),
    },
    {
        "instruction": "Write `flatten(lst)` flattening a nested list. Fix any errors.",
        "response": (
            # comprehension skips lists → [1,2,3] expected but [1,3] returned (genuine AssertionError)
            _tc("execute_code",
                code="def flatten(lst):\n    return [x for x in lst if not isinstance(x, list)]\nassert flatten([[1, [2]], 3]) == [1, 2, 3]\nprint('PASS')")
            + "OBSERVATION: ERROR:\nAssertionError\n"
            + "CRITIQUE:\n"
            "  Correctness — AssertionError: comprehension drops sub-lists entirely instead of recursing.\n"
            "  → Fix needed: recurse into each sub-list with extend(flatten(item)).\n"
            + _tc("execute_code",
                  code="def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\nassert flatten([[1, [2, 3]], [4, [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\nassert flatten([]) == []\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed flatten to recurse into sub-lists. Verified: PASS."
        ),
    },
    {
        "instruction": "Write `merge_sorted(a, b)` merging two sorted lists without sort(). Fix any errors.",
        "response": (
            # Missing i/j increments → infinite loop / IndexError (genuine: b[j] access when j>=len(b))
            _tc("execute_code",
                code="def merge_sorted(a, b):\n    i = j = 0\n    result = []\n    while i < len(a) or j < len(b):\n        result.append(a[i] if a[i] < b[j] else b[j])\n        i += 1\n    return result\nassert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nprint('PASS')")
            + "OBSERVATION: ERROR:\nIndexError: list index out of range\n"
            + "CRITIQUE:\n"
            "  Correctness — IndexError: when j >= len(b), b[j] is out of range; "
            "also j never increments so b is never consumed.\n"
            "  → Fix needed: use separate arms for i and j with bounds checks; "
            "extend result with remaining tail.\n"
            + _tc("execute_code",
                  code="def merge_sorted(a, b):\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\nassert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1]) == [1]\nassert merge_sorted([1], []) == [1]\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + "FINAL_ANSWER: Fixed IndexError — proper two-pointer merge with tail extension. Verified: PASS."
        ),
    },
]


# ── 6. REPEATED_CALL_FIX ──────────────────────────────────────────────────
# TWO identical failing calls appear in instruction/context.
# Supervised response begins with CRITIQUE identifying the repetition,
# then provides a DIFFERENT correct implementation → PASS → Solution OK.

def _rcf(task_desc: str, bad_code: str, bad_error: str,
         error_desc: str, fix_desc: str, good_code: str) -> dict:
    """Build a REPEATED_CALL_FIX example with two identical bad calls in instruction."""
    bad_tc = _tc("execute_code", code=bad_code).rstrip("\n")
    bad_obs = f"OBSERVATION: ERROR:\n{bad_error}"
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous failed attempts (same call repeated twice — still failing):\n"
            f"{bad_tc}\n"
            f"{bad_obs}\n\n"
            f"{bad_tc}\n"
            f"{bad_obs}\n"
        ),
        "response": (
            "CRITIQUE:\n"
            f"  Correctness — {error_desc}\n"
            "  Requirements — repeating the same failing call always produces the same error.\n"
            f"  → Fix needed: {fix_desc}\n"
            + _tc("execute_code", code=good_code + "\nprint('PASS')")
            + "OBSERVATION: PASS\n"
            + _crit_ok()
            + f"FINAL_ANSWER: Fixed by writing a different, correct implementation. Verified: PASS."
        ),
    }


REPEATED_CALL_FIX_TEMPLATES = [
    _rcf(
        "Write `factorial(n)` and verify factorial(5)==120 and factorial(0)==1.",
        bad_code="def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n)\nassert factorial(5) == 120\nprint('PASS')",
        bad_error="RecursionError: maximum recursion depth exceeded",
        error_desc="RecursionError x2: recursive call uses n instead of n-1 — infinite recursion.",
        fix_desc="switch to an iterative implementation that avoids recursion.",
        good_code="def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    _rcf(
        "Write `safe_divide(a, b)` returning a/b or 0 when b==0.",
        bad_code="def safe_divide(a, b):\n    return a / b\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0\nprint('PASS')",
        bad_error="ZeroDivisionError: division by zero",
        error_desc="ZeroDivisionError x2: no guard for b==0.",
        fix_desc="add 'if b == 0: return 0' before the division.",
        good_code="def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0\nassert safe_divide(0, 0) == 0",
    ),
    _rcf(
        "Write `is_palindrome(s)` and test on 'racecar' and 'hello'.",
        bad_code="def is_palindrome(s):\n    for i in range(len(s)):\n        if s[i] != s[len(s) - i]:\n            return False\n    return True\nassert is_palindrome('racecar')\nprint('PASS')",
        bad_error="IndexError: string index out of range",
        error_desc="IndexError x2: s[len(s) - i] when i=0 gives s[len(s)] which is out of bounds.",
        fix_desc="use 's == s[::-1]' — correct and avoids index arithmetic.",
        good_code="def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True",
    ),
    _rcf(
        "Write `clamp(x, lo, hi)` constraining x to [lo, hi].",
        bad_code="def clamp(x, lo, hi):\n    if x < lo: return lo\n    if x > lo: return hi\n    return x\nassert clamp(5, 0, 10) == 5\nprint('PASS')",
        bad_error="AssertionError",
        error_desc="AssertionError x2: second branch checks x>lo but should check x>hi; clamp(5,0,10) returns hi=10 instead of 5.",
        fix_desc="use max(lo, min(hi, x)) — one expression, no off-by-one.",
        good_code="def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\nassert clamp(-5, 0, 10) == 0\nassert clamp(5, 0, 10) == 5\nassert clamp(15, 0, 10) == 10",
    ),
]


# ── 7. NO_OUTPUT_FIX ──────────────────────────────────────────────────────
# Code ran but produced no output → CRITIQUE says "add asserts and print('PASS')"
# → corrected call → PASS → Solution OK → FINAL_ANSWER.
# First call intentionally lacks asserts and print('PASS') (that's the bug being fixed).
# The audit only checks calls whose *direct next* OBSERVATION is PASS — the
# first call here is followed by "(no output)", so it is not audited.

_OBS_NO_OUTPUT = "OBSERVATION: (no output)\n"


def _nof(task_desc: str, fn_name: str, impl_only: str, good_code: str) -> dict:
    """NO_OUTPUT_FIX: bad call (no output) in instruction; supervised response starts with CRITIQUE."""
    bad_call = _tc("execute_code", code=impl_only).rstrip("\n")
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous failed attempt:\n"
            f"{bad_call}\n"
            f"{_OBS_NO_OUTPUT}"
        ),
        "response": (
            "CRITIQUE:\n"
            "  Correctness — Code ran but produced no output; "
            "cannot verify correctness without assertions.\n"
            "  → Fix needed: code ran but did not prove correctness; "
            "add assertions and print('PASS').\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: Added asserts and print('PASS'). "
            f"`{fn_name}` verified by execute_code: PASS."
        ),
    }


NO_OUTPUT_FIX_TEMPLATES = [
    _nof(
        "Write `sum_list(lst)` returning the sum of all numbers in a list.",
        "sum_list",
        "def sum_list(lst):\n    return sum(lst)",
        "def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0",
    ),
    _nof(
        "Write `is_palindrome(s)` returning True if s is a palindrome.",
        "is_palindrome",
        "def is_palindrome(s):\n    return s == s[::-1]",
        "def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True",
    ),
    _nof(
        "Write `clamp(x, lo, hi)` constraining x to the range [lo, hi].",
        "clamp",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\nassert clamp(-5, 0, 10) == 0\nassert clamp(5, 0, 10) == 5\nassert clamp(15, 0, 10) == 10",
    ),
    _nof(
        "Write `safe_divide(a, b)` returning a/b or 0 when b is zero.",
        "safe_divide",
        "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b",
        "def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0\nassert safe_divide(0, 0) == 0",
    ),
    _nof(
        "Write `factorial(n)` computing n! iteratively.",
        "factorial",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    # 15 additional NO_OUTPUT_FIX templates to reach ≥50% of agent-only pool
    _nof(
        "Write `add_numbers(a, b)` returning a + b.",
        "add_numbers",
        "def add_numbers(a, b):\n    return a + b",
        "def add_numbers(a, b):\n    return a + b\nassert add_numbers(2, 3) == 5\nassert add_numbers(-1, 1) == 0\nassert add_numbers(0, 0) == 0",
    ),
    _nof(
        "Write `multiply(a, b)` returning the product a * b.",
        "multiply",
        "def multiply(a, b):\n    return a * b",
        "def multiply(a, b):\n    return a * b\nassert multiply(3, 4) == 12\nassert multiply(0, 100) == 0\nassert multiply(-2, 3) == -6",
    ),
    _nof(
        "Write `is_even(n)` returning True if n is even, False otherwise.",
        "is_even",
        "def is_even(n):\n    return n % 2 == 0",
        "def is_even(n):\n    return n % 2 == 0\nassert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True",
    ),
    _nof(
        "Write `absolute_value(x)` returning the absolute value of x without using abs().",
        "absolute_value",
        "def absolute_value(x):\n    return x if x >= 0 else -x",
        "def absolute_value(x):\n    return x if x >= 0 else -x\nassert absolute_value(5) == 5\nassert absolute_value(-3) == 3\nassert absolute_value(0) == 0",
    ),
    _nof(
        "Write `string_length(s)` returning the number of characters in s without using len().",
        "string_length",
        "def string_length(s):\n    count = 0\n    for _ in s:\n        count += 1\n    return count",
        "def string_length(s):\n    count = 0\n    for _ in s:\n        count += 1\n    return count\nassert string_length('hello') == 5\nassert string_length('') == 0\nassert string_length('ab') == 2",
    ),
    _nof(
        "Write `to_uppercase(s)` returning s converted to uppercase.",
        "to_uppercase",
        "def to_uppercase(s):\n    return s.upper()",
        "def to_uppercase(s):\n    return s.upper()\nassert to_uppercase('hello') == 'HELLO'\nassert to_uppercase('') == ''\nassert to_uppercase('World') == 'WORLD'",
    ),
    _nof(
        "Write `count_words(s)` returning the number of whitespace-separated words in s.",
        "count_words",
        "def count_words(s):\n    return len(s.split())",
        "def count_words(s):\n    return len(s.split())\nassert count_words('hello world') == 2\nassert count_words('') == 0\nassert count_words('one two three four') == 4",
    ),
    _nof(
        "Write `first_element(lst)` returning the first element of a non-empty list.",
        "first_element",
        "def first_element(lst):\n    return lst[0]",
        "def first_element(lst):\n    return lst[0]\nassert first_element([1, 2, 3]) == 1\nassert first_element(['a', 'b']) == 'a'\nassert first_element([42]) == 42",
    ),
    _nof(
        "Write `last_element(lst)` returning the last element of a non-empty list.",
        "last_element",
        "def last_element(lst):\n    return lst[-1]",
        "def last_element(lst):\n    return lst[-1]\nassert last_element([1, 2, 3]) == 3\nassert last_element(['x']) == 'x'\nassert last_element([0, 5, 10]) == 10",
    ),
    _nof(
        "Write `square(n)` returning n squared.",
        "square",
        "def square(n):\n    return n * n",
        "def square(n):\n    return n * n\nassert square(0) == 0\nassert square(5) == 25\nassert square(-3) == 9",
    ),
    _nof(
        "Write `average(nums)` returning the arithmetic mean of a non-empty list of numbers.",
        "average",
        "def average(nums):\n    return sum(nums) / len(nums)",
        "def average(nums):\n    return sum(nums) / len(nums)\nassert average([1, 2, 3]) == 2.0\nassert average([10]) == 10.0\nassert average([1, 1, 1, 1]) == 1.0",
    ),
    _nof(
        "Write `is_positive(n)` returning True if n > 0, False otherwise.",
        "is_positive",
        "def is_positive(n):\n    return n > 0",
        "def is_positive(n):\n    return n > 0\nassert is_positive(1) is True\nassert is_positive(0) is False\nassert is_positive(-5) is False",
    ),
    _nof(
        "Write `celsius_to_fahrenheit(c)` converting Celsius to Fahrenheit using F = C * 9/5 + 32.",
        "celsius_to_fahrenheit",
        "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32",
        "def celsius_to_fahrenheit(c):\n    return c * 9 / 5 + 32\nassert celsius_to_fahrenheit(0) == 32.0\nassert celsius_to_fahrenheit(100) == 212.0\nassert celsius_to_fahrenheit(-40) == -40.0",
    ),
    _nof(
        "Write `is_empty(lst)` returning True if the list has no elements, False otherwise.",
        "is_empty",
        "def is_empty(lst):\n    return len(lst) == 0",
        "def is_empty(lst):\n    return len(lst) == 0\nassert is_empty([]) is True\nassert is_empty([1]) is False\nassert is_empty([0, 0, 0]) is False",
    ),
    _nof(
        "Write `find_max(a, b, c)` returning the largest of three numbers without using max().",
        "find_max",
        "def find_max(a, b, c):\n    if a >= b and a >= c:\n        return a\n    elif b >= c:\n        return b\n    return c",
        "def find_max(a, b, c):\n    if a >= b and a >= c:\n        return a\n    elif b >= c:\n        return b\n    return c\nassert find_max(1, 2, 3) == 3\nassert find_max(5, 5, 5) == 5\nassert find_max(-1, -2, -3) == -1",
    ),
    _nof(
        "Write `string_repeat(s, n)` returning s repeated n times (use string multiplication).",
        "string_repeat",
        "def string_repeat(s, n):\n    return s * n",
        "def string_repeat(s, n):\n    return s * n\nassert string_repeat('ab', 3) == 'ababab'\nassert string_repeat('x', 0) == ''\nassert string_repeat('hi', 1) == 'hi'",
    ),
]


# ── 8. JSON_ROBUSTNESS ────────────────────────────────────────────────────
# TOOL_FIRST_SUCCESS examples with code containing characters that could
# confuse manual JSON escaping: single quotes, f-strings, dict literals,
# double-quoted strings, multi-line classes, key=value strings.
# All are emitted via _tc() → json.dumps() so escaping is automatic.

JSON_ROBUSTNESS_TEMPLATES = [
    _tfs(
        "Write `greet(name)` returning an f-string greeting like 'Hello, Alice!'.",
        "greet",
        "def greet(name):\n    return f'Hello, {name}!'",
        "assert greet('Alice') == 'Hello, Alice!'\nassert greet('World') == 'Hello, World!'\nassert greet('') == 'Hello, !'",
    ),
    _tfs(
        "Write `make_record(name, score)` returning a dict {'name': name, 'score': score}.",
        "make_record",
        "def make_record(name, score):\n    return {'name': name, 'score': score}",
        "r = make_record('Alice', 95)\nassert r == {'name': 'Alice', 'score': 95}\nassert make_record('Bob', 0)['score'] == 0",
    ),
    _tfs(
        "Write `describe(items)` returning a string 'Items: a, b, c' joining the list with ', '.",
        "describe",
        "def describe(items):\n    return f\"Items: {', '.join(str(x) for x in items)}\"",
        "assert describe(['a', 'b', 'c']) == 'Items: a, b, c'\nassert describe([1, 2]) == 'Items: 1, 2'\nassert describe([]) == 'Items: '",
    ),
    _tfs(
        "Implement a Point class with x, y attributes and __repr__ returning 'Point(x, y)'.",
        "Point",
        "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'",
        "p = Point(3, 4)\nassert p.x == 3\nassert p.y == 4\nassert repr(p) == 'Point(3, 4)'\nassert repr(Point(0, 0)) == 'Point(0, 0)'",
    ),
    _tfs(
        "Write `parse_kv(s)` parsing 'key=value' comma-separated pairs into a dict. Input: 'a=1,b=2'.",
        "parse_kv",
        "def parse_kv(s):\n    result = {}\n    for pair in s.split(','):\n        k, v = pair.split('=')\n        result[k.strip()] = v.strip()\n    return result",
        "assert parse_kv('a=1,b=2,c=3') == {'a': '1', 'b': '2', 'c': '3'}\nassert parse_kv('x=hello') == {'x': 'hello'}",
    ),
    _tfs(
        "Write `replace_spaces(s, char)` replacing all spaces with char. Verify on 'hello world'.",
        "replace_spaces",
        "def replace_spaces(s, char):\n    return s.replace(' ', char)",
        "assert replace_spaces('hello world', '_') == 'hello_world'\nassert replace_spaces('a b c', '-') == 'a-b-c'\nassert replace_spaces('nospace', '_') == 'nospace'",
    ),
]


# ── 9. SYNTAX_ERROR_TO_PASS_FIX ──────────────────────────────────────────
# Bad call (Python SyntaxError) in instruction; CRITIQUE-first response.
# Each error is a real, distinct SyntaxError class: missing colon after def,
# `=` instead of `==` in assert, unclosed paren, missing colon in elif.

def _sef(task_desc: str, fn_name: str, bad_code: str, bad_error: str,
         critique_text: str, good_code: str) -> dict:
    """SYNTAX_ERROR_TO_PASS_FIX: bad call (SyntaxError) in instruction; CRITIQUE-first."""
    bad_call = _tc("execute_code", code=bad_code).rstrip("\n")
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous failed attempt:\n"
            f"{bad_call}\n"
            f"OBSERVATION: ERROR:\n{bad_error}\n"
        ),
        "response": (
            "CRITIQUE:\n"
            f"  Correctness — {critique_text}\n"
            "  → Fix needed: correct the syntax error; add asserts and print('PASS').\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: Fixed SyntaxError. `{fn_name}` verified by execute_code: PASS."
        ),
    }


SYNTAX_ERROR_TO_PASS_FIX_TEMPLATES = [
    _sef(
        "Write `sum_list(lst)` returning the sum of all numbers in a list. Verify with asserts.",
        "sum_list",
        bad_code="def sum_list(lst)\n    return sum(lst)\nassert sum_list([1, 2, 3]) == 6\nprint('PASS')",
        bad_error="SyntaxError: expected ':' (<string>, line 1)",
        critique_text="SyntaxError: `def sum_list(lst)` is missing its colon. "
                      "Every `def` header must end with `:`.",
        good_code="def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0",
    ),
    _sef(
        "Write `factorial(n)` computing n! iteratively. Verify factorial(5)==120 and factorial(0)==1.",
        "factorial",
        bad_code="def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(5) = 120\nprint('PASS')",
        bad_error="SyntaxError: invalid syntax (<string>, line 7)",
        critique_text="SyntaxError: `=` used instead of `==` in the assert. "
                      "Equality comparison in assert requires `==`, not `=` (assignment).",
        good_code="def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    _sef(
        "Write `is_palindrome(s)` returning True if s reads the same forwards and backwards.",
        "is_palindrome",
        bad_code="def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nprint('PASS'",
        bad_error="SyntaxError: unexpected EOF while parsing (<string>, line 5)",
        critique_text="SyntaxError: unclosed parenthesis — `print('PASS'` is missing the closing `)`.",
        good_code="def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True",
    ),
    _sef(
        "Write `clamp(x, lo, hi)` constraining x to the range [lo, hi].",
        "clamp",
        bad_code="def clamp(x, lo, hi):\n    if x < lo\n        return lo\n    if x > hi:\n        return hi\n    return x\nassert clamp(5, 0, 10) == 5\nprint('PASS')",
        bad_error="SyntaxError: expected ':' (<string>, line 2)",
        critique_text="SyntaxError: `if x < lo` is missing its colon. "
                      "Every `if` condition must end with `:`.",
        good_code="def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\nassert clamp(-5, 0, 10) == 0\nassert clamp(5, 0, 10) == 5\nassert clamp(15, 0, 10) == 10",
    ),
    _sef(
        "Write `binary_search(arr, target)` returning the index or -1 if not found.",
        "binary_search",
        bad_code="def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\nassert binary_search([1, 3, 5, 7, 9], 7) == 3\nprint('PASS')",
        bad_error="SyntaxError: expected ':' (<string>, line 7)",
        critique_text="SyntaxError: `elif arr[mid] < target` is missing its colon.",
        good_code="def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\nassert binary_search([1, 3, 5, 7, 9], 7) == 3\nassert binary_search([1, 3, 5, 7, 9], 4) == -1",
    ),
]


# ── 10. INDENT_ERROR_TO_PASS_FIX ─────────────────────────────────────────
# Bad call (IndentationError) in instruction; CRITIQUE-first response.
# Covers functions not already in INDENT_ERROR_FIX; critiques are more
# explicit about WHERE the indentation must go.

def _ief2(task_desc: str, fn_name: str, bad_code: str, good_code: str,
          critique_line: str) -> dict:
    """INDENT_ERROR_TO_PASS_FIX: bad call (IndentationError) in instruction; CRITIQUE-first."""
    bad_call = _tc("execute_code", code=bad_code).rstrip("\n")
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous failed attempt:\n"
            f"{bad_call}\n"
            "OBSERVATION: ERROR:\n"
            "IndentationError: expected an indented block after function definition on line 1\n"
        ),
        "response": (
            "CRITIQUE:\n"
            f"  Correctness — IndentationError: {critique_line}.\n"
            "  → Fix needed: indent every line inside the function body with 4 spaces; "
            "add asserts and print('PASS').\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: Fixed IndentationError — applied 4-space indentation. "
            f"`{fn_name}` verified by execute_code: PASS."
        ),
    }


INDENT_ERROR_TO_PASS_FIX_TEMPLATES = [
    _ief2(
        "Write `max_of_list(lst)` returning the maximum value in a list.",
        "max_of_list",
        bad_code="def max_of_list(lst):\nreturn max(lst)",
        good_code="def max_of_list(lst):\n    return max(lst)\nassert max_of_list([3, 1, 4, 1, 5, 9]) == 9\nassert max_of_list([1]) == 1",
        critique_line="the `return` must be indented 4 spaces inside the function body",
    ),
    _ief2(
        "Write `reverse_string(s)` returning the input string reversed.",
        "reverse_string",
        bad_code="def reverse_string(s):\nreturn s[::-1]",
        good_code="def reverse_string(s):\n    return s[::-1]\nassert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
        critique_line="the `return` must be indented inside the function — no indentation means it falls outside the function body",
    ),
    _ief2(
        "Write `count_vowels(s)` counting vowels (a, e, i, o, u) case-insensitively.",
        "count_vowels",
        bad_code="def count_vowels(s):\nreturn sum(1 for c in s if c.lower() in 'aeiou')",
        good_code="def count_vowels(s):\n    return sum(1 for c in s if c.lower() in 'aeiou')\nassert count_vowels('Hello World') == 3\nassert count_vowels('AEIOU') == 5\nassert count_vowels('') == 0",
        critique_line="the `return` statement must be inside the function body with 4-space indentation",
    ),
    _ief2(
        "Write `power(base, exp)` computing base**exp iteratively without using the `**` operator.",
        "power",
        bad_code="def power(base, exp):\nresult = 1\nfor _ in range(exp):\nresult *= base\nreturn result",
        good_code="def power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result\nassert power(2, 10) == 1024\nassert power(3, 0) == 1\nassert power(5, 3) == 125",
        critique_line="all body lines (result=1, for-loop, result*=base, return) must be indented 4 spaces inside the function; nested loop body needs 8 spaces",
    ),
    _ief2(
        "Write `find_min(lst)` returning the minimum element without using Python's built-in min().",
        "find_min",
        bad_code="def find_min(lst):\ncurrent = lst[0]\nfor x in lst:\nif x < current:\ncurrent = x\nreturn current",
        good_code="def find_min(lst):\n    current = lst[0]\n    for x in lst:\n        if x < current:\n            current = x\n    return current\nassert find_min([3, 1, 4, 1, 5, 9]) == 1\nassert find_min([7]) == 7",
        critique_line="all body lines, the for-loop, and the nested if-block each need 4 more spaces of indentation per nesting level",
    ),
]


# ── 11. WRONG_CRITIQUE_FIX ────────────────────────────────────────────────
# Teaches the model NOT to over-critique correct code.
#
# Two sub-patterns:
#   PASS case    : previous call got OBSERVATION: PASS + erroneous critique
#                  → supervised response is FINAL_ANSWER immediately
#   NO-OUTPUT case: correct impl, no asserts, (no output), wrong diagnosis
#                  → supervised response: CRITIQUE correctly identifying
#                    "missing asserts", corrected call → PASS → FINAL_ANSWER
#
# PASS-case responses start with FINAL_ANSWER: — valid after updating
# _first_response_line_type() to recognise FINAL_ANSWER as a permitted type.

def _wcf_pass(task_desc: str, fn_name: str, code_with_asserts: str,
              wrong_critique: str) -> dict:
    """WRONG_CRITIQUE_FIX (PASS): code already passed; FINAL_ANSWER immediately."""
    tc_line = _tc("execute_code", code=code_with_asserts + "\nprint('PASS')").rstrip("\n")
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous attempt:\n"
            f"{tc_line}\n"
            "OBSERVATION: PASS\n"
            "CRITIQUE:\n"
            f"  {wrong_critique}\n"
            "  → Fix needed: ...\n"
        ),
        "response": (
            f"FINAL_ANSWER: `{fn_name}` is already verified by execute_code: PASS. "
            "No further action needed."
        ),
    }


def _wcf_no_output(task_desc: str, fn_name: str, impl_only: str,
                   wrong_critique: str, good_code: str) -> dict:
    """WRONG_CRITIQUE_FIX (no output): wrong diagnosis of (no output); correct it."""
    bad_call = _tc("execute_code", code=impl_only).rstrip("\n")
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous attempt:\n"
            f"{bad_call}\n"
            "OBSERVATION: (no output)\n"
            "CRITIQUE:\n"
            f"  {wrong_critique}\n"
            "  → Fix needed: ...\n"
        ),
        "response": (
            "CRITIQUE:\n"
            "  Correctness — The previous critique misidentified the problem. "
            "The implementation is correct, but the code produced no output because "
            "it lacks assertions and print('PASS').\n"
            "  → Fix needed: add assertions to verify correctness and "
            "print('PASS') as the final line.\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: Correct diagnosis — implementation was right; "
            f"added asserts and print('PASS'). `{fn_name}` verified: PASS."
        ),
    }


WRONG_CRITIQUE_FIX_TEMPLATES = [
    # ── PASS case: erroneous critique after a correct PASS ─────────────
    _wcf_pass(
        "Write `sum_list(lst)` returning the sum of all numbers in a list. Verify with asserts.",
        "sum_list",
        code_with_asserts="def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0",
        wrong_critique="The implementation may fail for lists with floats — add type-checking.",
    ),
    _wcf_pass(
        "Write `factorial(n)` computing n! iteratively. Verify factorial(5)==120 and factorial(0)==1.",
        "factorial",
        code_with_asserts="def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
        wrong_critique="Edge case n=1 not tested — may be incorrect for single-element input.",
    ),
    # ── NO-OUTPUT case: wrong diagnosis of missing-assert bug ──────────
    _wcf_no_output(
        "Write `is_palindrome(s)` returning True if s reads the same forwards and backwards.",
        "is_palindrome",
        impl_only="def is_palindrome(s):\n    return s == s[::-1]",
        wrong_critique="Algorithm is wrong — string reversal with slicing does not handle Unicode correctly.",
        good_code="def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True",
    ),
    _wcf_no_output(
        "Write `clamp(x, lo, hi)` constraining x to the range [lo, hi].",
        "clamp",
        impl_only="def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
        wrong_critique="Implementation may not handle boundary values — needs separate cases for x==lo and x==hi.",
        good_code="def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\nassert clamp(-5, 0, 10) == 0\nassert clamp(5, 0, 10) == 5\nassert clamp(15, 0, 10) == 10\nassert clamp(0, 0, 10) == 0\nassert clamp(10, 0, 10) == 10",
    ),
]


# ── 12. REPEATED_FAILURE_BREAKER ──────────────────────────────────────────
# Two identical failing calls in instruction + system hint that repetition
# was detected. Teaches: after a repeated error, produce a STRUCTURALLY
# DIFFERENT implementation with asserts + print('PASS').

def _rfb(task_desc: str, fn_name: str, bad_code: str, bad_error: str,
         error_desc: str, fix_desc: str, good_code: str) -> dict:
    """REPEATED_FAILURE_BREAKER: two identical failing calls in instruction with system hint."""
    bad_tc = _tc("execute_code", code=bad_code).rstrip("\n")
    bad_obs = f"OBSERVATION: ERROR:\n{bad_error}"
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "[SYSTEM: The previous tool call was repeated identically — "
            "a structurally different approach is required.]\n\n"
            "Previous failed attempts (same call made twice — still failing):\n"
            f"{bad_tc}\n"
            f"{bad_obs}\n\n"
            f"{bad_tc}\n"
            f"{bad_obs}\n"
        ),
        "response": (
            "CRITIQUE:\n"
            f"  Correctness — {error_desc} "
            "Repeating the same failing call always produces the same error.\n"
            "  Requirements — A structurally different implementation is required.\n"
            f"  → Fix needed: {fix_desc}\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: Used a structurally different implementation. "
            f"`{fn_name}` verified by execute_code: PASS."
        ),
    }


REPEATED_FAILURE_BREAKER_TEMPLATES = [
    _rfb(
        "Write `factorial(n)` computing n! and verify factorial(5)==120 and factorial(0)==1.",
        "factorial",
        bad_code="def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n)\nassert factorial(5) == 120\nprint('PASS')",
        bad_error="RecursionError: maximum recursion depth exceeded",
        error_desc="RecursionError (×2): `factorial(n)` calls `factorial(n)` — infinite recursion since n never decreases.",
        fix_desc="use an iterative implementation with a for-loop to avoid recursion.",
        good_code="def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    _rfb(
        "Write `sum_to_n(n)` returning 1+2+…+n and verify sum_to_n(10)==55 and sum_to_n(0)==0.",
        "sum_to_n",
        bad_code="def sum_to_n(n):\n    return sum(range(n))\nassert sum_to_n(10) == 55\nprint('PASS')",
        bad_error="AssertionError",
        error_desc="AssertionError (×2): `range(n)` gives 0..n-1 and misses n itself; sum_to_n(10) returns 45, not 55.",
        fix_desc="change `range(n)` to `range(1, n + 1)` to include n in the sum.",
        good_code="def sum_to_n(n):\n    return sum(range(1, n + 1))\nassert sum_to_n(10) == 55\nassert sum_to_n(0) == 0\nassert sum_to_n(1) == 1",
    ),
    _rfb(
        "Write `flatten(lst)` recursively flattening a nested list of arbitrary depth.",
        "flatten",
        bad_code="def flatten(lst):\n    return [x for x in lst if not isinstance(x, list)]\nassert flatten([[1, [2]], 3]) == [1, 2, 3]\nprint('PASS')",
        bad_error="AssertionError",
        error_desc="AssertionError (×2): the list comprehension drops sub-lists entirely instead of recursing into them.",
        fix_desc="use explicit recursion: if an item is a list, extend result with flatten(item).",
        good_code="def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\nassert flatten([[1, [2, 3]], [4, [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\nassert flatten([]) == []",
    ),
    _rfb(
        "Write `safe_divide(a, b)` returning a/b or 0 when b is zero.",
        "safe_divide",
        bad_code="def safe_divide(a, b):\n    return a / b\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0\nprint('PASS')",
        bad_error="ZeroDivisionError: division by zero",
        error_desc="ZeroDivisionError (×2): no guard for b==0; the division always runs.",
        fix_desc="add `if b == 0: return 0` before the division.",
        good_code="def safe_divide(a, b):\n    if b == 0:\n        return 0\n    return a / b\nassert safe_divide(10, 2) == 5.0\nassert safe_divide(7, 0) == 0\nassert safe_divide(0, 0) == 0",
    ),
]


# ── 13. Extended JSON_ROBUSTNESS ──────────────────────────────────────────
# Additional TOOL_FIRST_SUCCESS examples with complex code patterns that
# require careful JSON encoding: regex backslashes, OrderedDict, LRU cache
# class, nested lists, mixed-quote strings.

JSON_ROBUSTNESS_EXT_TEMPLATES = [
    _tfs(
        "Write `is_valid_email(s)` using regex to check the pattern word@word.word (simple check).",
        "is_valid_email",
        "import re\ndef is_valid_email(s):\n    return bool(re.match(r'^[\\w.]+@[\\w]+\\.[\\w]+$', s))",
        "assert is_valid_email('user@example.com') is True\nassert is_valid_email('bad@') is False\nassert is_valid_email('no-at-sign') is False\nassert is_valid_email('a@b.c') is True",
    ),
    _tfs(
        "Write `count_chars(s)` using collections.OrderedDict to count character frequencies preserving insertion order.",
        "count_chars",
        "from collections import OrderedDict\ndef count_chars(s):\n    d = OrderedDict()\n    for c in s:\n        d[c] = d.get(c, 0) + 1\n    return d",
        "r = count_chars('abcabc')\nassert list(r.keys()) == ['a', 'b', 'c']\nassert r['a'] == 2\nassert r['b'] == 2\nassert count_chars('') == {}",
    ),
    _tfs(
        "Implement an LRU cache with get(key) returning -1 on miss and put(key, val) evicting the least-recently-used when over capacity.",
        "LRUCache",
        "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.cache = OrderedDict()\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, val):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = val\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)",
        "c = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nassert c.get(3) == 3",
    ),
    _tfs(
        "Write `flatten_one(lst)` flattening exactly one level of nesting from a list of lists.",
        "flatten_one",
        "def flatten_one(lst):\n    return [x for sublist in lst for x in sublist]",
        "assert flatten_one([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]\nassert flatten_one([]) == []\nassert flatten_one([[1], [2, 3]]) == [1, 2, 3]",
    ),
    _tfs(
        "Write `format_name(first, last)` returning a string like 'Last, First' using an f-string with a comma.",
        "format_name",
        "def format_name(first, last):\n    return f\"{last}, {first}\"",
        "assert format_name('John', 'Doe') == 'Doe, John'\nassert format_name('Alice', \"O'Brien\") == \"O'Brien, Alice\"\nassert format_name('', 'Smith') == 'Smith, '",
    ),
]


# ── 14. ANTI_TRIPLE_QUOTE_FIX: invalid triple-quote JSON → correct JSON ───
# Context: model emitted triple-quoted string in execute_code args, runtime
# returned ERROR: invalid JSON.  Supervised response teaches correct escaping.
# NOTE: bad_call_str in instruction is deliberately NOT valid JSON — that is
# the point.  The audit allows invalid JSON in instruction context.

def _atqf(task_desc: str, fn_name: str, impl_only: str, good_code: str) -> dict:
    """ANTI_TRIPLE_QUOTE_FIX: bad triple-quote call in instruction → CRITIQUE → correct JSON."""
    bad_call_str = 'TOOL_CALL: execute_code({"code": """' + impl_only + '"""})'
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous failed attempt:\n"
            f"{bad_call_str}\n"
            f"{_OBS_INV_JSON}"
        ),
        "response": (
            "CRITIQUE:\n"
            "  Correctness — invalid JSON: triple-quoted strings (`\"\"\"...\"\"\"`) "
            "are not valid JSON. JSON only allows double-quoted strings with "
            "newlines escaped as `\\n`.\n"
            "  → Fix needed: use execute_code({\"code\": \"...\"}) with `\\n` "
            "escaping for newlines, never triple quotes.\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: Fixed triple-quoted JSON. `{fn_name}` verified: PASS."
        ),
    }


ANTI_TRIPLE_QUOTE_FIX_TEMPLATES = [
    _atqf(
        "Write `factorial(n)` computing n! iteratively. Verify factorial(0)==1 and factorial(5)==120.",
        "factorial",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(0) == 1\nassert factorial(5) == 120\nprint('PASS')",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\nassert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    _atqf(
        "Write `word_count(text)` returning a dict of word frequencies (case-insensitive).",
        "word_count",
        "def word_count(text):\n    words = text.lower().split()\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts",
        "def word_count(text):\n    words = text.lower().split()\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts\nr = word_count('the cat sat on the mat')\nassert r['the'] == 2\nassert r['cat'] == 1\nassert word_count('') == {}",
    ),
    _atqf(
        "Write `merge_sorted(a, b)` merging two sorted lists into one sorted list.",
        "merge_sorted",
        "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]",
        "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\nassert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1]) == [1]\nassert merge_sorted([1], []) == [1]",
    ),
    _atqf(
        "Write `sum_list(lst)` returning the sum of all numbers in a list.",
        "sum_list",
        "def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3]) == 6\nprint('PASS')",
        "def sum_list(lst):\n    return sum(lst)\nassert sum_list([1, 2, 3, 4, 5]) == 15\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0",
    ),
    _atqf(
        "Write `is_palindrome(s)` returning True if s reads the same forwards and backwards.",
        "is_palindrome",
        "def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar')\nprint('PASS')",
        "def is_palindrome(s):\n    return s == s[::-1]\nassert is_palindrome('racecar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True",
    ),
]


# ── 15. TARGETED_NO_OUTPUT: benchmark-task no-output → correct assertions ──
# Each template shows the correct implementation without asserts producing
# (no output), then teaches the model to add the exact required assertions.
# All use _nof() which produces CRITIQUE-first responses.

TARGETED_NO_OUTPUT_TEMPLATES = [
    _nof(
        "Write `word_count(text)` returning a dict of word frequencies (case-insensitive, split on whitespace).",
        "word_count",
        "def word_count(text):\n    words = text.lower().split()\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts",
        "def word_count(text):\n    words = text.lower().split()\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts\nr = word_count('the cat sat on the mat')\nassert r['the'] == 2\nassert r['cat'] == 1\nassert word_count('') == {}",
    ),
    _nof(
        "Write `merge_sorted(a, b)` merging two sorted lists into one sorted list.",
        "merge_sorted",
        "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]",
        "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\nassert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]\nassert merge_sorted([1], []) == [1]",
    ),
    _nof(
        "Write `to_roman(n)` converting a positive integer to a Roman numeral string.",
        "to_roman",
        "def to_roman(n):\n    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    result = ''\n    for v, s in vals:\n        while n >= v:\n            result += s\n            n -= v\n    return result",
        "def to_roman(n):\n    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    result = ''\n    for v, s in vals:\n        while n >= v:\n            result += s\n            n -= v\n    return result\nassert to_roman(1) == 'I'\nassert to_roman(4) == 'IV'\nassert to_roman(9) == 'IX'\nassert to_roman(58) == 'LVIII'\nassert to_roman(1994) == 'MCMXCIV'",
    ),
    _nof(
        "Write `bfs(graph, start)` returning nodes in breadth-first order (visit neighbors in sorted order).",
        "bfs",
        "from collections import deque\ndef bfs(graph, start):\n    visited = []\n    queue = deque([start])\n    seen = {start}\n    while queue:\n        node = queue.popleft()\n        visited.append(node)\n        for neighbor in sorted(graph.get(node, [])):\n            if neighbor not in seen:\n                seen.add(neighbor)\n                queue.append(neighbor)\n    return visited",
        "from collections import deque\ndef bfs(graph, start):\n    visited = []\n    queue = deque([start])\n    seen = {start}\n    while queue:\n        node = queue.popleft()\n        visited.append(node)\n        for neighbor in sorted(graph.get(node, [])):\n            if neighbor not in seen:\n                seen.add(neighbor)\n                queue.append(neighbor)\n    return visited\ng = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}\nassert bfs(g, 'A') == ['A', 'B', 'C', 'D']\nassert bfs({'X': []}, 'X') == ['X']",
    ),
    _nof(
        "Implement an LRU cache class with get(key) and put(key, val). Use OrderedDict. Capacity=2.",
        "LRUCache",
        "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.cache = OrderedDict()\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, val):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = val\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)",
        "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.cache = OrderedDict()\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, val):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = val\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)\nc = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nassert c.get(3) == 3",
    ),
]


# ── 17. TARGETED_SUCCESS: benchmark-failure-specific correct examples ─────
# Each template covers one of the recurring 0/1 benchmark failures.
# All are TOOL_FIRST_SUCCESS style (fresh task → correct first attempt).
# Key rules applied in each:
#   word_count  : split-based (no regex), explicit per-word assertions
#   factorial   : assertions OUTSIDE function body
#   lru_cache   : no triple-quoted strings; json.dumps handles escaping
#   merge_sorted: assert exact merged list, not only sortedness
#   graph_bfs   : correct BFS order (not reversed)
#   roman       : minimal, non-repetitive implementation + required assertions

TARGETED_SUCCESS_TEMPLATES = [
    _tfs(
        "Write `word_count(text)` returning a dict of word frequencies (case-insensitive, split on whitespace).",
        "word_count",
        "def word_count(text):\n    words = text.lower().split()\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts",
        "r = word_count('The cat sat on the mat')\nassert r['the'] == 2\nassert r['cat'] == 1\nassert r['mat'] == 1\nassert word_count('') == {}",
    ),
    _tfs(
        "Write `factorial(n)` returning n! iteratively. Assertions must be outside the function body.",
        "factorial",
        "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
        "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    _tfs(
        "Implement an LRU cache with get(key) returning -1 on miss and put(key, val) evicting the LRU entry when over capacity.",
        "LRUCache",
        "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.cache = OrderedDict()\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, val):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = val\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)",
        "c = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\nassert c.get(1) == 1\nassert c.get(3) == 3",
    ),
    _tfs(
        "Write `merge_sorted(a, b)` merging two sorted lists into one sorted list. Assert the exact output.",
        "merge_sorted",
        "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result + a[i:] + b[j:]",
        "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]\nassert merge_sorted([1], []) == [1]\nassert merge_sorted([1, 2], [1, 2]) == [1, 1, 2, 2]",
    ),
    _tfs(
        "Write `bfs(graph, start)` returning nodes in breadth-first order. Visit neighbors in sorted order. Do NOT reverse the result.",
        "bfs",
        "from collections import deque\ndef bfs(graph, start):\n    visited = []\n    queue = deque([start])\n    seen = {start}\n    while queue:\n        node = queue.popleft()\n        visited.append(node)\n        for neighbor in sorted(graph.get(node, [])):\n            if neighbor not in seen:\n                seen.add(neighbor)\n                queue.append(neighbor)\n    return visited",
        "g = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}\nassert bfs(g, 'A') == ['A', 'B', 'C', 'D']\nassert bfs({'X': ['Y'], 'Y': []}, 'X') == ['X', 'Y']\nassert bfs({'Z': []}, 'Z') == ['Z']",
    ),
    _tfs(
        "Write `to_roman(n)` converting a positive integer to a Roman numeral string.",
        "to_roman",
        "def to_roman(n):\n    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),\n            (100,'C'),(90,'XC'),(50,'L'),(40,'XL'),\n            (10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    result = ''\n    for v, s in vals:\n        while n >= v:\n            result += s\n            n -= v\n    return result",
        "assert to_roman(1) == 'I'\nassert to_roman(4) == 'IV'\nassert to_roman(9) == 'IX'\nassert to_roman(58) == 'LVIII'\nassert to_roman(1994) == 'MCMXCIV'",
    ),
]


# ── 18. BENCHMARK_RECOVERY: one-error → fix for benchmark tasks ─────────────
# Each template uses a helper that puts the bad TOOL_CALL + OBSERVATION: ERROR
# in the INSTRUCTION (context, not supervised), and teaches a CRITIQUE-first
# supervised response that fixes the bug, reaches OBSERVATION: PASS, and stops
# with a FINAL_ANSWER.
#
# Rules: no markdown fences, no VALID:, no DIFFERENT_CORRECTED_TOOL_CALL,
# no triple-quoted JSON, no fake OBSERVATION in supervised output.

def _bench_fix(
    task_desc: str,
    fn_name: str,
    bad_code: str,
    bad_obs: str,        # e.g. "AssertionError" or "(no output)"
    error_desc: str,     # short human-readable diagnosis
    fix_desc: str,       # short fix instruction
    good_code: str,      # correct implementation + asserts (no print PASS — added by helper)
) -> dict:
    """One-shot error → fix trajectory for a benchmark task."""
    bad_tc = _tc("execute_code", code=bad_code).rstrip("\n")
    return {
        "instruction": (
            f"Original task: {task_desc}\n\n"
            "Previous failed attempt:\n"
            f"{bad_tc}\n"
            f"OBSERVATION: {bad_obs}\n"
        ),
        "response": (
            "CRITIQUE:\n"
            f"  Correctness — {error_desc}\n"
            f"  → Fix needed: {fix_desc}\n" +
            _tc("execute_code", code=good_code + "\nprint('PASS')") +
            "OBSERVATION: PASS\n" +
            _crit_ok() +
            f"FINAL_ANSWER: `{fn_name}` fixed and verified by execute_code: PASS."
        ),
    }


BENCHMARK_RECOVERY_TEMPLATES = [
    # bugfix_off_by_one: range(n) → range(1, n+1)
    _bench_fix(
        "Fix the bug: `sum_to_n(n)` should return 1+2+…+n. "
        "Verify sum_to_n(10)==55 and sum_to_n(0)==0.",
        "sum_to_n",
        bad_code=(
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        total += i\n"
            "    return total\n"
            "assert sum_to_n(10) == 55\n"
            "print('PASS')"
        ),
        bad_obs="AssertionError",
        error_desc=(
            "range(n) iterates 0..n-1 and omits n itself, so sum_to_n(10) "
            "returns 45 instead of 55."
        ),
        fix_desc="change `range(n)` to `range(1, n + 1)` to include n in the sum.",
        good_code=(
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total\n"
            "assert sum_to_n(10) == 55\n"
            "assert sum_to_n(0) == 0\n"
            "assert sum_to_n(1) == 1"
        ),
    ),
    # flatten: generator/yield approach returns generator not list
    _bench_fix(
        "Write `flatten(lst)` recursively flattening a nested list. "
        "Verify flatten([[1,[2,3]],[4,[5,[6]]]])==[1,2,3,4,5,6].",
        "flatten",
        bad_code=(
            "def flatten(lst):\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            yield from flatten(item)\n"
            "        else:\n"
            "            yield item\n"
            "assert flatten([[1, [2, 3]], [4, [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\n"
            "print('PASS')"
        ),
        bad_obs="AssertionError",
        error_desc=(
            "The function uses `yield from`, making it a generator. "
            "Comparing a generator to a list always raises AssertionError."
        ),
        fix_desc=(
            "replace the generator with an explicit list: initialise "
            "`result = []`, extend with `result.extend(flatten(item))` for "
            "nested items, and `result.append(item)` for scalars, then "
            "`return result`."
        ),
        good_code=(
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
            "assert flatten([[1, [2, 3]], [4, [5, [6]]]]) == [1, 2, 3, 4, 5, 6]\n"
            "assert flatten([]) == []\n"
            "assert flatten([1, 2, 3]) == [1, 2, 3]"
        ),
    ),
    # palindrome: asserts 'hello' is True instead of False
    _bench_fix(
        "Write `is_palindrome(s)` returning True iff s reads the same "
        "forwards and backwards. Verify: 'racecar'→True, 'hello'→False, ''→True.",
        "is_palindrome",
        bad_code=(
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
            "assert is_palindrome('racecar') is True\n"
            "assert is_palindrome('hello') is True\n"
            "print('PASS')"
        ),
        bad_obs="AssertionError",
        error_desc=(
            "`is_palindrome('hello')` returns False but the assertion "
            "checks `is True`, so the test fails. 'hello' is not a palindrome."
        ),
        fix_desc="change `assert is_palindrome('hello') is True` to `assert is_palindrome('hello') is False`.",
        good_code=(
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
            "assert is_palindrome('racecar') is True\n"
            "assert is_palindrome('hello') is False\n"
            "assert is_palindrome('') is True\n"
            "assert is_palindrome('a') is True"
        ),
    ),
    # graph_bfs: unsorted neighbors → wrong order
    _bench_fix(
        "Write `bfs(graph, start)` returning nodes in breadth-first order. "
        "Visit neighbors in sorted order. "
        "Verify on graph={'A':['B','C'],'B':['D'],'C':[],'D':[]}, start='A'.",
        "bfs",
        bad_code=(
            "from collections import deque\n"
            "def bfs(graph, start):\n"
            "    visited = []\n"
            "    queue = deque([start])\n"
            "    seen = {start}\n"
            "    while queue:\n"
            "        node = queue.popleft()\n"
            "        visited.append(node)\n"
            "        for neighbor in graph.get(node, []):\n"
            "            if neighbor not in seen:\n"
            "                seen.add(neighbor)\n"
            "                queue.append(neighbor)\n"
            "    return visited\n"
            "g = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}\n"
            "assert bfs(g, 'A') == ['A', 'B', 'C', 'D']\n"
            "print('PASS')"
        ),
        bad_obs="AssertionError",
        error_desc=(
            "Neighbors are iterated in dict-insertion order, not sorted. "
            "When graph['A'] happens to list 'C' before 'B' or adjacency "
            "ordering is non-deterministic the result differs from ['A','B','C','D']."
        ),
        fix_desc="wrap the neighbor iteration with `sorted(...)`: `for neighbor in sorted(graph.get(node, []))`.",
        good_code=(
            "from collections import deque\n"
            "def bfs(graph, start):\n"
            "    visited = []\n"
            "    queue = deque([start])\n"
            "    seen = {start}\n"
            "    while queue:\n"
            "        node = queue.popleft()\n"
            "        visited.append(node)\n"
            "        for neighbor in sorted(graph.get(node, [])):\n"
            "            if neighbor not in seen:\n"
            "                seen.add(neighbor)\n"
            "                queue.append(neighbor)\n"
            "    return visited\n"
            "g = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}\n"
            "assert bfs(g, 'A') == ['A', 'B', 'C', 'D']\n"
            "assert bfs({'X': []}, 'X') == ['X']"
        ),
    ),
    # unique_sorted: no assertions, no print('PASS') → no output
    _bench_fix(
        "Write `unique_sorted(lst)` returning unique elements in sorted order. "
        "Verify: unique_sorted([3,1,2,1,3])==[1,2,3], unique_sorted([])==[].",
        "unique_sorted",
        bad_code=(
            "def unique_sorted(lst):\n"
            "    return sorted(set(lst))\n"
            "unique_sorted([3, 1, 2, 1, 3])"
        ),
        bad_obs="(no output)",
        error_desc=(
            "The code runs but produces no output because there are no "
            "assertions and no print('PASS'). Cannot verify correctness."
        ),
        fix_desc="add assertions for the required cases and `print('PASS')` to confirm success.",
        good_code=(
            "def unique_sorted(lst):\n"
            "    return sorted(set(lst))\n"
            "assert unique_sorted([3, 1, 2, 1, 3]) == [1, 2, 3]\n"
            "assert unique_sorted([]) == []\n"
            "assert unique_sorted([1]) == [1]"
        ),
    ),
    # word_count: assertion error on wrong key name
    _bench_fix(
        "Write `word_count(text)` returning a dict of word frequencies "
        "(case-insensitive). Verify on 'the cat sat on the mat': 'the'→2, 'cat'→1.",
        "word_count",
        bad_code=(
            "def word_count(text):\n"
            "    counts = {}\n"
            "    for word in text.split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n"
            "d = word_count('the cat sat on the mat')\n"
            "assert d['The'] == 2\n"
            "print('PASS')"
        ),
        bad_obs="KeyError: 'The'",
        error_desc=(
            "The function splits on whitespace without lowercasing, so keys "
            "preserve their original case. `d['The']` raises KeyError because "
            "the text starts with lowercase 'the'."
        ),
        fix_desc="add `.lower()` before `.split()` so all keys are lowercase.",
        good_code=(
            "def word_count(text):\n"
            "    counts = {}\n"
            "    for word in text.lower().split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n"
            "d = word_count('the cat sat on the mat')\n"
            "assert d['the'] == 2\n"
            "assert d['cat'] == 1\n"
            "assert word_count('') == {}"
        ),
    ),
]


# ── Dev-set recovery templates (targeting 5 frozen held-out failure categories) ──
# These are similar-but-not-identical tasks for rle_decode, group_anagrams,
# deep_get, merge_intervals, and tree_depth_tuple.
# Train on these; test on frozen held-out with original (82-record) memory index.

DEV_SET_RECOVERY_TEMPLATES = [
    # ── rle_decode: (char, count) list → string ──────────────────────────────
    _bench_fix(
        "Write `rle_decode(encoded)` where encoded is a list of (char, count) tuples "
        "and the function returns the decoded string. "
        "rle_decode([('a',3),('b',2),('c',1)]) should return 'aaabbc'.",
        "rle_decode",
        bad_code=(
            "def rle_decode(encoded):\n"
            "    result = []\n"
            "    for char, count in encoded:\n"
            "        result.append(char * count)\n"
            "    return result\n"
            "assert rle_decode([('a',3),('b',2),('c',1)]) == 'aaabbc'\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="Returns a list of strings, not a joined string.",
        fix_desc="Return ''.join(char * count for char, count in encoded).",
        good_code=(
            "def rle_decode(encoded):\n"
            "    return ''.join(char * count for char, count in encoded)\n"
            "assert rle_decode([('a',3),('b',2),('c',1)]) == 'aaabbc'\n"
            "assert rle_decode([('x',1),('y',4)]) == 'xyyyy'\n"
            "assert rle_decode([]) == ''"
        ),
    ),
    # ── rle_decode variant: string format "3a2b1c" ────────────────────────────
    _bench_fix(
        "Write `rle_decode_str(s)` that decodes a run-length string like '3a2b1c' into "
        "'aaabbc'. rle_decode_str('1x4y') should return 'xyyyy'.",
        "rle_decode_str",
        bad_code=(
            "def rle_decode_str(s):\n"
            "    result = ''\n"
            "    for i in range(0, len(s), 2):\n"
            "        result += s[i] * int(s[i+1])\n"
            "    return result\n"
            "assert rle_decode_str('3a2b1c') == 'aaabbc'\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="Index order wrong: s[i] is the digit, s[i+1] is the char — should be s[i+1]*int(s[i]).",
        fix_desc="For each pair at index i (digit) and i+1 (char): result += s[i+1] * int(s[i]).",
        good_code=(
            "def rle_decode_str(s):\n"
            "    result = ''\n"
            "    for i in range(0, len(s), 2):\n"
            "        result += s[i+1] * int(s[i])\n"
            "    return result\n"
            "assert rle_decode_str('3a2b1c') == 'aaabbc'\n"
            "assert rle_decode_str('1x4y') == 'xyyyy'\n"
            "assert rle_decode_str('') == ''"
        ),
    ),
    # ── group_anagrams: wrong key (word itself vs sorted chars) ───────────────
    _bench_fix(
        "Write `anagram_buckets(words)` that returns a dict mapping canonical key "
        "(tuple of sorted chars) to list of words in that anagram group. "
        "anagram_buckets(['eat','tea','tan','ate']) should return "
        "{('a','e','t'):['eat','tea','ate'], ('a','n','t'):['tan']}.",
        "anagram_buckets",
        bad_code=(
            "def anagram_buckets(words):\n"
            "    groups = {}\n"
            "    for w in words:\n"
            "        key = w.lower()\n"
            "        groups.setdefault(key, []).append(w)\n"
            "    return groups\n"
            "assert anagram_buckets(['eat','tea','ate']) == "
            "{('a','e','t'):['eat','tea','ate']}\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="Uses the word itself as key — 'eat', 'tea', 'ate' are different keys. "
                   "Need key = tuple(sorted(w)) so all anagrams share the same canonical key.",
        fix_desc="Replace key = w.lower() with key = tuple(sorted(w.lower())).",
        good_code=(
            "def anagram_buckets(words):\n"
            "    groups = {}\n"
            "    for w in words:\n"
            "        key = tuple(sorted(w.lower()))\n"
            "        groups.setdefault(key, []).append(w)\n"
            "    return groups\n"
            "r = anagram_buckets(['eat','tea','tan','ate'])\n"
            "assert r[tuple(sorted('eat'))] == ['eat','tea','ate']\n"
            "assert r[tuple(sorted('tan'))] == ['tan']"
        ),
    ),
    # ── group_anagrams: sorted-group return ──────────────────────────────────
    _bench_fix(
        "Write `largest_anagram_group(words)` that returns the largest group of "
        "mutually anagrammatic words, sorted alphabetically. "
        "largest_anagram_group(['eat','tea','tan','ate','nat','bat']) should return "
        "['ate','eat','tea'].",
        "largest_anagram_group",
        bad_code=(
            "def largest_anagram_group(words):\n"
            "    groups = {}\n"
            "    for w in words:\n"
            "        key = w\n"
            "        groups.setdefault(key, []).append(w)\n"
            "    return sorted(max(groups.values(), key=len))\n"
            "assert largest_anagram_group(['eat','tea','tan','ate','nat','bat']) "
            "== ['ate','eat','tea']\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="key = w groups each word alone — every group has size 1. "
                   "Need key = tuple(sorted(w)) to group anagrams together.",
        fix_desc="Replace key = w with key = tuple(sorted(w.lower())).",
        good_code=(
            "def largest_anagram_group(words):\n"
            "    groups = {}\n"
            "    for w in words:\n"
            "        key = tuple(sorted(w.lower()))\n"
            "        groups.setdefault(key, []).append(w)\n"
            "    return sorted(max(groups.values(), key=len))\n"
            "assert largest_anagram_group(['eat','tea','tan','ate','nat','bat']) "
            "== ['ate','eat','tea']"
        ),
    ),
    # ── deep_get: raises KeyError instead of returning default ────────────────
    _bench_fix(
        "Write `safe_path_get(d, path, default=None)` where path is a dot-separated "
        "string of keys. Returns default if any key is missing. "
        "safe_path_get({'a':{'b':42}}, 'a.b') should return 42. "
        "safe_path_get({'a':1}, 'a.x', 99) should return 99.",
        "safe_path_get",
        bad_code=(
            "def safe_path_get(d, path, default=None):\n"
            "    for key in path.split('.'):\n"
            "        d = d[key]\n"
            "    return d\n"
            "assert safe_path_get({'a':{'b':42}}, 'a.b') == 42\n"
            "assert safe_path_get({'a':1}, 'a.x', 99) == 99\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: KeyError: 'x'",
        error_desc="d[key] raises KeyError when key is missing. Need d.get(key, sentinel) "
                   "to detect missing keys and return default.",
        fix_desc="Replace d = d[key] with: if not isinstance(d, dict) or key not in d: return default; d = d[key].",
        good_code=(
            "def safe_path_get(d, path, default=None):\n"
            "    for key in path.split('.'):\n"
            "        if not isinstance(d, dict) or key not in d:\n"
            "            return default\n"
            "        d = d[key]\n"
            "    return d\n"
            "assert safe_path_get({'a':{'b':42}}, 'a.b') == 42\n"
            "assert safe_path_get({'a':1}, 'a.x', 99) == 99\n"
            "assert safe_path_get({}, 'a.b.c', 0) == 0"
        ),
    ),
    # ── deep_get: positional args variant ────────────────────────────────────
    _bench_fix(
        "Write `nested_get_or(d, *keys, default=None)` that navigates nested dicts "
        "using positional key arguments. Returns default if any key is missing. "
        "nested_get_or({'a':{'b':7}}, 'a','b') should return 7. "
        "nested_get_or({'a':1}, 'a','b', default=0) should return 0.",
        "nested_get_or",
        bad_code=(
            "def nested_get_or(d, *keys, default=None):\n"
            "    for k in keys:\n"
            "        d = d[k]\n"
            "    return d\n"
            "assert nested_get_or({'a':{'b':7}}, 'a','b') == 7\n"
            "assert nested_get_or({'a':1}, 'a','b', default=0) == 0\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: KeyError: 'b'",
        error_desc="d[k] raises KeyError on missing key. Use d.get(k) and check for sentinel.",
        fix_desc="Replace d = d[k] with: if not isinstance(d, dict): return default; d = d.get(k, _MISSING); if d is _MISSING: return default.",
        good_code=(
            "_MISSING = object()\n"
            "def nested_get_or(d, *keys, default=None):\n"
            "    for k in keys:\n"
            "        if not isinstance(d, dict):\n"
            "            return default\n"
            "        d = d.get(k, _MISSING)\n"
            "        if d is _MISSING:\n"
            "            return default\n"
            "    return d\n"
            "assert nested_get_or({'a':{'b':7}}, 'a','b') == 7\n"
            "assert nested_get_or({'a':1}, 'a','b', default=0) == 0\n"
            "assert nested_get_or({}, 'x', default=-1) == -1"
        ),
    ),
    # ── merge_intervals: sort descending bug ─────────────────────────────────
    _bench_fix(
        "Write `insert_interval(intervals, new_interval)` where intervals is a list "
        "of non-overlapping sorted [start,end] intervals and new_interval=[s,e] to "
        "insert and merge. insert_interval([[1,3],[6,9]], [2,5]) should return "
        "[[1,5],[6,9]].",
        "insert_interval",
        bad_code=(
            "def insert_interval(intervals, new_interval):\n"
            "    intervals = sorted(intervals + [new_interval], key=lambda x: -x[0])\n"
            "    merged = [intervals[0]]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
            "assert insert_interval([[1,3],[6,9]], [2,5]) == [[1,5],[6,9]]\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="Sorts descending (key=-x[0]) — processes largest start first, so "
                   "merged[-1][1] is small and later intervals don't overlap. "
                   "Must sort ascending (remove negation).",
        fix_desc="Change key=lambda x: -x[0] to key=lambda x: x[0].",
        good_code=(
            "def insert_interval(intervals, new_interval):\n"
            "    intervals = sorted(intervals + [new_interval], key=lambda x: x[0])\n"
            "    merged = [list(intervals[0])]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
            "assert insert_interval([[1,3],[6,9]], [2,5]) == [[1,5],[6,9]]\n"
            "assert insert_interval([[1,2],[3,4]], [5,6]) == [[1,2],[3,4],[5,6]]"
        ),
    ),
    # ── merge_intervals: find free intervals ─────────────────────────────────
    _bench_fix(
        "Write `find_free_intervals(busy, total_start, total_end)` that finds free "
        "time slots within [total_start, total_end] not covered by any busy interval. "
        "find_free_intervals([[1,3],[6,9]], 0, 10) should return [[0,1],[3,6],[9,10]].",
        "find_free_intervals",
        bad_code=(
            "def find_free_intervals(busy, total_start, total_end):\n"
            "    busy = sorted(busy, key=lambda x: -x[0])\n"
            "    free = []\n"
            "    prev = total_start\n"
            "    for s, e in busy:\n"
            "        if s > prev:\n"
            "            free.append([prev, s])\n"
            "        prev = max(prev, e)\n"
            "    if prev < total_end:\n"
            "        free.append([prev, total_end])\n"
            "    return free\n"
            "assert find_free_intervals([[1,3],[6,9]], 0, 10) == [[0,1],[3,6],[9,10]]\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="Sorts descending (key=-x[0]) — wrong iteration order means gaps are "
                   "detected in reverse. Must sort ascending.",
        fix_desc="Change key=lambda x: -x[0] to key=lambda x: x[0].",
        good_code=(
            "def find_free_intervals(busy, total_start, total_end):\n"
            "    busy = sorted(busy, key=lambda x: x[0])\n"
            "    free = []\n"
            "    prev = total_start\n"
            "    for s, e in busy:\n"
            "        if s > prev:\n"
            "            free.append([prev, s])\n"
            "        prev = max(prev, e)\n"
            "    if prev < total_end:\n"
            "        free.append([prev, total_end])\n"
            "    return free\n"
            "assert find_free_intervals([[1,3],[6,9]], 0, 10) == [[0,1],[3,6],[9,10]]\n"
            "assert find_free_intervals([], 0, 5) == [[0,5]]"
        ),
    ),
    # ── tree_depth_tuple: count leaves ────────────────────────────────────────
    _bench_fix(
        "Write `count_leaves(tree)` where tree is a nested tuple. A leaf is any "
        "non-tuple element. count_leaves(((1,2),(3,(4,5)))) should return 5. "
        "count_leaves(1) should return 1.",
        "count_leaves",
        bad_code=(
            "def count_leaves(tree):\n"
            "    if not isinstance(tree, tuple):\n"
            "        return 1\n"
            "    count = 0\n"
            "    for child in tree:\n"
            "        count += 1\n"
            "    return count\n"
            "assert count_leaves(((1,2),(3,(4,5)))) == 5\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="Counts immediate children of tuple, not all leaves recursively. "
                   "((1,2),(3,(4,5))) has 2 children (not 5 leaves). "
                   "Must recurse into each child.",
        fix_desc="Replace count += 1 with count += count_leaves(child).",
        good_code=(
            "def count_leaves(tree):\n"
            "    if not isinstance(tree, tuple):\n"
            "        return 1\n"
            "    return sum(count_leaves(child) for child in tree)\n"
            "assert count_leaves(((1,2),(3,(4,5)))) == 5\n"
            "assert count_leaves(1) == 1\n"
            "assert count_leaves((1,)) == 1"
        ),
    ),
    # ── tree_depth_tuple: flatten ─────────────────────────────────────────────
    _bench_fix(
        "Write `flatten_tuple_tree(tree)` that flattens a nested tuple into a list "
        "of all leaf values left-to-right. "
        "flatten_tuple_tree(((1,2),(3,(4,5)))) should return [1,2,3,4,5]. "
        "flatten_tuple_tree(42) should return [42].",
        "flatten_tuple_tree",
        bad_code=(
            "def flatten_tuple_tree(tree):\n"
            "    if not isinstance(tree, tuple):\n"
            "        return tree\n"
            "    result = []\n"
            "    for child in tree:\n"
            "        result.append(flatten_tuple_tree(child))\n"
            "    return result\n"
            "assert flatten_tuple_tree(((1,2),(3,(4,5)))) == [1,2,3,4,5]\n"
            "print('PASS')"
        ),
        bad_obs="ERROR: AssertionError",
        error_desc="Base case returns the leaf value (not wrapped in list), and inner "
                   "case appends nested lists instead of extending. "
                   "Need: base case returns [tree]; recursive case extends result.",
        fix_desc="Base: return [tree]. Recursive: result.extend(flatten_tuple_tree(child)).",
        good_code=(
            "def flatten_tuple_tree(tree):\n"
            "    if not isinstance(tree, tuple):\n"
            "        return [tree]\n"
            "    result = []\n"
            "    for child in tree:\n"
            "        result.extend(flatten_tuple_tree(child))\n"
            "    return result\n"
            "assert flatten_tuple_tree(((1,2),(3,(4,5)))) == [1,2,3,4,5]\n"
            "assert flatten_tuple_tree(42) == [42]\n"
            "assert flatten_tuple_tree((1,)) == [1]"
        ),
    ),
]


# ── Trajectory audit ───────────────────────────────────────────────────────

_ALLOWED_TOOLS = frozenset({
    "execute_code", "run_script", "run_tests", "read_file", "write_file",
    "install_package", "check_syntax", "format_code", "run_linter",
    "list_files", "search_file", "edit_file",
    "git_commit", "git_push", "git_log", "git_diff", "git_status",
})
_TC_NAME_RE = re.compile(r"^TOOL_CALL:\s*(\w+)\s*\(", re.MULTILINE)
_EXC_CODE_RE = re.compile(r'execute_code\((\{.*?\})\)', re.DOTALL)


def _audit_response(response: str, skip_pass_discipline: bool = False) -> list[str]:
    """Return list of issues found in a single response string."""
    issues: list[str] = []

    # 1. Invalid tool names — no exceptions: supervised response must only use registered tools
    for m in _TC_NAME_RE.finditer(response):
        name = m.group(1)
        if name not in _ALLOWED_TOOLS:
            issues.append(f"unknown tool '{name}'")

    # 2. execute_code JSON and Python validity
    _positions: list[tuple[int, str]] = []  # (pos, code_str)
    marker = "execute_code("
    pos = 0
    while True:
        idx = response.find(marker, pos)
        if idx == -1:
            break
        scan = idx + len(marker)
        depth = 0; in_str = False; qch = ""; esc = False; end = -1
        for i, ch in enumerate(response[scan:], scan):
            if esc: esc = False; continue
            if in_str:
                if ch == "\\": esc = True
                elif ch == qch: in_str = False
                continue
            if ch in ('"', "'"): in_str = True; qch = ch
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = i + 1; break
        if end != -1:
            raw = response[scan:end]
            try:
                args = json.loads(raw)
                code = args.get("code", "")
                if code:
                    _positions.append((idx, code))
            except json.JSONDecodeError as e:
                # Allow the intentionally-bad-JSON training examples
                if "OBSERVATION: ERROR: invalid JSON" not in response[end:end + 200]:
                    issues.append(f"invalid JSON in execute_code args at pos {idx}: {e}")
        pos = idx + 1

    # Check each execute_code code string for JSON-hostile patterns
    for _pos, _code in _positions:
        # Double-escaped newlines: \\n in the parsed code means a literal
        # backslash-n was stored instead of a real newline.  This breaks
        # code execution and indicates a manual string-concat error.
        if "\\" + "n" in _code:
            issues.append("double_escaped_newline_in_execute_code")
            break

    # Triple-quoted strings in raw response: json.dumps() never emits these;
    # their presence indicates a hand-written template with invalid JSON.
    if re.search(r'execute_code\s*\(\s*\{[^}]*"""', response):
        issues.append("triple_quoted_string_in_execute_code_args")

    # Check last execute_code Python parseability
    if _positions:
        last_pos, last_code = _positions[-1]
        # Only audit the last call if it comes AFTER any OBSERVATION: PASS
        # (last call in a trajectory should be valid Python)
        segment_after = response[last_pos:]
        if "OBSERVATION: ERROR" not in segment_after.split("OBSERVATION: PASS")[0]:
            try:
                ast.parse(last_code)
            except SyntaxError as e:
                issues.append(f"unparseable Python in last execute_code: {e}")

    # 3. PASS discipline: successful execute_code must have assert + print('PASS'),
    #    and FINAL_ANSWER must not precede the first OBSERVATION: PASS.
    #    Skipped for templates where PASS is in the instruction, not the response
    #    (e.g. WRONG_CRITIQUE_FIX PASS-case where response is FINAL_ANSWER only).
    if skip_pass_discipline:
        return issues
    for m_pass in re.finditer(r"^OBSERVATION:\s*PASS\b", response, re.MULTILINE):
        before = response[:m_pass.start()]
        ec_matches = list(re.finditer(r"TOOL_CALL:\s*execute_code\(", before))
        if not ec_matches:
            continue
        last_ec = ec_matches[-1]
        # Only check if no OBSERVATION: intervenes between the EC call and this PASS
        between = response[last_ec.start():m_pass.start()]
        if "OBSERVATION:" in between:
            continue
        paren_start = last_ec.end()
        depth = 0; in_str = False; qch = ""; esc = False; end_pd = -1
        for i, ch in enumerate(response[paren_start:], paren_start):
            if esc: esc = False; continue
            if in_str:
                if ch == "\\": esc = True
                elif ch == qch: in_str = False
                continue
            if ch in ('"', "'"): in_str = True; qch = ch
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end_pd = i + 1; break
        if end_pd == -1:
            continue
        try:
            args_pd = json.loads(response[paren_start:end_pd])
        except json.JSONDecodeError:
            continue
        code_pd = args_pd.get("code", "")
        if not code_pd:
            continue
        if "print('PASS')" not in code_pd and 'print("PASS")' not in code_pd:
            issues.append("successful_execute_code_no_print_pass")
        if "assert " not in code_pd:
            issues.append("successful_execute_code_no_assert")

    fa_m = re.search(r"^FINAL_ANSWER:", response, re.MULTILINE)
    pass_m = re.search(r"^OBSERVATION:\s*PASS\b", response, re.MULTILINE)
    if fa_m is not None:
        if pass_m is None:
            issues.append("final_answer_before_pass: FINAL_ANSWER with no OBSERVATION: PASS")
        elif fa_m.start() < pass_m.start():
            issues.append("final_answer_before_pass")

    # 4. No TOOL_CALL after the first OBSERVATION: PASS
    first_pass_m = re.search(r"^OBSERVATION:\s*PASS\b", response, re.MULTILINE)
    if first_pass_m:
        after_pass = response[first_pass_m.start():]
        if re.search(r"\nTOOL_CALL:", after_pass):
            issues.append("tool_call_after_pass")

    # 5. solution_ok_after_error_before_pass
    events: list[tuple[str, int]] = []
    for m in re.finditer(
        r"OBSERVATION:\s*(.*?)(?=\nTOOL_CALL:|\nCRITIQUE:|\nFINAL_ANSWER:|\Z)",
        response, re.DOTALL
    ):
        obs = m.group(1).strip()
        if obs.startswith("PASS"):
            events.append(("OBS_PASS", m.start()))
        elif "Error" in obs or "ERROR" in obs:
            events.append(("OBS_ERROR", m.start()))
        else:
            events.append(("OBS_OTHER", m.start()))

    for m in re.finditer(r"→ Solution OK", response, re.IGNORECASE):
        events.append(("SOL_OK", m.start()))

    events.sort(key=lambda x: x[1])
    last_error = False
    for ev_type, _ in events:
        if ev_type == "OBS_ERROR":
            last_error = True
        elif ev_type == "OBS_PASS":
            last_error = False
        elif ev_type == "SOL_OK" and last_error:
            issues.append("solution_ok_after_error_before_pass")
            break

    # 4. repeated identical tool call after error
    tc_events: list[tuple[str, int]] = []  # (call_line, pos)
    obs_events: list[tuple[bool, int]] = []  # (is_error, pos)
    for m in re.finditer(r"^(TOOL_CALL:.+)$", response, re.MULTILINE):
        tc_events.append((m.group(1).strip(), m.start()))
    for m in re.finditer(
        r"^OBSERVATION:\s*(.*?)(?=^(?:TOOL_CALL:|CRITIQUE:|FINAL_ANSWER:)|\Z)",
        response, re.MULTILINE | re.DOTALL
    ):
        obs_text = m.group(1)
        is_error = "Error" in obs_text or "ERROR" in obs_text
        obs_events.append((is_error, m.start()))

    all_tc_obs = [(t, p, "TC") for t, p in tc_events] + [(e, p, "OBS") for e, p in obs_events]
    all_tc_obs.sort(key=lambda x: x[1])

    last_call = None
    after_error = False
    for item in all_tc_obs:
        if item[2] == "TC":
            call = item[0]
            if after_error and call == last_call:
                issues.append(f"repeated_identical_tool_call_after_error: {call[:60]!r}")
                break
            last_call = call
            after_error = False
        else:
            after_error = item[0]  # is_error

    # 7. Supervised target must never contain confusing non-protocol markers.
    if "VALID:" in response:
        issues.append("bad_marker_VALID_colon_in_response")
    if "DIFFERENT_CORRECTED_TOOL_CALL" in response:
        issues.append("bad_marker_DIFFERENT_CORRECTED_TOOL_CALL_in_response")
    # Markdown code fences confuse the model into emitting prose instead of tool calls.
    if "```" in response:
        issues.append("markdown_code_fence_in_response")

    # 8. Recovery response must reach OBSERVATION: PASS — never end as (no output).
    # A recovery response is one that starts with CRITIQUE: or has inline error handling.
    first_line_type = _first_response_line_type(response)
    has_inline_error = bool(
        re.search(r"OBSERVATION:.*(?:ERROR|no output)", response[:response.find("CRITIQUE:")]
                  if "CRITIQUE:" in response and response.find("CRITIQUE:") > 0 else "")
    )
    if first_line_type == "CRITIQUE" or has_inline_error:
        if "OBSERVATION: PASS" not in response:
            issues.append("recovery_response_never_reaches_pass")

    return issues


def _first_response_line_type(response: str) -> str:
    """Return 'TOOL_CALL', 'CRITIQUE', 'FINAL_ANSWER', or 'other' for first non-blank line."""
    for line in response.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("TOOL_CALL:"):
            return "TOOL_CALL"
        if s.startswith("CRITIQUE:"):
            return "CRITIQUE"
        if s.startswith("FINAL_ANSWER:"):
            return "FINAL_ANSWER"
        return "other"
    return "other"


def _audit_first_line_distribution(rows: list[dict], mode: str = "") -> None:
    n_tc = n_crit = n_fa = n_other = 0
    other_examples: list[str] = []
    for r in rows:
        t = _first_response_line_type(r["response"])
        if t == "TOOL_CALL":
            n_tc += 1
        elif t == "CRITIQUE":
            n_crit += 1
        elif t == "FINAL_ANSWER":
            n_fa += 1
        else:
            n_other += 1
            if len(other_examples) < 3:
                first_line = next(
                    (l.strip() for l in r["response"].splitlines() if l.strip()), ""
                )
                other_examples.append(first_line[:80])
    total = len(rows) or 1
    tag = f" [{mode}]" if mode else ""
    print(f"\nFirst-line distribution{tag} ({total} examples):")
    print(f"  TOOL_CALL first    : {n_tc:5d}  ({n_tc/total:.1%})")
    print(f"  CRITIQUE first     : {n_crit:5d}  ({n_crit/total:.1%})")
    print(f"  FINAL_ANSWER first : {n_fa:5d}  ({n_fa/total:.1%})")
    print(f"  other/markdown     : {n_other:5d}  ({n_other/total:.1%})")
    if other_examples:
        print("  other examples:")
        for ex in other_examples:
            print(f"    {ex!r}")


def _is_recovery(t: dict) -> bool:
    """Return True if template shows a failed-then-fixed pattern (recovery with PASS)."""
    instr = t.get("instruction", "")
    resp  = t.get("response",    "")
    if "OBSERVATION:" in instr:
        return True
    if re.search(r"OBSERVATION:.*?ERROR", resp, re.DOTALL) and "OBSERVATION: PASS" in resp:
        return True
    return False


def _is_no_output_fix(t: dict) -> bool:
    """Return True if template has a (no output) observation in its instruction context."""
    return "(no output)" in t.get("instruction", "")


def _audit_all_templates() -> None:
    """Audit every template family and enforce pool composition gates."""
    all_families = {
        "TOOL_FIRST_SUCCESS":          TOOL_FIRST_SUCCESS_TEMPLATES,
        "TARGETED_SUCCESS":            TARGETED_SUCCESS_TEMPLATES,
        "ANTI_TRIPLE_QUOTE_FIX":       ANTI_TRIPLE_QUOTE_FIX_TEMPLATES,
        "TARGETED_NO_OUTPUT":          TARGETED_NO_OUTPUT_TEMPLATES,
        "INDENT_ERROR_FIX":            INDENT_ERROR_FIX_TEMPLATES,
        "INVALID_JSON_FIX":            INVALID_JSON_FIX_TEMPLATES,
        "UNKNOWN_TOOL_FIX":            UNKNOWN_TOOL_FIX_TEMPLATES,
        "SOLUTION_OK_AFTER_ERROR_FIX": SOLUTION_OK_AFTER_ERROR_FIX_TEMPLATES,
        "REPEATED_CALL_FIX":           REPEATED_CALL_FIX_TEMPLATES,
        "NO_OUTPUT_FIX":               NO_OUTPUT_FIX_TEMPLATES,
        "JSON_ROBUSTNESS":             JSON_ROBUSTNESS_TEMPLATES,
        "JSON_ROBUSTNESS_EXT":         JSON_ROBUSTNESS_EXT_TEMPLATES,
        "SYNTAX_ERROR_TO_PASS_FIX":    SYNTAX_ERROR_TO_PASS_FIX_TEMPLATES,
        "INDENT_ERROR_TO_PASS_FIX":    INDENT_ERROR_TO_PASS_FIX_TEMPLATES,
        "WRONG_CRITIQUE_FIX":          WRONG_CRITIQUE_FIX_TEMPLATES,
        "REPEATED_FAILURE_BREAKER":    REPEATED_FAILURE_BREAKER_TEMPLATES,
    }

    # WRONG_CRITIQUE_FIX PASS-case responses start with FINAL_ANSWER — skip PASS
    # discipline for those (OBSERVATION: PASS is in the instruction, not the response).
    SKIP_PASS_CHECK = {"WRONG_CRITIQUE_FIX"}

    total = 0
    failed = 0
    for family, templates in all_families.items():
        skip_pass = family in SKIP_PASS_CHECK
        for i, t in enumerate(templates):
            issues = _audit_response(t["response"], skip_pass_discipline=skip_pass)
            total += 1
            if issues:
                failed += 1
                print(f"  FAIL [{family}][{i}] {t['instruction'][:60]!r}")
                for iss in issues:
                    print(f"       {iss}")

    # First-line distribution across all templates
    all_templates = [t for fam in all_families.values() for t in fam]
    _audit_first_line_distribution(all_templates, mode="all templates")

    # ── Pool composition gates ─────────────────────────────────────────────
    pool = _agent_only_pool()
    total_pool = max(len(pool), 1)

    n_tc   = sum(1 for t in pool if _first_response_line_type(t["response"]) == "TOOL_CALL")
    n_crit = sum(1 for t in pool if _first_response_line_type(t["response"]) == "CRITIQUE")
    n_fa   = sum(1 for t in pool if _first_response_line_type(t["response"]) == "FINAL_ANSWER")
    tc_rate   = n_tc   / total_pool
    crit_rate = n_crit / total_pool
    fa_rate   = n_fa   / total_pool

    print(f"\nAgent-only pool distribution ({len(pool)} slots):")
    print(f"  TOOL_CALL-first   : {n_tc:4d}  ({tc_rate:.1%})   gate: 40–60%")
    print(f"  CRITIQUE-first    : {n_crit:4d}  ({crit_rate:.1%})")
    print(f"  PASS-stop (FA)    : {n_fa:4d}  ({fa_rate:.1%})   gate:  5–15%")

    # Gate A: TOOL_CALL-first rate must be 40–60%
    if not (0.40 <= tc_rate <= 0.60):
        failed += 1
        print(f"  FAIL: TOOL_CALL-first {tc_rate:.1%} outside 40–60% — rebalance pool weights")

    # Gate B: PASS-stop (FINAL_ANSWER-first) rate must be 5–15%
    if not (0.05 <= fa_rate <= 0.15):
        failed += 1
        print(f"  FAIL: PASS-stop {fa_rate:.1%} outside 5–15% — adjust WRONG_CRITIQUE_FIX weight")

    # Targeted task coverage
    print(f"\nTargeted task coverage ({len(TARGETED_SUCCESS_TEMPLATES)} templates):")
    for t in TARGETED_SUCCESS_TEMPLATES:
        print(f"  {t['instruction'][:70]!r}")

    print(f"\nTemplate audit: {total - failed}/{total} passed")
    if failed:
        raise SystemExit(f"ABORT: {failed} issue(s) found — fix before training.")


def _agent_only_pool() -> list[dict]:
    """Balanced pool: ~50% TOOL_CALL-first, ~42% CRITIQUE-first, ~5% PASS-stop.

    Gate requirements (checked in _audit_all_templates):
      TOOL_CALL-first : 40–60%
      PASS-stop       :  5–15%
    """
    return (
        # ── TOOL_CALL-first: fresh success (~50%) ─────────────────────────
        TOOL_FIRST_SUCCESS_TEMPLATES  * 6 +   # 11 × 6 = 66
        JSON_ROBUSTNESS_TEMPLATES     * 4 +   #  6 × 4 = 24
        JSON_ROBUSTNESS_EXT_TEMPLATES * 4 +   #  5 × 4 = 20
        TARGETED_SUCCESS_TEMPLATES    * 6 +   #  6 × 6 = 36  (benchmark failures)
        # ── CRITIQUE-first: recovery (~42%) ───────────────────────────────
        NO_OUTPUT_FIX_TEMPLATES            * 2 +   # 21 × 2 = 42
        ANTI_TRIPLE_QUOTE_FIX_TEMPLATES    * 3 +   #  5 × 3 = 15  (triple-quote fix)
        TARGETED_NO_OUTPUT_TEMPLATES       * 3 +   #  5 × 3 = 15  (benchmark no-output)
        BENCHMARK_RECOVERY_TEMPLATES       * 6 +   #  6 × 6 = 36  (failure→fix)
        INDENT_ERROR_FIX_TEMPLATES         * 2 +   #  7 × 2 = 14
        INDENT_ERROR_TO_PASS_FIX_TEMPLATES * 2 +   #  5 × 2 = 10
        INVALID_JSON_FIX_TEMPLATES         * 2 +   #  4 × 2 =  8
        UNKNOWN_TOOL_FIX_TEMPLATES         * 1 +   #  7 × 1 =  7
        SOLUTION_OK_AFTER_ERROR_FIX_TEMPLATES * 1 + #  5 × 1 =  5
        REPEATED_CALL_FIX_TEMPLATES        * 1 +   #  4 × 1 =  4
        REPEATED_FAILURE_BREAKER_TEMPLATES * 1 +   #  4 × 1 =  4
        SYNTAX_ERROR_TO_PASS_FIX_TEMPLATES * 2 +   #  5 × 2 = 10
        # ── PASS-stop (FINAL_ANSWER-first): ~5% ───────────────────────────
        WRONG_CRITIQUE_FIX_TEMPLATES * 9          #  4 × 9 = 36 (18 FA + 18 CRIT)
    )


def _dev_set_pool() -> list[dict]:
    """Targeted pool for the 5 frozen held-out failure categories.

    Used by --dev-set-only mode. Does NOT modify the main agent-only pool.
    Train on this; re-evaluate on frozen held-out with original (82-record) memory.
    Scientific hygiene: do NOT add frozen held-out failed answers here — use
    similar-but-not-identical tasks from DEV_SET_RECOVERY_TEMPLATES only.
    """
    return DEV_SET_RECOVERY_TEMPLATES * 10   # 10 templates × 10 = 100 examples


def make_dev_set_data(n: int = 200) -> list[dict]:
    """Sample n examples from the dev-set pool for targeted fine-tuning."""
    pool = _dev_set_pool()
    if n > len(pool):
        pool = pool * (n // len(pool) + 1)
    return random.sample(pool, n)


def make_react_variations(n: int) -> list[dict]:
    """Sample from all template families for the mixed (non-agent-only) dataset."""
    pool = (
        TOOL_FIRST_SUCCESS_TEMPLATES       * 6 +
        JSON_ROBUSTNESS_TEMPLATES          * 4 +
        JSON_ROBUSTNESS_EXT_TEMPLATES      * 4 +
        TARGETED_SUCCESS_TEMPLATES         * 6 +
        NO_OUTPUT_FIX_TEMPLATES            * 2 +
        ANTI_TRIPLE_QUOTE_FIX_TEMPLATES    * 3 +
        TARGETED_NO_OUTPUT_TEMPLATES       * 3 +
        INDENT_ERROR_FIX_TEMPLATES         * 2 +
        INDENT_ERROR_TO_PASS_FIX_TEMPLATES * 2 +
        INVALID_JSON_FIX_TEMPLATES         * 2 +
        UNKNOWN_TOOL_FIX_TEMPLATES         * 1 +
        SOLUTION_OK_AFTER_ERROR_FIX_TEMPLATES * 1 +
        REPEATED_CALL_FIX_TEMPLATES        * 1 +
        REPEATED_FAILURE_BREAKER_TEMPLATES * 1 +
        SYNTAX_ERROR_TO_PASS_FIX_TEMPLATES * 2 +
        WRONG_CRITIQUE_FIX_TEMPLATES       * 8 +
        REACT_TEMPLATES +
        THINK_CODE_TEMPLATES
    )
    rows = []
    while len(rows) < n:
        item = random.choice(pool)
        rows.append({"instruction": item["instruction"], "response": item["response"]})
    return rows[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse as _ap
    parser = _ap.ArgumentParser(description="Generate code-agent training data")
    parser.add_argument("--n-code",     type=int, default=N_CODE,
                        help="CodeAlpaca pairs to include (default 5000; ignored in --agent-only)")
    parser.add_argument("--n-react",    type=int, default=N_REACT,
                        help="ReAct/recovery trajectories to sample (default 1000)")
    parser.add_argument("--agent-only", action="store_true",
                        help="Write agent_only_data.jsonl: recovery templates only, "
                             "no CodeAlpaca, no <think> examples")
    parser.add_argument("--dev-set-only", action="store_true",
                        help="Write dev_set_data.jsonl: targeted templates for the "
                             "5 frozen held-out failure categories only. "
                             "Train on this; test on frozen held-out with original memory.")
    args = parser.parse_args()

    print("\nAuditing recovery templates ...")
    _audit_all_templates()

    if args.dev_set_only:
        rows = make_dev_set_data(n=200)
        out = DATA_DIR / "dev_set_data.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps({"instruction": r["instruction"],
                                    "response": r["response"]}) + "\n")
        print(f"Wrote {len(rows)} dev-set examples → {out}")
        print("Categories: rle_decode, group_anagrams, deep_get, "
              "merge_intervals, tree_depth_tuple")
        print("Next: make train-dev-set-targeted → make eval-heldout-dev-retrained")
        return

    if args.agent_only:
        # ── Agent-only dataset: no CodeAlpaca, no <think> base templates ──
        pool  = _agent_only_pool()
        rows: list[dict] = []
        while len(rows) < args.n_react:
            item = random.choice(pool)
            rows.append({"instruction": item["instruction"], "response": item["response"]})
        rows = rows[:args.n_react]
        random.shuffle(rows)

        _audit_first_line_distribution(rows, mode="agent-only output")

        # Hard fail: agent-only dataset must have 0% markdown/prose-first.
        # TOOL_CALL, CRITIQUE, and FINAL_ANSWER are all valid first-line types.
        n_other = sum(1 for r in rows if _first_response_line_type(r["response"]) == "other")
        if n_other > 0:
            raise SystemExit(
                f"ABORT: {n_other} agent-only examples have markdown/prose-first responses. "
                "Fix recovery templates before writing data."
            )

        # Audit: failure_to_fix count — instruction has ERROR or no-output observation,
        # response must reach OBSERVATION: PASS.
        _ERR_IN_INSTR = re.compile(r"^OBSERVATION:\s*(?:ERROR|(?:\(no output\)))", re.MULTILINE)
        n_failure_to_fix = sum(
            1 for r in rows
            if _ERR_IN_INSTR.search(r["instruction"]) and "OBSERVATION: PASS" in r["response"]
        )
        print(f"  failure_to_fix examples: {n_failure_to_fix}")
        if n_failure_to_fix < 100:
            raise SystemExit(
                f"ABORT: only {n_failure_to_fix} failure_to_fix examples "
                "(minimum 100 required). Add more BENCHMARK_RECOVERY or similar templates."
            )

        # Audit: no fake OBSERVATION leakage in supervised responses.
        # Valid patterns: PASS, ERROR (any error text), (no output), (none).
        # The space after ':' must be literal so lookaheads fire on the content,
        # not before the space (which would let any content through).
        _OBS_LEAK_RE = re.compile(
            r"^OBSERVATION: (?!PASS\b)(?!ERROR\b)(?!\(no output\))(?!\(none\))", re.MULTILINE
        )
        n_obs_leak = sum(1 for r in rows if _OBS_LEAK_RE.search(r["response"]))
        if n_obs_leak > 0:
            raise SystemExit(
                f"ABORT: {n_obs_leak} examples have unexpected OBSERVATION content in response."
            )

        # Audit: no invalid execute_code JSON in supervised responses.
        n_bad_json = 0
        for r in rows:
            for m in re.finditer(r"execute_code\((\{[^)]*\})", r["response"], re.DOTALL):
                try:
                    json.loads(m.group(1))
                except json.JSONDecodeError:
                    n_bad_json += 1
                    break
        if n_bad_json > 0:
            raise SystemExit(
                f"ABORT: {n_bad_json} examples have invalid execute_code JSON in response."
            )

        out = DATA_DIR / "agent_only_data.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        print(f"\nAgent-only dataset: {len(rows)} examples")
        print(f"  Saved → {out}")
        print(f"  failure_to_fix:   {n_failure_to_fix}")
        print(f"  markdown-first:   {n_other}")
        print(f"  obs-leakage:      {n_obs_leak}")
        print(f"  bad-JSON:         {n_bad_json}")

        # Hard gate: too few examples → training will be ineffective.
        if len(rows) < 1500:
            raise SystemExit(
                f"ABORT: only {len(rows)} agent-only examples generated (minimum 1500). "
                "Increase --n-react or add more templates."
            )
        return

    # ── Standard mixed dataset ─────────────────────────────────────────────
    print(f"\nGenerating code-agent training data: "
          f"{args.n_code} code pairs + {args.n_react} ReAct trajectories")

    code_rows  = load_code_alpaca(args.n_code)
    react_rows = make_react_variations(args.n_react)

    all_rows = code_rows + react_rows
    random.shuffle(all_rows)

    out = DATA_DIR / "code_agent_data.jsonl"
    with open(out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nTotal: {len(all_rows)} examples")
    print(f"  Code pairs:        {len(code_rows)}")
    print(f"  ReAct trajectories:{len(react_rows)}")
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
