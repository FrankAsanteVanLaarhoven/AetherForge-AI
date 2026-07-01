"""
scripts/solution_body_verifier.py — v2.35 strict solution-body verifier (deterministic).

v2.34 recovered tool-call emission (no_tool_call 26→1) but the frozen 32-task score stayed at 5/32
because the emitted execute_code bodies were incomplete, weakly asserted, or incorrect. This module
verifies a code body against the TASK'S REAL benchmark assertions — never the model's own
`print("PASS")` — and classifies why a body fails. It fabricates nothing and injects no solutions.

It rejects (never counts as pass):
  - unconditional print("PASS") with no real verification (fake pass),
  - missing / model-invented asserts (only benchmark-owned assertions are trusted),
  - empty / incomplete implementations (no required function definition),
  - code that only satisfies model-invented tests or bypasses the task contract.

Public API (pure, deterministic):
    defines_function(code, func)                 -> bool
    is_unconditional_pass(code)                  -> bool
    strip_model_verification(code)               -> str
    verify_solution_body(code, func, tests)      -> dict(status, reason)
    classify_body(code, func, tests)             -> str   (strict_pass | fake_pass | incomplete_no_def
                                                           | assertion_failure | no_benchmark_tests)
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REASON_STRICT_PASS = "strict_pass"
REASON_FAKE_PASS = "fake_pass"
REASON_INCOMPLETE = "incomplete_no_def"
REASON_ASSERT_FAIL = "assertion_failure"
REASON_NO_TESTS = "no_benchmark_tests"


def defines_function(code, func):
    """True if `code` contains a top-level def for `func` (a required implementation)."""
    if not func:
        return True
    return bool(re.search(r"(?m)^\s*def\s+" + re.escape(func) + r"\s*\(", code or ""))


def _assert_lines(code):
    return [ln for ln in (code or "").splitlines() if ln.strip().startswith("assert ")]


def is_unconditional_pass(code):
    """True if the body prints a PASS sentinel with NO asserts guarding it (a fake-pass tell)."""
    c = code or ""
    prints_pass = "print('PASS')" in c or 'print("PASS")' in c
    return prints_pass and not _assert_lines(c)


def strip_model_verification(code):
    """Drop the model's own assert / print lines; keep function and helper definitions."""
    out = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        if s.startswith("assert ") or s.startswith("print(") or s.startswith("# Test"):
            continue
        out.append(ln)
    return "\n".join(out)


def verify_solution_body(code, func, tests, timeout=10):
    """Verify `code` against the benchmark-owned `tests` (list of assertion expressions).

    Returns dict(status 'pass'|'reject', reason). Never trusts the model's print('PASS'):
    model verification is stripped and only the benchmark assertions decide the outcome.
    """
    if func and not defines_function(code, func):
        return {"status": "reject", "reason": REASON_INCOMPLETE}
    if not tests:
        return {"status": "reject", "reason": REASON_NO_TESTS}
    body = strip_model_verification(code or "")
    prog = body + "\n" + "\n".join(f"assert {t}" for t in tests) + "\nprint('PASS')\n"
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "c.py"
        fp.write_text(prog)
        try:
            r = subprocess.run([sys.executable, str(fp)], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "reject", "reason": REASON_ASSERT_FAIL}
    if r.returncode == 0 and "PASS" in r.stdout:
        return {"status": "pass", "reason": REASON_STRICT_PASS}
    return {"status": "reject", "reason": REASON_ASSERT_FAIL}


def classify_body(code, func, tests):
    """Classify a body: strict_pass / fake_pass / incomplete_no_def / assertion_failure / no_benchmark_tests."""
    if not tests:
        return REASON_NO_TESTS
    v = verify_solution_body(code, func, tests)
    if v["status"] == "pass":
        return REASON_STRICT_PASS
    if func and not defines_function(code, func):
        return REASON_INCOMPLETE
    if is_unconditional_pass(code):
        return REASON_FAKE_PASS
    return REASON_ASSERT_FAIL


if __name__ == "__main__":
    good = "def add(a,b):\n    return a+b"
    fake = "def add(a,b):\n    return a\nprint('PASS')"
    tests = ["add(1,2)==3", "add(0,0)==0"]
    for name, c in (("good", good), ("fake", fake), ("empty", "print('PASS')")):
        print(f"{name:6} -> {classify_body(c, 'add', tests)}  fake={is_unconditional_pass(c)}")
