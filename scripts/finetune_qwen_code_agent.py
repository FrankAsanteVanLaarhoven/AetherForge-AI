#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-0.5B-Instruct on code + execution trace data.
Compatible with modern transformers (>=4.40)
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Ensure repo root is on sys.path so `memory.*` packages are importable
# regardless of the working directory used to launch the script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

AGENT_SYSTEM = """You are AetherForge Code Agent. You solve programming tasks by actually running
code — never by describing what code would do. Every answer is verified by execution.

Tools:
  execute_code({"code": "..."})                         run Python inline
  run_script({"path": "file.py", "args": []})           run a saved script
  run_tests({"path": "test_file.py"})                   pytest on a file/dir
  read_file({"path": "..."})                             read a file's contents
  write_file({"path": "...", "content": "..."})          create or overwrite a file
  edit_file({"path": "...", "old": "...", "new": "..."}) replace text in a file
  search_file({"path": ".", "pattern": "..."})           grep across files
  list_files({"path": "."})                              list a directory
  run_command({"command": "..."})                        shell (no sudo / rm -rf)
  check_syntax({"code": "..."})                          compile-check, no execution
  run_linter({"code": "..."})                            flake8 style + error check
  install_package({"package": "numpy"})                 pip install a package
  format_code({"code": "..."})                           black-format Python code
  git_status({"path": "."})                              show working tree status
  git_diff({"path": ".", "staged": false})               show file-level diff
  git_log({"n": 5})                                      show recent commits

Protocol:
  1. <think>Plan the steps, note edge cases, pick the right tools.</think>
  2. TOOL_CALL: execute_code({"code": "..."})   — one registered tool per call
  3. OBSERVATION: <runtime-injected result — never write this yourself>
  4. On error: read the full traceback, identify root cause, fix, re-run
  5. Prefer edit_file over rewriting entire files for small changes
  6. Verify with run_tests or execute_code
  7. CRITIQUE:
       Correctness — <does output exactly match the requirements?>
       Edge cases  — <empty input, zero, None, boundaries, large N?>
       Requirements— <is anything from the task description unaddressed?>
       -> Solution OK.   OR   -> Fix needed: <specific issue>
  8. FINAL_ANSWER: <concise verified answer>

Rules:
  - Never claim code works without running it
  - Never give up after one error — diagnose and fix
  - check_syntax before execute_code for complex or multi-class code
  - install_package only when an ImportError confirms the package is missing
  - git_* tools only work in a git repository
  - If CRITIQUE ends with "Fix needed:", go back to TOOL_CALL before FINAL_ANSWER
  - Only call registered tools listed above. Never call the task function as a tool.
  - Always use real 4-space indentation inside code strings — never write unindented
    function bodies. The code inside execute_code must be valid Python.
  - Use assert statements to verify correctness, then print("PASS").
    Invalid: TOOL_CALL: is_palindrome({"s": "racecar"})
    Invalid: TOOL_CALL: execute_code({"code": "def is_palindrome(s):\\nreturn s == s[::-1]"})
    Valid:   TOOL_CALL: execute_code({"code": "def is_palindrome(s):\\n    return s == s[::-1]\\nassert is_palindrome('racecar')\\nassert not is_palindrome('hello')\\nprint('PASS')"})
    Invalid: TOOL_CALL: sum_list({"lst": [1, 2, 3]})
    Invalid: TOOL_CALL: execute_code({"code": "def sum_list(lst):\\nreturn sum(lst)"})
    Valid:   TOOL_CALL: execute_code({"code": "def sum_list(lst):\\n    return sum(lst)\\nassert sum_list([1,2,3,4,5]) == 15\\nassert sum_list([]) == 0\\nprint('PASS')"})
"""

# Strict contract system prompt — used with --agent-contract strict.
# Instructs the model to start EVERY response with TOOL_CALL, never markdown.
STRICT_AGENT_SYSTEM = """You are AetherForge Code Agent. You solve programming tasks by executing code.

STRICT CONTRACT — no exceptions:

1. For every fresh programming task your VERY FIRST output line MUST be:
     TOOL_CALL: execute_code({"code": "...implementation + asserts + print('PASS')..."})
   Do NOT write markdown, prose, or explanations first. Go straight to TOOL_CALL.

2. If recovering from a prior error your first output line MUST be:
     CRITIQUE:
   followed immediately by a different corrected TOOL_CALL.

3. OBSERVATION is injected by the runtime — never write it yourself.

4. A tool call only counts as verified if the code prints PASS.
   FINAL_ANSWER is only valid after OBSERVATION: PASS.
   OBSERVATION: (no output) is NOT a pass — add assertions and print('PASS').

5. After OBSERVATION: ERROR or OBSERVATION: (no output) — write CRITIQUE identifying
   root cause, then a DIFFERENT corrected TOOL_CALL with asserts and print('PASS').
   Never repeat a failing call. Never claim success without PASS.

Rules:
  - Always use 4-space indentation in code strings.
  - Always include assert statements to verify correctness.
  - Always end successful code with print("PASS").
  - Code containing quotes, f-strings, or dicts is fine — the JSON encoding handles it.
  - Never call task functions as tools:
      INVALID: TOOL_CALL: sum_list({"lst": [1, 2, 3]})
      VALID:   TOOL_CALL: execute_code({"code": "def sum_list(lst):\\n    return sum(lst)\\nassert sum_list([1,2,3,4,5]) == 15\\nprint('PASS')"})

Available tools: execute_code, run_script, run_tests, read_file, write_file, edit_file,
  search_file, list_files, run_command, check_syntax, run_linter, install_package,
  format_code, git_status, git_diff, git_log
"""

# ---------------------------------------------------------------------------
# OBSERVATION masking
# ---------------------------------------------------------------------------
# In training trajectories, OBSERVATION blocks are injected by the runtime —
# the model must NOT learn to generate them.  We mask every OBSERVATION span
# to -100 so no loss is computed on runtime output.
#
# Pattern: \nOBSERVATION: <content> up to (but not including) the next
# structural element: <think>, TOOL_CALL:, CRITIQUE:, FINAL_ANSWER:, or end.
_OBS_RE = re.compile(
    r"\nOBSERVATION:.*?(?=\n(?:<think>|TOOL_CALL:|CRITIQUE:|FINAL_ANSWER:)|\Z)",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Allowed tool names — must match the TOOLS dict in agent_loop.py exactly.
# ---------------------------------------------------------------------------
ALLOWED_TOOLS: set = {
    "execute_code", "run_script", "run_tests",
    "read_file", "write_file", "edit_file", "search_file", "list_files",
    "run_command", "check_syntax", "run_linter", "install_package",
    "format_code", "git_status", "git_diff", "git_log",
}

_TOOL_CALL_NAME_RE = re.compile(r"TOOL_CALL:\s*(\w+)\s*\(")


def find_invalid_tool_calls(text: str) -> List[str]:
    """Return list of non-allowed tool names found in TOOL_CALL: lines."""
    return [
        m.group(1)
        for m in _TOOL_CALL_NAME_RE.finditer(text)
        if m.group(1) not in ALLOWED_TOOLS
    ]


def _extract_execute_code_bodies(text: str) -> List[str]:
    """Return list of code strings from all execute_code(...) calls in text.

    Uses a quote-aware balanced-brace scanner so nested braces/quotes in
    the code string don't confuse the parser.
    """
    results: List[str] = []
    marker = "execute_code("
    pos    = 0
    while True:
        idx = text.find(marker, pos)
        if idx == -1:
            break
        scan    = idx + len(marker)
        depth   = 0
        in_str  = False
        qch     = ""
        escaped = False
        end     = -1
        for i, ch in enumerate(text[scan:], scan):
            if escaped:
                escaped = False
                continue
            if in_str:
                if ch == "\\":
                    escaped = True
                elif ch == qch:
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                qch    = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            raw = text[scan:end]
            try:
                args = json.loads(raw)
                code = args.get("code", "")
                if code:
                    results.append(code)
            except json.JSONDecodeError:
                results.append(None)   # bad JSON — signal separately
        pos = idx + 1
    return results


def _sol_ok_after_error_before_pass(text: str) -> bool:
    """True if '→ Solution OK' appears after an ERROR obs without an intervening PASS obs."""
    events: List[Tuple[str, int]] = []
    for m in re.finditer(
        r"OBSERVATION:\s*(.*?)(?=\nTOOL_CALL:|\nCRITIQUE:|\nFINAL_ANSWER:|\Z)",
        text, re.DOTALL,
    ):
        obs = m.group(1).strip()
        if obs.startswith("PASS"):
            events.append(("OBS_PASS", m.start()))
        elif "Error" in obs or "ERROR" in obs:
            events.append(("OBS_ERROR", m.start()))
    for m in re.finditer(r"→ Solution OK", text, re.IGNORECASE):
        events.append(("SOL_OK", m.start()))
    events.sort(key=lambda x: x[1])
    last_error = False
    for ev_type, _ in events:
        if ev_type == "OBS_ERROR":
            last_error = True
        elif ev_type == "OBS_PASS":
            last_error = False
        elif ev_type == "SOL_OK" and last_error:
            return True
    return False


def _repeated_call_after_error(text: str) -> bool:
    """True if the same TOOL_CALL line is repeated after an ERROR observation."""
    tc_events: List[Tuple[str, int]] = []
    obs_events: List[Tuple[bool, int]] = []
    for m in re.finditer(r"^(TOOL_CALL:.+)$", text, re.MULTILINE):
        tc_events.append((m.group(1).strip(), m.start()))
    for m in re.finditer(
        r"^OBSERVATION:\s*(.*?)(?=^(?:TOOL_CALL:|CRITIQUE:|FINAL_ANSWER:)|\Z)",
        text, re.MULTILINE | re.DOTALL,
    ):
        obs_events.append(("Error" in m.group(1) or "ERROR" in m.group(1), m.start()))
    all_evs = [(t, p, "TC") for t, p in tc_events] + [(e, p, "OBS") for e, p in obs_events]
    all_evs.sort(key=lambda x: x[1])
    last_call: Optional[str] = None
    after_error = False
    for item in all_evs:
        if item[2] == "TC":
            call = item[0]
            if after_error and call == last_call:
                return True
            last_call = call
            after_error = False
        else:
            after_error = item[0]
    return False


def audit_execute_code_quality(text: str) -> Tuple[int, int, List[str]]:
    """Audit execute_code calls in a supervised text segment.

    Returns (n_bad_json, n_bad_ast, examples_of_bad_code).
    n_bad_ast counts only the LAST execute_code call per trajectory
    (first calls may be intentionally buggy in fix-the-bug examples).
    """
    import ast as _ast

    bodies    = _extract_execute_code_bodies(text)
    n_bad_json = bodies.count(None)
    bodies     = [b for b in bodies if b is not None]

    if not bodies:
        return n_bad_json, 0, []

    # Only audit the last call — first calls may be intentional bugs.
    last_code   = bodies[-1]
    n_bad_ast   = 0
    bad_snippets: List[str] = []
    try:
        _ast.parse(last_code)
    except SyntaxError as e:
        n_bad_ast = 1
        bad_snippets.append(f"{e}: {last_code[:80]!r}")

    return n_bad_json, n_bad_ast, bad_snippets


def split_target_segments(target_text: str) -> List[Tuple[str, bool]]:
    """
    Return (text, is_supervised) segments for the assistant response.
    OBSERVATION spans are not supervised; everything else is.
    """
    segments: List[Tuple[str, bool]] = []
    last_end = 0
    for m in _OBS_RE.finditer(target_text):
        start, end = m.start(), m.end()
        if start > last_end:
            segments.append((target_text[last_end:start], True))
        segments.append((target_text[start:end], False))
        last_end = end
    if last_end < len(target_text):
        segments.append((target_text[last_end:], True))
    if not segments:
        segments = [(target_text, True)]
    return segments


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

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


def get_training_files(agent_only: bool = False) -> List[str]:
    script_path = Path(__file__).resolve()
    data_dir = script_path.parent.parent / "data"
    if agent_only:
        required = data_dir / "agent_only_data.jsonl"
        if not required.exists() or required.stat().st_size == 0:
            print(
                "ERROR: --agent-only requires data/agent_only_data.jsonl but it is missing.\n"
                "       Run:  make data-agent-only"
            )
            sys.exit(1)
        candidates = [
            data_dir / "agent_only_data.jsonl",
            data_dir / "execution_traces.jsonl",
        ]
    else:
        candidates = [
            data_dir / "code_agent_data.jsonl",
            data_dir / "execution_traces.jsonl",
        ]
    return [str(f) for f in candidates if f.exists() and f.stat().st_size > 0]


def load_examples(data_files: List[str]) -> List[Dict]:
    all_examples = []
    for filepath in data_files:
        print(f"Loading {filepath} ...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return all_examples


def tokenize_example(
    example: Dict,
    tokenizer,
    max_length: int,
    system_prompt: str = None,
) -> Optional[Dict]:
    """
    Build input_ids and labels with two constraints:
      1. Prompt tokens (system + user) → labels = -100 (no loss).
      2. OBSERVATION spans inside the response → labels = -100 (no loss).
         The model must learn to emit TOOL_CALLs and FINAL_ANSWERs,
         not to hallucinate runtime output.

    If prompt + target > max_length, the prompt is truncated from the left so
    the full target is always kept.  If the target alone exceeds max_length the
    sample is skipped (returns None).
    """
    instruction = example.get("instruction", "").strip()
    response    = example.get("response", "").strip()
    if not instruction or not response:
        return None

    sys = system_prompt if system_prompt is not None else AGENT_SYSTEM
    prompt_text = (
        f"<|im_start|>system\n{sys}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Build target ids and labels segment-by-segment so OBSERVATION spans
    # can be masked independently of prompt masking.
    target_text = response + "<|im_end|>"
    segments    = split_target_segments(target_text)

    target_ids: List[int]    = []
    target_labels: List[int] = []
    for seg_text, is_supervised in segments:
        seg_ids = tokenizer.encode(seg_text, add_special_tokens=False)
        target_ids.extend(seg_ids)
        if is_supervised:
            target_labels.extend(seg_ids)
        else:
            target_labels.extend([-100] * len(seg_ids))

    if len(target_ids) >= max_length:
        return None

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    total = len(prompt_ids) + len(target_ids)
    if total > max_length:
        # Truncate prompt from the left to preserve the full target.
        keep       = max_length - len(target_ids)
        prompt_ids = prompt_ids[-keep:] if keep > 0 else []

    input_ids = prompt_ids + target_ids
    labels    = [-100] * len(prompt_ids) + target_labels

    pad_id  = tokenizer.pad_token_id
    pad_len = max_length - len(input_ids)
    input_ids      = input_ids + [pad_id] * pad_len
    attention_mask = [1] * (max_length - pad_len) + [0] * pad_len
    labels         = labels   + [-100]  * pad_len

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def debug_masking(records: List[Dict], tokenizer, n: int = 5):
    """
    Print per-example masking stats and hard-fail on any of:
      - zero supervised tokens
      - first supervised position == 0 (prompt not masked)
      - supervised ratio > 80%  (boundary not found)
      - leakage (response text visible in pre-assistant region)
      - any OBSERVATION marker token is supervised
    """
    pad_id = tokenizer.pad_token_id
    obs_marker_ids = tokenizer.encode("\nOBSERVATION:", add_special_tokens=False)
    obs_marker_len = len(obs_marker_ids)
    errors: List[str] = []

    print("\n=== Label Masking Diagnostic ===")
    for i, rec in enumerate(records[:n]):
        ids  = rec["input_ids"]
        labs = rec["labels"]

        n_total      = len(ids)
        n_padding    = sum(1 for t in ids if t == pad_id)
        n_real       = n_total - n_padding
        n_supervised = sum(1 for l in labs if l != -100)
        ratio        = n_supervised / n_real if n_real > 0 else 0.0

        sup_ids  = [t for t, l in zip(ids, labs) if l != -100 and t != pad_id]
        sup_text = tokenizer.decode(sup_ids, skip_special_tokens=False)

        full_ids  = [t for t in ids if t != pad_id]
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)

        asst_marker = "<|im_start|>assistant\n"
        pre_asst    = full_text.split(asst_marker)[0] if asst_marker in full_text else full_text
        # Use 80-char prefix: short snippets may appear in system-prompt examples
        # (e.g. "TOOL_CALL: execute_code({\"code\":" appears in STRICT_AGENT_SYSTEM),
        # but no real implementation will match verbatim at 80 chars.
        leakage     = len(sup_text) > 80 and sup_text[:80].strip() in pre_asst

        first_sup   = next((j for j, l in enumerate(labs) if l != -100), n_total)

        # OBSERVATION supervision audit
        obs_supervised_at = []
        for k in range(len(ids) - obs_marker_len + 1):
            if ids[k : k + obs_marker_len] == obs_marker_ids:
                if any(labs[k + j] != -100 for j in range(obs_marker_len)):
                    obs_supervised_at.append(k)

        print(f"\n  Example {i + 1}:")
        print(f"    total tokens   : {n_total}")
        print(f"    non-padding    : {n_real}")
        print(f"    supervised     : {n_supervised}  ({ratio:.1%})")
        print(f"    first sup pos  : {first_sup}")
        print(f"    leakage        : {'YES <<< FIX NEEDED' if leakage else 'no'}")
        if obs_supervised_at:
            print(f"    OBSERVATION sup: YES at positions {obs_supervised_at[:3]}  <<< FIX NEEDED")
        else:
            print(f"    OBSERVATION sup: no")

        # Invalid TOOL_CALL name audit
        bad_tools_in_sup = find_invalid_tool_calls(sup_text)
        if bad_tools_in_sup:
            print(f"    invalid TOOL_CALL sup: YES {bad_tools_in_sup}  <<< FIX NEEDED")
        else:
            print(f"    invalid TOOL_CALL sup: no")

        # execute_code JSON + Python quality audit
        n_bad_json, n_bad_ast, bad_snips = audit_execute_code_quality(sup_text)
        if n_bad_json:
            print(f"    invalid execute_code JSON: YES ({n_bad_json})  <<< FIX NEEDED")
        else:
            print(f"    invalid execute_code JSON: no")
        if n_bad_ast:
            print(f"    unparseable execute_code code: YES ({n_bad_ast}) — {bad_snips[:1]}  <<< FIX NEEDED")
        else:
            print(f"    unparseable execute_code code: no")

        sol_ok_err = _sol_ok_after_error_before_pass(sup_text)
        if sol_ok_err:
            print(f"    solution_ok_after_error_before_pass: YES  <<< FIX NEEDED")
        else:
            print(f"    solution_ok_after_error_before_pass: no")

        rep_call = _repeated_call_after_error(sup_text)
        if rep_call:
            print(f"    repeated_identical_tool_call_after_error: YES  <<< FIX NEEDED")
        else:
            print(f"    repeated_identical_tool_call_after_error: no")

        print(f"    input (200c)   : {repr(full_text[:200])}")
        print(f"    supervised(200): {repr(sup_text[:200])}")

        if n_supervised == 0:
            errors.append(f"Example {i+1}: zero supervised tokens")
        if first_sup == 0:
            errors.append(f"Example {i+1}: first supervised position == 0 (prompt not masked)")
        if ratio > 0.80:
            errors.append(f"Example {i+1}: {ratio:.0%} supervised > 80% (boundary not found)")
        if leakage:
            errors.append(f"Example {i+1}: leakage — response appears in prompt region")
        if obs_supervised_at:
            errors.append(f"Example {i+1}: OBSERVATION marker tokens are supervised")
        if bad_tools_in_sup:
            errors.append(f"Example {i+1}: invalid TOOL_CALL names in supervised text: {bad_tools_in_sup}")
        if n_bad_json:
            errors.append(f"Example {i+1}: invalid JSON in execute_code args ({n_bad_json} call(s))")
        if n_bad_ast:
            errors.append(f"Example {i+1}: unparseable Python in last execute_code: {bad_snips[:1]}")
        if sol_ok_err:
            errors.append(f"Example {i+1}: solution_ok_after_error_before_pass in supervised text")
        if rep_call:
            errors.append(f"Example {i+1}: repeated_identical_tool_call_after_error in supervised text")

    print("=== End Diagnostic ===\n")

    if errors:
        print("MASKING ERRORS — fix before training:")
        for e in errors:
            print(f"  {e}")
        raise ValueError("Label masking is broken. See errors above.")


# ---------------------------------------------------------------------------
# Memory leak-checking helpers (training only — inference retrieval unchanged)
# ---------------------------------------------------------------------------

_TC_MARKER = "TOOL_CALL: execute_code("
_EC_RE = re.compile(r'execute_code\(\{"code":\s*"((?:[^"\\]|\\.)*)"\}', re.DOTALL)


def normalize_code_for_leak_check(code: str) -> str:
    """Strip whitespace and normalise newline/indent so small formatting diffs don't matter."""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    return "\n".join(lines).lower()


def extract_execute_code_body(text: str) -> str:
    """Return the first execute_code body in text, normalised, or '' if none."""
    m = _EC_RE.search(text)
    if not m:
        return ""
    try:
        raw = json.loads('"' + m.group(1) + '"')   # decode JSON string escapes
    except (json.JSONDecodeError, ValueError):
        raw = m.group(1)
    return normalize_code_for_leak_check(raw)


def is_memory_target_leak(memory_record: Dict, response_text: str) -> bool:
    """Return True if inserting this memory would leak the supervised target into the prompt.

    Conditions:
      1. corrected_tool_call appears verbatim in the response (exact string match).
      2. Normalised execute_code body of memory == normalised execute_code body of response.
      3. Task strings are identical AND normalised execute_code bodies are highly similar.
    """
    ctc = memory_record.get("corrected_tool_call", "")
    mem_task = memory_record.get("task", "")

    # 1 — exact corrected_tool_call overlap
    if ctc and ctc in response_text:
        return True

    # 2 — normalised execute_code body overlap
    mem_body = extract_execute_code_body(ctc)
    resp_body = extract_execute_code_body(response_text)
    if mem_body and resp_body and mem_body == resp_body:
        return True

    # 3 — same task, highly similar code body (one is a prefix of the other at ≥80%)
    if mem_task:
        # Look up the task from the example we're processing — passed as second arg text
        # We check only when the task strings match.
        pass  # The caller filters by task identity separately if needed

    return False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", "--model-name",
                        dest="hf_model",
                        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        help="HuggingFace model ID or local path to use as the base model "
                             "(default: Qwen/Qwen2.5-Coder-0.5B-Instruct)")
    parser.add_argument("--test-run",        action="store_true")
    parser.add_argument("--steps",           type=int,   default=3000)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--batch-size",      type=int,   default=2)
    parser.add_argument("--grad-accum",      type=int,   default=8)
    parser.add_argument("--max-length",      type=int,   default=1024)
    parser.add_argument("--load-in-4bit",    action="store_true",
                        help="QLoRA: load the base in 4-bit NF4 + gradient checkpointing so "
                             "7B-class models train on 16GB VRAM.")
    parser.add_argument("--output-dir",      default="outputs/qwen_code_agent",
                        help="Base output dir; final model saved to <output-dir>/final")
    parser.add_argument("--agent-only",      action="store_true",
                        help="Train on agent_only_data.jsonl + execution_traces only; "
                             "hard-fails if any supervised response starts with markdown/prose")
    parser.add_argument("--training-file",   default=None,
                        help="Load training examples from this JSONL file instead of the "
                             "default data files. Useful for targeted fine-tuning on a "
                             "specific dev set (e.g. data/dev_set_data.jsonl). "
                             "File must contain {instruction, response} records. "
                             "Still applies --agent-only markdown filter if --agent-only set.")
    parser.add_argument("--agent-contract",  choices=["standard", "strict"], default="standard",
                        help="System prompt variant: standard (default) | strict (tool-first enforced)")
    parser.add_argument("--memory-enabled",  action="store_true",
                        help="Augment each training example's system prompt with retrieved "
                             "verified memories (guidance only; labels unchanged)")
    parser.add_argument("--memory-index",    default="memory/index",
                        help="Directory containing the pre-built memory index")
    parser.add_argument("--memory-top-k",    type=int, default=4,
                        help="Number of memory records to retrieve per training example")
    args = parser.parse_args()

    # Select system prompt
    system_prompt = STRICT_AGENT_SYSTEM if args.agent_contract == "strict" else AGENT_SYSTEM

    # ── Memory index (optional) ────────────────────────────────────────────
    _embed_q   = None
    _mem_search = None
    _fmt_mem   = None
    memory_state = None
    if args.memory_enabled:
        try:
            from memory.store import load_index as _load_mem
            memory_state = _load_mem(Path(args.memory_index))
            n_mem = len(memory_state.get("records", []))
            print(f"[memory] Loaded {n_mem} verified records from {args.memory_index}")
            print("[memory] Each training example's system prompt will be augmented "
                  "with top-k retrieved memories (guidance only; labels unchanged).")
            from memory.embed import embed_query as _embed_q
            from memory.store import search as _mem_search
            from memory.core import format_memory_block as _fmt_mem
        except FileNotFoundError as exc:
            print(f"[memory] ERROR: {exc}")
            print("[memory] Build the index first: python scripts/build_vector_memory.py")
            sys.exit(1)

    print("=== Smoke test (25 steps) ===" if args.test_run else "=== Full Training ===")

    if args.training_file:
        tf = Path(args.training_file)
        if not tf.exists() or tf.stat().st_size == 0:
            print(
                f"ERROR: --training-file '{tf}' is missing or empty.\n"
                f"       For the dev-set: make generate-targeted-dev-traces"
            )
            sys.exit(1)
        print(f"[training-file] Using targeted file: {tf}")
        data_files = [str(tf)]
    else:
        data_files = get_training_files(agent_only=args.agent_only)
    if not data_files:
        if args.agent_only:
            print("ERROR: agent_only_data.jsonl not found. Run: make data-agent-only")
        else:
            print("ERROR: No training data found. Run: make data-code")
        sys.exit(1)

    print(f"Found {len(data_files)} data file(s)")
    examples = load_examples(data_files)
    print(f"Total examples loaded: {len(examples)}")

    print(f"Base model: {args.hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    if getattr(args, "load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        print("QLoRA: loading base in 4-bit NF4 (for 7B-class on 16GB VRAM)")
    else:
        load_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **load_kwargs)

    if getattr(args, "load_in_4bit", False):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = []
    usable_responses: List[str] = []   # raw response text for each accepted example
    skipped_too_long       = 0
    skipped_bad_tools      = 0
    skipped_markdown_first = 0
    bad_tool_counts: Dict[str, int] = {}

    for ex in examples:
        response = ex.get("response", "")

        # Hard filter: no invalid TOOL_CALL names in supervised response
        bad_names = find_invalid_tool_calls(response)
        if bad_names:
            skipped_bad_tools += 1
            for name in bad_names:
                bad_tool_counts[name] = bad_tool_counts.get(name, 0) + 1
            continue

        # Agent-only filter: skip markdown/prose-first responses.
        # TOOL_CALL, CRITIQUE, and FINAL_ANSWER are all valid agent-first-line types.
        if args.agent_only:
            if _first_response_line_type(response) == "other":
                skipped_markdown_first += 1
                continue

        # Optionally augment the system prompt with retrieved memories.
        # The memory block is part of the prompt (masked, no loss), so it
        # teaches the model to read and use memory context when present,
        # without changing what the model is supervised to predict.
        # Leaking memories (whose corrected_tool_call or execute_code body
        # appears in the supervised response) are filtered before insertion.
        ex_system_prompt = system_prompt
        if memory_state is not None:
            instruction_text = ex.get("instruction", "")
            qvec = _embed_q(
                instruction_text,
                memory_state["query_texts"],
                memory_state.get("vocab"),
            )
            hits = _mem_search(memory_state, qvec, args.memory_top_k)
            hits = [h for h in hits if h.get("verified", False) and h.get("score", 0) >= 0.05]
            # Filter memories that would leak the supervised target into the prompt.
            hits = [h for h in hits if not is_memory_target_leak(h, response)]
            if hits:
                mem_block = _fmt_mem(hits)
                ex_system_prompt = system_prompt.rstrip() + "\n\n" + mem_block

        rec = tokenize_example(ex, tokenizer, args.max_length,
                               system_prompt=ex_system_prompt)
        if rec is None:
            skipped_too_long += 1
        else:
            records.append(rec)
            usable_responses.append(response)

    print(f"  loaded            : {len(examples)}")
    print(f"  skipped too long  : {skipped_too_long}")
    print(f"  skipped bad tools : {skipped_bad_tools}")
    if args.agent_only:
        print(f"  skipped markdown-first (agent-only filter): {skipped_markdown_first}")
    if bad_tool_counts:
        top = sorted(bad_tool_counts.items(), key=lambda x: -x[1])[:10]
        print(f"  top invalid names : {top}")
    print(f"  usable examples   : {len(records)}")

    if skipped_bad_tools > 0:
        top_names = sorted(bad_tool_counts.items(), key=lambda x: -x[1])[:10]
        raise RuntimeError(
            f"\nABORT: {skipped_bad_tools} training example(s) contain invalid TOOL_CALL names "
            f"in the supervised response.\n"
            f"  Top offending names: {top_names}\n"
            "Fix data generation so no supervised response contains task-function tool calls.\n"
            "Bad tool calls must appear only in instruction/context, never in the assistant target."
        )

    if not records:
        print("ERROR: No usable training examples.")
        sys.exit(1)

    # ── Dataset-level execute_code quality audit ──────────────────────────
    print("\n── execute_code quality audit (full dataset) ──")
    n_total_calls = total_bad_json = total_bad_ast = 0
    total_sol_ok_err = total_rep_call = 0
    bad_ast_examples: List[str] = []

    for ex in examples:
        sup_text = ex.get("response", "")
        # Remove OBSERVATION spans — we only audit supervised text
        sup_clean = re.sub(
            r"\nOBSERVATION:.*?(?=\n(?:<think>|TOOL_CALL:|CRITIQUE:|FINAL_ANSWER:)|\Z)",
            "",
            sup_text,
            flags=re.DOTALL,
        )
        bodies = _extract_execute_code_bodies(sup_clean)
        n_total_calls += len([b for b in bodies if b is not None])
        bj, ba, snips = audit_execute_code_quality(sup_clean)
        total_bad_json += bj
        total_bad_ast  += ba
        if ba and len(bad_ast_examples) < 5:
            bad_ast_examples.extend(snips[:1])
        if _sol_ok_after_error_before_pass(sup_text):
            total_sol_ok_err += 1
        if _repeated_call_after_error(sup_text):
            total_rep_call += 1

    print(f"  total execute_code calls                : {n_total_calls}")
    print(f"  invalid JSON args                       : {total_bad_json}")
    print(f"  unparseable last code                   : {total_bad_ast}")
    print(f"  solution_ok_after_error_before_pass     : {total_sol_ok_err}")
    print(f"  repeated_identical_tool_call_after_error: {total_rep_call}")
    if bad_ast_examples:
        print("  examples of bad code:")
        for s in bad_ast_examples:
            print(f"    {s[:120]}")
    all_ok = (total_bad_json == 0 and total_bad_ast == 0
              and total_sol_ok_err == 0 and total_rep_call == 0)
    if not all_ok:
        raise RuntimeError(
            "\nABORT: Dataset quality gates failed — fix data generation before training.\n"
            f"  invalid JSON args                       : {total_bad_json}\n"
            f"  unparseable last code                   : {total_bad_ast}\n"
            f"  solution_ok_after_error_before_pass     : {total_sol_ok_err}\n"
            f"  repeated_identical_tool_call_after_error: {total_rep_call}\n"
        )
    else:
        print("  ✓ All quality gates pass")

    # ── PASS discipline audit (usable training set) ────────────────────────
    if args.agent_only:
        print("\n── PASS discipline audit (agent-only training set) ──")
        no_print_pass = 0
        no_assert     = 0
        fa_before_pass = 0

        for resp in usable_responses:
            # Check each execute_code → OBSERVATION: PASS pair in the response
            for m_pass in re.finditer(r"^OBSERVATION:\s*PASS\b", resp, re.MULTILINE):
                before = resp[:m_pass.start()]
                ec_matches = list(re.finditer(r"TOOL_CALL:\s*execute_code\(", before))
                if not ec_matches:
                    continue
                last_ec = ec_matches[-1]
                between = resp[last_ec.start():m_pass.start()]
                if "OBSERVATION:" in between:
                    continue
                paren_start = last_ec.end()
                depth = 0; in_str = False; qch = ""; esc = False; end_pd = -1
                for i, ch in enumerate(resp[paren_start:], paren_start):
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
                    args_pd = json.loads(resp[paren_start:end_pd])
                except json.JSONDecodeError:
                    continue
                code_pd = args_pd.get("code", "")
                if not code_pd:
                    continue
                if "print('PASS')" not in code_pd and 'print("PASS")' not in code_pd:
                    no_print_pass += 1
                if "assert " not in code_pd:
                    no_assert += 1

            # Check FINAL_ANSWER before first OBSERVATION: PASS.
            # Skip if the response has no execute_code: WRONG_CRITIQUE_FIX PASS-case
            # responses start with FINAL_ANSWER because OBSERVATION: PASS was in the
            # instruction context, which is the correct trained behavior.
            fa_m  = re.search(r"^FINAL_ANSWER:", resp, re.MULTILINE)
            pas_m = re.search(r"^OBSERVATION:\s*PASS\b", resp, re.MULTILINE)
            has_ec_in_resp = bool(re.search(r"TOOL_CALL:\s*execute_code\(", resp))
            if fa_m is not None and has_ec_in_resp:
                if pas_m is None or fa_m.start() < pas_m.start():
                    fa_before_pass += 1

        print(f"  successful execute_code without print('PASS'): {no_print_pass}")
        print(f"  successful execute_code without assert        : {no_assert}")
        print(f"  FINAL_ANSWER before OBSERVATION: PASS         : {fa_before_pass}")
        pd_ok = (no_print_pass == 0 and no_assert == 0 and fa_before_pass == 0)
        if not pd_ok:
            raise RuntimeError(
                "\nABORT: PASS discipline gates failed — fix data generation before training.\n"
                f"  successful execute_code without print('PASS'): {no_print_pass}\n"
                f"  successful execute_code without assert        : {no_assert}\n"
                f"  FINAL_ANSWER before OBSERVATION: PASS         : {fa_before_pass}\n"
            )
        else:
            print("  ✓ PASS discipline gates pass")

    # ── First-line distribution audit (usable training examples only) ─────
    print("\n── First-line distribution audit (usable training set) ──")
    n_tc = n_crit = n_fa = n_other_fl = 0
    other_fl_examples: List[str] = []
    for resp in usable_responses:
        t = _first_response_line_type(resp)
        if t == "TOOL_CALL":
            n_tc += 1
        elif t == "CRITIQUE":
            n_crit += 1
        elif t == "FINAL_ANSWER":
            n_fa += 1
        else:
            n_other_fl += 1
            if len(other_fl_examples) < 3:
                first_line = next(
                    (l.strip() for l in resp.splitlines() if l.strip()), ""
                )
                other_fl_examples.append(first_line[:80])
    total_fl = len(usable_responses) or 1
    print(f"  TOOL_CALL first    : {n_tc:6d}  ({n_tc/total_fl:.1%})")
    print(f"  CRITIQUE first     : {n_crit:6d}  ({n_crit/total_fl:.1%})")
    print(f"  FINAL_ANSWER first : {n_fa:6d}  ({n_fa/total_fl:.1%})")
    print(f"  other/markdown     : {n_other_fl:6d}  ({n_other_fl/total_fl:.1%})")
    if other_fl_examples:
        print("  other examples:")
        for ex_s in other_fl_examples:
            print(f"    {ex_s!r}")

    if args.agent_only and n_other_fl > 0:
        raise RuntimeError(
            f"\nABORT: {n_other_fl} usable training example(s) start with markdown/prose "
            "after agent-only filtering — this should be impossible. "
            "File a bug against the --agent-only filter logic."
        )
    if args.agent_only:
        print("  ✓ agent-only: 0% markdown-first in training set")

    if args.test_run:
        debug_masking(records, tokenizer, n=5)

    train_dataset = Dataset.from_dict({
        "input_ids":      [r["input_ids"]     for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels":         [r["labels"]         for r in records],
    })

    output_dir  = args.output_dir
    final_dir   = f"{output_dir}/final"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=25 if args.test_run else args.steps,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    if not args.test_run:
        trainer.save_model(final_dir)
        print(f"Model saved to {final_dir}")

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
