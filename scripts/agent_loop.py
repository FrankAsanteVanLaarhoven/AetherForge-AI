"""
scripts/agent_loop.py — Agentic Software Engineering loop for AetherForge.

Implements the pattern that makes Cursor feel magical:
  Plan → Write → Execute → Read error → Fix → Re-execute → Verify → Done

The model generates structured text; this loop:
  1. Parses TOOL_CALL lines and executes real tools (subprocess / file I/O)
  2. Injects the OBSERVATION back into context
  3. Detects test-pass or FINAL_ANSWER and stops

Tool-use format (model generates; loop injects OBSERVATION):
  <think>reasoning</think>
  TOOL_CALL: execute_code({"code": "..."})
  OBSERVATION: <stdout/stderr injected here>
  TOOL_CALL: write_file({"path": "...", "content": "..."})
  OBSERVATION: OK — wrote N bytes
  FINAL_ANSWER: <answer to the user>

Usage:
    # Interactive REPL
    conda run -n ml-torch python scripts/agent_loop.py --interactive

    # Single task
    conda run -n ml-torch python scripts/agent_loop.py \
        --task "Write a function to merge two sorted lists and verify it with tests."

    # With a specific AetherForge checkpoint
    conda run -n ml-torch python scripts/agent_loop.py \
        --checkpoint outputs/aetherforge_code_agent/final/model.pt \
        --config     outputs/aetherforge_code_agent/final/config.json \
        --task "..."

    # Fast path: HuggingFace model (Qwen2.5-0.5B LoRA fine-tuned)
    conda run -n ml-torch python scripts/agent_loop.py \
        --hf-model outputs/qwen_code_agent/final \
        --interactive

    # Fast path: raw base model (no fine-tuning, useful for debugging)
    conda run -n ml-torch python scripts/agent_loop.py \
        --hf-model Qwen/Qwen2.5-0.5B-Instruct \
        --task "Write a fizzbuzz function and run it."
"""

import argparse
import json
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from aetherforge.model import AetherForge

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_AGENT_STEPS = 10
MAX_GEN_TOKENS  = 256
TEMPERATURE     = 0.3
TOP_P           = 0.92
REP_PENALTY     = 1.15
TOOL_TIMEOUT    = 20   # seconds per execution

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_execute_code(args: dict) -> str:
    """Run Python code in a subprocess, return stdout + stderr."""
    code = args.get("code", "").strip()
    if not code:
        return "ERROR: empty code"
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=TOOL_TIMEOUT,
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if err and not out:
            return f"ERROR:\n{err}"
        if err:
            return f"{out}\n\nSTDERR:\n{err}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {TOOL_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


def tool_run_script(args: dict) -> str:
    """Run a .py script file, return stdout + stderr."""
    path = args.get("path", "")
    if not Path(path).exists():
        return f"ERROR: file not found: {path}"
    try:
        r = subprocess.run(
            [sys.executable, path] + args.get("args", []),
            capture_output=True, text=True, timeout=TOOL_TIMEOUT,
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if err and not out:
            return f"ERROR:\n{err}"
        return f"{out}\n{err}".strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {TOOL_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


def tool_run_tests(args: dict) -> str:
    """Run pytest on a file or directory; prepend a PASS/FAIL summary line."""
    target = args.get("path", args.get("test_file", "."))
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", target, "--tb=short", "-q", "--no-header"],
            capture_output=True, text=True, timeout=60,
        )
        output = (r.stdout + r.stderr).strip()
        lines  = output.splitlines()

        # Parse pytest exit code: 0=all passed, 1=some failed, 2=interrupted,
        # 3=internal error, 4=bad usage, 5=no tests collected.
        EXIT_LABEL = {0: "ALL PASSED", 1: "SOME FAILED", 2: "INTERRUPTED",
                      3: "INTERNAL ERROR", 4: "USAGE ERROR", 5: "NO TESTS COLLECTED"}
        label = EXIT_LABEL.get(r.returncode, f"EXIT {r.returncode}")

        # Extract the short summary line if present (e.g. "3 passed in 0.12s")
        summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l
                        or "error" in l.lower()), "")
        header  = f"[{label}] {summary}".strip()

        # Keep the last 45 lines of body for context
        body = "\n".join(lines[-45:]) if len(lines) > 45 else output
        return f"{header}\n{body}" if body else header
    except subprocess.TimeoutExpired:
        return "ERROR: pytest timed out (60s)"
    except FileNotFoundError:
        return "ERROR: pytest not installed (pip install pytest)"
    except Exception as e:
        return f"ERROR: {e}"


def tool_read_file(args: dict) -> str:
    path = Path(args.get("path", ""))
    if not path.exists():
        return f"ERROR: not found: {path}"
    try:
        content = path.read_text(errors="replace")
        if len(content) > 3000:
            return content[:3000] + f"\n[...truncated — {len(content)} total chars]"
        return content
    except Exception as e:
        return f"ERROR: {e}"


def tool_write_file(args: dict) -> str:
    path    = Path(args.get("path", ""))
    content = args.get("content", "")
    if not path.parent.exists():
        try:
            path.parent.mkdir(parents=True)
        except Exception as e:
            return f"ERROR: cannot create directory {path.parent}: {e}"
    try:
        path.write_text(content)
        return f"OK — wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_list_files(args: dict) -> str:
    path = Path(args.get("path", "."))
    if not path.exists():
        return f"ERROR: not found: {path}"
    try:
        entries = sorted(
            f"{'[dir] ' if p.is_dir() else ''}{p.name}"
            for p in path.iterdir()
        )
        return "\n".join(entries) if entries else "(empty)"
    except Exception as e:
        return f"ERROR: {e}"


def tool_run_command(args: dict) -> str:
    """Run a shell command (restricted: no rm -rf, no sudo)."""
    cmd = args.get("command", "").strip()
    if not cmd:
        return "ERROR: empty command"
    # Basic safety guard
    BLOCKED = ["rm -rf", "sudo", "chmod 777", "dd if=", "> /dev/"]
    for b in BLOCKED:
        if b in cmd:
            return f"ERROR: blocked command pattern '{b}'"
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=TOOL_TIMEOUT, cwd=str(Path.cwd()),
        )
        out = (r.stdout + r.stderr).strip()
        return out[:2000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {TOOL_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


def tool_edit_file(args: dict) -> str:
    """Replace the first occurrence of old_str with new_str in a file."""
    path    = Path(args.get("path", ""))
    old_str = args.get("old", "")
    new_str = args.get("new", "")
    if not path.exists():
        return f"ERROR: not found: {path}"
    if not old_str:
        return "ERROR: 'old' string is empty — provide the text to replace"
    try:
        content = path.read_text(errors="replace")
        if old_str not in content:
            return (f"ERROR: string not found in {path}\n"
                    f"Hint: use read_file first to see exact contents")
        new_content = content.replace(old_str, new_str, 1)
        path.write_text(new_content)
        return f"OK — replaced 1 occurrence in {path}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_search_file(args: dict) -> str:
    """Grep for a pattern in a file or directory (Python files by default)."""
    pattern = args.get("pattern", "")
    path    = args.get("path", ".")
    if not pattern:
        return "ERROR: provide 'pattern'"
    try:
        r = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, path],
            capture_output=True, text=True, timeout=10,
        )
        result = (r.stdout + r.stderr).strip()
        if not result:
            return f"No matches for '{pattern}' in {path}"
        lines = result.splitlines()
        if len(lines) > 30:
            result = "\n".join(lines[:30]) + f"\n[...{len(lines)-30} more lines truncated]"
        return result
    except FileNotFoundError:
        return "ERROR: grep not found — try run_command with 'grep'"
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out"
    except Exception as e:
        return f"ERROR: {e}"


def tool_install_package(args: dict) -> str:
    """Install a Python package via pip (user scope)."""
    package = args.get("package", "").strip()
    if not package:
        return "ERROR: provide 'package'"
    # Block shell metacharacters and known dangerous names.
    if any(c in package for c in (";", "&", "|", ">", "<", "\n", " ")):
        return f"ERROR: invalid package name '{package}'"
    BLOCKED = {"rm", "wget", "curl", "bash", "sh", "os", "sys"}
    if package.lower().split("[")[0] in BLOCKED:
        return f"ERROR: blocked name '{package}'"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--user", package],
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            return f"pip error:\n{out[-600:]}"
        return out[-400:] if len(out) > 400 else (out or f"Successfully installed {package}")
    except subprocess.TimeoutExpired:
        return "ERROR: install timed out (120s)"
    except Exception as e:
        return f"ERROR: {e}"


def tool_format_code(args: dict) -> str:
    """Format Python code with black. Returns the formatted source."""
    import tempfile, os
    code = args.get("code", "").strip()
    if not code:
        return "ERROR: empty code"
    fd, fname = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        Path(fname).write_text(code)
        r = subprocess.run(
            [sys.executable, "-m", "black", "--quiet", "--line-length", "88", fname],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            err = (r.stdout + r.stderr).strip()
            os.unlink(fname)
            return f"black error: {err}"
        formatted = Path(fname).read_text()
        os.unlink(fname)
        return formatted
    except FileNotFoundError:
        try:
            os.unlink(fname)
        except OSError:
            pass
        return "black not installed — run: install_package({\"package\": \"black\"})"
    except Exception as e:
        try:
            os.unlink(fname)
        except OSError:
            pass
        return f"ERROR: {e}"


def tool_git_status(args: dict) -> str:
    """Show git working tree status (short format)."""
    path = args.get("path", ".")
    try:
        r = subprocess.run(
            ["git", "-C", path, "status", "--short"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return f"git error: {r.stderr.strip()}"
        return r.stdout.strip() or "Nothing to commit, working tree clean."
    except FileNotFoundError:
        return "ERROR: git not available in this environment"
    except subprocess.TimeoutExpired:
        return "ERROR: git status timed out"
    except Exception as e:
        return f"ERROR: {e}"


def tool_git_diff(args: dict) -> str:
    """Show unstaged (or --staged) diff for a path."""
    path   = args.get("path", ".")
    staged = bool(args.get("staged", False))
    try:
        base = ["git", "diff"]
        if staged:
            base.append("--cached")

        # Stat first for a compact summary.
        stat_r = subprocess.run(
            base + ["--stat", path], capture_output=True, text=True, timeout=10,
        )
        diff_r = subprocess.run(
            base + [path], capture_output=True, text=True, timeout=10,
        )
        stat = stat_r.stdout.strip()
        diff = diff_r.stdout.strip()

        if not stat and not diff:
            return "No changes."
        if diff:
            lines = diff.splitlines()
            if len(lines) > 50:
                diff = "\n".join(lines[:50]) + f"\n[...{len(lines)-50} more lines]"
        return f"{stat}\n\n{diff}".strip() if stat else diff
    except FileNotFoundError:
        return "ERROR: git not available"
    except Exception as e:
        return f"ERROR: {e}"


def tool_git_log(args: dict) -> str:
    """Show recent git commits (oneline format)."""
    n    = min(int(args.get("n", 5)), 20)
    path = args.get("path", ".")
    try:
        r = subprocess.run(
            ["git", "-C", path, "log", f"--max-count={n}",
             "--oneline", "--no-decorate"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return f"git log error: {r.stderr.strip()}"
        return r.stdout.strip() or "No commits yet."
    except FileNotFoundError:
        return "ERROR: git not available"
    except Exception as e:
        return f"ERROR: {e}"


def tool_check_syntax(args: dict) -> str:
    """Compile-check Python code without running it."""
    import ast
    code = args.get("code", "").strip()
    if not code:
        return "ERROR: empty code"
    try:
        ast.parse(code)
        return "Syntax OK."
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


def tool_run_linter(args: dict) -> str:
    """Run flake8 on code (inline or from a file path)."""
    import ast as _ast, tempfile, os
    code = args.get("code", "").strip()
    path = args.get("path", "").strip()
    if code:
        fd, fname = tempfile.mkstemp(suffix=".py")
        os.close(fd)
        try:
            Path(fname).write_text(code)
        except Exception as e:
            os.unlink(fname)
            return f"ERROR: {e}"
    elif path:
        fname = path
    else:
        return "ERROR: provide 'code' or 'path'"
    try:
        r = subprocess.run(
            [sys.executable, "-m", "flake8",
             "--max-line-length=100", "--ignore=E501,W503", fname],
            capture_output=True, text=True, timeout=15,
        )
        result = (r.stdout + r.stderr).strip()
        if code:
            # Replace tempfile path with <code> for readability
            result = result.replace(fname, "<code>")
            os.unlink(fname)
        return result or "No linting issues found."
    except FileNotFoundError:
        if code:
            os.unlink(fname)
        return "flake8 not installed (pip install flake8)"
    except subprocess.TimeoutExpired:
        return "ERROR: linter timed out"
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = {
    "execute_code":    tool_execute_code,
    "run_script":      tool_run_script,
    "run_tests":       tool_run_tests,
    "read_file":       tool_read_file,
    "write_file":      tool_write_file,
    "edit_file":       tool_edit_file,
    "search_file":     tool_search_file,
    "list_files":      tool_list_files,
    "run_command":     tool_run_command,
    "check_syntax":    tool_check_syntax,
    "run_linter":      tool_run_linter,
    "install_package": tool_install_package,
    "format_code":     tool_format_code,
    "git_status":      tool_git_status,
    "git_diff":        tool_git_diff,
    "git_log":         tool_git_log,
}


def dispatch_tool(call_str: str) -> str:
    """Parse 'name({"key": "val"})' and invoke the tool."""
    m = re.match(r"(\w+)\s*\((.+)\)\s*$", call_str.strip(), re.DOTALL)
    if not m:
        return f"ERROR: cannot parse: {call_str!r}"
    name, raw = m.group(1), m.group(2).strip()
    if name not in TOOLS:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt single-quote fix
        try:
            args = json.loads(raw.replace("'", '"'))
        except Exception:
            return f"ERROR: invalid JSON in tool args: {raw!r}"
    return TOOLS[name](args)

def _extract_pending_tool_call(
    context: str,
    last_processed_tool_pos: int,
) -> tuple[int, str, int]:
    """Find the next unprocessed TOOL_CALL in the full context.

    Uses a quote-aware, escape-aware character scanner so that parentheses,
    stop markers, and the word OBSERVATION inside quoted string arguments
    are never miscounted.  Handles code strings such as:
        execute_code({"code": "def f(x):\\n    return x\\nprint(f(''))"})
        execute_code({"code": "x = (1 + 2"})   # unbalanced paren inside string
        execute_code({"code": "print('OBSERVATION: fake')"})

    Scanner state (only active outside a quoted string):
      depth    — paren nesting; starts at 0, becomes 0 again at call end
      in_str   — True while inside a single- or double-quoted string
      quote_ch — which quote char opened the current string
      escaped  — True when previous in-string char was an unescaped backslash;
                 next char is skipped regardless of its value

    Returns (marker_pos, call_str, abs_end) or (-1, "", -1) when not ready.
    """
    marker = "TOOL_CALL:"
    pos = context.find(marker, last_processed_tool_pos + 1)
    if pos == -1:
        return -1, "", -1

    start = pos + len(marker)
    tail  = context[start:]

    # Skip leading whitespace (including newlines); record offset for abs_end.
    stripped   = tail.lstrip()
    leading_ws = len(tail) - len(stripped)

    if not stripped:
        return pos, "", -1  # call body not yet generated

    # Extract tool name (contiguous word chars).
    name_end = 0
    while name_end < len(stripped) and (stripped[name_end].isalnum() or stripped[name_end] == "_"):
        name_end += 1
    if not name_end:
        return pos, "", -1

    # Find opening paren; allow optional horizontal whitespace.
    after_name = stripped[name_end:]
    aw = len(after_name) - len(after_name.lstrip(" \t"))
    if aw >= len(after_name) or after_name[aw] != "(":
        return pos, "", -1  # opening paren not yet in context

    scan_start = name_end + aw  # index of '(' in `stripped`

    # ── Quote-aware, escape-aware paren scanner ──────────────────────────
    depth    = 0
    in_str   = False
    quote_ch = ""
    escaped  = False
    call_end = -1

    for i, ch in enumerate(stripped[scan_start:], scan_start):
        if escaped:
            escaped = False
            continue
        if in_str:
            if ch == "\\":
                escaped = True   # next char is a literal — skip it
            elif ch == quote_ch:
                in_str = False   # found the closing quote
            # All other chars (parens, braces, markers) are ignored in strings.
            continue
        # ── Outside a quoted string ───────────────────────────────────────
        if ch in ('"', "'"):
            in_str   = True
            quote_ch = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                call_end = i + 1   # exclusive, relative to `stripped`
                break

    if call_end == -1:
        return pos, "", -1  # unclosed paren — call body not yet complete in context

    call_str = stripped[:call_end]
    abs_end  = start + leading_ws + call_end  # absolute position in `context`

    return pos, call_str, abs_end


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)
    model = AetherForge(**cfg).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=False
    )
    return model, tokenizer


@torch.no_grad()
def generate_next(model, tokenizer, ids: list[int], max_new: int) -> str:
    """Generate up to max_new tokens, stop at EOS or a STOP_AT string."""
    STOP_AT = ["TOOL_CALL:", "FINAL_ANSWER:", "\nOBSERVATION:"]

    t = torch.tensor([ids], device=DEVICE, dtype=torch.long)
    attn = torch.ones_like(t)
    out = model.generate(
        t,
        attention_mask=attn,
        max_new_tokens=max_new,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=REP_PENALTY,
    )
    new_ids = out[0, len(ids):].tolist()
    text    = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Trim at first stop string (keep the trigger itself — agent loop needs it)
    earliest = len(text)
    trigger  = ""
    for stop in STOP_AT:
        idx = text.find(stop)
        if 0 <= idx < earliest:
            earliest = idx
            trigger  = stop
    if trigger:
        text = text[:earliest + len(trigger)]
    return text


# ---------------------------------------------------------------------------
# HuggingFace model path (fast path: Qwen2.5-0.5B-Instruct + optional LoRA)
# ---------------------------------------------------------------------------

def load_hf_model(model_name_or_path: str, lora_path: str = None):
    """Load any HF CausalLM (+ optional PEFT LoRA adapter) for the agent loop."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_bf16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    dtype    = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name_or_path}  (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map=DEVICE,
        trust_remote_code=False,
    )

    if lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def _hf_initial_context(
    tokenizer,
    task: str,
    system_prompt: str = None,
    memory_block: str = None,
) -> str:
    """Format the opening context using the model's chat template.

    memory_block: if provided, appended to the system prompt as guidance.
    It appears as instructions, NOT as an OBSERVATION.
    """
    sys_content = system_prompt if system_prompt is not None else SYSTEM
    if memory_block:
        sys_content = sys_content.rstrip() + "\n\n" + memory_block
    msgs = [
        {"role": "system", "content": sys_content},
        {"role": "user",   "content": task},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"{sys_content}\n### Task\n{task}\n\n### Agent\n"


@torch.no_grad()
def generate_next_hf(model, tokenizer, context: str, max_new: int) -> str:
    """Generate the next agent chunk from a HuggingFace CausalLM."""
    STOP_AT = ["TOOL_CALL:", "FINAL_ANSWER:", "\nOBSERVATION:"]

    enc = tokenizer(context, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(DEVICE)
    attn = enc.attention_mask.to(DEVICE) if hasattr(enc, "attention_mask") else torch.ones_like(ids)

    out = model.generate(
        ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, ids.shape[1]:].tolist()
    text    = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Trim at the first stop string, but keep the trigger so the loop can react.
    earliest = len(text)
    trigger  = ""
    for stop in STOP_AT:
        idx = text.find(stop)
        if 0 <= idx < earliest:
            earliest = idx
            trigger  = stop
    if trigger:
        text = text[:earliest + len(trigger)]
    return text


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM = textwrap.dedent("""\
You are AetherForge Code Agent. You solve programming tasks by actually running
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
       → Solution OK.   OR   → Fix needed: <specific issue>
  8. FINAL_ANSWER: <concise verified answer>

Rules:
  - Never claim code works without running it
  - Never give up after one error — diagnose and fix
  - check_syntax before execute_code for complex or multi-class code
  - install_package only when an ImportError confirms the package is missing
  - git_* tools only work in a git repository
  - If CRITIQUE ends with "Fix needed:", go back to TOOL_CALL before FINAL_ANSWER
  - Only call registered tools listed above. Never call the task function as a tool.
    Invalid: TOOL_CALL: is_palindrome({"s": "racecar"})
    Valid:   TOOL_CALL: execute_code({"code": "def is_palindrome(s): return s == s[::-1]\\nassert is_palindrome('racecar')\\nassert not is_palindrome('hello')\\nprint('PASS')"})
    Invalid: TOOL_CALL: sum_list({"lst": [1, 2, 3]})
    Valid:   TOOL_CALL: execute_code({"code": "def sum_list(lst): return sum(lst)\\nassert sum_list([1,2,3]) == 6\\nprint('PASS', sum_list([1,2,3]))"})
  - Use assert statements to verify correctness — do not only print results.
""")

# Strict contract: forces tool-first behaviour, disallows markdown preambles.
STRICT_SYSTEM = textwrap.dedent("""\
You are AetherForge Code Agent. You solve programming tasks by executing code.

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
      INVALID: TOOL_CALL: sum_list({"lst": [1,2,3]})
      VALID:   TOOL_CALL: execute_code({"code": "def sum_list(lst):\\n    return sum(lst)\\nassert sum_list([1,2,3,4,5]) == 15\\nprint('PASS')"})

Available tools: execute_code, run_script, run_tests, read_file, write_file, edit_file,
  search_file, list_files, run_command, check_syntax, run_linter, install_package,
  format_code, git_status, git_diff, git_log
""")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

class AgentResult:
    def __init__(self, answer: str, steps: int, tool_calls: list[dict],
                 critique: str = "", assistant_text: str = ""):
        self.answer         = answer
        self.steps          = steps
        self.tool_calls     = tool_calls
        self.critique       = critique        # last CRITIQUE: block, if any
        self.assistant_text = assistant_text  # model-generated portion only (no system/user prompt)

    def __str__(self):
        return self.answer


_MAX_CONSECUTIVE_ERRORS = 3  # inject hint after this many consecutive ERRORs
_MAX_CALL_REPEATS       = 4  # break loop if the identical call repeats this many times


def run_agent(
    model, tokenizer,
    task: str,
    verbose: bool = True,
    max_steps: int = MAX_AGENT_STEPS,
    is_hf: bool = False,
    force_critique: bool = True,
    system_prompt: str = None,
    stop_after_pass: bool = False,
    memory_state: dict = None,
    memory_top_k: int = 4,
) -> AgentResult:
    """Run the agent loop.

    force_critique: if True and the model skips CRITIQUE: before FINAL_ANSWER,
    intercept and inject a CRITIQUE: prompt to make it self-check first.
    system_prompt: override the default SYSTEM prompt (e.g. STRICT_SYSTEM).
    stop_after_pass: if True, immediately emit FINAL_ANSWER when OBSERVATION: PASS
    is received, without waiting for the model to generate one.
    memory_state: pre-loaded memory index state from memory.store.load_index().
    When provided, top-k verified examples are retrieved and inserted as
    RETRIEVED_VERIFIED_MEMORY: guidance in the system prompt.  This is NOT an
    OBSERVATION; the model must still produce a real TOOL_CALL.
    """
    # ── Optional memory retrieval ─────────────────────────────────────────
    memory_block = None
    if memory_state is not None:
        try:
            from memory.embed import embed_query
            from memory.store import search
            from memory.core import format_memory_block
            qv = embed_query(task, memory_state["query_texts"], memory_state.get("vocab"))
            hits = search(memory_state, qv, memory_top_k)
            hits = [h for h in hits if h.get("verified", False)]
            if hits:
                memory_block = format_memory_block(hits)
                if verbose:
                    print(f"\n[memory] Retrieved {len(hits)} verified example(s)")
        except Exception as exc:
            if verbose:
                print(f"\n[memory] Warning: retrieval failed ({exc}), continuing without memory")

    sys = system_prompt if system_prompt is not None else SYSTEM
    context = (
        _hf_initial_context(tokenizer, task, system_prompt=sys, memory_block=memory_block)
        if is_hf
        else (
            (sys + ("\n\n" + memory_block if memory_block else ""))
            + f"\n### Task\n{task}\n\n### Agent\n"
        )
    )
    initial_len      = len(context)   # length of system+user prompt — everything after is assistant text
    tool_calls: list[dict]      = []
    call_counts: dict[str, int] = {}
    consecutive_errs = 0
    final_ans        = ""
    # Ignore TOOL_CALL examples already present in the initial prompt.
    # Only execute TOOL_CALL markers generated after this point.
    last_processed_tool_pos = context.rfind("TOOL_CALL:")

    if verbose:
        bar = "=" * 64
        print(f"\n{bar}\nTask: {task}\n{bar}")

    for step in range(max_steps):
        # Generate next agent chunk
        if is_hf:
            text = generate_next_hf(model, tokenizer, context, MAX_GEN_TOKENS)
        else:
            ids  = tokenizer.encode(context, add_special_tokens=False)
            text = generate_next(model, tokenizer, ids, MAX_GEN_TOKENS)

        if not text.strip():
            break
        if verbose:
            print(text, end="", flush=True)
        context += text

        # Determine which marker comes first in this generated chunk so we
        # never exit on FINAL_ANSWER when a TOOL_CALL precedes it.
        tc_in_chunk = text.find("TOOL_CALL:")
        fa_in_chunk = text.find("FINAL_ANSWER:")
        tool_first  = tc_in_chunk != -1 and (fa_in_chunk == -1 or tc_in_chunk < fa_in_chunk)

        # ── TOOL_CALL (scan full context; handles split body across steps) ──
        pending_pos, pending_call, pending_end = _extract_pending_tool_call(
            context, last_processed_tool_pos
        )
        if pending_call:
            call_line = pending_call
            last_processed_tool_pos = pending_pos

            # Truncate context to the end of the call body, removing any
            # model-hallucinated OBSERVATION / FINAL_ANSWER that follows.
            # The runtime is the only source of truth for OBSERVATION text.
            context = context[:pending_end].rstrip()

            call_counts[call_line] = call_counts.get(call_line, 0) + 1
            if call_counts[call_line] >= _MAX_CALL_REPEATS:
                hint = (
                    f"\n[SYSTEM HINT: the same tool call has been repeated "
                    f"{_MAX_CALL_REPEATS} times without progress. "
                    f"Try a completely different approach.]\n"
                )
                context += hint
                if verbose:
                    print(hint, flush=True)
                break

            t0      = time.time()
            obs     = dispatch_tool(call_line)
            elapsed = time.time() - t0

            tool_calls.append({"call": call_line, "obs": obs, "sec": round(elapsed, 2)})

            if obs.strip().startswith("PASS") and stop_after_pass:
                obs_block = f"\nOBSERVATION: {obs}\n"
                if verbose:
                    print(obs_block, end="", flush=True)
                context += obs_block
                final_ans = "Verified with execute_code: PASS."
                break

            if obs.startswith("ERROR"):
                consecutive_errs += 1
                if consecutive_errs >= _MAX_CONSECUTIVE_ERRORS:
                    obs += (
                        "\n[SYSTEM HINT: multiple consecutive errors. "
                        "Re-read the full traceback above, identify the root cause, "
                        "and try a different fix.]"
                    )
                    consecutive_errs = 0
            else:
                consecutive_errs = 0

            obs_block = f"\nOBSERVATION: {obs}\n"
            if verbose:
                print(obs_block, end="", flush=True)
            context += obs_block
            continue

        # TOOL_CALL: visible in context but body not yet parseable (split across
        # generation steps, or paren not closed yet).
        if pending_pos != -1:
            # Requirement: strip any model-written OBSERVATION that appears after
            # an incomplete TOOL_CALL so it cannot mislead the next generation.
            fake_obs = context.find("\nOBSERVATION:", pending_pos)
            if fake_obs != -1:
                context = context[:fake_obs]
                if verbose:
                    print(
                        "\n[DEBUG] Stripped model-written OBSERVATION after "
                        "incomplete TOOL_CALL — awaiting real dispatch.",
                        flush=True,
                    )

        if "TOOL_CALL:" in text:
            if verbose:
                print(
                    f"\n[DEBUG] TOOL_CALL found in chunk (offset {tc_in_chunk}) "
                    "but body incomplete — waiting for next generation step.",
                    flush=True,
                )
            if tool_first:
                # TOOL_CALL precedes FINAL_ANSWER in this chunk; wait for the
                # body rather than exiting on the answer prematurely.
                continue

        # ── FINAL_ANSWER ─────────────────────────────────────────────────
        # Skipped entirely when TOOL_CALL appears before FINAL_ANSWER in the
        # same chunk — the tool must execute first.
        if "FINAL_ANSWER:" in text and not tool_first:
            # Intercept: if no CRITIQUE in recent context, force one first.
            if force_critique and "CRITIQUE:" not in context[-600:]:
                _fa_idx = text.rfind("FINAL_ANSWER:")
                # Roll context back to just before FINAL_ANSWER:
                context = context[: len(context) - (len(text) - _fa_idx)]
                context += "\nCRITIQUE:"
                if verbose:
                    print("\nCRITIQUE:", end="", flush=True)
                if is_hf:
                    crit = generate_next_hf(model, tokenizer, context, 220)
                else:
                    ids2 = tokenizer.encode(context, add_special_tokens=False)
                    crit = generate_next(model, tokenizer, ids2, 220)
                context += crit
                if verbose:
                    print(crit, end="", flush=True)

                # If CRITIQUE demands a fix, nudge toward TOOL_CALL so the
                # model doesn't immediately re-emit FINAL_ANSWER.
                if "fix needed" in crit.lower():
                    context += "\nTOOL_CALL:"
                    if verbose:
                        print("\nTOOL_CALL:", end="", flush=True)
                continue

            tail_start = text.rfind("FINAL_ANSWER:") + len("FINAL_ANSWER:")
            tail = text[tail_start:].strip()
            if not tail:
                if is_hf:
                    extra = generate_next_hf(model, tokenizer, context, 256)
                else:
                    ids2  = tokenizer.encode(context, add_special_tokens=False)
                    extra = generate_next(model, tokenizer, ids2, 256)
                context += extra
                tail = extra.strip()
                if verbose:
                    print(extra, end="", flush=True)
            final_ans = tail
            break

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"Steps: {step + 1}  |  Tool calls: {len(tool_calls)}")

    # Extract the last CRITIQUE: block from context so _score_result can use it.
    critique = ""
    if "CRITIQUE:" in context:
        crit_start = context.rfind("CRITIQUE:") + len("CRITIQUE:")
        crit_end   = context.find("FINAL_ANSWER:", crit_start)
        critique   = context[crit_start: crit_end if crit_end != -1 else len(context)].strip()

    # assistant_text: everything generated after the initial system+user prompt.
    assistant_text = context[initial_len:]

    # answer: text after the last FINAL_ANSWER: — never the full context.
    if "FINAL_ANSWER:" in context:
        answer = final_ans or context.split("FINAL_ANSWER:")[-1].strip()
    else:
        answer = final_ans or ""

    return AgentResult(
        answer         = answer,
        steps          = step + 1,
        tool_calls     = tool_calls,
        critique       = critique,
        assistant_text = assistant_text,
    )


# ---------------------------------------------------------------------------
# Best-of-N: run N independent trajectories, score by execution success
# ---------------------------------------------------------------------------

def _score_result(result: AgentResult) -> tuple:
    """Higher is better. Tuple comparison gives lexicographic priority."""
    calls = result.tool_calls

    # Primary: did run_tests report all-passing? (checks new [ALL PASSED] header
    # and legacy "X passed / no failed" pytest output)
    def _tests_ok(obs: str) -> bool:
        lo = obs.lower()
        return ("[all passed]" in lo
                or ("passed" in lo and "failed" not in lo and "error" not in lo))

    tests_passed = any(
        c["call"].startswith("run_tests") and _tests_ok(c["obs"])
        for c in calls
    )

    # Secondary: CRITIQUE concluded positively (→ Solution OK.)
    critique_ok = "→ solution ok" in result.critique.lower()

    # Tertiary: did we actually reach FINAL_ANSWER?
    has_answer = bool(result.answer.strip())

    # Quaternary: last execute_code succeeded
    last_exec_ok = any(
        c["call"].startswith("execute_code") and not c["obs"].startswith("ERROR")
        for c in reversed(calls[-4:])
    )

    # Quinary: net successful tool calls
    n_ok  = sum(1 for c in calls if not c["obs"].startswith("ERROR"))
    n_err = sum(1 for c in calls if c["obs"].startswith("ERROR"))

    # Tiebreaker: fewer steps
    return (tests_passed, critique_ok, has_answer, last_exec_ok, n_ok - n_err, -result.steps)


def run_agent_best_of_n(
    model, tokenizer,
    task: str,
    n: int = 3,
    verbose: bool = True,
    max_steps: int = MAX_AGENT_STEPS,
    is_hf: bool = False,
    force_critique: bool = True,
    early_stop: bool = True,
    system_prompt: str = None,
    stop_after_pass: bool = False,
    memory_state: dict = None,
    memory_top_k: int = 4,
) -> AgentResult:
    """Generate up to N independent agent trajectories and return the best.

    Scoring: passing tests > positive CRITIQUE > has answer > last exec ok >
    net successful calls > fewest steps.

    early_stop: stop as soon as a candidate has all-passing tests — no need
    to run more candidates once we have a provably correct solution.
    """
    if verbose:
        print(f"\n[Best-of-{n}] sampling up to {n} trajectories …")

    _PERFECT = (True,)  # tests_passed=True is a perfect score on the primary key

    candidates: list[AgentResult] = []
    for i in range(n):
        result = run_agent(
            model, tokenizer, task,
            verbose=False,
            max_steps=max_steps,
            is_hf=is_hf,
            force_critique=force_critique,
            system_prompt=system_prompt,
            stop_after_pass=stop_after_pass,
            memory_state=memory_state,
            memory_top_k=memory_top_k,
        )
        score = _score_result(result)
        candidates.append(result)

        if verbose:
            tests_ok, crit_ok, has_ans, exec_ok, net_ok, _ = score
            print(
                f"  candidate {i+1}/{n}: "
                f"steps={result.steps}  "
                f"tests={'✓' if tests_ok else '✗'}  "
                f"critique={'✓' if crit_ok else '✗'}  "
                f"exec={'✓' if exec_ok else '✗'}  "
                f"net_ok={net_ok}"
            )

        # Early stop: tests pass → this is provably correct, no need to try more.
        if early_stop and score[0]:  # tests_passed
            if verbose:
                print(f"  [early stop] candidate {i+1} passes all tests")
            break

    best = max(candidates, key=_score_result)

    if verbose:
        winner_idx = candidates.index(best)
        print(f"\n[Best-of-{n}] Selected candidate {winner_idx + 1}  "
              f"(steps={best.steps}, tool_calls={len(best.tool_calls)})")
        print("\n" + "=" * 64)
        print("FINAL ANSWER:", best.answer)
        print("=" * 64)

    return best


# ---------------------------------------------------------------------------
# Benchmark: run on a set of coding tasks and measure pass rate
# ---------------------------------------------------------------------------

BENCHMARK_TASKS = [
    {
        "task": "Write a Python function `fizzbuzz(n)` that returns a list of strings "
                "'FizzBuzz' for multiples of 15, 'Fizz' for 3, 'Buzz' for 5, else the "
                "number as a string. Verify with execute_code.",
        "check": lambda obs_list: any("FizzBuzz" in o for o in obs_list),
    },
    {
        "task": "Write and test a function `is_prime(n)` that returns True for primes. "
                "Test it on 2, 3, 4, 17, 18.",
        "check": lambda obs_list: any("True" in o and "False" in o for o in obs_list),
    },
    {
        "task": "Write a Python function `binary_search(arr, target)` and demonstrate "
                "it finds 7 in [1, 3, 5, 7, 9, 11].",
        "check": lambda obs_list: any("3" in o or "True" in o for o in obs_list),
    },
    {
        "task": "Fix the bug in this code and verify it runs:\n"
                "```python\ndef sum_list(lst):\n    total = 0\n    for i in lst\n"
                "        total += i\n    return total\nprint(sum_list([1,2,3]))\n```",
        "check": lambda obs_list: any("6" in o for o in obs_list),
    },
    {
        "task": "Calculate the first 10 Fibonacci numbers using a generator "
                "and print them with execute_code.",
        "check": lambda obs_list: any("55" in o or "34" in o for o in obs_list),
    },
]


def run_benchmark(model, tokenizer, verbose: bool = False,
                  is_hf: bool = False) -> dict:
    passed = 0
    results = []
    for i, item in enumerate(BENCHMARK_TASKS):
        print(f"\nBenchmark {i+1}/{len(BENCHMARK_TASKS)}: {item['task'][:60]}...")
        result = run_agent(model, tokenizer, item["task"],
                           verbose=verbose, is_hf=is_hf)
        obs_list = [tc["obs"] for tc in result.tool_calls]
        ok = item["check"](obs_list)
        passed += ok
        results.append({"task": item["task"][:60], "passed": ok,
                        "steps": result.steps, "tool_calls": len(result.tool_calls)})
        print(f"  {'PASS' if ok else 'FAIL'}  ({result.steps} steps, "
              f"{len(result.tool_calls)} tool calls)")

    print(f"\nBenchmark: {passed}/{len(BENCHMARK_TASKS)} passed "
          f"({100*passed//len(BENCHMARK_TASKS)}%)")
    return {"pass_rate": passed / len(BENCHMARK_TASKS), "details": results}


# ---------------------------------------------------------------------------
# Unit test: chunk ordering
# ---------------------------------------------------------------------------

def _run_agent_with_chunks(chunks: list[str], task: str = "test",
                           max_steps: int = 8) -> "AgentResult":
    """Helper: run run_agent with a fake model that yields `chunks` in order."""
    import types
    import scripts.agent_loop as _al

    seq = [0]

    def _fake_generate(model, tokenizer, context, max_new):
        n = seq[0]
        seq[0] += 1
        if n < len(chunks):
            return chunks[n]
        return "FINAL_ANSWER: done"

    orig = _al.generate_next_hf
    _al.generate_next_hf = _fake_generate
    tok = types.SimpleNamespace(
        encode=lambda s, add_special_tokens=False: [0],
        eos_token_id=0,
    )
    try:
        return _al.run_agent(object(), tok, task,
                             verbose=False, is_hf=True,
                             force_critique=False, max_steps=max_steps)
    finally:
        _al.generate_next_hf = orig


def test_chunk_ordering() -> None:
    """TOOL_CALL + fake OBSERVATION + FINAL_ANSWER in one chunk.

    The loop must dispatch the real tool (obs '123') before processing
    FINAL_ANSWER.  Expected: 1 tool_call, obs contains '123', not 'fake'.
    """
    CHUNK = (
        'TOOL_CALL: execute_code({"code": "print(123)"})\n'
        'OBSERVATION: fake\n'
        'FINAL_ANSWER: done'
    )
    result = _run_agent_with_chunks([CHUNK])
    n = len(result.tool_calls)
    assert n == 1, f"Expected 1 tool call, got {n}"
    obs = result.tool_calls[0]["obs"]
    assert "123" in obs, f"Expected '123' in obs, got {obs!r}"
    assert obs != "fake", "Got fake obs — runtime injection did not occur"
    print("test_chunk_ordering: PASSED")


def test_quote_aware_parser() -> None:
    """Two sub-tests for the quote-aware paren scanner.

    A: code string with nested function calls and single-quote args.
       The unbalanced `(` inside the outer JSON string must not confuse depth.
    B: code string contains literal text 'OBSERVATION: fake'.
       The runtime must execute the real code, not use the model-written OBSERVATION.
    Both must produce exactly one real tool_call with non-fake observation.
    """
    sq = chr(39)  # single-quote char, avoids backslash escape confusion

    # ── A: parens + single-quoted args inside the JSON code string ────────
    # Actual TOOL_CALL text:
    #   TOOL_CALL: execute_code({"code": "def f(x):\n    return x\nprint(f('42'))"})
    # Single-quote delimiters inside the JSON value are valid (only " needs escaping).
    code_a = "def f(x):\\n    return x\\nprint(f(" + sq + "42" + sq + "))"
    CHUNK_A = (
        'TOOL_CALL: execute_code({"code": "' + code_a + '"})\n'
        + "OBSERVATION: fake_a\n"
        + "FINAL_ANSWER: done_a"
    )
    result_a = _run_agent_with_chunks([CHUNK_A])
    n_a = len(result_a.tool_calls)
    assert n_a == 1, f"[A] Expected 1 tool call, got {n_a}"
    obs_a = result_a.tool_calls[0]["obs"]
    assert obs_a != "fake_a", "[A] Got fake obs — runtime injection did not occur"
    print(f"test_quote_aware_parser [A] nested-parens: PASSED  (obs={obs_a!r})")

    # ── B: OBSERVATION: inside the code string + fake OBSERVATION after ───
    # Actual TOOL_CALL text:
    #   TOOL_CALL: execute_code({"code": "print('OBSERVATION: fake')"})
    code_b = "print(" + sq + "OBSERVATION: fake" + sq + ")"
    CHUNK_B = (
        'TOOL_CALL: execute_code({"code": "' + code_b + '"})\n'
        + "OBSERVATION: fake_b\n"
        + "FINAL_ANSWER: done_b"
    )
    result_b = _run_agent_with_chunks([CHUNK_B])
    n_b = len(result_b.tool_calls)
    assert n_b == 1, f"[B] Expected 1 tool call, got {n_b}"
    obs_b = result_b.tool_calls[0]["obs"]
    assert obs_b != "fake_b", "[B] Got fake obs — runtime injection did not occur"
    print(f"test_quote_aware_parser [B] OBSERVATION-in-code: PASSED  (obs={obs_b!r})")


def test_split_chunk() -> None:
    """TOOL_CALL: marker and call body arrive in separate generation steps.

    Step 1: model yields 'TOOL_CALL:'  (just the marker)
    Step 2: model yields the call body with nested parens and an unbalanced
            paren inside the code string: execute_code({"code": "x = (1+2"})
    The call must be dispatched (producing an ERROR is fine — but it must
    be counted as one tool_call entry, not zero).
    """
    CHUNK_1 = "TOOL_CALL:"
    CHUNK_2 = ' execute_code({"code": "x = (1+2"})\nFINAL_ANSWER: done'
    result = _run_agent_with_chunks([CHUNK_1, CHUNK_2])
    n = len(result.tool_calls)
    assert n == 1, f"Expected 1 tool call (possibly with ERROR), got {n}"
    print(f"test_split_chunk: PASSED  (obs={result.tool_calls[0]['obs'][:60]!r})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CKPT   = "outputs/aetherforge_1B_code_agent/final/model.pt"
DEFAULT_CONFIG = "outputs/aetherforge_1B_bpe/init/config.json"
FALLBACK_CKPT  = "outputs/aetherforge_code_agent/final/model.pt"
FALLBACK_CFG   = "outputs/aetherforge_code_agent/final/config.json"
FALLBACK2_CKPT = "outputs/aetherforge_distill_5k/final/model.pt"
FALLBACK2_CFG  = "outputs/aetherforge_distill_5k/final/config.json"


def main():
    p = argparse.ArgumentParser()
    # ── HF fast path ────────────────────────────────────────────────────
    p.add_argument("--hf-model",    default=None, dest="hf_model",
                   help="HuggingFace model ID or local path "
                        "(e.g. outputs/qwen_code_agent/final). "
                        "When set, --checkpoint / --config are ignored.")
    p.add_argument("--hf-lora",     default=None, dest="hf_lora",
                   help="PEFT LoRA adapter path to load on top of --hf-model")
    # ── AetherForge path ─────────────────────────────────────────────────
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--config",     default=DEFAULT_CONFIG)
    # ── Common ───────────────────────────────────────────────────────────
    p.add_argument("--task",       default="")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--benchmark",   action="store_true",
                   help="Run the built-in 5-task benchmark and report pass rate")
    p.add_argument("--test",        action="store_true",
                   help="Run the chunk-ordering smoke test (no model required)")
    p.add_argument("--verbose",     action="store_true", default=True)
    # ── Memory ───────────────────────────────────────────────────────────
    p.add_argument("--memory-enabled", action="store_true",
                   help="Enable offline vector memory retrieval")
    p.add_argument("--memory-index",   default="memory/index",
                   help="Directory containing the pre-built memory index")
    p.add_argument("--memory-top-k",   type=int, default=4,
                   help="Number of memory records to retrieve per task")
    args = p.parse_args()

    # ── Smoke test (no model needed) ─────────────────────────────────────
    if args.test:
        test_chunk_ordering()
        test_quote_aware_parser()
        test_split_chunk()
        sys.exit(0)

    # ── Memory loading ────────────────────────────────────────────────────
    memory_state = None
    if args.memory_enabled:
        try:
            from memory.store import load_index as _load_mem
            memory_state = _load_mem(Path(args.memory_index))
            n_mem = len(memory_state.get("records", []))
            print(f"[memory] Loaded {n_mem} verified records from {args.memory_index}")
        except FileNotFoundError as exc:
            print(f"[memory] ERROR: {exc}")
            print("[memory] Build the index first: python scripts/build_vector_memory.py")
            sys.exit(1)
        except Exception as exc:
            print(f"[memory] ERROR loading index: {exc}")
            sys.exit(1)

    # ── Model loading ────────────────────────────────────────────────────
    is_hf = bool(args.hf_model)

    if is_hf:
        model, tokenizer = load_hf_model(args.hf_model, lora_path=args.hf_lora)
        n_params = sum(p.numel() for p in model.parameters())
        vram     = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
        print(f"  {n_params/1e6:.1f}M params  {vram:.2f} GB VRAM  device={DEVICE}\n")
    else:
        # Cascade through AetherForge checkpoints: 1B → 128M code-agent → 128M distilled
        ckpt = args.checkpoint
        cfg  = args.config
        if not Path(ckpt).exists():
            print(f"Note: {ckpt} not found, trying 128M code-agent fallback")
            ckpt, cfg = FALLBACK_CKPT, FALLBACK_CFG
        if not Path(ckpt).exists():
            print(f"Note: {ckpt} not found, trying 128M distilled fallback")
            ckpt, cfg = FALLBACK2_CKPT, FALLBACK2_CFG
        if not Path(ckpt).exists():
            print(f"ERROR: no checkpoint found. Options:\n"
                  f"  AetherForge: make finetune-1B-code-agent\n"
                  f"  HF fast path: make finetune-qwen-code-agent")
            sys.exit(1)

        print(f"Loading {ckpt} ...")
        model, tokenizer = load_model(ckpt, cfg)
        n_params = sum(p.numel() for p in model.parameters())
        vram     = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
        print(f"  {n_params/1e6:.1f}M params  {vram:.2f} GB VRAM  device={DEVICE}\n")

    # ── Run ──────────────────────────────────────────────────────────────
    _mem_kwargs = {"memory_state": memory_state, "memory_top_k": args.memory_top_k}

    if args.benchmark:
        run_benchmark(model, tokenizer, verbose=False, is_hf=is_hf)

    elif args.interactive:
        label = "HF" if is_hf else "AetherForge"
        if memory_state:
            label += " + memory"
        print(f"Code Agent ({label}) — type your task, or 'quit'.\n")
        while True:
            try:
                task = input("Task> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task.lower() in ("quit", "exit", "q"):
                break
            if not task:
                continue
            r = run_agent(model, tokenizer, task, verbose=True, is_hf=is_hf, **_mem_kwargs)
            if r.answer:
                print(f"\nAnswer: {r.answer}\n")

    elif args.task:
        r = run_agent(model, tokenizer, args.task,
                      verbose=args.verbose, is_hf=is_hf, **_mem_kwargs)
        if r.answer:
            print(f"\nAnswer: {r.answer}")

    else:
        print("Provide --task, --interactive, --benchmark, or --test")
        sys.exit(1)


if __name__ == "__main__":
    main()
