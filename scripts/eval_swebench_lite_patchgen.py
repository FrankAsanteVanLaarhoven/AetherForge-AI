#!/usr/bin/env python3
"""eval_swebench_lite_patchgen.py — SWE-bench Lite Phase 2 patch generation.

IMPORTANT — what this script does and does NOT do:
  - DOES: load SWE-bench Lite tasks from Hugging Face
  - DOES: prepare a local repo workspace for each task (clone or stub)
  - DOES: run an AetherForge agent with repo-level tools (list_files,
          read_file, search_code, write_file, run_command, git_diff)
  - DOES: add a continuation guard so the agent is nudged past inspection
          into edit/diff steps when the patch remains empty after tool use
  - DOES: save predictions in the official SWE-bench format
  - DOES NOT: run the official Docker harness (required for verified scores)
  - DOES NOT: claim any issue is resolved

Official evaluation: https://github.com/princeton-nlp/SWE-bench
Phase plan: docs/SWEBENCH_PHASE2_PLAN.md

TODO — limitations at current AetherForge capability level:
  - The model was trained on function-level code tasks, not repo-level edits.
  - Expected resolution rate on official harness: 0% or near-0%.
  - That is expected and acceptable at this milestone.
  - True SWE-bench resolution requires better repo-level training data.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from swebench_repo_tools import dispatch, git_diff

_TRANSFORMERS_OK = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    _TRANSFORMERS_OK = True
except ImportError:
    pass

MAX_AGENT_STEPS = 12
MAX_NEW_TOKENS = 512
TOOL_CALL_RE = re.compile(r'TOOL_CALL:\s*(\w+)\s*\((\{.*?\})\)', re.DOTALL)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    """Load model + tokenizer. Returns (model, tokenizer) or (None, None)."""
    if not _TRANSFORMERS_OK:
        print("[warn] transformers/peft unavailable — stub mode")
        return None, None
    try:
        adapter_cfg = Path(model_path) / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg) as f:
                cfg = json.load(f)
            base = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
            print(f"[model] Loading base {base} + LoRA {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            print(f"[model] Loading {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
        model.eval()
        return model, tokenizer
    except Exception as exc:
        print(f"[warn] Model load failed: {exc}\n[warn] Running in stub mode")
        return None, None


# ---------------------------------------------------------------------------
# Repo workspace setup
# ---------------------------------------------------------------------------

def _prepare_workspace(output_dir: Path, instance_id: str,
                       repo: str, base_commit: str, stub: bool) -> tuple[Path, bool]:
    """Clone the repo at base_commit into a local workspace.

    Returns (workspace_path, cloned_ok).
    In stub mode always returns (workspace_path, False) without cloning.
    """
    workspace = output_dir / "workspaces" / instance_id
    workspace.mkdir(parents=True, exist_ok=True)

    if stub:
        return workspace, False

    if (workspace / ".git").exists():
        print(f"  [workspace] Already cloned: {workspace}")
        return workspace, True

    repo_url = f"https://github.com/{repo}.git"
    print(f"  [workspace] Cloning {repo_url} @ {base_commit} …")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(workspace)],
            check=True, capture_output=True, timeout=120,
        )
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", base_commit],
            check=True, capture_output=True, timeout=60, cwd=str(workspace),
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            check=True, capture_output=True, timeout=30, cwd=str(workspace),
        )
        return workspace, True
    except Exception as exc:
        print(f"  [workspace] Clone failed: {exc}")
        print("  [workspace] Continuing in stub mode for this task.")
        return workspace, False


# ---------------------------------------------------------------------------
# Agent prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert software engineer solving a real GitHub issue.
    You have access to the repository and the following tools:

    TOOL_CALL: list_files({"subdir": "."})
    TOOL_CALL: read_file({"path": "path/to/file.py"})
    TOOL_CALL: search_code({"pattern": "function_name", "include_glob": "*.py"})
    TOOL_CALL: write_file({"path": "path/to/file.py", "content": "..."})
    TOOL_CALL: run_command({"command": "python -m pytest tests/test_foo.py -x"})
    TOOL_CALL: git_diff({})

    Rules:
    - Always call tools using the exact TOOL_CALL format above.
    - Read relevant files before editing them.
    - After editing, call git_diff({}) to confirm your changes.
    - Do not guess; read the code first.
    - Do not claim PASS unless a test run confirms it.
""")


def _build_prompt(repo: str, problem_statement: str, hints: str,
                  workspace_path: Path, cloned: bool) -> str:
    repo_info = (
        f"Repository cloned to: {workspace_path}\nUse the tools to inspect it."
        if cloned else
        "NOTE: Repository is not available locally (clone failed or stub mode). "
        "Generate your best guess patch based on the issue description alone."
    )
    hints_block = f"\nHints:\n{hints[:800]}" if hints and hints.strip() else ""

    return textwrap.dedent(f"""\
        {_SYSTEM_PROMPT}

        Repository: {repo}
        {repo_info}
        {hints_block}

        Issue to fix:
        {problem_statement[:3000]}

        Start by searching for the relevant code, then read the relevant file(s),
        then write a minimal fix, then call git_diff({{}}) to show the patch.
    """)


def _extract_search_paths(search_output: str, max_paths: int = 5) -> list[str]:
    """Parse 'file:line: text' search output into unique relative file paths."""
    seen: list[str] = []
    for line in search_output.splitlines():
        parts = line.split(":")
        if len(parts) >= 2:
            candidate = parts[0].strip()
            if candidate and not candidate.startswith("/") and candidate not in seen:
                seen.append(candidate)
        if len(seen) >= max_paths:
            break
    return seen


def _continuation_message(tool_log: list[dict], last_search_output: str = "") -> str:
    """Generate a context-aware continuation nudge, including exact file paths."""
    successful = [e["tool"] for e in tool_log if e.get("ok")]
    failed_reads = [e for e in tool_log if e["tool"] == "read_file" and not e.get("ok")]

    # Extract exact paths from search results to counter hallucinated prefixes
    path_hint = ""
    if last_search_output:
        found_paths = _extract_search_paths(last_search_output)
        if found_paths:
            paths_str = ", ".join(f"'{p}'" for p in found_paths[:3])
            path_hint = (
                f" IMPORTANT: use the EXACT relative path from the search results: {paths_str}. "
                "Do NOT add 'src/' or any other prefix."
            )

    if "write_file" in successful:
        return (
            "You edited a file. Now call git_diff({}) to produce the patch. "
            "The patch will be submitted for evaluation."
        )
    if "read_file" in successful:
        return (
            f"You read a file.{path_hint} Now write a minimal fix using "
            "write_file({\"path\": \"...\", \"content\": \"...\"}), "
            "then call git_diff({}) to show the patch."
        )
    if failed_reads:
        # The model tried to read but failed — give it the exact correct path
        return (
            f"Your read_file calls failed.{path_hint} "
            "Use only the relative path shown in the search results above. "
            "Then write_file to apply the fix, then git_diff({})."
        )
    if "search_code" in successful or "list_files" in successful:
        return (
            f"You inspected the repository but have not produced a patch.{path_hint} "
            "Read the file with read_file, write the minimal fix with write_file, "
            "then call git_diff({}) to show the patch."
        )
    return (
        "You have not yet produced a patch. Search for the relevant code, "
        "read the file using its exact relative path, write the fix, "
        "then call git_diff({})."
    )


# ---------------------------------------------------------------------------
# Agent loop with continuation guard
# ---------------------------------------------------------------------------

def _run_one_step(model, tokenizer, context: str) -> str:
    """Generate one model step. Returns the new text."""
    inputs = tokenizer(context, return_tensors="pt",
                       truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _run_agent(
    model, tokenizer, prompt: str,
    workspace: Path, cloned: bool,
    max_steps: int = MAX_AGENT_STEPS,
    max_repo_steps: int = 6,
    continue_after_empty_patch: bool = True,
) -> tuple[str, list[dict], str, int]:
    """Run the agent loop with continuation guard.

    Returns (final_patch, tool_call_log, stopped_reason, continuation_attempts).
    """
    if model is None:
        return "", [], "stub_mode", 0

    context = prompt
    tool_log: list[dict] = []
    stopped_reason = "max_steps"
    _seen_calls: set[str] = set()  # duplicate detection
    _last_search_output: str = ""  # for continuation path hints

    def _dispatch_and_log(step: int, tool_name: str, tool_args: dict,
                          continuation: bool = False) -> tuple[dict, str]:
        """Run one tool call, inject observation, append to log. Returns (result, obs_text)."""
        if cloned:
            result = dispatch(str(workspace), tool_name, tool_args)
        else:
            result = {"ok": False, "stdout": "", "stderr": "Workspace not available (stub mode)"}

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        obs_text = stdout or stderr or "(no output)"
        if not result["ok"] and not obs_text.strip():
            obs_text = f"Tool failed: {result.get('stderr', 'unknown error')}"

        # Give read_file more context so the model can see the relevant code
        truncate = 4000 if tool_name == "read_file" else 2000
        context_line = f"\nOBSERVATION: {obs_text[:truncate]}\n"
        tool_log.append({
            "step": step, "tool": tool_name, "args": tool_args,
            "ok": result["ok"], "output_chars": len(obs_text),
            **({"continuation": True} if continuation else {}),
        })
        return result, obs_text, context_line

    # ── Primary loop ─────────────────────────────────────────────────────────
    for step in range(max_steps):
        generated = _run_one_step(model, tokenizer, context)
        context += generated

        m = TOOL_CALL_RE.search(generated)
        if not m:
            stopped_reason = "no_tool_call"
            break

        tool_name = m.group(1)
        try:
            tool_args = json.loads(m.group(2))
        except json.JSONDecodeError:
            observation = f"ERROR: malformed tool args: {m.group(2)[:200]}"
            context += f"\nOBSERVATION: {observation}\n"
            tool_log.append({"step": step, "tool": tool_name, "ok": False, "error": observation})
            continue

        # Break out of spin loops (same tool+args seen before)
        call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        if call_key in _seen_calls:
            context += (
                f"\nOBSERVATION: ERROR — you already tried this exact call and it failed. "
                f"Try a different approach: use a different path or a different tool.\n"
            )
            tool_log.append({"step": step, "tool": tool_name, "args": tool_args,
                              "ok": False, "error": "duplicate_call_skipped"})
            continue
        _seen_calls.add(call_key)

        result, obs_text, ctx_line = _dispatch_and_log(step, tool_name, tool_args)
        context += ctx_line

        if tool_name == "search_code" and result["ok"]:
            _last_search_output = obs_text

        if tool_name == "git_diff" and result["ok"] and result.get("has_diff"):
            stopped_reason = "git_diff_success"
            return result["stdout"], tool_log, stopped_reason, 0

    # ── Continuation guard ───────────────────────────────────────────────────
    # If the primary loop ended without a patch but the agent successfully used
    # at least one repo tool, inject a targeted nudge and run more steps.
    successful_tool_calls = [e for e in tool_log if e.get("ok")]
    continuation_attempts = 0

    if (
        continue_after_empty_patch
        and cloned
        and successful_tool_calls
        and len(tool_log) < max_steps + max_repo_steps
    ):
        nudge = _continuation_message(tool_log, _last_search_output)
        context += f"\n\nCONTINUATION REQUIRED: {nudge}\n"
        print(f"  [continuation] Nudging agent: {nudge[:100]}…")

        steps_used = len(tool_log)
        for repo_step in range(max_repo_steps):
            continuation_attempts += 1
            generated = _run_one_step(model, tokenizer, context)
            context += generated

            m = TOOL_CALL_RE.search(generated)
            if not m:
                # If the last continuation tool succeeded, re-nudge rather than stop
                last_cont = next(
                    (e for e in reversed(tool_log) if e.get("continuation")), None
                )
                if last_cont and last_cont.get("ok") and repo_step < max_repo_steps - 1:
                    re_nudge = _continuation_message(tool_log, _last_search_output)
                    context += f"\nCONTINUATION REQUIRED: {re_nudge}\n"
                    print(f"  [re-nudge] after successful {last_cont['tool']}: {re_nudge[:80]}…")
                    continue
                stopped_reason = "continuation_no_tool_call"
                break

            tool_name = m.group(1)
            step_idx = steps_used + repo_step
            try:
                tool_args = json.loads(m.group(2))
            except json.JSONDecodeError:
                observation = f"ERROR: malformed tool args: {m.group(2)[:200]}"
                context += f"\nOBSERVATION: {observation}\n"
                tool_log.append({"step": step_idx, "tool": tool_name,
                                  "ok": False, "error": observation, "continuation": True})
                continue

            # Spin-loop guard in continuation
            call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if call_key in _seen_calls:
                context += (
                    f"\nOBSERVATION: ERROR — you already tried this exact call. "
                    f"The file path may be wrong. Check the search results and use "
                    f"just the relative path (e.g. 'astropy/modeling/separable.py'). "
                    f"Then call write_file to fix the bug, then git_diff.\n"
                )
                tool_log.append({"step": step_idx, "tool": tool_name, "args": tool_args,
                                  "ok": False, "error": "duplicate_call_skipped",
                                  "continuation": True})
                continue
            _seen_calls.add(call_key)

            result, obs_text, ctx_line = _dispatch_and_log(step_idx, tool_name, tool_args,
                                                            continuation=True)
            context += ctx_line

            if tool_name == "git_diff" and result["ok"] and result.get("has_diff"):
                stopped_reason = "git_diff_success_after_continuation"
                return result["stdout"], tool_log, stopped_reason, continuation_attempts

    # Final fallback: check git_diff regardless
    if cloned:
        diff_result = git_diff(str(workspace))
        if diff_result["ok"] and diff_result.get("has_diff"):
            stopped_reason = "fallback_git_diff"
            return diff_result["stdout"], tool_log, stopped_reason, continuation_attempts

    stopped_reason = stopped_reason if stopped_reason != "max_steps" else "empty_patch"
    return "", tool_log, stopped_reason, continuation_attempts


# ---------------------------------------------------------------------------
# SWE-bench dataset loading
# ---------------------------------------------------------------------------

def _load_tasks(limit: int, instance_id: str | None) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading princeton-nlp/SWE-bench_Lite (split=test, limit={limit}) …")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks = list(ds)

    if instance_id:
        tasks = [t for t in tasks if t["instance_id"] == instance_id]
        if not tasks:
            print(f"ERROR: instance_id {instance_id!r} not found in dataset.")
            sys.exit(1)
    else:
        tasks = tasks[:limit]

    print(f"Loaded {len(tasks)} task(s)")
    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite Phase 2: repo-level patch generation"
    )
    parser.add_argument("--model", default="outputs/qwen15b_memory_300steps/final")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--instance-id", default=None,
                        help="Run a specific SWE-bench instance by ID")
    parser.add_argument("--output-dir", default="outputs/swebench_lite_phase2")
    parser.add_argument("--memory-enabled", action="store_true")
    parser.add_argument("--memory-index", default="memory/index")
    parser.add_argument("--stub-only", action="store_true",
                        help="Skip cloning and model loading; produces empty patches")
    parser.add_argument("--max-steps", type=int, default=MAX_AGENT_STEPS,
                        help="Max primary agent steps per task (default: 12)")
    parser.add_argument("--max-repo-steps", type=int, default=6,
                        help="Max continuation steps after empty patch (default: 6)")
    parser.add_argument("--require-nonempty-patch", action="store_true",
                        help="Exit with error if no patch produced (for CI)")
    parser.add_argument("--continue-after-empty-patch", action="store_true",
                        default=True,
                        help="Nudge agent to continue if patch empty after tool use (default: on)")
    parser.add_argument("--no-continue-after-empty-patch", dest="continue_after_empty_patch",
                        action="store_false",
                        help="Disable continuation guard")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    metadata_path = out_dir / "run_metadata.json"

    print()
    print("=" * 72)
    print("AetherForge-AI  —  SWE-bench Lite Phase 2: Repo-Level Patch Generation")
    print("=" * 72)
    print("IMPORTANT: Patch generation only; official harness evaluation still required.")
    print("  See: https://github.com/princeton-nlp/SWE-bench")
    print()
    print("Phase 2 adds: list_files, read_file, search_code, write_file,")
    print("  run_command, git_diff — real repo tools (see scripts/swebench_repo_tools.py)")
    print(f"Continuation guard: {'on' if args.continue_after_empty_patch else 'off'}"
          f"  max_repo_steps={args.max_repo_steps}")
    print()
    print("Known limitation:")
    print("  AetherForge was trained on function-level code tasks.")
    print("  Expect 0% or near-0% resolution on the official harness at this stage.")
    print()

    model, tokenizer = (None, None) if args.stub_only else _load_model(args.model)
    model_name = f"{args.model}{' [stub-only]' if args.stub_only else ''}"

    tasks = _load_tasks(args.limit, args.instance_id)

    predictions = []
    run_meta = []

    for i, task in enumerate(tasks, 1):
        instance_id = task["instance_id"]
        repo = task.get("repo", "unknown/repo")
        problem = task.get("problem_statement", "")
        hints = task.get("hints_text", "")
        base_commit = task.get("base_commit", "HEAD")

        print(f"\n[{i:02d}/{len(tasks)}] {instance_id}")
        print(f"  repo:   {repo}")
        print(f"  commit: {base_commit}")
        print(f"  issue:  {problem[:120].strip()}…")

        t0 = time.time()

        workspace, cloned = _prepare_workspace(
            out_dir, instance_id, repo, base_commit, stub=args.stub_only
        )

        prompt = _build_prompt(repo, problem, hints, workspace, cloned)
        patch, tool_log, stopped_reason, continuation_attempts = _run_agent(
            model, tokenizer, prompt, workspace, cloned,
            max_steps=args.max_steps,
            max_repo_steps=args.max_repo_steps,
            continue_after_empty_patch=args.continue_after_empty_patch,
        )

        elapsed = time.time() - t0
        has_diff = bool(patch.strip()) and ("diff --git" in patch or "---" in patch)
        last_entry = tool_log[-1] if tool_log else {}
        status = (
            "patch with diff" if has_diff else
            "non-empty patch (no diff header)" if patch.strip() else
            "empty patch"
        )
        print(f"  status: {status}  patch_chars={len(patch)}  elapsed={elapsed:.1f}s")
        print(f"  tool calls: {len(tool_log)}  continuations: {continuation_attempts}"
              f"  stopped: {stopped_reason}")

        record = {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        }
        predictions.append(record)

        run_meta.append({
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "workspace_cloned": cloned,
            "patch_chars": len(patch),
            "has_diff": has_diff,
            "tool_calls": len(tool_log),
            "continuation_attempts": continuation_attempts,
            "last_tool_name": last_entry.get("tool", ""),
            "last_tool_ok": last_entry.get("ok", False),
            "stopped_reason": stopped_reason,
            "tool_call_log": tool_log,
            "elapsed_s": round(elapsed, 2),
            "official_harness_run": False,
        })

    # Write outputs
    with open(predictions_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    with open(metadata_path, "w") as f:
        json.dump(run_meta, f, indent=2)

    print()
    print("=" * 72)
    print(f"Wrote {len(predictions)} prediction(s)  → {predictions_path}")
    print(f"Run metadata                            → {metadata_path}")
    print()
    patches_with_diff = sum(1 for m in run_meta if m["has_diff"])
    total_tool_calls = sum(m["tool_calls"] for m in run_meta)
    total_continuations = sum(m["continuation_attempts"] for m in run_meta)
    print(f"Patches with diff header  : {patches_with_diff}/{len(predictions)}")
    print(f"Total tool calls          : {total_tool_calls}")
    print(f"Total continuation steps  : {total_continuations}")
    print()

    if args.require_nonempty_patch and patches_with_diff == 0:
        print("ERROR: --require-nonempty-patch set but no diff produced.")
        sys.exit(1)

    print("Next step — official harness (requires Docker):")
    print("  pip install swebench")
    print(f"  python -m swebench.harness.run_evaluation \\")
    print(f"    --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"    --predictions_path {predictions_path} \\")
    print(f"    --max_workers 1 \\")
    print(f"    --run_id aetherforge_phase2")
    print()
    print("IMPORTANT: Patch generation only; official harness evaluation still required.")
    print("=" * 72)


if __name__ == "__main__":
    main()
