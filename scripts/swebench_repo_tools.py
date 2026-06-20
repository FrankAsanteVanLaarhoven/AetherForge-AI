"""swebench_repo_tools.py — Safe repo-level tools for SWE-bench Phase 2.

These tools give an agent the ability to inspect and edit a local repository
checkout without escaping the workspace directory.

Safety guarantees:
  - All paths are resolved and validated to stay inside repo_dir.
  - Absolute paths that escape the repo are rejected.
  - run_command uses an allowlist; dangerous commands are blocked.
  - No OBSERVATION: PASS is faked; only real command output is returned.

Return format:
  Every tool returns a dict with at minimum:
    {"ok": bool, "stdout": str, "stderr": str}
  Plus tool-specific metadata fields.
"""

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowed commands for run_command (prefix-match on the command name)
# ---------------------------------------------------------------------------
_ALLOWED_PREFIXES = (
    "python", "python3",
    "pytest", "py.test",
    "grep", "find", "ls", "cat",
    "git diff", "git log", "git status", "git show",
    "sed", "head", "tail", "wc",
)

_BLOCKED_PATTERNS = (
    r"\brm\b", r"\brmdir\b", r"\bsudo\b", r"\bchmod\b",
    r"\bcurl\b", r"\bwget\b", r"\bdd\b",
    r">\s*/", r">\s*~",  # redirect to absolute/home paths
    r"\.\./\.\./\.\./",  # deep traversal
)


def _resolve_path(repo_dir: Path, rel_path: str) -> Path | None:
    """Resolve a path, returning None if it escapes repo_dir.

    Always works with the resolved absolute repo_dir so that rglob output
    (which is always absolute) compares correctly.
    """
    try:
        repo_resolved = repo_dir.resolve()
        # Strip workspace prefix if the model generated an absolute path
        # that begins with the repo directory (a common model error).
        rp = rel_path
        repo_str = str(repo_resolved)
        if rp.startswith(repo_str + "/") or rp.startswith(repo_str + os.sep):
            rp = rp[len(repo_str) + 1:]
        resolved = (repo_resolved / rp).resolve()
        if not str(resolved).startswith(repo_str):
            return None
        return resolved
    except Exception:
        return None


def list_files(repo_dir: str, subdir: str = ".") -> dict:
    """List files recursively under subdir (default: repo root).

    Returns at most 500 entries to avoid context overflow.
    """
    repo = Path(repo_dir).resolve()  # always absolute so relative_to works
    target = _resolve_path(repo, subdir)
    if target is None:
        return {"ok": False, "stdout": "", "stderr": f"Path {subdir!r} escapes repo", "files": []}
    if not target.exists():
        return {"ok": False, "stdout": "", "stderr": f"Path does not exist: {subdir}", "files": []}

    files = []
    for p in sorted(target.rglob("*")):
        if p.is_file():
            try:
                files.append(str(p.relative_to(repo)))
            except ValueError:
                pass
        if len(files) >= 500:
            break

    stdout = "\n".join(files) if files else "(no files found)"
    return {"ok": True, "stdout": stdout, "stderr": "", "files": files, "count": len(files)}


def read_file(repo_dir: str, path: str, max_chars: int = 20_000) -> dict:
    """Read a file from the repo workspace.

    Truncates at max_chars to keep context manageable.
    Accepts both relative paths and absolute paths that start with repo_dir.
    """
    repo = Path(repo_dir).resolve()
    target = _resolve_path(repo, path)
    if target is None:
        return {"ok": False, "stdout": "", "stderr": f"Path {path!r} escapes repo", "path": path}
    if not target.exists():
        return {"ok": False, "stdout": "", "stderr": f"File not found: {path}", "path": path}
    if not target.is_file():
        return {"ok": False, "stdout": "", "stderr": f"Not a file: {path}", "path": path}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc), "path": path}

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars] + f"\n... [truncated at {max_chars} chars]"

    return {
        "ok": True,
        "stdout": content,
        "stderr": "",
        "path": path,
        "chars": len(content),
        "truncated": truncated,
    }


def search_code(repo_dir: str, pattern: str,
                include_glob: str = "*.py", max_results: int = 50) -> dict:
    """Search for pattern across files matching include_glob.

    Uses Python's built-in re so no subprocess shell injection is possible.
    """
    repo = Path(repo_dir)
    if not repo.exists():
        return {"ok": False, "stdout": "", "stderr": f"Repo dir not found: {repo_dir}", "matches": []}

    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return {"ok": False, "stdout": "", "stderr": f"Invalid regex: {exc}", "matches": []}

    matches = []
    for p in sorted(repo.rglob("*")):
        if not p.is_file():
            continue
        if not fnmatch.fnmatch(p.name, include_glob):
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(lines, 1):
            if rx.search(line):
                rel = str(p.relative_to(repo))
                matches.append({"file": rel, "line": lineno, "text": line})
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    lines_out = [f"{m['file']}:{m['line']}: {m['text']}" for m in matches]
    stdout = "\n".join(lines_out)
    return {"ok": True, "stdout": stdout, "stderr": "", "matches": matches, "count": len(matches)}


def write_file(repo_dir: str, path: str, content: str) -> dict:
    """Write content to a file inside the repo workspace.

    Creates parent directories if needed. Never writes outside repo_dir.
    Accepts both relative paths and absolute paths that start with repo_dir.
    """
    repo = Path(repo_dir).resolve()
    target = _resolve_path(repo, path)
    if target is None:
        return {"ok": False, "stdout": "", "stderr": f"Path {path!r} escapes repo", "path": path}

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc), "path": path}

    return {
        "ok": True,
        "stdout": f"Wrote {len(content)} chars to {path}",
        "stderr": "",
        "path": path,
        "chars": len(content),
    }


def run_command(repo_dir: str, command: str, timeout: int = 60) -> dict:
    """Run a whitelisted shell command inside repo_dir.

    Only commands matching _ALLOWED_PREFIXES are permitted.
    Commands matching _BLOCKED_PATTERNS are rejected outright.
    Working directory is set to repo_dir.
    """
    repo = Path(repo_dir)

    # Block check
    for pat in _BLOCKED_PATTERNS:
        if re.search(pat, command):
            return {
                "ok": False, "stdout": "",
                "stderr": f"Command blocked by safety filter (matched: {pat!r}): {command!r}",
                "command": command,
            }

    # Allowlist check
    cmd_stripped = command.strip()
    allowed = any(
        cmd_stripped == p or cmd_stripped.startswith(p + " ")
        for p in _ALLOWED_PREFIXES
    )
    if not allowed:
        return {
            "ok": False, "stdout": "",
            "stderr": (
                f"Command not in allowlist: {command!r}\n"
                f"Allowed prefixes: {_ALLOWED_PREFIXES}"
            ),
            "command": command,
        }

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo),
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout[:10_000],
            "stderr": result.stderr[:4_000],
            "returncode": result.returncode,
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False, "stdout": "",
            "stderr": f"Command timed out after {timeout}s: {command!r}",
            "command": command,
        }
    except Exception as exc:
        return {
            "ok": False, "stdout": "",
            "stderr": str(exc),
            "command": command,
        }


def git_diff(repo_dir: str) -> dict:
    """Return the unified diff of all uncommitted changes in the repo."""
    repo = Path(repo_dir)
    result = run_command(str(repo), "git diff", timeout=30)
    diff_text = result.get("stdout", "")
    has_diff = bool(diff_text.strip())
    return {
        "ok": result["ok"],
        "stdout": diff_text,
        "stderr": result.get("stderr", ""),
        "has_diff": has_diff,
        "patch_chars": len(diff_text),
    }


# ---------------------------------------------------------------------------
# Tool registry — used by the agent loop in eval_swebench_lite_patchgen.py
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "list_files":  list_files,
    "read_file":   read_file,
    "search_code": search_code,
    "write_file":  write_file,
    "run_command": run_command,
    "git_diff":    git_diff,
}


def dispatch(repo_dir: str, tool_name: str, tool_args: dict) -> dict:
    """Dispatch a tool call by name, injecting repo_dir as first argument."""
    if tool_name not in TOOL_REGISTRY:
        return {
            "ok": False, "stdout": "",
            "stderr": f"Unknown tool: {tool_name!r}. Available: {list(TOOL_REGISTRY)}",
        }
    fn = TOOL_REGISTRY[tool_name]
    return fn(repo_dir, **tool_args)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # write_file
        r = write_file(td, "hello.py", "print('hello')\n")
        assert r["ok"], r
        # read_file
        r = read_file(td, "hello.py")
        assert r["ok"] and "hello" in r["stdout"], r
        # list_files
        r = list_files(td)
        assert r["ok"] and "hello.py" in r["files"], r
        # search_code
        r = search_code(td, r"print", include_glob="*.py")
        assert r["ok"] and r["count"] == 1, r
        # path escape rejection
        r = read_file(td, "../etc/passwd")
        assert not r["ok"], r
        # blocked command
        r = run_command(td, "rm -rf .", timeout=5)
        assert not r["ok"], r
        # non-allowlist command
        r = run_command(td, "curl http://example.com", timeout=5)
        assert not r["ok"], r

    print("All swebench_repo_tools self-tests passed.")
