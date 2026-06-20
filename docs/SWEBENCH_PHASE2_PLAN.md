# SWE-bench Lite Phase 2 Plan — AetherForge-AI

## Overview

This document describes Phase 2 of the SWE-bench Lite integration for
AetherForge-AI: real repo-level patch generation.

SWE-bench results are reported **separately** from the function-level
verified-agent benchmark and must NOT be compared directly.

| Benchmark | Type | Current best |
|---|---|---|
| Function-level verified-agent (frozen held-out) | Code task execution | 75.0% clean |
| Function-level verified-agent (adapted memory) | Code task execution | 82.1% adaptation |
| SWE-bench Lite | Repo-level issue resolution | Phase 2 in progress |

---

## What SWE-bench Is

SWE-bench evaluates whether a language model can resolve real GitHub issues
by producing a patch (unified diff) that makes failing tests pass when applied
to the real repository, verified inside a Docker container running the full
test suite.

| Variant | Tasks | Description |
|---|---|---|
| SWE-bench Full | 2,294 | Full dataset, Python repos |
| **SWE-bench Lite** | **300** | Curated subset, faster iteration |
| SWE-bench Verified | 500 | Human-validated subset |

**Key properties:**
- Tasks are real GitHub issues with linked pull requests (ground truth patches)
- Evaluation applies the model's patch to the real repo and runs the test suite
- Docker is required for reproducible environment isolation
- Official harness: `https://github.com/princeton-nlp/SWE-bench`

---

## Phase Plan

### Phase 1 — Stub Format Validation (Complete)

**Goal:** Can AetherForge produce syntactically valid SWE-bench prediction records?

**Script:** `scripts/eval_swebench_lite_smoke.py`

**Result:** Format validation passed. Correct fields: `instance_id`,
`model_name_or_path`, `model_patch`. Empty patch (stub) in correct format.

**Claim:** Patch generation only. No resolution claims.

---

### Phase 2 — Real Repo-Level Patch Generation (Current)

**Goal:** Can AetherForge inspect a real repository, edit files, and produce
a non-empty unified diff for one SWE-bench Lite task?

**Script:** `scripts/eval_swebench_lite_patchgen.py`

**Repo tools:** `scripts/swebench_repo_tools.py`

Available tools for the agent:
- `list_files({"subdir": "."})` — list repo files
- `read_file({"path": "..."})` — read a source file
- `search_code({"pattern": "...", "include_glob": "*.py"})` — grep across repo
- `write_file({"path": "...", "content": "..."})` — write/edit a file
- `run_command({"command": "python -m pytest tests/ -x"})` — run safe commands
- `git_diff({})` — get unified diff of all changes

**Command:**
```bash
make eval-swebench-lite-patchgen-one
```

**Output:**
- `outputs/swebench_lite_phase2/predictions.jsonl` — official prediction format
- `outputs/swebench_lite_phase2/run_metadata.json` — per-task metadata

**Safety constraints:**
- All file paths validated to stay inside the workspace directory
- `run_command` uses an allowlist (python, pytest, grep, find, git diff, etc.)
- Blocked commands: rm, sudo, curl, wget, and path-escape patterns

**Expected outcome at current capability:**
- The model may produce a patch, but it is unlikely to resolve the issue.
- Expected resolution rate on official harness: 0% or near-0%.
- This is expected. The model was trained on function-level code tasks.
- The milestone is: does the agent call repo tools and produce a non-empty diff?

**Claim:** Patch generation only. No resolution claims until official harness.

---

### Phase 3 — Official Harness Evaluation (1 Task)

**Goal:** Submit Phase 2 predictions to the official SWE-bench harness and
get a verified resolution rate on one task.

**Prerequisites:**
- Docker installed and configured
- `pip install swebench`
- Phase 2 produces a non-empty patch

**Command (once prerequisites met):**
```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path outputs/swebench_lite_phase2/predictions.jsonl \
  --max_workers 1 \
  --run_id aetherforge_phase2
```

**Claim boundary:** Only after this step may any SWE-bench resolution rate
be reported. Until then: patch generation only.

---

### Phase 4 — Limited SWE-bench Lite Sample (5–10 Tasks)

**Goal:** Run 5–10 tasks through the official harness and report a verified
SWE-bench Lite resolution rate.

**Claim boundary:** This result is a **repository-level issue-resolution rate**,
separate from the function-level verified-agent results.

---

### Phase 5 — Full SWE-bench Lite (300 Tasks)

**Goal:** Run all 300 tasks and report a verified SWE-bench Lite %.

**Do not run until Phase 4 is complete and the agent is meaningfully producing patches.**

---

## Result Reporting Boundaries

| Metric | Benchmark | Claim |
|---|---|---|
| Function-level verified-agent | Frozen held-out (28 tasks) | 75.0% clean generalization |
| Function-level verified-agent | Adapted memory (28 tasks) | 82.1% adaptation result |
| SWE-bench Phase 1 | 1 task, stub | Patch format validation only |
| SWE-bench Phase 2 | 1 task, real repo | Patch generation only, not verified |
| SWE-bench Phase 3+ | Official harness | Verified resolution rate |

**Never mix SWE-bench results with the function-level benchmark.**

---

## AetherForge Current Capability vs SWE-bench Requirements

| Requirement | AetherForge Phase 2 | Gap |
|---|---|---|
| Read repo files | `read_file` tool ✓ | Needs Phase 2 training |
| Search repo | `search_code` tool ✓ | Needs Phase 2 training |
| Edit files | `write_file` tool ✓ | Needs better edit prompting |
| Run real tests | `run_command` tool ✓ | Needs Docker + installed deps |
| Generate git diff | `git_diff` tool ✓ | Available when repo cloned |
| Produce patch format | Saved as `model_patch` ✓ | — |
| Docker harness | Not configured | Phase 3 prerequisite |
| Verified resolution | 0% expected | Phase 3+ |

---

## Scientific Boundary Statement

> AetherForge-AI Phase 2 SWE-bench evaluation generates candidate patches using
> repo-level tools on locally cloned repositories. These patches have not been
> verified by the official Docker-based harness. No resolution claims are made
> until official harness evaluation completes (Phase 3+).
>
> Function-level verified-agent results (75.0% frozen held-out, 82.1% adapted
> memory) are a separate benchmark track and must not be compared directly to
> SWE-bench resolution rates.

---

## References

- SWE-bench paper: Jimenez et al., 2024
- Dataset: `princeton-nlp/SWE-bench_Lite` on Hugging Face
- Official harness: `https://github.com/princeton-nlp/SWE-bench`
