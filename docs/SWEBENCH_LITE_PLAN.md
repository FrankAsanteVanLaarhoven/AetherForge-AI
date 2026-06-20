# SWE-bench Lite Integration Plan — AetherForge-AI

## Overview

This document describes the phased plan for evaluating AetherForge-AI on
SWE-bench Lite as a repository-level issue-resolution benchmark.

SWE-bench Lite results are reported **separately** from the existing
function-level verified-agent benchmark and must NOT be compared directly.

---

## What SWE-bench Is

**SWE-bench** evaluates whether a language model can resolve real GitHub issues
by producing a patch (unified diff) that makes failing tests pass in the actual
repository, evaluated in a Docker container running the repo's full test suite.

| Variant | Tasks | Description |
|---|---|---|
| SWE-bench Full | 2,294 | Full dataset, Python repos from GitHub |
| **SWE-bench Lite** | **300** | Curated subset, faster iteration |
| SWE-bench Verified | 500 | Human-validated subset, higher quality |

**Key properties:**
- Tasks are real GitHub issues with linked pull requests (ground truth patches)
- Evaluation applies the model's patch to the real repo and runs the test suite
- Docker is required for reproducible environment isolation
- Official harness: `https://github.com/princeton-nlp/SWE-bench`

---

## AetherForge Current Capability vs SWE-bench Requirements

| Requirement | AetherForge Current State | Gap |
|---|---|---|
| Read repo files | Not implemented | Needs `read_file` tool |
| Search repo | Not implemented | Needs `search_repo` tool |
| Edit files | Not implemented | Needs `edit_file` tool |
| Run real tests | Not implemented | Needs `run_tests` tool |
| Generate git diff | Not implemented | Needs `generate_diff` tool |
| Produce patch format | Partial (model output only) | Needs structure enforcement |
| Docker harness | Not configured | Needs Docker setup |
| Function-level code | Fully implemented | Current benchmark strength |

**Expected score at current capability:** 0% or near-0% resolution on the
official harness. This is expected and acceptable at the smoke-test milestone.

---

## Phased Plan

### Phase 1 — Smoke Test (Current)

**Goal:** Can AetherForge produce syntactically valid SWE-bench predictions?

**Scope:** 1–5 tasks, stub or minimal model output.

**Script:** `scripts/eval_swebench_lite_smoke.py`

**Command:**
```bash
make eval-swebench-lite-smoke
```

**Output:** `outputs/swebench_lite_smoke/predictions.jsonl`

**Claim boundary:** Patch generation only. No resolution claims until official
harness verifies.

---

### Phase 2 — Repo-Level Tool Implementation

**Goal:** Implement the four repo-level tools the agent needs.

**Required tools:**
1. `read_file(path)` — read a file from the cloned repository
2. `search_repo(pattern)` — grep/find across repo files
3. `edit_file(path, old_text, new_text)` — apply an in-place edit
4. `run_tests(test_ids)` — run specific tests and return pass/fail
5. `generate_diff()` — produce unified diff of all edits

**New agent protocol:** The TOOL_CALL/OBSERVATION loop must be extended to
support these tools alongside `execute_code`.

---

### Phase 3 — Official Harness Evaluation (5–10 tasks)

**Goal:** Submit predictions to the official SWE-bench harness and get a
verified resolution rate.

**Prerequisites:**
- Docker installed and configured
- Official `swebench` Python package installed
- Phase 2 tools implemented

**Command (once prerequisites met):**
```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path outputs/swebench_lite_smoke/predictions.jsonl \
  --max_workers 1 \
  --run_id aetherforge_smoke
```

---

### Phase 4 — Full SWE-bench Lite (300 tasks)

**Goal:** Run all 300 SWE-bench Lite tasks and report a verified resolution %.

**Claim boundary:** This result is a **repository-level issue-resolution rate**,
separate from the function-level verified-agent benchmark (75.0% held-out).

---

## Result Reporting Boundaries

| Metric | Benchmark | Claim |
|---|---|---|
| Function-level verified-agent | Frozen held-out (28 tasks) | 75.0% clean generalization |
| Function-level verified-agent | Adapted memory (28 tasks) | 82.1% adaptation result |
| SWE-bench Lite | Phase 1 smoke (≤5 tasks) | Patch generation only, not verified |
| SWE-bench Lite | Phase 3+ (official harness) | Verified resolution rate |

**Never mix SWE-bench results with the function-level benchmark.**

---

## Scientific Boundary Statement

> AetherForge-AI Phase 1 SWE-bench evaluation produces candidate patches using
> the current function-level model. These patches have not been verified by the
> official Docker-based harness. No resolution claims are made at this stage.
> True SWE-bench resolution rates will be reported only after official harness
> evaluation (Phase 3+).

---

## References

- SWE-bench paper: Jimenez et al., 2024 — "SWE-bench: Can Language Models Resolve Real-world GitHub Issues?"
- Dataset: `princeton-nlp/SWE-bench_Lite` on Hugging Face
- Official harness: `https://github.com/princeton-nlp/SWE-bench`
- SWE-bench Lite: 300-task curated subset for faster iteration
- SWE-bench Verified: 500-task human-validated subset
