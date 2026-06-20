#!/usr/bin/env python3
"""Inspect the most recent SWE-bench Phase 2 prediction file."""
import json
import sys
from pathlib import Path

paths = sorted(Path("outputs").glob("swebench_lite_phase2*/predictions.jsonl"))
if not paths:
    print("No predictions.jsonl found under outputs/swebench_lite_phase2*/")
    sys.exit(1)

p = paths[-1]
print(f"Reading: {p}")
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    patch = r.get("model_patch", "")
    has_diff = "diff --git" in patch or "---" in patch
    print(f"instance_id       : {r.get('instance_id')}")
    print(f"model_name_or_path: {r.get('model_name_or_path')}")
    print(f"patch_chars       : {len(patch)}")
    print(f"has_diff          : {has_diff}")
    if patch.strip():
        print("--- patch preview (first 500 chars) ---")
        print(patch[:500])
    print()

meta_path = p.parent / "run_metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    for m in meta:
        print(f"workspace_cloned  : {m.get('workspace_cloned')}")
        print(f"tool_calls        : {m.get('tool_calls')}")
        print(f"official_harness_run: {m.get('official_harness_run')}")
        print(f"elapsed_s         : {m.get('elapsed_s')}")
