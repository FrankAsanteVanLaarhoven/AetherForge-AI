#!/usr/bin/env bash
#
# scripts/run_v233_gpu_runbook.sh — v2.33 end-to-end GPU runbook (scaffold-first).
#
# Scaffold-only tool-call preservation SFT on a GPU host: build scaffold dataset -> train adapter ->
# eval tool-use preservation -> frozen 32-task benchmark gate -> summarise. Refuses without a CUDA GPU.
# Protects the champion + memory indexes (read-only tamper check). Artifacts local-only. No fabricated
# metrics. v2.33 success is tool-call preservation without regression — NOT repair improvement.
#
# Usage: bash scripts/run_v233_gpu_runbook.sh [BASE_MODEL] [MAX_STEPS]

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASE="${1:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
MAX_STEPS="${2:-60}"
OUT="outputs/v233_scaffold_first_sft"

PROTECTED=(
  "outputs/qwen15b_memory_300steps/final"
  "memory/index_adapted" "memory/index_adapted_v29"
  "memory/dense_index_v219" "memory/dense_index_v221" "memory/dense_index_v222"
  "memory/dense_index_v226" "memory/dense_index_v227"
)

echo "=== v2.33 GPU runbook (scaffold-first) ===  base=$BASE max_steps=$MAX_STEPS out=$OUT (local-only)"

if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "REFUSING: no CUDA GPU available. Run on a GPU host. Nothing trained; no metrics produced."
  exit 1
fi
echo "[gpu] CUDA available."

case "$OUT" in
  outputs/qwen15b_memory_300steps*|outputs/qwen3b*|outputs/qwen7b*|memory/*)
    echo "REFUSING: output path '$OUT' overlaps a protected asset."; exit 1;;
esac

manifest() {
  for p in "${PROTECTED[@]}"; do
    for d in "$ROOT/$p"*; do
      [ -e "$d" ] && find "$d" -type f -printf '%s %p\n' 2>/dev/null || true
    done
  done | LC_ALL=C sort
}
BEFORE="$(manifest | sha256sum | cut -d' ' -f1)"
echo "[guard] protected-asset manifest hash: $BEFORE"

echo "[1/5] build scaffold dataset (no repair; local-only)…"
make build-v233-scaffold-dataset
echo "[2/5] train scaffold adapter (separate path)…"
python3 scripts/train_v233_scaffold_sft.py --base "$BASE" --max-steps "$MAX_STEPS"
echo "[3/5] eval tool-use preservation…"
python3 scripts/eval_v233_scaffold_sft.py --base "$BASE" --adapter "$OUT/adapter"
echo "[4/5] frozen 32-task benchmark gate (+ no_tool_call breakdown)…"
python3 scripts/eval_v233_scaffold_sft.py --benchmarks --base "$BASE" --adapter "$OUT/adapter"
echo "[5/5] summarise v2.33…"
make summarise-v233

AFTER="$(manifest | sha256sum | cut -d' ' -f1)"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "ERROR: protected-asset manifest changed during the run ($BEFORE -> $AFTER)."; exit 2
fi
echo "[guard] protected assets unchanged."

cat <<'GATES'

=== v2.33 promotion gates (all must hold to PROMOTE) ===
  [ ] training completed without instability
  [ ] tool-use preservation >= 80%
  [ ] frozen 32-task benchmark does NOT materially regress vs champion 23/28
  [ ] tree_serialize 3/3 format-control preserved
  [ ] no_tool_call is NOT the dominant failure mode
  [ ] contamination 0; no protected artifact overwritten
v2.33 success = tool-call preservation without regression (NOT repair improvement).
Only after this passes should v2.34 reintroduce repair traces.
GATES
echo "Done. Review results/v233_scaffold_first/summary.md before promoting."
