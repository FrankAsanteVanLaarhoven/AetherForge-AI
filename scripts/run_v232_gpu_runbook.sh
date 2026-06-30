#!/usr/bin/env bash
#
# scripts/run_v232_gpu_runbook.sh — v2.32 end-to-end GPU runbook.
#
# Mixed repair + tool-use-preservation split-loss SFT on a GPU host: build mixed dataset -> train ->
# eval (repair + preservation) -> benchmark gate -> summarise. Refuses without a CUDA GPU. Protects
# the champion + memory indexes (read-only tamper check). All training/eval artifacts local-only.
# Fabricates no metrics.
#
# Usage: bash scripts/run_v232_gpu_runbook.sh [BASE_MODEL] [MAX_STEPS]

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASE="${1:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
MAX_STEPS="${2:-80}"
OUT="outputs/v232_tool_use_preservation_sft"

PROTECTED=(
  "outputs/qwen15b_memory_300steps/final"
  "memory/index_adapted" "memory/index_adapted_v29"
  "memory/dense_index_v219" "memory/dense_index_v221" "memory/dense_index_v222"
  "memory/dense_index_v226" "memory/dense_index_v227"
)

echo "=== v2.32 GPU runbook ===  base=$BASE max_steps=$MAX_STEPS out=$OUT (local-only)"

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

echo "[1/5] build mixed dataset (repair + tool-use preservation, local-only)…"
make build-v232-mixed-dataset
echo "[2/5] train split-loss adapter (separate path)…"
python3 scripts/train_v232_mixed_sft.py --base "$BASE" --max-steps "$MAX_STEPS"
echo "[3/5] eval repair + tool-use preservation…"
python3 scripts/eval_v232_mixed_sft.py --base "$BASE" --adapter "$OUT/adapter"
echo "[4/5] benchmark gate (frozen 32-task / hard-tree / tree_serialize)…"
python3 scripts/eval_v232_mixed_sft.py --benchmarks --base "$BASE" --adapter "$OUT/adapter"
echo "[5/5] summarise v2.32…"
make summarise-v232

AFTER="$(manifest | sha256sum | cut -d' ' -f1)"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "ERROR: protected-asset manifest changed during the run ($BEFORE -> $AFTER)."; exit 2
fi
echo "[guard] protected assets unchanged."

cat <<'GATES'

=== v2.32 promotion gates (all must hold to PROMOTE) ===
  [ ] training completed without instability
  [ ] repair validation IMPROVES (adapter > base)
  [ ] tool-use preservation held (adapter preservation pass-rate >= 80%)
  [ ] frozen 32-task benchmark does NOT materially regress vs champion 23/28
  [ ] tree_serialize 3/3 format-control preserved
  [ ] contamination guard 0; no protected artifact overwritten
GATES
echo "Done. Review results/v232_tool_use_preservation/summary.md before promoting."
