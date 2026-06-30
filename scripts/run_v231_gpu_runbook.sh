#!/usr/bin/env bash
#
# scripts/run_v231_gpu_runbook.sh — v2.31 end-to-end GPU runbook.
#
# Runs the tiny repair-trace SFT pilot on a GPU host: export dataset -> train 1.5B LoRA smoke ->
# evaluate -> summarise. Refuses to run without a CUDA GPU. Protects the frozen champion and memory
# indexes (read-only tamper check). All training/eval artifacts stay local-only (gitignored).
#
# This script does NOT fabricate metrics: if no GPU is present it exits without training.
#
# Usage:
#   bash scripts/run_v231_gpu_runbook.sh [BASE_MODEL] [MAX_STEPS]
# Defaults: BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct  MAX_STEPS=60

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASE="${1:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
MAX_STEPS="${2:-60}"
OUT="outputs/v231_tiny_repair_trace_sft"   # separate, local-only, gitignored

PROTECTED=(
  "outputs/qwen15b_memory_300steps/final"
  "memory/index_adapted" "memory/index_adapted_v29"
  "memory/dense_index_v219" "memory/dense_index_v221" "memory/dense_index_v222"
  "memory/dense_index_v226" "memory/dense_index_v227"
)

echo "=== v2.31 GPU runbook ==="
echo "base=$BASE  max_steps=$MAX_STEPS  out=$OUT  (local-only)"

# ── 0. GPU gate ──────────────────────────────────────────────────────────────
if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "REFUSING: no CUDA GPU available. This runbook must run on a GPU host."
  echo "Nothing was trained; no metrics were produced."
  exit 1
fi
echo "[gpu] CUDA available."

# ── 1. Protected-asset read-only snapshot (champion + memory indexes) ─────────
manifest() {
  for p in "${PROTECTED[@]}"; do
    for d in "$ROOT/$p"*; do
      [ -e "$d" ] && find "$d" -type f -printf '%s %p\n' 2>/dev/null || true
    done
  done | LC_ALL=C sort
}
BEFORE="$(manifest | sha256sum | cut -d' ' -f1)"
echo "[guard] protected-asset manifest hash: $BEFORE"

# refuse if the output path collides with any protected path
case "$OUT" in
  outputs/qwen15b_memory_300steps*|outputs/qwen3b*|outputs/qwen7b*|memory/*)
    echo "REFUSING: output path '$OUT' overlaps a protected asset."; exit 1;;
esac

# ── 2. Pipeline: export -> train -> eval -> summarise ────────────────────────
echo "[1/4] export SFT dataset (local-only)…"
make build-v231-sft-dataset

echo "[2/4] train 1.5B LoRA smoke (separate adapter path)…"
python3 scripts/train_v231_repair_sft.py --base "$BASE" --max-steps "$MAX_STEPS"

echo "[3/4] evaluate v2.31 adapter vs base (repair validation + benchmark gate)…"
python3 scripts/eval_v231_repair_sft.py --base "$BASE" --adapter "$OUT/adapter"
python3 scripts/eval_v231_repair_sft.py --benchmarks --base "$BASE" --adapter "$OUT/adapter"

echo "[4/4] summarise v2.31…"
make summarise-v231

# ── 3. Verify protected assets were not modified ─────────────────────────────
AFTER="$(manifest | sha256sum | cut -d' ' -f1)"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "ERROR: protected-asset manifest changed during the run ($BEFORE -> $AFTER)."
  echo "Investigate before trusting results; champion/memory indexes must be read-only."
  exit 2
fi
echo "[guard] protected assets unchanged."

# ── 4. Promotion gates (decision is made by a human reading the summary) ─────
cat <<'GATES'

=== v2.31 promotion gates (all must hold to PROMOTE) ===
  [ ] training completed without instability (loss decreases, no NaN/divergence)
  [ ] contamination guard remains 0 violations
  [ ] repair validation improves OR preserves high repair performance (adapter >= base)
  [ ] frozen 32-task benchmark does NOT materially regress vs champion 23/28
  [ ] tree_serialize 3/3 format-control result preserved
  [ ] no protected champion adapter or memory index overwritten
Strong: repair validation improves AND >=1 previously-weak repair category improves
        AND no broad regression AND tree_serialize 3/3 preserved.
Reject/HOLD if: overfits tiny corpus, benchmark regresses, repair validation does not
        improve, adapter only memorizes controlled perturbations, or artifacts staged.

Delegated benchmark evals (run on this GPU host, compare to champion 23/28):
  evaluate_code_agent.py over the frozen 32-task set + hard tree subset + tree_serialize.
  Keep all outputs local-only; commit ONLY the small results/v231_tiny_repair_trace_sft summary.
GATES

echo "Done. Review results/v231_tiny_repair_trace_sft/summary.md before promoting."
