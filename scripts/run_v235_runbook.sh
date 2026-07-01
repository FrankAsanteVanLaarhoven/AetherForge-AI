#!/usr/bin/env bash
#
# scripts/run_v235_runbook.sh — v2.35 solution-body generation runbook (inference-time, no training).
#
# Verifies protected assets, runs the v2.35 solution-body evaluation (v2.34 tool-call control + strict
# benchmark-assertion verification over the frozen 32-task transcripts), summarises, then re-verifies
# protected assets. No GPU/training required. Refuses if the v2.33 transcripts are missing. Outputs
# local-only (gitignored). No fabricated metrics.
#
# Usage: bash scripts/run_v235_runbook.sh

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BASELINE_CSV="outputs/v233_scaffold_first_sft/eval_32task_adapter/best_of_3.csv"

PROTECTED=(
  "outputs/qwen15b_memory_300steps/final"
  "memory/index_adapted" "memory/index_adapted_v29"
  "memory/dense_index_v219" "memory/dense_index_v221" "memory/dense_index_v222"
  "memory/dense_index_v226" "memory/dense_index_v227"
)

echo "=== v2.35 runbook (solution-body generation; inference-time) ==="
if [ ! -f "$BASELINE_CSV" ]; then
  echo "REFUSING: v2.33 benchmark transcripts missing ($BASELINE_CSV). Run make v233-gpu-runbook first."
  exit 1
fi

manifest() {
  for p in "${PROTECTED[@]}"; do
    for d in "$ROOT/$p"*; do
      [ -e "$d" ] && find "$d" -type f -printf '%s %p\n' 2>/dev/null || true
    done
  done | LC_ALL=C sort
}
BEFORE="$(manifest | sha256sum | cut -d' ' -f1)"
echo "[guard] protected-asset manifest hash: $BEFORE"

echo "[1/2] run v2.35 solution-body evaluation…"
make eval-v235
echo "[2/2] summarise v2.35…"
make summarise-v235

AFTER="$(manifest | sha256sum | cut -d' ' -f1)"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "ERROR: protected-asset manifest changed during the run ($BEFORE -> $AFTER)."; exit 2
fi
echo "[guard] protected assets unchanged."

cat <<'GATES'

=== v2.35 promotion gates (all must hold to PROMOTE) ===
  [ ] strict-verified 32-task score improves over v2.34's 5/32
  [ ] tool-call rate remains improved (v2.34 control retained)
  [ ] no_tool_call remains non-dominant
  [ ] strict verifier rejects fake PASS (never counted as a pass)
  [ ] tree_serialize remains preserved
  [ ] contamination 0; protected artifacts unchanged
v2.35 makes NO claim of model/repair improvement, SOTA, SWE-bench, or production readiness.
GATES
echo "Done. Review results/v235_solution_body_generation/summary.md."
