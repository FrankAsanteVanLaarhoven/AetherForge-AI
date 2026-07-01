#!/usr/bin/env bash
#
# scripts/run_v234_runbook.sh — v2.34 tool-call format-control runbook (inference-time, no training).
#
# Verifies protected assets, runs the deterministic tool-call format-control evaluation over the v2.33
# benchmark transcripts, summarises, then re-verifies protected assets. No GPU/training required — this
# is an offline control experiment. Refuses if the v2.33 benchmark transcripts are missing. Outputs are
# local-only (gitignored). No fabricated metrics.
#
# Usage: bash scripts/run_v234_runbook.sh

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

echo "=== v2.34 runbook (tool-call format control; inference-time) ==="
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

echo "[1/2] run v2.34 controlled evaluation…"
make eval-v234
echo "[2/2] summarise v2.34…"
make summarise-v234

AFTER="$(manifest | sha256sum | cut -d' ' -f1)"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "ERROR: protected-asset manifest changed during the run ($BEFORE -> $AFTER)."; exit 2
fi
echo "[guard] protected assets unchanged."

cat <<'GATES'

=== v2.34 promotion gates (all must hold to PROMOTE) ===
  [ ] no_tool_call is no longer the dominant failure mode
  [ ] tool-call rate materially improves over v2.33
  [ ] 32-task score improves over v2.33's 5/32
  [ ] tree_serialize remains preserved
  [ ] format controller does not fabricate unsafe tool calls
  [ ] contamination 0; protected artifacts unchanged
v2.34 makes NO claim of model/repair improvement, SOTA, SWE-bench, or production readiness.
GATES
echo "Done. Review results/v234_tool_call_format_control/summary.md."
