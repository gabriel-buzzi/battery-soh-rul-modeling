#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGETS=(SOH RUL)

run_stage() {
  local stage="$1"
  local target="$2"
  shift 2
  echo
  echo "==> stage=${stage} target=${target}"
  "$PYTHON_BIN" -m severson_features_soh_rul.modeling.pipeline \
    stage="$stage" \
    target="$target" \
    "$@"
}

for target in "${TARGETS[@]}"; do
  run_stage optimize "$target"
  run_stage fit_final_model "$target"
  run_stage predict "$target"
  run_stage rank "$target"
  run_stage topk_sweep "$target"
  run_stage robustness_protocol_lopo "$target"
done

echo
echo "All pipeline stages completed."
