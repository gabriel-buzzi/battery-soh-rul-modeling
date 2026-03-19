#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIXI_BIN="${PIXI_BIN:-pixi}"

STAGES=(
  optimize
  fit_final_model
  predict
  permutation_importance
  rank
  topk_sweep
  robustness_protocol_lopo
)

run_stage() {
  local stage="$1"
  shift 1
  echo
  echo "==> stage=${stage}"
  "$PIXI_BIN" run python -m severson_features_soh_rul.modeling.pipeline \
    stage="$stage" \
    "$@"
}

for stage in "${STAGES[@]}"; do
  run_stage "$stage"
done

echo
echo "All pipeline stages completed."
