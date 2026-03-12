#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CAMPAIGN_ID="${CAMPAIGN_ID:-revision_round1}"

TARGETS=(SOH RUL)
FINAL_EVAL_FEATURE_SETS=(
  full_all
  full_topk
  full_no_temp
  charge_all
  charge_topk
  charge_no_temp
)
TRACK_FEATURE_SETS=(full_all charge_all)

run_exp() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  "$PYTHON_BIN" -m src.experiments.runner \
    artifacts.campaign_id="$CAMPAIGN_ID" \
    "$@"
}

for target in "${TARGETS[@]}"; do
  echo
  echo "############################################"
  echo "Running revision campaign for target=${target}"
  echo "campaign_id=${CAMPAIGN_ID}"
  echo "############################################"

  run_exp "feature_analysis/full_cycle (${target})" \
    track=full_cycle_feature_analysis \
    target="$target"

  run_exp "feature_analysis/charge_only (${target})" \
    track=charge_only_feature_analysis \
    target="$target"

  for feature_set in "${FINAL_EVAL_FEATURE_SETS[@]}"; do
    run_exp "final_eval (${target}, ${feature_set})" \
      track=final_eval \
      target="$target" \
      features.set_id="$feature_set"
  done

  for feature_set in "${TRACK_FEATURE_SETS[@]}"; do
    run_exp "uncertainty (${target}, ${feature_set})" \
      track=uncertainty \
      target="$target" \
      features.set_id="$feature_set"

    run_exp "diagnostics (${target}, ${feature_set})" \
      track=diagnostics \
      target="$target" \
      features.set_id="$feature_set"

    run_exp "protocol_robustness (${target}, ${feature_set})" \
      track=protocol_robustness \
      target="$target" \
      features.set_id="$feature_set"
  done

  echo
  echo "==> Exporting paper tables for target=${target}"
  "$PYTHON_BIN" -m src.experiments.export_paper_tables \
    artifacts.campaign_id="$CAMPAIGN_ID" \
    target="$target"
done

echo
echo "All tracks completed for campaign_id=${CAMPAIGN_ID}"
