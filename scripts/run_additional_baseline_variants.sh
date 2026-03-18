#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running item 1: charge-only baseline test runs"
python -m severson_features_soh_rul.modeling.runner -m \
  track=final_eval \
  target=SOH,RUL \
  features.set_id=charge_all

echo "Running item 2: full-cycle no-temperature test runs"
python -m severson_features_soh_rul.modeling.runner -m \
  track=final_eval \
  target=SOH,RUL \
  features.set_id=full_no_temp

echo "Running item 3: full-cycle compact-subset test runs"
python -m severson_features_soh_rul.modeling.runner -m \
  track=final_eval \
  target=SOH,RUL \
  features.set_id=full_topk

echo "Running item 4: charge-only no-temperature test runs"
python -m severson_features_soh_rul.modeling.runner -m \
  track=final_eval \
  target=SOH,RUL \
  features.set_id=charge_no_temp

echo "Running item 5: charge-only compact-subset test runs"
python -m severson_features_soh_rul.modeling.runner -m \
  track=final_eval \
  target=SOH,RUL \
  features.set_id=charge_topk

echo "All requested runs completed."
