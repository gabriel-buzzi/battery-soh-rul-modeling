#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Running baseline flow for SOH"
"$PYTHON_BIN" -m severson_features_soh_rul.modeling.pipeline \
  stage=baseline_flow \
  target=SOH

echo "Running baseline flow for RUL"
"$PYTHON_BIN" -m severson_features_soh_rul.modeling.pipeline \
  stage=baseline_flow \
  target=RUL

echo "All requested runs completed."
