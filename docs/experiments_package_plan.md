# Experiments Package Implementation Plan

## Objective
Build a new experiment package that reuses existing data preparation outputs, but replaces current modeling/optimization code with a reproducible, track-based pipeline aligned with the revision experiment plan.

Current extension to objective:
- support multiple targets (`SOH`, `RUL`, `RUL_THROUGHPUT`);
- support both full-cycle and charge-derived feature spaces;
- produce paper-ready artifacts that can be consumed directly by AI writing workflows.

## Scope Boundaries
- Reuse from `src/data`: raw loading, cell/cycle processing, feature extraction.
- Do not reuse from `src/modeling` except small generic utilities if truly needed.
- New source of truth for experiments will be `src/experiments`.
- Notebooks are analysis/reporting clients only; they must not contain training logic.

## Package Layout (target)
```text
src/experiments/
  __init__.py
  runner.py
  dataset.py
  split.py
  cv.py
  models.py
  optimize.py
  ranking.py
  ablations.py
  uncertainty.py
  diagnostics.py
  protocol_robustness.py
  io.py
  schemas.py
```

## Configuration Layout (target)
```text
src/conf/experiments/
  base.yaml
  tracks/full_cycle.yaml
  tracks/charge_only.yaml
  tracks/uncertainty.yaml
  tracks/protocol_robustness.yaml
```

## Phase Plan

### Phase 1: Foundations and Data Contracts
Deliverables:
- `split.py`: deterministic train/test cell split artifact (`train_cells.json`, `test_cells.json`).
- `dataset.py`: feature loading and validated feature views:
  - full-cycle all features
  - top-k feature subsets (provided externally at first)
  - no-temperature variants
- `io.py`: run directory creation and artifact writing conventions.

Acceptance criteria:
- Same split is reused across all runs when given the same seed.
- Group leakage checks pass (no cell overlap train/test).
- Input schema validation fails fast on missing columns.

### Phase 2: Core Full-Cycle Track (MVP)
Deliverables:
- `models.py`: ExtraTrees factory + parameter handling.
- `cv.py`: grouped 5-fold by `cell`, shared scoring for SOH and RUL.
- `optimize.py`: TPE optimization over grouped CV.
- `runner.py`: command entrypoint for full-cycle track.

Artifacts per run:
- `resolved_config.yaml`
- `metrics_cv.csv`
- `metrics_cv.json`
- `best_params.json`
- `predictions_test.csv`
- `predictions_test.json`
- `metrics_test.json`
- `optimization_history.csv`
- `optimization_history.json`
- `debug_optimization_loss.png`
- `debug_test_scatter.png`
- `run_summary.json`
- `split_manifest.json`
- `feature_manifest.json`
- `per_cell_test_metrics.csv`
- `per_cell_test_metrics.json`
- `residual_summary.json`
- `table_main_metrics.csv`
- `table_main_metrics.json`
- `table_cv_metrics.csv`
- `table_cv_metrics.json`
- `table_test_metrics.csv`
- `table_test_metrics.json`
- `artifacts_index.json`

Acceptance criteria:
- End-to-end run from prepared features to held-out test metrics.
- Reproducible metrics within expected random-seed variance.

### Phase 3: Feature Analysis Track
Deliverables:
- `ranking.py`: permutation importance across `(seed, fold)` with mean/std.
- `ablations.py`:
  - top-k sweep (`k = 16, 12, 8, 6, 4, 2`)
  - leave-one-feature-out within selected top-k
  - no-temperature comparison

Artifacts per run:
- `feature_ranking_permutation.csv`
- `feature_ranking_permutation.json`
- `feature_ranking_intrinsic.csv`
- `feature_ranking_intrinsic.json`
- `topk_sweep_metrics.csv`
- `topk_sweep_metrics.json`
- `loo_metrics.csv`
- `loo_metrics.json`
- `no_temp_metrics.json`
- `feature_analysis_summary.json`
- `topk_vs_val_rmse.png`
- `topk_vs_relative_gap.png`

Acceptance criteria:
- Stable ranking table with variability estimate.
- Compact subset selection supported by explicit metrics.
- LOO executed in the same run using `selected_k` resolved from either:
  - explicit integer config, or
  - `selected_k: heuristics`.
- Full-feature baseline included in top-k sweep without retraining:
  - baseline row always present (`k = total_features`);
  - `val_rmse_delta_from_baseline = 0` for baseline row.

### Phase 4A: Charge-Only Feature Analysis
Deliverables:
- charge-only feature view integration in `dataset.py`.
- charge-only track config and execution path.
- charge-only ranking/top-k/LOO/no-temperature analysis with the same structure as full-cycle feature analysis.

Artifacts per run:
- `feature_ranking_permutation.csv`
- `feature_ranking_permutation.json`
- `feature_ranking_intrinsic.csv`
- `feature_ranking_intrinsic.json`
- `topk_sweep_metrics.csv`
- `topk_sweep_metrics.json`
- `loo_metrics.csv`
- `loo_metrics.json`
- `no_temp_metrics.json`
- `topk_vs_val_rmse.png`
- `topk_vs_relative_gap.png`

Acceptance criteria:
- Comparable artifact structure between full-cycle and charge-only feature-analysis tracks.

### Phase 4B: Uncertainty Track
Deliverables:
- `uncertainty.py`: repeated-seed retraining on fixed train/test split.
- uncertainty reporting both overall and by difficult regions.

Artifacts per run:
- `predictions_repeated.csv`
- `predictions_repeated.json`
- `uncertainty_summary.json`
- `uncertainty_by_region.csv` (including near-EoL slices)
- `uncertainty_by_region.json` (including near-EoL slices)

Acceptance criteria:
- Sample-level uncertainty computed from repeated predictions.

### Phase 5: Diagnostics and Robustness
Deliverables:
- `diagnostics.py`: largest-error cell analysis and life-region concentration checks.
- `protocol_robustness.py`: hold-out protocol-family evaluation (cell-grouped).

Artifacts per run:
- `error_cells_summary.csv`
- `error_cells_summary.json`
- `protocol_family_results.csv`
- `protocol_family_results.json`

Acceptance criteria:
- Reviewer-facing robustness and difficult-cell evidence generated directly from pipeline outputs.

## Cross-Cutting Standards
- Single run directory per execution: `results/experiments/<track>/<run_id>/`.
- Every run stores resolved config, code version (git commit hash if available), and random seeds.
- Keep metric definitions centralized and reused across all tracks.
- Avoid hidden preprocessing in modeling code (no implicit smoothing).
- Every run includes a machine-readable artifact contract (`artifacts_index.json`) and schema version.

## Initial Execution Order
1. Implement Phase 1 + Phase 2 only.
2. Validate with SOH and RUL on full-cycle baseline.
3. Add Phase 3 feature analysis.
4. Add charge-only and uncertainty.
5. Add diagnostics and protocol robustness last.

## Implementation Status
- [x] Step 1 completed: skeleton + MVP full-cycle runner.
  - Added package skeleton under `src/experiments`.
  - Added config scaffold under `src/conf/experiments`.
  - Implemented deterministic split artifact creation/reuse.
  - Implemented grouped CV + TPE optimization for `ExtraTrees`.
  - Implemented final train/test evaluation.
  - Implemented paper-ready artifact contract and AI-friendly summaries/manifests.
- [x] Step 2 completed: full-cycle feature analysis track.
  - Added permutation and intrinsic feature ranking (`ranking.py`).
  - Added top-k sweep, leave-one-out, and no-temperature ablations (`ablations.py`).
  - Extended runner to support `track=full_cycle_feature_analysis`.
  - Added top-k performance debug plots (RMSE and relative gap).
  - Added automatic `selected_k` heuristic mode for single-run execution.
  - Added cached full-feature baseline handling in top-k sweep (no duplicated CV training).
  - Added Step 2 artifact exports:
    - `feature_ranking_permutation.csv/json`
    - `feature_ranking_intrinsic.csv/json`
    - `topk_sweep_metrics.csv/json`
    - `loo_metrics.csv/json`
    - `no_temp_metrics.json`
- [x] Data processing updates integrated in pipeline scope:
  - new throughput target (`RUL_THROUGHPUT`);
  - charge-step prefixed features (`charge_*`) in extracted datasets.
- [x] Step 3 completed: Phase 4A charge-only feature analysis.
  - Added `track=charge_only_feature_analysis` in runner.
  - Added `src/conf/experiments/tracks/charge_only_feature_analysis.yaml`.
  - Reused the same ranking/top-k/LOO/no-temperature flow as full-cycle analysis.
  - Reused the same artifact interface (`feature_ranking_*`, `topk_sweep_*`, `loo_*`, `no_temp_metrics.json`, top-k plots).
- [x] Step 4 completed: Phase 4B uncertainty.
  - Added `track=uncertainty` branch in runner.
  - Implemented repeated-seed retraining with fixed split and fixed optimized hyperparameters.
  - Added uncertainty artifacts:
    - `predictions_repeated.csv/json`
    - `uncertainty_summary.json`
    - `uncertainty_by_region.csv/json`
  - Added uncertainty-aware `run_summary.json` and `artifacts_index.json`.
- [x] Optimization caching integrated across tracks.
  - Baseline optimization is persisted and reused when target + feature set + split/config key is unchanged.
  - Re-optimization is triggered automatically when key inputs change (e.g., target or feature columns).
- [x] Step 5 completed: Phase 5 diagnostics/protocol robustness.
  - Added `track=diagnostics` branch:
    - exports `error_cells_summary.csv/json`
    - exports `diagnostics_summary.json`
  - Added `track=protocol_robustness` branch:
    - protocol families from per-cell average charging C-rate
    - leave-one-family-out evaluation
    - exports `protocol_family_results.csv/json`
    - exports `protocol_robustness_summary.json`
  - Integrated both with `run_summary.json` and `artifacts_index.json`.

## How To Test Current State
### Prerequisites
- Environment with project dependencies installed (`hydra`, `optuna`, `pandas`, `scikit-learn`, `pyarrow`).
- A features parquet at minimum containing:
  - metadata/targets: `cell`, `cycle`, `SOH`, `RUL`, `RUL_THROUGHPUT`
  - full-cycle features: `V_*`, `I_*`, `T_*` columns listed in `src/experiments/schemas.py`.
  - optional charge features for charge tracks: `charge_*`.

### Smoke test command
Run from repository root:
```bash
python -m severson_features_soh_rul.modeling.runner \
  data.features_data_path=./data/interim/features.parquet \
  data.split_dir=./results/experiments/splits \
  artifacts.root_dir=./results/experiments \
  target=SOH \
  optimize.n_trials=5 \
  cv.n_splits=5 \
  model.n_jobs=1 \
  random_seed=42
```

### Expected outputs
Under `results/experiments/full_cycle/<run_id>/`:
- `resolved_config.yaml`
- `best_params.json`
- `optimization_history.csv`
- `optimization_history.json`
- `metrics_cv.csv`
- `metrics_cv.json`
- `predictions_test.csv`
- `predictions_test.json`
- `metrics_test.json`
- `run_metadata.json`
- `debug_optimization_loss.png`
- `debug_test_scatter.png`

And under `results/experiments/splits/`:
- `train_cells.json`
- `test_cells.json`

### Determinism check
Run the same command twice with:
- same `random_seed`
- same split files (do not set `data.force_recreate_split=true`)

Then confirm:
- the split files are unchanged
- metrics remain within expected model stochastic tolerance

## Immediate Next Task (Step 6)
Consolidate and harden full revision pipeline.

### Step 6A: Revision Bundle Orchestration
Status: `[x]`

Scope:
- Add one orchestration entrypoint that runs the required revision tracks in sequence:
  - `full_cycle_feature_analysis`
  - `charge_only_feature_analysis`
  - `uncertainty`
  - `diagnostics`
  - `protocol_robustness`
- Ensure all child runs share:
  - same split artifact
  - same target
  - same root output directory
  - same optimization cache directory

Deliverables:
- `src/experiments/revision_bundle.py` (or equivalent runner mode).
- `src/conf/experiments/tracks/revision_bundle.yaml`.
- `results/experiments/revision_bundle/<run_id>/bundle_summary.json` with:
  - child run ids/paths
  - execution order
  - success/failure status per child run
  - total runtime

Acceptance criteria:
- One command triggers all required tracks end-to-end.
- Failed child run reports clear error without hiding prior successful outputs.

### Step 6B: Integration Tests
Status: `[x]`

Scope:
- Add lightweight integration checks for artifact contract and cache behavior.

Deliverables:
- test module(s), for example under `tests/experiments/`, covering:
  - artifact presence by track (`csv/json/png` as applicable)
  - required columns for key artifacts:
    - feature analysis: `topk_sweep_metrics`, `feature_ranking_permutation`, `loo_metrics`
    - uncertainty: `predictions_repeated`, `uncertainty_by_region`
    - diagnostics: `error_cells_summary`
    - protocol robustness: `protocol_family_results`
  - optimization cache hit behavior:
    - first run creates cache
    - second identical run reuses cache

Acceptance criteria:
- Tests pass on a minimal synthetic or reduced dataset fixture.
- Failures identify missing artifact/column names explicitly.

### Step 6C: Paper Table Export
Status: `[x]`

Scope:
- Add one export script that merges outputs from all relevant tracks into manuscript-ready tables.

Deliverables:
- `src/experiments/export_paper_tables.py` (or equivalent).
- outputs under `results/paper_tables/<bundle_or_run_id>/`:
  - `table_main_comparison.csv/json`
  - `table_feature_analysis.csv/json`
  - `table_uncertainty.csv/json`
  - `table_robustness.csv/json`
- optional LaTeX-ready CSV formatting helper for direct inclusion.

Acceptance criteria:
- Tables are generated from run artifacts only (no manual edits).
- Column names/units are stable and match manuscript language.

### Suggested Implementation Order
1. Implement Step 6A orchestration first.
2. Implement Step 6B tests against current tracks and new bundle mode.
3. Implement Step 6C export script once bundle outputs are stable.
