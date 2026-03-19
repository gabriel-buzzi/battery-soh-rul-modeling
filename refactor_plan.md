# Refactor Plan: Simple Prediction-First Modeling Pipeline

## 1) Goal
Create a new modeling pipeline that is intentionally minimal:
- Run only what is needed to produce prediction artifacts.
- Keep analysis/reporting concerns outside this implementation plan.
- Keep reproducibility guarantees (deterministic split, config snapshot, cache keys).

This plan supersedes the previous multi-track artifact-heavy workflow.

## 2) Scope Decisions
### In scope
- Unified optimization on train split only.
- Permutation ranking with conformal uncertainty (MAPIE-based).
- Top-k sweep from permutation ranking to test compact feature subsets.
- Test-set prediction with conformal intervals.
- Protocol robustness reframed as leave-one-protocol-out (LOPO), not protocol-family grouping.
- Reweighting as a first-class configurable option in training (`sample_weight`/resampling).
- Single, unified config tree.
- One shared code path for both targets (`SOH`, `RUL`) and any feature set specified explicitly in config.

### Out of scope
- Smoothing-window ablation (smoothing removed from active pipeline).
- Early-life-only training track (explicitly not pursued for this revision).
- New chemistry datasets (not feasible in current dataset/code constraints).
- Reviewer items in the list above are intentionally excluded from this refactor objective and are not acceptance blockers.

## 3) Design Principles
- One input table, one split strategy, one model family (ExtraTrees), few commands.
- Save raw predictions first-class; metrics are optional side effects.
- Keep modules pure and composable; avoid track-specific duplication.
- Strict train/test boundaries by cell for all baseline tasks.
- Every run writes only small, essential artifacts + resolved config.
- `features.columns` is the only feature source of truth in config.
- Artifact lookup must be metadata-driven (deterministic run key), never hardcoded file paths in user config.
- Stage execution should be idempotent by default (reuse existing artifacts unless explicit overwrite is requested).

## 4) Minimal Pipeline Stages
1. `optimize`
- Inputs: train split, explicit feature list, target, search_space, cv setup.
- Method: grouped CV on train only.
- Objective: overfitting-aware CV objective, not pure RMSE.
- Per-fold terms:
  - `rmse_val`
  - `overfit_gap = max(0, rmse_val - rmse_train) / max(rmse_val, eps)`
- Aggregate objective:
  - `obj = mean(rmse_val) + lambda_gap * mean(max(0, overfit_gap - tau_gap))`
- Recommended defaults:
  - `tau_gap = 0.05` (5% tolerated relative gap)
  - `lambda_gap = 0.5` (penalty strength; tune by sensitivity)
- Weighting: if enabled in config, apply the selected weighting strategy in fold training.
- Output: `best_params.json` and optional `cv_trials.csv`.
- Stability artifacts (mandatory):
  - `cv_fold_metrics.csv` (per-fold train/val RMSE and objective components)
  - `cv_aggregate_metrics.json` (mean/std across folds, including gap penalty terms)
- Cache key sensitive to: target, feature signature (hash of configured feature list), split seed, cv folds, search space, model fixed params, weighting strategy/config.

2. `rank`
- Inputs: train split, optimized params, explicit feature list, target.
- Method: grouped CV + permutation importance with MAPIE model wrapper.
- Weighting: use the same training weighting policy as optimization for internal consistency.
- Outputs:
  - `predictions_rank_val.csv` (val predictions + intervals, fold-level)
  - `ranking_permutation_rmse.csv`
  - `ranking_permutation_interval_width.csv`
  - `ranking_composite.csv` (single sorted ranking used by top-k sweep)
  - `ranking_stability.csv` (per-feature mean/std across folds and repeats)

3. `topk_sweep`
- Inputs: train split, ranking output, target, explicit feature list.
- Method: evaluate predefined k values using grouped CV with MAPIE outputs.
- Outputs:
  - `topk_sweep_cv.csv`
  - `topk_selection.json` (selected k and rule inputs)

4. `fit_final_model`
- Inputs: train split, resolved params artifact, selected feature set (explicit full list or selected top-k list), target.
- Method: load best hyperparameters from file (same artifact-resolution logic pattern used by `predict`), fit on full train split, save model file.
- Output: `model.best.joblib`.
- Requirement: fail fast if no unique matching optimize artifacts are found.

5. `predict`
- Inputs: saved model file, selected feature set (full or top-k), dataset split to predict.
- Method: load fitted model from disk and run prediction with intervals.
- Output: `predictions_test.csv`.
- Requirement: fail fast if no unique matching fitted model artifact is found.

6. `robustness_protocol_lopo`
- Inputs: all rows, protocol id column, explicit feature list, target, optimized params.
- Method: for each protocol value P, train on protocols != P, test on protocol == P.
- Output: `predictions_protocol_lopo.csv` (one row per sample prediction, with held_out_protocol tag).

7. `baseline_flow` (orchestrator)
- Inputs: same config used by baseline stages.
- Method: run sequentially `optimize -> fit_final_model -> predict`.
- Output: no new artifact type; only stage orchestration.

## 5) Artifact Contract (Prediction-First)
For every run directory, always save:
- `config.resolved.yaml`
- `run_info.json` (stage, target, feature_set_id, feature_hash, seed, git sha, timestamps)
- `predictions*.csv` (stage-dependent)

Prediction schema (mandatory columns):
- `cell`
- `cycle`
- `target`
- `feature_set_id`
- `feature_hash`
- `split_seed`
- `y_true`
- `y_pred`
- `y_pred_lo` (if conformal)
- `y_pred_hi` (if conformal)
- `stage` (`rank_val`, `test`, `protocol_lopo`, etc.)

Optional metadata sidecars:
- `best_params.json`
- `model.best.joblib`
- `cv_trials.csv`
- `cv_fold_metrics.csv`
- `cv_aggregate_metrics.json`
- `ranking_*.csv`
- `ranking_stability.csv`
- `topk_selection.json`

## 6) Unified Config Proposal
Create one config root at `config/modeling.yaml` with these top-level sections:
- `data`
- `split`
- `features`
- `model`
- `optimize`
- `conformal`
- `ranking`
- `robustness`
- `weighting`
- `artifacts`

Example contract fields:
- `data.features_path`
- `data.protocol_column` (default: `charge_policy`)
- `split.train_cells_proportion`
- `split.seed`
- `features.columns` (explicit list of feature column names; source of truth)
- `features.id` (optional human-friendly name; if null use `feature_hash`)
- `features.hash_mode` (`order_invariant` default; `order_sensitive` optional)
- `model.name` (start with `extratrees` only)
- `optimize.enabled`, `optimize.n_trials`, `optimize.cv_folds`
- `optimize.objective.lambda_gap`, `optimize.objective.tau_gap`
- `conformal.enabled`, `conformal.alpha`
- `ranking.n_repeats`
- `robustness.mode: protocol_lopo`
- `weighting.enabled`, `weighting.strategy`
- `features.selection_mode` (`base` or `topk`)
- `features.topk.k_values`
- `features.topk.selection_rule`
- `artifacts.root_dir`
- `artifacts.naming_key` (`feature_id_or_hash` default)
- `artifacts.run_key_fields` (deterministic artifact identity fields)
- `artifacts.overwrite` (`false` by default)
- `artifacts.require_exact_match` (`true` by default)

## 7) Code Organization (Target State)
Proposed nested package layout inside `src/severson_features_soh_rul/modeling/`:

```text
modeling/
  __init__.py
  pipeline.py                       # CLI entrypoint + stage dispatch

  config/
    __init__.py
    schema.py                       # typed config schema
    defaults.py                     # config defaults/constants

  data/
    __init__.py
    features.py                     # feature list validation + signatures
    split.py                        # deterministic cell split

  core/
    __init__.py
    models.py                       # model builders (ExtraTrees, etc.)
    conformal.py                    # MAPIE wrappers/utilities
    weighting.py                    # sample-weight/resampling strategies

  stages/
    __init__.py
    optimize.py                     # optimization stage
    rank.py                         # permutation ranking stage
    topk_sweep.py                   # top-k sweep + selection
    fit_final_model.py              # full-train fit from optimize artifacts
    predict.py                      # inference stage from saved model
    robustness_protocol_lopo.py     # LOPO robustness stage
    baseline_flow.py                # orchestration stage

  artifacts/
    __init__.py
    writer.py                       # artifact write helpers + run_info
    resolver.py                     # metadata/run-key artifact resolution
    run_key.py                      # deterministic run-key construction

  metrics/
    __init__.py
    regression.py                   # RMSE/MAE/R2 helpers
    objectives.py                   # overfitting-aware objective helpers
```

Package boundary rules:
- `stages/*` orchestrate; they should not contain low-level business logic.
- `core/*` and `metrics/*` are pure reusable logic.
- `artifacts/*` is the only layer allowed to define file/folder conventions.
- `data/*` is the only layer that knows about input schema details (`cell`, `cycle`, feature columns).
- `pipeline.py` imports only stage-level interfaces, not low-level internals.

Remove `tracks/*` once the new pipeline commands are validated.

## 8) Execution Surface (Simple)
Single CLI module:
- `python -m severson_features_soh_rul.modeling.pipeline stage=optimize ...`
- `python -m severson_features_soh_rul.modeling.pipeline stage=rank ...`
- `python -m severson_features_soh_rul.modeling.pipeline stage=topk_sweep ...`
- `python -m severson_features_soh_rul.modeling.pipeline stage=fit_final_model ...`
- `python -m severson_features_soh_rul.modeling.pipeline stage=predict ...`
- `python -m severson_features_soh_rul.modeling.pipeline stage=robustness_protocol_lopo ...`
- `python -m severson_features_soh_rul.modeling.pipeline stage=baseline_flow ...`

No separate "export paper tables" step in core pipeline.

Artifact resolution strategy:
- Resolve artifacts by metadata match (`target`, `feature_hash`, `split.seed`, `weighting`, `stage`, `model`) instead of fragile filename patterns.
- Use `features.id` only as human-readable label; use `feature_hash` for deterministic matching.
- Fail on 0 or >1 matches when `artifacts.require_exact_match=true`.

Artifact folder layout:
- `<root>/<run_key>/optimize/`
- `<root>/<run_key>/rank/`
- `<root>/<run_key>/topk_sweep/`
- `<root>/<run_key>/fit_final_model/`
- `<root>/<run_key>/predict/`
- `<root>/<run_key>/robustness_protocol_lopo/`

Run key (deterministic):
- `target`
- `feature_hash`
- `split.seed`
- `model.name`
- `weighting.strategy` (or `none`)
- optional: `topk.k_selected` when `features.selection_mode=topk`

Stage dependency contract:
- `fit_final_model` requires exactly one matching `optimize` artifact set.
- `predict` requires exactly one matching `fit_final_model` model artifact.
- `robustness_protocol_lopo` can either:
  - consume `best_params` from optimize and refit per holdout, or
  - consume a selected final model where methodologically valid (default: refit per holdout).

## 9) Protocol Robustness Reframe (Important)
Replace family binning with strict LOPO:
- Unit of holdout: exact protocol identifier (`charge_policy`).
- Allow tiny holdout protocols (1-2 cells), but tag counts in outputs.
- Keep this as prediction artifact generation only; analysis notebook can aggregate per protocol.

## 10) Weighting/Resampling Strategy
Weighting is pipeline-native (not a separate stage). Change weighting config and rerun the same stages.

Supported strategy candidates:
- `sample_weight_inverse_life_density`
- `sample_weight_long_life_boost`
- `cell_level_oversample_long_life`

Each run writes the same prediction schema, enabling direct artifact-to-artifact comparison across weighting policies.

## 11) Top-k Sweep Policy (Dual Objective)
Context:
- Ranking now has two criteria: prediction error impact (RMSE) and uncertainty impact (interval width / spread).
- We need one ordered list for sweep and one deterministic rule to select k.

Proposed approach:
1. Build a composite ranking score per feature:
- Rescale both impacts before combining (required because units/scales differ).
- Default rescaling: quantile-clipped min-max to [0, 1]:
  - clip each metric to `[q05, q95]` across features,
  - then min-max normalize.
- Optional robust alternative: rank-normalization to percentile scores.
- `score = w_rmse * impact_rmse + w_uncertainty * impact_uncertainty`
- Default weights: `w_rmse=0.7`, `w_uncertainty=0.3`.
2. Use this ordering to run top-k CV sweep.
3. Build feasible set using baseline-relative constraints:
- `rmse_mean(k) <= rmse_mean(full) * (1 + tau_rmse)`
- `interval_width_mean(k) <= interval_width_mean(full) * (1 + tau_width)`
- Recommended defaults: `tau_rmse=0.05`, `tau_width=0.10`.
4. Selection rule:
- If feasible set is not empty: choose smallest `k` in feasible set (explicit compactness preference).
- If feasible set is empty: pick k with best lexicographic tuple:
- minimize `rmse_mean`
- then minimize `interval_width_mean`
- then minimize `k`

Why not pure lexicographic always:
- Pure lexicographic tends to favor larger k for tiny RMSE gains.
- Feasible-set + smallest-k makes the compactness goal explicit and reviewer-defensible.

## 12) Migration Plan
Phase A: foundation
- Freeze current split behavior and verify seed-42 reproducibility.
- Implement unified config and minimal artifact writer.
- Implement run-key builder + resolver and stage dependency checks.

Phase B: core parity
- Implement `optimize`, `fit_final_model`, and `predict` stages.
- Validate outputs against current baseline predictions.
- Implement `baseline_flow` orchestrator.

Phase C: ranking + uncertainty
- Implement MAPIE-based `rank` stage and uncertainty-aware ranking outputs.
- Implement `topk_sweep` stage and deterministic k-selection.

Phase D: protocol + weighting
- Implement LOPO robustness stage.
- Implement weighting strategies in shared training flow.

Phase E: deprecation cleanup
- Remove old `tracks/*`, old uncertainty track, old diagnostics track, and paper-table export coupling.
- Update README and scripts to new CLI.
- Remove obsolete config trees after cutover verification.

## 13) Acceptance Criteria
- Baseline command path is 3 required stages: optimize -> fit_final_model -> predict.
- `baseline_flow` performs the same chain with one command and identical artifacts.
- All produced CSVs contain `cell`, `cycle`, `y_true`, `y_pred` (+ interval columns when enabled).
- Re-running with same seed/config yields identical split and deterministic outputs (except expected model stochasticity controlled by seed).
- LOPO stage runs end-to-end with one artifact file (`predictions_protocol_lopo.csv`).
- `fit_final_model` fails clearly when no unique matching optimize artifacts are found.
- `predict` fails clearly when no unique matching fitted model artifact is found.
- Top-k selection is reproducible and exported (`topk_selection.json`).
- Optimization stability evidence is exported (`cv_fold_metrics.csv`, `cv_aggregate_metrics.json`).
- Ranking stability evidence is exported (`ranking_stability.csv`).
- Resolver fails deterministically on ambiguous artifact matches.

## 14) Engineering Implementation Details
Coding standards:
- All new/refactored code must include explicit type annotations compatible with project Python version `3.10`.
- Use NumPy-style docstrings for public modules, classes, and functions.
- Keep type syntax/libraries compatible with Python `3.10` (for example, avoid features requiring newer versions).
- Ensure docstrings and typing remain compliant with repository lint configuration (`ruff` + `pydocstyle` NumPy convention).

Error handling:
- Every stage must validate required input columns and config keys at startup.
- Error messages must include stage name, missing requirement, and expected run-key fields.

File I/O semantics:
- Write all artifacts atomically (tmp file + rename) to avoid partial artifacts on interruption.
- Include `run_info.json` with schema version and producer code version.

Logging:
- Structured logs per stage with: stage, run_key, target, feature_hash, split seed, weighting, elapsed seconds.

Overwrites:
- Default `artifacts.overwrite=false`.
- If output exists and overwrite is false, stage should skip and return existing artifact metadata.

## 15) Testing & QA (Minimum)
- Unit: feature signature/hash generation (`order_invariant` and `order_sensitive` modes).
- Unit: run-key generation and resolver exact-match behavior.
- Unit: stage dependency checks (`predict` without fitted model, `fit_final_model` without optimize artifacts).
- Integration: deterministic split reproduction for fixed seed.
- Integration: `baseline_flow` produces same outputs as manual stage chain.
- Integration: top-k selection artifact generated and parseable.

## 16) Notes / Caveats
- MAPIE is expected to run with grouped splitting using `GroupKFold`; keep this as the default conformal CV splitter in this project.
- If conformal prediction runtime is too high, keep a switch to disable it while preserving the same output schema (interval columns nullable).
