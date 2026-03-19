```python

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from mapie.regression import SplitConformalRegressor

# ==================== INPUTS ====================
# X         : pd.DataFrame        (features only)
# y         : pd.Series or np.ndarray  (target)
# groups    : pd.Series or np.ndarray  (grouping column, same length as X/y)
# estimator : your fitted sklearn-compatible regressor (will be cloned)

n_outer_splits = 5
n_permutations = 10             # number of permutations per feature per fold
confidence_level = 0.95         # → 95% prediction intervals

gkf = GroupKFold(n_splits=n_outer_splits)

# Storage for aggregated impacts
n_features = X.shape[1]
delta_rmse      = np.zeros(n_features)
delta_ci_width  = np.zeros(n_features)
baseline_rmse_list = []
baseline_ci_list   = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
    X_test  = X.iloc[test_idx]
    y_test  = y.iloc[test_idx]  if isinstance(y, pd.Series) else y[test_idx]

    # Clone and fit fresh estimator on this outer train fold
    est = clone(estimator)
    est.fit(X_train, y_train)

    # ── MAPIE Split Conformal (recommended inside outer CV) ──
    mapie = SplitConformalRegressor(
        estimator=est,
        confidence_level=confidence_level,
        conformity_score="absolute",           # alternatives: "residual_normalized", "scaled"
        prefit=True,
        random_state=42 + fold                 # optional: different seed per fold
    )

    # Conformalize using the training data of this fold
    mapie.conformalize(X_train, y_train)

    # Baseline performance on this test fold
    y_pred, y_pi = mapie.predict_interval(X_test)
    # y_pi shape: (n_test, 2) → lower, upper

    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred))
    ci_width_base = (y_pi[:, 1] - y_pi[:, 0]).mean()

    baseline_rmse_list.append(rmse_base)
    baseline_ci_list.append(ci_width_base)

    # ── Permutation per feature ──
    for feat_idx, feature_name in enumerate(X.columns):
        deltas_rmse = []
        deltas_ci   = []

        for _ in range(n_permutations):
            X_test_perm = X_test.copy()
            # Shuffle the column in-place (fastest pandas way)
            X_test_perm[feature_name] = np.random.permutation(X_test_perm[feature_name].values)

            y_pred_perm, y_pi_perm = mapie.predict_interval(X_test_perm)

            rmse_perm = np.sqrt(mean_squared_error(y_test, y_pred_perm))
            ci_width_perm = (y_pi_perm[:, 1] - y_pi_perm[:, 0]).mean()

            deltas_rmse.append(rmse_perm - rmse_base)
            deltas_ci.append(ci_width_perm - ci_width_base)

        # Average over permutations → accumulate over folds
        delta_rmse[feat_idx]     += np.mean(deltas_rmse)
        delta_ci_width[feat_idx] += np.mean(deltas_ci)

# Final averaging across folds
delta_rmse     /= n_outer_splits
delta_ci_width /= n_outer_splits

avg_baseline_rmse = np.mean(baseline_rmse_list)
avg_baseline_ci   = np.mean(baseline_ci_list)

print(f"Average baseline RMSE:      {avg_baseline_rmse: .4f}")
print(f"Average baseline CI width:  {avg_baseline_ci: .4f}\n")

```