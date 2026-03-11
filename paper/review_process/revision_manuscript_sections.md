# Methodology

## Baseline Full-Cycle Modeling

The revised experimental study adopts a single reference model family, Extremely Randomized Trees (`ExtraTrees`), to isolate the contribution of feature design from the confounding effects of broad model-family comparison. This choice is consistent with the central objective of the study: to determine how much prognostic information can be extracted from lightweight statistics computed on a single diagnostic cycle, and how robust those statistics remain under feature reduction, partial-cycle observation, and cross-family generalization.

All experiments were conducted on the same cycle-level feature table and used a fixed cell-wise train-test split. Entire cells, rather than individual cycles, were assigned to the training or test partition. This design prevents leakage of cell-specific degradation trajectories across partitions and ensures that every reported test result reflects generalization to previously unseen cells. The resulting split contains 99 training cells and 25 held-out test cells, corresponding to 79,001 training cycles and 20,184 test cycles.

Hyperparameter optimization was performed only on the training cells through grouped 5-fold cross-validation, with folds defined by cell identity so that all cycles from a given cell remain together within each fold. The optimization objective combined predictive accuracy and overfitting control:

$$
\mathrm{Objective} = \mathrm{RMSE}_{\mathrm{val}} + \frac{\left| \mathrm{RMSE}_{\mathrm{train}} - \mathrm{RMSE}_{\mathrm{val}} \right|}{\mathrm{RMSE}_{\mathrm{val}}}.
$$

This objective favors configurations with low validation error while penalizing large train-validation discrepancies. Such a criterion is particularly relevant for battery prognosis because cycle-level datasets contain many highly correlated samples within each cell, and models can otherwise appear strong while relying excessively on cell-specific structure.

The baseline full-cycle configuration used the complete set of 16 statistical features extracted from voltage, current, and temperature over the full diagnostic cycle. These baseline experiments establish a consistent reference point against which all subsequent feature-selection, charge-only, uncertainty, diagnostics, and robustness analyses are interpreted.

## Full-Cycle Feature Analysis

The full-cycle feature-analysis track was designed to answer three linked scientific questions: which single-cycle statistics are most informative for prognosis, how much redundancy exists within the 16-feature representation, and how compact a deployable feature subset can become before predictive performance deteriorates appreciably.

The first step was permutation-based feature ranking under grouped cross-validation. For each of five model seeds and each of five grouped folds, the optimized `ExtraTrees` model was trained on the training portion of the fold, evaluated on the corresponding validation portion, and then re-evaluated after shuffling one feature at a time within the validation fold. The increase in validation RMSE caused by shuffling a feature provides a direct measure of that feature's contribution to predictive performance under the fitted multivariate model. Intrinsic tree-based importances were also recorded in the same runs, but only as supporting evidence. Permutation importance was treated as the primary ranking criterion because it quantifies the consequence of destroying the information carried by a feature in the actual predictive setting.

Once ranked, the features were evaluated through a top-$k$ sweep using the ordered subsets $k \in \{16, 12, 10, 8, 6, 4, 2\}$. Each subset was evaluated with grouped 5-fold cross-validation using the same optimized model hyperparameters. This design avoids confounding changes in feature subset with repeated hyperparameter re-optimization and reveals how predictive performance degrades as the feature representation becomes progressively more compact.

The final subset size was selected by a complexity-performance heuristic rather than by choosing the absolute lowest validation RMSE. Specifically, the smallest subset whose validation RMSE remained within 10% of the 16-feature baseline was selected. This criterion frames compactness as a controlled trade-off rather than an arbitrary reduction, favoring feature parsimony when the predictive penalty remains modest.

After the compact subset was selected, a leave-one-feature-out analysis was run inside that subset to determine whether all retained features contributed meaningfully or whether some remained redundant even after ranking and pruning. Finally, a no-temperature ablation was performed by removing all temperature-derived features while keeping the same optimized model structure. This last step was motivated by the well-known practical difficulty of obtaining stable temperature measurements in laboratory and field settings, and by the question of whether a voltage-current-only representation is already sufficient for reliable prognosis.

## Charge-Only Feature Analysis

The charge-only analysis repeated the same methodology on a feature table computed only from the charging process. The ranking, top-$k$ sweep, compact-subset heuristic, leave-one-feature-out analysis, and no-temperature ablation were all preserved unchanged. This parallel design ensures that any observed difference between full-cycle and charge-only performance can be attributed to the information content of the cycle segment itself, rather than to changes in model class, optimization strategy, or validation protocol.

The scientific motivation for this track is practical rather than purely algorithmic. Full diagnostic cycles are informative but operationally restrictive. In many real battery systems, acquiring both charge and discharge trajectories under controlled conditions is burdensome, whereas charge segments are more naturally observed. The charge-only track therefore tests whether a reduced observational window can preserve enough statistical structure for useful cross-cell prognosis.

## Uncertainty Analysis

The uncertainty track quantified the stability of the optimized `ExtraTrees` predictor with respect to repeated retraining under different random seeds. Starting from the optimized hyperparameter set, the model was retrained 20 times on the same full training partition and used to generate repeated predictions for each held-out test cycle. The resulting ensemble of predictions was summarized by its mean, standard deviation, and selected quantiles.

This procedure does not estimate all sources of predictive uncertainty, but it provides a useful measure of model instability induced by stochastic tree construction. The analysis was further stratified by degradation stage using SoH-defined regions: Early-Life (95-100%), Mid-Life (85-95%), and Aged (80-85%). This regionalization was motivated by the expectation that both signal regularity and prognostic difficulty change across the degradation trajectory. The aim was therefore not only to quantify average uncertainty, but also to determine whether uncertainty systematically increases or decreases with aging.

## Difficult-Cell Diagnostics

Aggregate test metrics can conceal the extent to which errors are concentrated in a small subset of cells. To address this, the diagnostics track computed per-cell prediction errors on the held-out test set and ranked cells by RMSE. For each cell, the analysis recorded RMSE, MAE, signed bias, high-percentile absolute error, and the life region in which the cell's errors were most concentrated. Early-, mid-, and late-life regions were defined according to normalized cycle position within each cell, thereby distinguishing whether a cell was consistently difficult or only problematic during a specific stage of its trajectory.

The scientific motivation for this analysis is to separate diffuse model weakness from localized failure modes. If most error is concentrated in a small number of atypical cells, then the main methodological question shifts from average predictive performance to understanding heterogeneity, outliers, and the representativeness of the training data.

## Protocol-Family Robustness

The protocol-robustness track evaluated whether the learned feature-target relationships generalize across families of charging aggressiveness rather than only across randomly held-out cells. Cells were grouped into protocol families based on binned maximum charge C-rate, with sparse families merged to ensure sufficient support. A leave-one-family-out evaluation was then performed: for each family, the model was trained on all remaining families and tested only on the held-out family.

This experiment was motivated by the fact that battery prognostic signals are not only cell-dependent but also protocol-dependent. If a feature representation is truly robust, its predictive relationships should remain informative when the distribution of charging aggressiveness shifts. The family-holdout design therefore probes a stronger notion of generalization than random cell holdout alone.

# Results

## Baseline Full-Cycle Performance

Table 1 summarizes the baseline full-cycle experiments for the two primary targets. The held-out test performance confirms that the feature-based `ExtraTrees` approach remains viable for cross-cell prognosis, although the two targets are not equally difficult. SoH estimation remains comparatively accurate, with test RMSE near 1.07 percentage points and strong test-set $R^2$. By contrast, cycle-based RUL prediction is substantially harder, with a test RMSE of 145.66 cycles and a notably larger train-validation gap already visible in cross-validation. This asymmetry indicates that the single-cycle statistics capture present health more directly than long-horizon lifetime.

**Table 1. Baseline full-cycle performance**  
Source artifacts: `full_cycle/*/run_summary.json`

| Target | Feature view | n_features | CV val RMSE | CV relative gap | Test RMSE | Test MAE | Test R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOH | full_all | 16 | 1.1433 | 0.3333 | 1.0660 | 0.8037 | 0.9409 |
| RUL | full_all | 16 | 112.0319 | 0.7339 | 145.6603 | 89.5122 | 0.8660 |

## Full-Cycle Feature Relevance and Compactness

The full-cycle ranking results show a clear hierarchy of prognostic relevance. For SoH, the dominant features are voltage entropy, voltage standard deviation, current interquartile range, and voltage interquartile range. For RUL, the ranking is led by voltage interquartile range and current standard deviation, followed by voltage entropy and voltage standard deviation. In both tasks, voltage-distribution descriptors dominate the upper portion of the ranking, while temperature features are absent from the most influential positions. This pattern indicates that the main single-cycle prognostic signal is encoded in the shape and dispersion of the voltage trajectory, with current statistics providing additional but secondary information.

**Table 2. Full-cycle feature ranking summary**  
Source artifacts: `full_cycle_feature_analysis/*/feature_ranking_permutation.csv`

| Target | Rank | Feature | Mean RMSE increase | Stability std |
| --- | ---: | --- | ---: | ---: |
| SOH | 1 | `V_entropy` | 1.8799 | 0.1338 |
| SOH | 2 | `V_std` | 0.7130 | 0.2623 |
| SOH | 3 | `I_iqr` | 0.4884 | 0.0473 |
| SOH | 4 | `V_iqr` | 0.4813 | 0.0941 |
| SOH | 5 | `I_kurtosis` | 0.4787 | 0.0817 |
| SOH | 6 | `V_median` | 0.4205 | 0.0594 |
| RUL | 1 | `V_iqr` | 104.4589 | 26.7059 |
| RUL | 2 | `I_std` | 61.3457 | 45.1326 |
| RUL | 3 | `V_entropy` | 48.4977 | 10.9364 |
| RUL | 4 | `V_std` | 47.0014 | 17.9436 |
| RUL | 5 | `I_mean` | 41.4087 | 9.3307 |
| RUL | 6 | `I_median` | 38.2823 | 20.5138 |

The top-$k$ sweep clarifies how much redundancy exists in the 16-feature representation. For SoH, validation RMSE improves slightly when the 16-feature baseline is reduced to 10 or 12 features, indicating that the full representation contains some distracting or weakly relevant features. The compact 6-feature subset does incur a validation penalty relative to the best-performing larger subsets, but remains within the pre-defined 10% tolerance. For RUL, the pattern is similar but more pronounced: performance is best around 10 features, while the 6-feature subset remains only marginally worse than the 16-feature baseline and far better than the 4- or 2-feature variants. These results support the use of six-feature compact representations as a balanced compromise between predictive power and deployment simplicity.

**Table 3. Full-cycle top-k sweep**  
Source artifacts: `full_cycle_feature_analysis/*/topk_sweep_metrics.csv`

| Target | k | Val RMSE | Relative gap | Delta from 16-feature baseline |
| --- | ---: | ---: | ---: | ---: |
| SOH | 16 | 1.1433 | 0.3333 | 0.0000 |
| SOH | 12 | 1.0912 | 0.2716 | -0.0521 |
| SOH | 10 | 1.0879 | 0.2612 | -0.0554 |
| SOH | 8 | 1.1151 | 0.2499 | -0.0282 |
| SOH | 6 | 1.2422 | 0.2208 | 0.0989 |
| SOH | 4 | 1.5707 | 0.1479 | 0.4274 |
| SOH | 2 | 2.2847 | 0.0988 | 1.1414 |
| RUL | 16 | 112.0319 | 0.7339 | 0.0000 |
| RUL | 12 | 107.4671 | 0.6820 | -4.5648 |
| RUL | 10 | 102.6867 | 0.5671 | -9.3453 |
| RUL | 8 | 105.8965 | 0.5281 | -6.1354 |
| RUL | 6 | 112.2295 | 0.5191 | 0.1976 |
| RUL | 4 | 137.7590 | 0.4560 | 25.7271 |
| RUL | 2 | 252.6106 | 0.2421 | 140.5787 |

[Figure Placeholder: Full-cycle top-k performance curves]

The heuristic compact-subset selection chose $k=6$ for both targets, even though the absolute minimum validation RMSE appears at larger $k$. This decision is methodologically defensible because the purpose of the sweep was not to identify the single numerically best subset, but to identify the smallest subset that preserves nearly all of the predictive value of the larger representation. Under that criterion, the selected SoH subset was `V_entropy`, `V_std`, `I_iqr`, `V_iqr`, `I_kurtosis`, and `V_median`, while the selected RUL subset was `V_iqr`, `I_std`, `V_entropy`, `V_std`, `I_mean`, and `I_median`.

The no-temperature ablation further strengthens the conclusion that temperature contributes weakly and inconsistently in this dataset. Removing all temperature features improved validation RMSE for both full-cycle SoH and full-cycle RUL. The gain is especially relevant for RUL, where the no-temperature configuration reduced validation RMSE from 112.03 to 101.99 cycles while also reducing the generalization gap. This result suggests that the voltage-current statistics already encode most of the useful prognostic information, and that temperature measurements in this dataset may add noise rather than complementary signal.

**Table 4. Full-cycle no-temperature ablation**  
Source artifacts: `full_cycle_feature_analysis/*/no_temp_metrics.json`

| Target | Configuration | n_features | Val RMSE | Relative gap | Objective score |
| --- | --- | ---: | ---: | ---: | ---: |
| SOH | Full-cycle baseline | 16 | 1.1433 | 0.3333 | 1.4766 |
| SOH | No-temperature | 11 | 1.0869 | 0.2657 | 1.3526 |
| RUL | Full-cycle baseline | 16 | 112.0319 | 0.7339 | 112.7658 |
| RUL | No-temperature | 11 | 101.9910 | 0.5853 | 102.5763 |

## Charge-Only Prognostics

The charge-only results preserve the same qualitative feature hierarchy but with a more compact dominant core. Voltage median, current median, voltage entropy, and current or voltage spread statistics occupy the top of the ranking for both targets. For SoH, the six-feature charge-only subset includes `charge_V_median`, `charge_I_median`, `charge_V_entropy`, `charge_V_std`, `charge_V_iqr`, and `charge_I_std`. For RUL, the four-feature subset `charge_V_median`, `charge_I_median`, `charge_V_entropy`, and `charge_I_std` is already sufficient under the compactness heuristic. This compression relative to the full-cycle case indicates that the charging segment concentrates much of the most actionable single-cycle signal, albeit not all of it.

**Table 5. Charge-only feature ranking and compact subset summary**  
Source artifacts: `charge_only_feature_analysis/*/feature_ranking_permutation.csv`, `feature_analysis_summary.json`

| Target | Rank | Feature | Mean RMSE increase | Stability std |
| --- | ---: | --- | ---: | ---: |
| SOH | 1 | `charge_V_median` | 1.4980 | 0.2744 |
| SOH | 2 | `charge_I_median` | 1.3504 | 0.1144 |
| SOH | 3 | `charge_V_entropy` | 0.9521 | 0.2044 |
| SOH | 4 | `charge_V_std` | 0.5725 | 0.1228 |
| SOH | 5 | `charge_V_iqr` | 0.1956 | 0.0317 |
| SOH | 6 | `charge_I_std` | 0.1494 | 0.0550 |
| RUL | 1 | `charge_V_median` | 213.8374 | 32.8777 |
| RUL | 2 | `charge_I_median` | 68.0544 | 27.7644 |
| RUL | 3 | `charge_V_entropy` | 32.4691 | 14.7522 |
| RUL | 4 | `charge_I_std` | 31.4898 | 34.7147 |
| RUL | Selected compact subset | `k=6` for SOH, `k=4` for RUL |  |  |

Charge-only validation performance is consistently worse than the corresponding full-cycle performance, especially for RUL. Nevertheless, the degradation is not catastrophic. For SoH, the 16-feature charge-only baseline reaches a validation RMSE of 1.2178, compared with 1.1433 for the full-cycle baseline. For RUL, the corresponding values are 136.03 and 112.03 cycles. This gap confirms that discharge information remains valuable, but it also shows that the charging segment alone still contains a non-trivial amount of prognostic structure.

The charge-only top-$k$ sweep reinforces the compactness argument. For SoH, larger subsets provide slightly better validation performance, but the six-feature subset remains close enough to the baseline to justify its selection. For RUL, the compact 4-feature subset performs worse than the best 8- or 10-feature subsets, yet still retains the dominant information carriers and therefore offers a parsimonious approximation when feature count is operationally constrained.

**Table 6. Charge-only top-k and no-temperature comparison**  
Source artifacts: `charge_only_feature_analysis/*/topk_sweep_metrics.csv`, `no_temp_metrics.json`

| Target | Configuration | n_features | Val RMSE | Relative gap | Delta from charge-only baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| SOH | Charge-only baseline | 16 | 1.2178 | 0.2459 | 0.0000 |
| SOH | Charge-only compact subset | 6 | 1.3203 | 0.1842 | 0.1024 |
| SOH | Charge-only no-temperature | 11 | 1.2277 | 0.2254 | 0.0098 |
| RUL | Charge-only baseline | 16 | 136.0298 | 0.7673 | 0.0000 |
| RUL | Charge-only compact subset | 4 | 145.8439 | 0.5031 | 9.8141 |
| RUL | Charge-only no-temperature | 11 | 130.5267 | 0.6692 | -5.5032 |

[Figure Placeholder: Charge-only vs full-cycle compactness comparison]

The no-temperature charge-only results again indicate that thermal features are not consistently beneficial. The effect is nearly neutral for SoH and beneficial for RUL, where removing temperature reduces validation RMSE from 136.03 to 130.53 cycles. Taken together with the full-cycle ablations, this pattern suggests that temperature is not a reliable pillar of the present feature representation and should not be treated as a prerequisite for deployment-oriented variants.

## Uncertainty Across Life Regions

The repeated-seed uncertainty analysis shows that prediction spread is generally modest compared with absolute predictive error, but is strongly stage-dependent. For SoH, the mean prediction standard deviation increases from 0.0298 percentage points in the Early-Life region to 0.0632 in the Aged region. The corresponding predictive RMSE also worsens steadily, from 0.7308 to 1.8385. Thus, both predictive instability and predictive inaccuracy increase as the cell approaches end-of-life. The negative $R^2$ in the Aged region indicates that this final stage remains difficult despite the relatively narrow SoH range.

For RUL, the pattern differs. The mean prediction standard deviation is largest in early life (6.11 cycles) and smallest in the aged region (1.26 cycles). Absolute predictive error follows the same trend, with RMSE falling from 162.70 cycles in early life to 34.31 cycles near end-of-life. This does not imply that late-life RUL is intrinsically easier in a relative sense; rather, the remaining lifetime itself becomes smaller, which shrinks the absolute error scale. The important distinction is that prediction spread and predictive error both vary systematically with degradation stage, but not identically across targets.

**Table 7. Uncertainty by life region**  
Source artifacts: `uncertainty/*/uncertainty_by_region.csv`

| Target | Region | RMSE of mean prediction | MAE | R² | Mean prediction std | q90 prediction std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SOH | Early-Life | 0.7308 | 0.5911 | 0.4708 | 0.0298 | 0.0449 |
| SOH | Mid-Life | 1.2622 | 1.0044 | 0.7941 | 0.0570 | 0.0859 |
| SOH | Aged | 1.8385 | 1.3771 | -0.6453 | 0.0632 | 0.1140 |
| RUL | Early-Life | 162.7012 | 105.2180 | 0.8291 | 6.1061 | 14.0029 |
| RUL | Mid-Life | 134.4387 | 80.5313 | 0.7090 | 4.3451 | 10.2050 |
| RUL | Aged | 34.3064 | 19.4232 | -0.2420 | 1.2629 | 2.9507 |

[Figure Placeholder: Prediction uncertainty across life regions]

## Difficult-Cell Diagnostics

Per-cell diagnostics show that prediction error is not evenly distributed across the held-out test set. Instead, a limited subset of cells contributes disproportionately to the overall error, especially for RUL. For SoH, the ten most difficult cells have a mean RMSE of 1.3444, whereas the remaining cells average 0.7060. For RUL, the separation is much sharper: the ten most difficult cells average 144.56 cycles RMSE, compared with 42.90 cycles for the rest. This concentration of error indicates that the model is not uniformly weak; rather, it struggles with a specific subset of trajectories.

Several cells recur across both SoH and RUL diagnostics, including `b3c7`, `b1c3`, `b2c1`, `b1c0`, and `b3c39`. Their repeated appearance suggests that the most difficult cases are not target-specific accidents but persistent outliers in the held-out population. The dominant error region also differs by target. For SoH, difficult cells are frequently dominated by late-life errors, consistent with the worsening aged-region behavior seen in the uncertainty analysis. For RUL, difficult cells are more often dominated by early- or mid-life errors, consistent with the larger absolute scale of remaining life in those regions.

**Table 8. Difficult-cell diagnostics**  
Source artifacts: `diagnostics/*/error_cells_summary.csv`, `diagnostics_summary.json`

| Target | Cell | RMSE | MAE | Dominant error region |
| --- | --- | ---: | ---: | --- |
| SOH | `b1c20` | 1.5963 | 1.2951 | late_life |
| SOH | `b1c0` | 1.5944 | 1.2275 | late_life |
| SOH | `b2c1` | 1.5439 | 1.3335 | late_life |
| SOH | `b3c7` | 1.4120 | 1.2658 | mid_life |
| SOH | `b2c42` | 1.4018 | 1.1319 | late_life |
| SOH | `b1c3` | 1.2853 | 0.8706 | late_life |
| SOH | `b3c39` | 1.2494 | 1.1687 | late_life |
| SOH | `b1c45` | 1.1830 | 1.0070 | late_life |
| SOH | `b2c24` | 1.1171 | 0.8746 | mid_life |
| SOH | `b2c43` | 1.0605 | 0.8556 | early_life |
| RUL | `b3c7` | 368.4326 | 317.8861 | early_life |
| RUL | `b1c3` | 210.1822 | 173.6432 | mid_life |
| RUL | `b2c1` | 163.1153 | 145.1395 | early_life |
| RUL | `b1c0` | 155.9591 | 123.3410 | mid_life |
| RUL | `b3c25` | 110.8863 | 83.6649 | early_life |
| RUL | `b3c39` | 101.6881 | 87.1706 | mid_life |
| RUL | `b1c36` | 98.2172 | 81.3563 | mid_life |
| RUL | `b3c8` | 83.5518 | 70.8162 | early_life |
| RUL | `b1c37` | 78.2553 | 61.7117 | early_life |
| RUL | `b1c40` | 75.2902 | 56.7190 | early_life |

[Figure Placeholder: Per-cell error distribution / difficult cells]

## Protocol-Family Robustness

The family-holdout evaluation shows that the learned feature-target relationships do generalize beyond a random cell split, but not uniformly across charging aggressiveness families. For SoH, mean family-holdout performance remains reasonably stable, with an average RMSE of 1.4847 and family-level $R^2$ values between 0.8470 and 0.9219. This indicates that the feature representation captures degradation signatures that are at least partly portable across families of charging conditions.

RUL is more sensitive to protocol shift. Family-holdout RMSE ranges from 66.08 cycles for `bin_1__no_rest` to 196.95 cycles for `bin_2__no_rest`, with lower $R^2$ in the more difficult families. The strongest deterioration occurs in higher-rate families with longer and more variable cycle-life distributions. These results imply that the model's cross-cell generalization is meaningful but incomplete: robustness degrades when the held-out family differs more strongly in aggressiveness and lifetime profile from the training families.

**Table 9. Protocol-family robustness**  
Source artifacts: `protocol_robustness/*/protocol_family_results.csv`, `protocol_robustness_summary.json`

| Target | Held-out family | RMSE | MAE | R² | Avg charge C-rate | Max charge C-rate | Mean cycle life |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOH | `bin_0__no_rest` | 1.2578 | 0.8944 | 0.9219 | 2.0696 | 2.3702 | 740.94 |
| SOH | `bin_1__no_rest` | 1.8110 | 0.9972 | 0.8470 | 2.2369 | 2.5662 | 638.94 |
| SOH | `bin_2__no_rest` | 1.4331 | 1.0989 | 0.8813 | 2.4077 | 2.9162 | 970.32 |
| SOH | `bin_3__no_rest` | 1.4367 | 0.8843 | 0.8840 | 2.7015 | 3.2504 | 847.68 |
| RUL | `bin_0__no_rest` | 116.0194 | 76.4622 | 0.9272 | 2.0696 | 2.3702 | 740.94 |
| RUL | `bin_1__no_rest` | 66.0759 | 47.3833 | 0.9158 | 2.2369 | 2.5662 | 638.94 |
| RUL | `bin_2__no_rest` | 196.9549 | 113.4280 | 0.7976 | 2.4077 | 2.9162 | 970.32 |
| RUL | `bin_3__no_rest` | 175.2911 | 103.8071 | 0.7571 | 2.7015 | 3.2504 | 847.68 |

[Figure Placeholder: Protocol-family holdout performance]

# Discussion

The revised experiments indicate that single-cycle, statistics-based prognosis remains scientifically meaningful, but the evidence is more nuanced than a purely performance-centered reading would suggest. The baseline full-cycle results show that present health can be estimated more reliably than long-horizon remaining life. This asymmetry is expected: SoH is more directly tied to the immediate shape of the observed diagnostic signals, whereas RUL compresses the entire future degradation path into one target and is therefore more sensitive to heterogeneity across cells and protocols.

The feature-analysis results provide the clearest substantive conclusion of the study. Across both full-cycle and charge-only settings, the most informative variables are primarily voltage-distribution features, especially entropy, spread, and median-related descriptors. Current-distribution features contribute additional signal, particularly for RUL, but they rarely displace the leading voltage features. Temperature features, by contrast, are weak and inconsistent. Their absence from the leading ranks and the repeated improvement of no-temperature ablations indicate that thermal statistics should not be treated as indispensable in the present representation. This finding is especially important because it shifts the interpretation of the method away from a fully tri-modal voltage-current-temperature representation and toward a more robust voltage-current core.

The compactness analysis also changes the methodological emphasis of the work. The best-performing subsets are not necessarily the smallest ones, and the strongest validation RMSE often occurs at 8 to 12 features rather than at the selected compact subset. However, choosing the compact subset through a 10% performance tolerance is still scientifically justified. The aim of this criterion is not to claim that six features are universally optimal, but to demonstrate that most of the predictive value can be retained with a much smaller representation. In this sense, the selected compact subsets are evidence of redundancy in the larger feature space and of a realistic complexity-performance trade-off for deployable diagnostics.

The charge-only results further support this interpretation. Restricting the analysis to charging segments degrades predictive accuracy, especially for RUL, but does not destroy the prognostic signal. The fact that charge-only SoH remains close to the full-cycle baseline, and that compact charge-only subsets still capture the dominant features, suggests that partial-cycle prognosis is feasible when operational constraints make discharge data unavailable. At the same time, the quantitative gap relative to full-cycle performance is large enough that charge-only variants should be framed as practical approximations rather than as equivalent substitutes.

The uncertainty and diagnostics tracks reveal that predictive reliability is strongly stage- and cell-dependent. For SoH, both prediction spread and prediction error worsen as cells approach the aged region, indicating that late-stage degradation is not fully captured by the present single-cycle statistics. For RUL, the largest absolute errors occur early in life, when small differences in degradation trajectory correspond to large differences in remaining cycles. This difference between targets emphasizes that predictive spread and predictive accuracy should not be conflated: a model may be stable across repeated retraining and still be systematically inaccurate in a region where the target itself is difficult to infer from present-state information.

The difficult-cell analysis shows that most of the predictive weakness is concentrated in a limited set of cells rather than distributed uniformly throughout the test population. This concentration is informative. It suggests that the dominant limitation of the current methodology is not the absence of signal in the majority of cells, but rather the inability to accommodate specific atypical trajectories. Some held-out cells appear consistently difficult across both SoH and RUL, implying that these cells are structurally unusual relative to the training population. Such behavior is compatible with variation in protocol response, hidden experimental irregularities, or degradation pathways that are underrepresented in the training data.

The protocol-family robustness results reinforce this interpretation. Generalization across charging families is meaningful but not uniform, and the RUL deterioration in the more aggressive families shows that cross-cell generalization alone is an incomplete measure of robustness. A feature representation that performs well under random cell holdout may still lose fidelity when the operational family changes. This matters for any practical deployment setting in which batteries are exposed to differing charge-rate regimes, because it indicates that feature relevance is partly conditioned by the policy family under which degradation unfolds.

These findings define a more constrained but more defensible scope for the method. The present approach supports the idea of lightweight prognosis from a single diagnostic cycle, especially for SoH and for approximate RUL stratification, but it should not be interpreted as a complete solution for real-world battery management. The study remains limited to a single chemistry, laboratory-controlled cycling, and diagnostic cycles that may not be naturally available in continuous operation. The results therefore support periodic diagnostic assessment under controlled conditions more strongly than they support direct online deployment in electric vehicles or other irregular duty-cycle systems.

Within that scope, however, the main contribution is still significant. The experiments show that a carefully ranked and pruned set of simple cycle statistics can retain substantial prognostic value, that the dominant information is largely carried by voltage-current structure rather than temperature, and that useful charge-only variants are possible even though they remain less accurate than full-cycle models. These conclusions are more valuable than a simple leaderboard result because they clarify what aspects of the signal matter, which parts of the cycle are most informative, and where the present methodology fails.

Although not emphasized in the main narrative, the throughput-based RUL supplementary runs follow the same qualitative patterns as cycle-based RUL: voltage and current features dominate, no-temperature variants remain competitive, and protocol-family robustness is uneven. This consistency suggests that the main conclusions are tied to the feature representation itself rather than to one particular lifetime scale.
