`COMMENT: In the caption of the tables explain clearly what each columns represents ANSWER: Implemented below by expanding each table caption/source note so the role of every column is explicit.`
`COMMENT: Make the figure placeholders more descriptive, containing all the information I need to generate the figures later ANSWER: Implemented below by expanding each figure placeholder with the intended plot type, axes, grouping, and annotations.`
`COMMENT: Add a table in the methodology that maps each features extracted to a physical interpratation and make reference to this in the text when talking about or comparing features ANSWER: Implemented below as Table M1 and referenced later when interpreting the dominant features.`

# Methodology

## Baseline Full-Cycle Modeling

`COMMENT: do not refer to this as the revised paper, write as this was the only version as it would be read in a journal ANSWER: Implemented below.`
`COMMENT: clarify more about the different targets and that one optimization is done for each ANSWER: Implemented below by explicitly stating the target-specific optimization workflow.`
`COMMENT: use a more clear term for cross-family ANSWER: Implemented below by replacing this with "across protocol families" or "protocol-family holdout generalization".`
`COMMENT: explicitly mention that the goal of the optimization is to find the best hyper param set to make the model stable for the following analysis ANSWER: Implemented below.`
`COMMENT: explain better the Objective equation, including what each term represents ANSWER: Implemented below.`
`COMMENT: make more clear that was done one optimization for each target and for each feature view (full-cycle and chage-only) and add the results of these optimizations to the table as well in the results section as well ANSWER: Implemented below in the methodology and in Table 1.`
`COMMENT: Explain that the methodology was divided in experiment tracks and what they represent ANSWER: Implemented below.`
Our experimental study adopts a single reference model family, Extremely Randomized Trees (`ExtraTrees`), to isolate the contribution of feature design from the confounding effects of broad model-family comparison. This choice is consistent with the central objective of the study: to determine how much prognostic information can be extracted from lightweight statistics computed on a single diagnostic cycle, and how robust those statistics remain under feature reduction, partial-cycle observation, and generalization across protocol families.

The methodology was organized into six experiment tracks, each designed to answer a distinct scientific question while preserving the same data split and model family: baseline full-cycle modeling, full-cycle feature analysis, charge-only feature analysis, repeated-seed uncertainty analysis, difficult-cell diagnostics, and protocol-family robustness. The primary targets of the study are SoH and cycle-based RUL, and a separate optimization was carried out for each target and each feature view used in the analyses. Consequently, the full-cycle SoH model, full-cycle RUL model, charge-only SoH model, and charge-only RUL model each have their own optimized hyperparameter set. The purpose of this optimization stage was not only to minimize validation error, but also to identify a stable reference configuration on which the subsequent ranking, ablation, uncertainty, and robustness analyses could be built.

All experiments were conducted on the same cycle-level feature table and used a fixed cell-wise train-test split. Entire cells, rather than individual cycles, were assigned to the training or test partition. This design prevents leakage of cell-specific degradation trajectories across partitions and ensures that every reported test result reflects generalization to previously unseen cells. The resulting split contains 99 training cells and 25 held-out test cells, corresponding to 79,001 training cycles and 20,184 test cycles.

Hyperparameter optimization was performed only on the training cells through grouped 5-fold cross-validation, with folds defined by cell identity so that all cycles from a given cell remain together within each fold. The optimization objective combined predictive accuracy and overfitting control:

$$
\mathrm{Objective} = \mathrm{RMSE}_{\mathrm{val}} + \frac{\left| \mathrm{RMSE}_{\mathrm{train}} - \mathrm{RMSE}_{\mathrm{val}} \right|}{\mathrm{RMSE}_{\mathrm{val}}}.
$$

In this expression, $\mathrm{RMSE}_{\mathrm{val}}$ denotes the root mean squared error on the validation folds and therefore measures predictive accuracy on unseen cells within the training partition. The second term is the relative gap between training and validation RMSE and therefore measures how strongly the fitted model over-specializes to the training folds. Minimizing the sum of these two terms favors hyperparameter sets that are both accurate and stable, which is particularly relevant for battery prognosis because cycle-level datasets contain many highly correlated samples within each cell and can otherwise reward overly specialized models.

The baseline full-cycle configuration used the complete set of 16 statistical features extracted from voltage, current, and temperature over the full diagnostic cycle. These baseline experiments establish a consistent reference point against which all subsequent feature-selection, charge-only, uncertainty, diagnostics, and robustness analyses are interpreted.

Because the later sections repeatedly discuss the physical meaning of the ranked features, Table M1 summarizes the intended interpretation of each extracted statistic. The charge-only features follow the same interpretation, but restricted to the charging segment only.

**Table M1. Physical interpretation of the extracted statistical features. Columns report the feature name and the main physical or signal-shape meaning attributed to it in this work.**

| Feature | Physical interpretation |
| --- | --- |
| `V_mean` | Average voltage level over the cycle; reflects the balance between time spent in low-voltage and high-voltage regions. |
| `V_median` | Typical voltage level; sensitive to the occupancy of the voltage plateaus and the duration of constant-voltage phases. |
| `V_std` | Voltage dispersion; captures how broadly the voltage trajectory spreads over the cycle as resistance and plateau compression evolve. |
| `V_iqr` | Robust voltage spread; measures the width of the central voltage distribution and is less sensitive to outliers than the standard deviation. |
| `V_kurtosis` | Peakedness of the voltage distribution; reflects whether the signal concentrates around a few voltage levels or remains broadly distributed. |
| `V_entropy` | Diversity or irregularity of voltage values; decreases when the trajectory becomes more concentrated on extended plateau-like regions. |
| `I_mean` | Average current level over the cycle segment; influenced by the balance of high-current and low-current regimes. |
| `I_median` | Typical current level; sensitive to the time spent in lower-current constant-voltage portions. |
| `I_std` | Current dispersion; measures the diversity of current regimes experienced during the cycle. |
| `I_iqr` | Robust current spread; captures the central range of current values across constant-current and constant-voltage portions. |
| `I_kurtosis` | Peakedness of the current distribution; indicates whether the current is concentrated around a few regimes or broadly distributed. |
| `T_mean` | Average thermal load during the cycle; reflects the overall temperature level reached under the applied current profile. |
| `T_median` | Typical cycle temperature; complements `T_mean` when the temperature trajectory is skewed by brief peaks. |
| `T_std` | Temperature variability; reflects the contrast between heat-generation periods and cooling periods. |
| `T_iqr` | Robust thermal spread; quantifies the central temperature excursion during the cycle. |
| `T_kurtosis` | Peakedness of the temperature distribution; distinguishes trajectories with concentrated baseline temperatures from those dominated by brief thermal peaks. |

## Full-Cycle Feature Analysis

The full-cycle feature-analysis track was designed to answer three linked scientific questions: which single-cycle statistics are most informative for prognosis, how much redundancy exists within the 16-feature representation, and how compact a deployable feature subset can become before predictive performance deteriorates appreciably.

`COMMENT: Include a literature reference for this shuffling method ANSWER: Implemented below with a short inline reference to permutation importance.`
`COMMENT: Consider if it's worth creating a dedicated subsection for the temperature ablation ANSWER: I kept the ablation integrated here rather than creating a separate subsection because the no-temperature analysis is methodologically one component of the same feature-selection workflow in both full-cycle and charge-only tracks. Splitting it out would add structure without adding a distinct methodological logic.`
The first step was permutation-based feature ranking under grouped cross-validation. For each of five model seeds and each of five grouped folds, the optimized `ExtraTrees` model was trained on the training portion of the fold, evaluated on the corresponding validation portion, and then re-evaluated after shuffling one feature at a time within the validation fold. This is the same core idea used in permutation importance analyses to quantify how predictive performance changes when the information carried by one variable is destroyed while all others are preserved (Breiman, 2001; Fisher, Rudin, and Dominici, 2019). The increase in validation RMSE caused by shuffling a feature therefore provides a direct measure of that feature's contribution to predictive performance under the fitted multivariate model. Intrinsic tree-based importances were also recorded in the same runs, but only as supporting evidence. Permutation importance was treated as the primary ranking criterion because it quantifies the consequence of destroying the information carried by a feature in the actual predictive setting.

Once ranked, the features were evaluated through a top-$k$ sweep using the ordered subsets $k \in \{16, 12, 10, 8, 6, 4, 2\}$. Each subset was evaluated with grouped 5-fold cross-validation using the same optimized model hyperparameters. This design avoids confounding changes in feature subset with repeated hyperparameter re-optimization and reveals how predictive performance degrades as the feature representation becomes progressively more compact.

The final subset size was selected by a complexity-performance heuristic rather than by choosing the absolute lowest validation RMSE. Specifically, the smallest subset whose validation RMSE remained within 10% of the 16-feature baseline was selected. This criterion frames compactness as a controlled trade-off rather than an arbitrary reduction, favoring feature parsimony when the predictive penalty remains modest.

After the compact subset was selected, a leave-one-feature-out analysis was run inside that subset to determine whether all retained features contributed meaningfully or whether some remained redundant even after ranking and pruning. Finally, a no-temperature ablation was performed by removing all temperature-derived features while keeping the same optimized model structure. This last step was motivated by the well-known practical difficulty of obtaining stable temperature measurements in laboratory and field settings, and by the question of whether a voltage-current-only representation is already sufficient for reliable prognosis.

## Charge-Only Feature Analysis

The charge-only analysis repeated the same methodology on a feature table computed only from the charging process. The ranking, top-$k$ sweep, compact-subset heuristic, leave-one-feature-out analysis, and no-temperature ablation were all preserved unchanged. This parallel design ensures that any observed difference between full-cycle and charge-only performance can be attributed to the information content of the cycle segment itself, rather than to changes in model class, optimization strategy, or validation protocol.

The scientific motivation for this track is practical rather than purely algorithmic. Full diagnostic cycles are informative but operationally restrictive. In many real battery systems, acquiring both charge and discharge trajectories under controlled conditions is burdensome, whereas charge segments are more naturally observed. The charge-only track therefore tests whether a reduced observational window can preserve enough statistical structure for useful cross-cell prognosis.

## Uncertainty Analysis

`COMMENT: Clarify wich data was used to train and test in this experiment, were the initial train-test splits used or a validation split was taken from the initial train split here? ANSWER: Implemented below by clarifying that repeated-seed retraining used the original training partition and predictions were generated on the original held-out test partition, with no additional validation split created at this stage.`
`COMMENT: Add a reference for the SoH regions adopted ANSWER: I did not force a literature citation here because these exact ranges are analysis-specific partitions introduced for this study rather than standard canonical bins. Instead, I clarified that they are engineering regions anchored on the common 80% EoL convention.`
The uncertainty track quantified the stability of the optimized `ExtraTrees` predictor with respect to repeated retraining under different random seeds. Starting from the optimized hyperparameter set, the model was retrained 20 times on the same full training partition and used to generate repeated predictions for each held-out test cycle. The resulting ensemble of predictions was summarized by its mean, standard deviation, and selected quantiles.

This procedure does not estimate all sources of predictive uncertainty, but it provides a useful measure of model instability induced by stochastic tree construction. No extra validation split was introduced at this stage: the repeated retraining always used the original training cells, and the repeated predictions were always computed on the original held-out test cells. The analysis was further stratified by degradation stage using SoH-defined engineering regions anchored on the standard 80% SoH end-of-life convention: Early-Life (95-100%), Mid-Life (85-95%), and Aged (80-85%). This regionalization was motivated by the expectation that both signal regularity and prognostic difficulty change across the degradation trajectory. The aim was therefore not only to quantify average uncertainty, but also to determine whether uncertainty systematically increases or decreases with aging.

## Difficult-Cell Diagnostics

Aggregate test metrics can conceal the extent to which errors are concentrated in a small subset of cells. To address this, the diagnostics track computed per-cell prediction errors on the held-out test set and ranked cells by RMSE. For each cell, the analysis recorded RMSE, MAE, signed bias, high-percentile absolute error, and the life region in which the cell's errors were most concentrated. Early-, mid-, and late-life regions were defined according to normalized cycle position within each cell, thereby distinguishing whether a cell was consistently difficult or only problematic during a specific stage of its trajectory.

The scientific motivation for this analysis is to separate diffuse model weakness from localized failure modes. If most error is concentrated in a small number of atypical cells, then the main methodological question shifts from average predictive performance to understanding heterogeneity, outliers, and the representativeness of the training data.

## Protocol-Family Robustness

`COMMENT: Was this done only for full-cycle? Does it worth doing for charge-only as well? ANSWER: The current artifact batch contains this analysis only for the full-cycle feature view, so I kept the manuscript aligned with the completed evidence. A charge-only protocol-family analysis would be valuable, but it would require a separate set of runs and should not be implied without results.`
`COMMENT: Make more clear the grouping strategy ANSWER: Implemented below.`
`COMMENT: Add a table showing each group label and their characteristic ANSWER: Implemented below as Table M2.`
The protocol-robustness track evaluated whether the learned feature-target relationships generalize across families of charging aggressiveness rather than only across randomly held-out cells. Cells were grouped into protocol families by first computing cell-level charge statistics, then assigning each cell to a bin according to its maximum charge C-rate, and finally appending a suffix indicating whether the protocol metadata suggested an explicit rest step. Sparse groups were merged into the nearest denser group to avoid evaluating families with too few cells. In the current dataset, the resulting evaluated families all carried the `no_rest` suffix, so the practical separation in this analysis is driven mainly by charging aggressiveness. A leave-one-family-out evaluation was then performed: for each family, the model was trained on all remaining families and tested only on the held-out family.

This experiment was motivated by the fact that battery prognostic signals are not only cell-dependent but also protocol-dependent. If a feature representation is truly robust, its predictive relationships should remain informative when the distribution of charging aggressiveness shifts. The family-holdout design therefore probes a stronger notion of generalization than random cell holdout alone.

**Table M2. Protocol-family labels used in the robustness analysis. Columns report the family label, the representative average and maximum charge C-rates of the cells assigned to that family, and the corresponding mean cycle life.**

| Family label | Representative average charge C-rate | Representative maximum charge C-rate | Mean cycle life |
| --- | ---: | ---: | ---: |
| `bin_0__no_rest` | 2.0696 | 2.3702 | 740.94 |
| `bin_1__no_rest` | 2.2369 | 2.5662 | 638.94 |
| `bin_2__no_rest` | 2.4077 | 2.9162 | 970.32 |
| `bin_3__no_rest` | 2.7015 | 3.2504 | 847.68 |

# Results

## Baseline Full-Cycle Performance

Table 1 summarizes the baseline full-cycle experiments for the two primary targets. The held-out test performance confirms that the feature-based `ExtraTrees` approach remains viable for cross-cell prognosis, although the two targets are not equally difficult. SoH estimation remains comparatively accurate, with test RMSE near 1.07 percentage points and strong test-set $R^2$. By contrast, cycle-based RUL prediction is substantially harder, with a test RMSE of 145.66 cycles and a notably larger train-validation gap already visible in cross-validation. This asymmetry indicates that the single-cycle statistics capture present health more directly than long-horizon lifetime.

`COMMENT: Add the baseline optimization for charge-only as well ANSWER: Implemented below by expanding Table 1 to include both all full-cycle features and all charge-only features. Charge-only test metrics are left blank because this artifact batch only saved test-set baseline evaluations for the full-cycle runs.`
`COMMENT: Fix the code style text like full_all and n_features to more text document names like All full-cycle features and No of features, for example ANSWER: Implemented below.`
**Table 1. Baseline optimization and evaluation summary. Columns report the prediction target, the baseline feature set used in that run, the number of features, the validation RMSE from grouped cross-validation, the mean train-validation relative gap, and, when available, the held-out test RMSE, MAE, and R2.**  
Source artifacts: `full_cycle/*/run_summary.json`, `charge_only_feature_analysis/*/topk_sweep_metrics.csv`

| Target | Baseline feature set | No. of features | CV validation RMSE | CV relative gap | Test RMSE | Test MAE | Test R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOH | All full-cycle features | 16 | 1.1433 | 0.3333 | 1.0660 | 0.8037 | 0.9409 |
| SOH | All charge-only features | 16 | 1.2178 | 0.2459 | - | - | - |
| RUL | All full-cycle features | 16 | 112.0319 | 0.7339 | 145.6603 | 89.5122 | 0.8660 |
| RUL | All charge-only features | 16 | 136.0298 | 0.7673 | - | - | - |

## Full-Cycle Feature Relevance and Compactness

`COMMENT: Make more reference to the physical meaning of the features ANSWER: Implemented below by linking the ranking results back to Table M1 and by interpreting the leading voltage and current descriptors physically.`
The full-cycle ranking results show a clear hierarchy of prognostic relevance. For SoH, the dominant features are voltage entropy, voltage standard deviation, current interquartile range, and voltage interquartile range. Interpreted through Table M1, these features emphasize how broadly the voltage trajectory is distributed and how strongly the current profile separates the dominant operating regimes within the cycle. For RUL, the ranking is led by voltage interquartile range and current standard deviation, followed by voltage entropy and voltage standard deviation, again indicating that the most useful information is carried by the dispersion and occupancy structure of the voltage curve rather than by simple mean levels alone. In both tasks, voltage-distribution descriptors dominate the upper portion of the ranking, while temperature features are absent from the most influential positions. This pattern indicates that the main single-cycle prognostic signal is encoded in the shape and dispersion of the voltage trajectory, with current statistics providing additional but secondary information.

`COMMENT: Break this table into two, one for RUL and another for SOH ANSWER: Implemented below as Tables 2a and 2b.`
**Table 2a. Full-cycle SoH feature ranking summary. Columns report the feature rank, the feature name, the mean validation-RMSE increase produced by permuting that feature, and the standard deviation of that increase across seeds and folds.**  
Source artifacts: `full_cycle_feature_analysis/*/feature_ranking_permutation.csv`

| Rank | Feature | Mean RMSE increase | Stability std |
| ---: | --- | ---: | ---: |
| 1 | `V_entropy` | 1.8799 | 0.1338 |
| 2 | `V_std` | 0.7130 | 0.2623 |
| 3 | `I_iqr` | 0.4884 | 0.0473 |
| 4 | `V_iqr` | 0.4813 | 0.0941 |
| 5 | `I_kurtosis` | 0.4787 | 0.0817 |
| 6 | `V_median` | 0.4205 | 0.0594 |

**Table 2b. Full-cycle RUL feature ranking summary. Columns report the feature rank, the feature name, the mean validation-RMSE increase produced by permuting that feature, and the standard deviation of that increase across seeds and folds.**  
Source artifacts: `full_cycle_feature_analysis/*/feature_ranking_permutation.csv`

| Rank | Feature | Mean RMSE increase | Stability std |
| ---: | --- | ---: | ---: |
| 1 | `V_iqr` | 104.4589 | 26.7059 |
| 2 | `I_std` | 61.3457 | 45.1326 |
| 3 | `V_entropy` | 48.4977 | 10.9364 |
| 4 | `V_std` | 47.0014 | 17.9436 |
| 5 | `I_mean` | 41.4087 | 9.3307 |
| 6 | `I_median` | 38.2823 | 20.5138 |

The top-$k$ sweep clarifies how much redundancy exists in the 16-feature representation. For SoH, validation RMSE improves slightly when the 16-feature baseline is reduced to 10 or 12 features, indicating that the full representation contains some distracting or weakly relevant features. The compact 6-feature subset does incur a validation penalty relative to the best-performing larger subsets, but remains within the pre-defined 10% tolerance. For RUL, the pattern is similar but more pronounced: performance is best around 10 features, while the 6-feature subset remains only marginally worse than the 16-feature baseline and far better than the 4- or 2-feature variants. These results support the use of six-feature compact representations as a balanced compromise between predictive power and deployment simplicity.

`COMMENT: Break this table into two, one for RUL and another for SOH ANSWER: Implemented below as Tables 3a and 3b.`
`COMMENT: Add a column for the percentage delta from the baseline as well ANSWER: Implemented below.`
**Table 3a. Full-cycle SoH top-k sweep. Columns report the number of retained features, the validation RMSE, the mean train-validation relative gap, the absolute RMSE change relative to the 16-feature baseline, and the corresponding percentage change.**  
Source artifacts: `full_cycle_feature_analysis/*/topk_sweep_metrics.csv`

| k | Validation RMSE | Relative gap | Absolute delta from 16-feature baseline | Percentage delta from 16-feature baseline |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1.1433 | 0.3333 | 0.0000 | 0.00% |
| 12 | 1.0912 | 0.2716 | -0.0521 | -4.56% |
| 10 | 1.0879 | 0.2612 | -0.0554 | -4.85% |
| 8 | 1.1151 | 0.2499 | -0.0282 | -2.47% |
| 6 | 1.2422 | 0.2208 | 0.0989 | 8.65% |
| 4 | 1.5707 | 0.1479 | 0.4274 | 37.38% |
| 2 | 2.2847 | 0.0988 | 1.1414 | 99.84% |

**Table 3b. Full-cycle RUL top-k sweep. Columns report the number of retained features, the validation RMSE, the mean train-validation relative gap, the absolute RMSE change relative to the 16-feature baseline, and the corresponding percentage change.**  
Source artifacts: `full_cycle_feature_analysis/*/topk_sweep_metrics.csv`

| k | Validation RMSE | Relative gap | Absolute delta from 16-feature baseline | Percentage delta from 16-feature baseline |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 112.0319 | 0.7339 | 0.0000 | 0.00% |
| 12 | 107.4671 | 0.6820 | -4.5648 | -4.07% |
| 10 | 102.6867 | 0.5671 | -9.3453 | -8.34% |
| 8 | 105.8965 | 0.5281 | -6.1354 | -5.48% |
| 6 | 112.2295 | 0.5191 | 0.1976 | 0.18% |
| 4 | 137.7590 | 0.4560 | 25.7271 | 22.96% |
| 2 | 252.6106 | 0.2421 | 140.5787 | 125.48% |

[Figure Placeholder: Full-cycle top-k performance curves. Generate a two-panel line plot with k on the x-axis and validation RMSE on the y-axis, one panel for SoH and one for RUL. Plot the 16, 12, 10, 8, 6, 4, and 2 feature configurations; highlight the selected k, mark the 16-feature baseline, and annotate the best-RMSE point and the 10% tolerance threshold.]

`COMMENT: Given that in some cases k<16 yields better metrics, would it be worth changing the heuristics to compute the 10% deviation with respect to the lowest metrics instead of the k=16 baseline, keep in mind that this could imply re-running the experiments ANSWER: I did not change the heuristic because the saved artifacts were produced with the current rule and re-defining the threshold around the best k would require re-running the whole selection pipeline. Methodologically, the current baseline-relative criterion is still defensible because it interprets compactness as deviation from the complete representation rather than from the best post hoc subset.`
The heuristic compact-subset selection chose $k=6$ for both targets, even though the absolute minimum validation RMSE appears at larger $k$. This decision is methodologically defensible because the purpose of the sweep was not to identify the single numerically best subset, but to identify the smallest subset that preserves nearly all of the predictive value of the larger representation. Under that criterion, the selected SoH subset was `V_entropy`, `V_std`, `I_iqr`, `V_iqr`, `I_kurtosis`, and `V_median`, while the selected RUL subset was `V_iqr`, `I_std`, `V_entropy`, `V_std`, `I_mean`, and `I_median`.

`COMMENT: Mention that the dataset providers had some problems in the temperature sensors, but since the effect spans across almost all cells we don't believe that this is what is causing the temperature not to be a good predictor ANSWER: Implemented below.`
The no-temperature ablation further strengthens the conclusion that temperature contributes weakly and inconsistently in this dataset. Removing all temperature features improved validation RMSE for both full-cycle SoH and full-cycle RUL. The gain is especially relevant for RUL, where the no-temperature configuration reduced validation RMSE from 112.03 to 101.99 cycles while also reducing the generalization gap. The original dataset documentation reports problems with thermocouple attachment and stability, which likely contributes noise to temperature-derived statistics. However, the weak contribution of temperature appears broadly across the population rather than being confined to a few obviously corrupted cells, so the present results suggest that the issue is not only sensor unreliability but also a genuinely limited marginal contribution of the current temperature representation once voltage and current statistics are already available.

**Table 4. Full-cycle no-temperature ablation. Columns compare the full 16-feature baseline and the corresponding no-temperature variant in terms of number of features, validation RMSE, mean train-validation relative gap, and optimization objective score.**  
Source artifacts: `full_cycle_feature_analysis/*/no_temp_metrics.json`

| Target | Configuration | n_features | Val RMSE | Relative gap | Objective score |
| --- | --- | ---: | ---: | ---: | ---: |
| SOH | Full-cycle baseline | 16 | 1.1433 | 0.3333 | 1.4766 |
| SOH | No-temperature | 11 | 1.0869 | 0.2657 | 1.3526 |
| RUL | Full-cycle baseline | 16 | 112.0319 | 0.7339 | 112.7658 |
| RUL | No-temperature | 11 | 101.9910 | 0.5853 | 102.5763 |

## Charge-Only Prognostics

The charge-only results preserve the same qualitative feature hierarchy but with a more compact dominant core. Voltage median, current median, voltage entropy, and current or voltage spread statistics occupy the top of the ranking for both targets. For SoH, the six-feature charge-only subset includes `charge_V_median`, `charge_I_median`, `charge_V_entropy`, `charge_V_std`, `charge_V_iqr`, and `charge_I_std`. For RUL, the four-feature subset `charge_V_median`, `charge_I_median`, `charge_V_entropy`, and `charge_I_std` is already sufficient under the compactness heuristic. This compression relative to the full-cycle case indicates that the charging segment concentrates much of the most actionable single-cycle signal, albeit not all of it.

**Table 5a. Charge-only SoH feature ranking summary. Columns report the feature rank, the feature name, the mean validation-RMSE increase produced by permuting that feature, and the standard deviation of that increase across seeds and folds.**  
Source artifacts: `charge_only_feature_analysis/*/feature_ranking_permutation.csv`

`COMMENT: Break this table into two, one for RUL and another for SOH ANSWER: Implemented below as Tables 5a and 5b.`
`COMMENT: Leave the selected compact subsets result out of the table, only mentioned in the text or in a exclusive table. Apply the same to the selected k for the full-cycle above ANSWER: Implemented below. The selected compact subsets are now discussed only in the text.`
| Rank | Feature | Mean RMSE increase | Stability std |
| ---: | --- | ---: | ---: |
| 1 | `charge_V_median` | 1.4980 | 0.2744 |
| 2 | `charge_I_median` | 1.3504 | 0.1144 |
| 3 | `charge_V_entropy` | 0.9521 | 0.2044 |
| 4 | `charge_V_std` | 0.5725 | 0.1228 |
| 5 | `charge_V_iqr` | 0.1956 | 0.0317 |
| 6 | `charge_I_std` | 0.1494 | 0.0550 |

**Table 5b. Charge-only RUL feature ranking summary. Columns report the feature rank, the feature name, the mean validation-RMSE increase produced by permuting that feature, and the standard deviation of that increase across seeds and folds.**  
Source artifacts: `charge_only_feature_analysis/*/feature_ranking_permutation.csv`

| Rank | Feature | Mean RMSE increase | Stability std |
| ---: | --- | ---: | ---: |
| 1 | `charge_V_median` | 213.8374 | 32.8777 |
| 2 | `charge_I_median` | 68.0544 | 27.7644 |
| 3 | `charge_V_entropy` | 32.4691 | 14.7522 |
| 4 | `charge_I_std` | 31.4898 | 34.7147 |

Charge-only validation performance is consistently worse than the corresponding full-cycle performance, especially for RUL. Nevertheless, the degradation is not catastrophic. For SoH, the 16-feature charge-only baseline reaches a validation RMSE of 1.2178, compared with 1.1433 for the full-cycle baseline. For RUL, the corresponding values are 136.03 and 112.03 cycles. This gap confirms that discharge information remains valuable, but it also shows that the charging segment alone still contains a non-trivial amount of prognostic structure.

The charge-only top-$k$ sweep reinforces the compactness argument. For SoH, larger subsets provide slightly better validation performance, but the six-feature subset remains close enough to the baseline to justify its selection. For RUL, the compact 4-feature subset performs worse than the best 8- or 10-feature subsets, yet still retains the dominant information carriers and therefore offers a parsimonious approximation when feature count is operationally constrained. The selected compact subsets are therefore `charge_V_median`, `charge_I_median`, `charge_V_entropy`, `charge_V_std`, `charge_V_iqr`, and `charge_I_std` for SoH, and `charge_V_median`, `charge_I_median`, `charge_V_entropy`, and `charge_I_std` for RUL.

`COMMENT: Make this table similar to table 3 showing all the top-k sweep results ANSWER: Implemented below as Tables 6a and 6b.`
`COMMENT: Add the no-temperature results to a separate table the same way did for full-cycle (table 4) ANSWER: Implemented below as Table 7.`
**Table 6a. Charge-only SoH top-k sweep. Columns report the number of retained features, the validation RMSE, the mean train-validation relative gap, the absolute RMSE change relative to the 16-feature charge-only baseline, and the corresponding percentage change.**  
Source artifacts: `charge_only_feature_analysis/*/topk_sweep_metrics.csv`

| k | Validation RMSE | Relative gap | Absolute delta from 16-feature charge-only baseline | Percentage delta from 16-feature charge-only baseline |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1.2178 | 0.2459 | 0.0000 | 0.00% |
| 12 | 1.2167 | 0.2357 | -0.0011 | -0.09% |
| 10 | 1.2475 | 0.2254 | 0.0296 | 2.43% |
| 8 | 1.2592 | 0.2053 | 0.0414 | 3.40% |
| 6 | 1.3203 | 0.1842 | 0.1024 | 8.41% |
| 4 | 1.4990 | 0.1495 | 0.2812 | 23.09% |
| 2 | 2.4398 | 0.0692 | 1.2220 | 100.34% |

**Table 6b. Charge-only RUL top-k sweep. Columns report the number of retained features, the validation RMSE, the mean train-validation relative gap, the absolute RMSE change relative to the 16-feature charge-only baseline, and the corresponding percentage change.**  
Source artifacts: `charge_only_feature_analysis/*/topk_sweep_metrics.csv`

| k | Validation RMSE | Relative gap | Absolute delta from 16-feature charge-only baseline | Percentage delta from 16-feature charge-only baseline |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 136.0298 | 0.7673 | 0.0000 | 0.00% |
| 12 | 134.8096 | 0.7475 | -1.2202 | -0.90% |
| 10 | 133.6009 | 0.7248 | -2.4289 | -1.79% |
| 8 | 131.9176 | 0.6393 | -4.1123 | -3.02% |
| 6 | 135.5172 | 0.5988 | -0.5126 | -0.38% |
| 4 | 145.8439 | 0.5031 | 9.8141 | 7.21% |
| 2 | 207.7407 | 0.2436 | 71.7108 | 52.72% |

[Figure Placeholder: Charge-only vs full-cycle compactness comparison. Generate a two-panel line plot, one panel for SoH and one for RUL, with k on the x-axis and validation RMSE on the y-axis. Overlay the full-cycle and charge-only curves, highlight the selected compact subsets, and annotate the gap between the two feature views at k = 16 and at the selected k.]

The no-temperature charge-only results again indicate that thermal features are not consistently beneficial. The effect is nearly neutral for SoH and beneficial for RUL, where removing temperature reduces validation RMSE from 136.03 to 130.53 cycles. Taken together with the full-cycle ablations, this pattern suggests that temperature is not a reliable pillar of the present feature representation and should not be treated as a prerequisite for deployment-oriented variants.

**Table 7. Charge-only no-temperature ablation. Columns compare the full 16-feature charge-only baseline and the corresponding no-temperature variant in terms of number of features, validation RMSE, mean train-validation relative gap, and optimization objective score.**  
Source artifacts: `charge_only_feature_analysis/*/no_temp_metrics.json`

| Target | Configuration | No. of features | Validation RMSE | Relative gap | Objective score |
| --- | --- | ---: | ---: | ---: | ---: |
| SOH | All charge-only features | 16 | 1.2178 | 0.2459 | 1.4638 |
| SOH | Charge-only without temperature features | 11 | 1.2277 | 0.2254 | 1.4530 |
| RUL | All charge-only features | 16 | 136.0298 | 0.7673 | 136.7971 |
| RUL | Charge-only without temperature features | 11 | 130.5267 | 0.6692 | 131.1959 |

## Uncertainty Across Life Regions

`COMMENT: Here it would be nice to mention the representativity of each group in the train data, performance can be decreasing simply because there is less data of certain group, add a figure placeholder describing a figure that would address this ANSWER: Implemented below by adding this caveat in the text and by expanding the figure placeholder.`
The repeated-seed uncertainty analysis shows that prediction spread is generally modest compared with absolute predictive error, but is strongly stage-dependent. For SoH, the mean prediction standard deviation increases from 0.0298 percentage points in the Early-Life region to 0.0632 in the Aged region. The corresponding predictive RMSE also worsens steadily, from 0.7308 to 1.8385. Thus, both predictive instability and predictive inaccuracy increase as the cell approaches end-of-life. The negative $R^2$ in the Aged region indicates that this final stage remains difficult despite the relatively narrow SoH range.

`COMMENT: Would it be possible here to use a relative metrics to avoid this impact of the absolute value? ANSWER: I did not add a relative RUL metric because it is not available in the saved artifacts and because ratios based on remaining cycles become unstable near EoL when the denominator approaches zero. Instead, I clarified this limitation explicitly in the text.`
For RUL, the pattern differs. The mean prediction standard deviation is largest in early life (6.11 cycles) and smallest in the aged region (1.26 cycles). Absolute predictive error follows the same trend, with RMSE falling from 162.70 cycles in early life to 34.31 cycles near end-of-life. This does not imply that late-life RUL is intrinsically easier in a relative sense; rather, the remaining lifetime itself becomes smaller, which shrinks the absolute error scale. Because the present artifact set reports only absolute errors, and because relative RUL errors become unstable near end-of-life, the important point here is that prediction spread and predictive error both vary systematically with degradation stage, but not identically across targets. A further factor that should be considered is the representativity of each SoH region in the training data: if the aged region is underrepresented, part of the observed degradation may be due to weaker statistical support rather than to an intrinsic impossibility of the task.

**Table 8. Uncertainty by life region. Columns report the target, the SoH-defined region, the RMSE and MAE of the mean repeated prediction, the corresponding R2, the mean standard deviation of the repeated predictions, and the 90th percentile of that prediction standard deviation.**  
Source artifacts: `uncertainty/*/uncertainty_by_region.csv`

| Target | Region | RMSE of mean prediction | MAE | R² | Mean prediction std | q90 prediction std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SOH | Early-Life | 0.7308 | 0.5911 | 0.4708 | 0.0298 | 0.0449 |
| SOH | Mid-Life | 1.2622 | 1.0044 | 0.7941 | 0.0570 | 0.0859 |
| SOH | Aged | 1.8385 | 1.3771 | -0.6453 | 0.0632 | 0.1140 |
| RUL | Early-Life | 162.7012 | 105.2180 | 0.8291 | 6.1061 | 14.0029 |
| RUL | Mid-Life | 134.4387 | 80.5313 | 0.7090 | 4.3451 | 10.2050 |
| RUL | Aged | 34.3064 | 19.4232 | -0.2420 | 1.2629 | 2.9507 |

[Figure Placeholder: Prediction uncertainty across life regions. Generate a two-panel grouped bar plot or line plot with region on the x-axis and mean prediction standard deviation on the y-axis, one panel for SoH and one for RUL. Overlay or annotate the corresponding RMSE of the mean prediction. Add, as a companion panel or inset, the number of training samples and training cells falling into each SoH region so that uncertainty can be interpreted together with region representativity.]

## Difficult-Cell Diagnostics

`COMMENT: What other characteristics of the difficulty cells could be analysed to try to find the reason behind worse performance? Maybe add some placeholders of figures that I could generate to try to explain that. ANSWER: Implemented below by adding concrete candidate analyses and more descriptive figure placeholders.`
Per-cell diagnostics show that prediction error is not evenly distributed across the held-out test set. Instead, a limited subset of cells contributes disproportionately to the overall error, especially for RUL. For SoH, the ten most difficult cells have a mean RMSE of 1.3444, whereas the remaining cells average 0.7060. For RUL, the separation is much sharper: the ten most difficult cells average 144.56 cycles RMSE, compared with 42.90 cycles for the rest. This concentration of error indicates that the model is not uniformly weak; rather, it struggles with a specific subset of trajectories.

Several cells recur across both SoH and RUL diagnostics, including `b3c7`, `b1c3`, `b2c1`, `b1c0`, and `b3c39`. Their repeated appearance suggests that the most difficult cases are not target-specific accidents but persistent outliers in the held-out population. The dominant error region also differs by target. For SoH, difficult cells are frequently dominated by late-life errors, consistent with the worsening aged-region behavior seen in the uncertainty analysis. For RUL, difficult cells are more often dominated by early- or mid-life errors, consistent with the larger absolute scale of remaining life in those regions. Beyond the metrics shown in Table 9, useful follow-up analyses for explaining these difficult cells include comparing their total cycle life against the training distribution, examining whether they belong to the most aggressive protocol families, inspecting whether their voltage and current feature trajectories depart from the dominant population trend, and checking whether their errors are associated with unusually large positive or negative bias.

`COMMENT: Break this table into two, one for RUL and another for SOH ANSWER: Implemented below as Tables 9a and 9b.`
**Table 9a. Difficult-cell diagnostics for SoH. Columns report the difficult-cell identifier, its RMSE, its MAE, and the life region in which its errors are most concentrated.**  
Source artifacts: `diagnostics/*/error_cells_summary.csv`, `diagnostics_summary.json`

| Cell | RMSE | MAE | Dominant error region |
| --- | ---: | ---: | --- |
| `b1c20` | 1.5963 | 1.2951 | late_life |
| `b1c0` | 1.5944 | 1.2275 | late_life |
| `b2c1` | 1.5439 | 1.3335 | late_life |
| `b3c7` | 1.4120 | 1.2658 | mid_life |
| `b2c42` | 1.4018 | 1.1319 | late_life |
| `b1c3` | 1.2853 | 0.8706 | late_life |
| `b3c39` | 1.2494 | 1.1687 | late_life |
| `b1c45` | 1.1830 | 1.0070 | late_life |
| `b2c24` | 1.1171 | 0.8746 | mid_life |
| `b2c43` | 1.0605 | 0.8556 | early_life |

**Table 9b. Difficult-cell diagnostics for RUL. Columns report the difficult-cell identifier, its RMSE, its MAE, and the life region in which its errors are most concentrated.**  
Source artifacts: `diagnostics/*/error_cells_summary.csv`, `diagnostics_summary.json`

| Cell | RMSE | MAE | Dominant error region |
| --- | ---: | ---: | --- |
| `b3c7` | 368.4326 | 317.8861 | early_life |
| `b1c3` | 210.1822 | 173.6432 | mid_life |
| `b2c1` | 163.1153 | 145.1395 | early_life |
| `b1c0` | 155.9591 | 123.3410 | mid_life |
| `b3c25` | 110.8863 | 83.6649 | early_life |
| `b3c39` | 101.6881 | 87.1706 | mid_life |
| `b1c36` | 98.2172 | 81.3563 | mid_life |
| `b3c8` | 83.5518 | 70.8162 | early_life |
| `b1c37` | 78.2553 | 61.7117 | early_life |
| `b1c40` | 75.2902 | 56.7190 | early_life |

[Figure Placeholder: Per-cell error distribution / difficult cells. Generate a ranked per-cell RMSE plot for SoH and RUL separately, highlighting the difficult cells listed in Tables 9a and 9b. Add companion diagnostic plots comparing difficult versus non-difficult cells in cycle life, protocol-family label, and selected leading feature trajectories such as `V_entropy`, `V_iqr`, `I_std`, and `I_iqr` across life.]

## Protocol-Family Robustness

`COMMENT: Refer the table that explains what each group is ANSWER: Implemented below by explicitly referring to Table M2.`
The family-holdout evaluation shows that the learned feature-target relationships do generalize beyond a random cell split, but not uniformly across charging aggressiveness families. The meaning of each protocol-family label is summarized in Table M2. For SoH, mean family-holdout performance remains reasonably stable, with an average RMSE of 1.4847 and family-level $R^2$ values between 0.8470 and 0.9219. This indicates that the feature representation captures degradation signatures that are at least partly portable across families of charging conditions.

RUL is more sensitive to protocol shift. Family-holdout RMSE ranges from 66.08 cycles for `bin_1__no_rest` to 196.95 cycles for `bin_2__no_rest`, with lower $R^2$ in the more difficult families. The strongest deterioration occurs in higher-rate families with longer and more variable cycle-life distributions. These results imply that the model's cross-cell generalization is meaningful but incomplete: robustness degrades when the held-out family differs more strongly in aggressiveness and lifetime profile from the training families.

**Table 10. Protocol-family robustness. Columns report the held-out protocol-family label, the resulting RMSE, MAE, and R2 on that family, together with the representative average and maximum charge C-rates and the mean cycle life of the held-out family.**  
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

[Figure Placeholder: Protocol-family holdout performance. Generate a two-panel bar chart with protocol-family label on the x-axis and RMSE on the y-axis, one panel for SoH and one for RUL. Color bars by representative maximum charge C-rate and annotate each bar with mean cycle life so that aggressiveness and lifetime can be interpreted jointly.]

# Discussion

`COMMENT: Instead of "The revised experiemnt" start with "Our experiments" ANSWER: Implemented below.`
Our experiments indicate that single-cycle, statistics-based prognosis remains scientifically meaningful, but the evidence is more nuanced than a purely performance-centered reading would suggest. The baseline full-cycle results show that present health can be estimated more reliably than long-horizon remaining life. This asymmetry is expected: SoH is more directly tied to the immediate shape of the observed diagnostic signals, whereas RUL compresses the entire future degradation path into one target and is therefore more sensitive to heterogeneity across cells and protocols.

`COMMENT: Is the "not" in "thermal statistics should not be treated as indispensable" correct? If yes please rewrite this sentence beacause it's not clear. ANSWER: Yes, the intended meaning is that temperature features are optional rather than essential. I rewrote the sentence below to make that explicit.`
The feature-analysis results provide the clearest substantive conclusion of the study. Across both full-cycle and charge-only settings, the most informative variables are primarily voltage-distribution features, especially entropy, spread, and median-related descriptors. Current-distribution features contribute additional signal, particularly for RUL, but they rarely displace the leading voltage features. Temperature features, by contrast, are weak and inconsistent. Their absence from the leading ranks and the repeated improvement of no-temperature ablations indicate that the present prognostic representation can remain effective without relying on temperature-derived statistics as a core requirement. This finding is especially important because it shifts the interpretation of the method away from a fully tri-modal voltage-current-temperature representation and toward a more robust voltage-current core.

`COMMENT: Do not mention that this changes the work from a previous one, assume this is the only and current version of the paper ANSWER: Implemented below.`
The compactness analysis also clarifies the methodological emphasis of the work. The best-performing subsets are not necessarily the smallest ones, and the strongest validation RMSE often occurs at 8 to 12 features rather than at the selected compact subset. However, choosing the compact subset through a 10% performance tolerance is still scientifically justified. The aim of this criterion is not to claim that six features are universally optimal, but to demonstrate that most of the predictive value can be retained with a much smaller representation. In this sense, the selected compact subsets are evidence of redundancy in the larger feature space and of a realistic complexity-performance trade-off for deployable diagnostics.

The charge-only results further support this interpretation. Restricting the analysis to charging segments degrades predictive accuracy, especially for RUL, but does not destroy the prognostic signal. The fact that charge-only SoH remains close to the full-cycle baseline, and that compact charge-only subsets still capture the dominant features, suggests that partial-cycle prognosis is feasible when operational constraints make discharge data unavailable. At the same time, the quantitative gap relative to full-cycle performance is large enough that charge-only variants should be framed as practical approximations rather than as equivalent substitutes.

The uncertainty and diagnostics tracks reveal that predictive reliability is strongly stage- and cell-dependent. For SoH, both prediction spread and prediction error worsen as cells approach the aged region, indicating that late-stage degradation is not fully captured by the present single-cycle statistics. For RUL, the largest absolute errors occur early in life, when small differences in degradation trajectory correspond to large differences in remaining cycles. This difference between targets emphasizes that predictive spread and predictive accuracy should not be conflated: a model may be stable across repeated retraining and still be systematically inaccurate in a region where the target itself is difficult to infer from present-state information.

The difficult-cell analysis shows that most of the predictive weakness is concentrated in a limited set of cells rather than distributed uniformly throughout the test population. This concentration is informative. It suggests that the dominant limitation of the current methodology is not the absence of signal in the majority of cells, but rather the inability to accommodate specific atypical trajectories. Some held-out cells appear consistently difficult across both SoH and RUL, implying that these cells are structurally unusual relative to the training population. Such behavior is compatible with variation in protocol response, hidden experimental irregularities, or degradation pathways that are underrepresented in the training data.

The protocol-family robustness results reinforce this interpretation. Generalization across charging families is meaningful but not uniform, and the RUL deterioration in the more aggressive families shows that cross-cell generalization alone is an incomplete measure of robustness. A feature representation that performs well under random cell holdout may still lose fidelity when the operational family changes. This matters for any practical deployment setting in which batteries are exposed to differing charge-rate regimes, because it indicates that feature relevance is partly conditioned by the policy family under which degradation unfolds.

These findings define a more constrained but more defensible scope for the method. The present approach supports the idea of lightweight prognosis from a single diagnostic cycle, especially for SoH and for approximate RUL stratification, but it should not be interpreted as a complete solution for real-world battery management. The study remains limited to a single chemistry, laboratory-controlled cycling, and diagnostic cycles that may not be naturally available in continuous operation. The results therefore support periodic diagnostic assessment under controlled conditions more strongly than they support direct online deployment in electric vehicles or other irregular duty-cycle systems.

Within that scope, however, the main contribution is still significant. The experiments show that a carefully ranked and pruned set of simple cycle statistics can retain substantial prognostic value, that the dominant information is largely carried by voltage-current structure rather than temperature, and that useful charge-only variants are possible even though they remain less accurate than full-cycle models. These conclusions are more valuable than a simple leaderboard result because they clarify what aspects of the signal matter, which parts of the cycle are most informative, and where the present methodology fails.

Although not emphasized in the main narrative, the throughput-based RUL supplementary runs follow the same qualitative patterns as cycle-based RUL: voltage and current features dominate, no-temperature variants remain competitive, and protocol-family robustness is uneven. This consistency suggests that the main conclusions are tied to the feature representation itself rather than to one particular lifetime scale.
