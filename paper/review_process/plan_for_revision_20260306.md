# Revision Plan

This document summarizes the experiment plan for the manuscript revision after consolidating the reviewer feedback and the revised paper scope.

The central structural change of the paper is a shift:
- from a manuscript centered mainly on model comparison and predictive performance;
- to a manuscript centered mainly on single-cycle feature relevance, interpretability, robustness and practical applicability.

This change makes sense given the reviewer feedback because the main unresolved points are not about ranking many regressors, but about:
- which features are truly informative;
- how compact the feature representation can become;
- how sensitive the method is to temperature information;
- whether the predictions can be accompanied by uncertainty estimates;
- and whether the method remains useful when only the charge process is available.

Under this revised framing, the model is no longer the main contribution of the paper. Instead, the model becomes an analysis instrument used to study the feature set. For that reason, the plan below adopts one main reference model, `ExtraTrees`, and organizes the experiments around feature analysis, uncertainty, reduced diagnostic information and robustness checks rather than broad model benchmarking.

The plan is organized into:
- the main full-cycle feature-analysis track;
- the charge-only comparison track;
- the uncertainty track;
- targeted diagnostic and robustness extensions;
- and a final mapping between reviewer comments and the planned experimental evidence.
Below these tracks are detailed.

### 1. Full-cycle feature analysis track

1. Once features are ready, split the cells into train cells and held-out test cells.

2. Using only the train cells, optimize an ExtraTrees with all full-cycle features from voltage, current and temperature signals. The optimization should be done with grouped 5-fold per cell, averaging the RMSE of the 5 validation folds and informing the TPE sampler. Save the selected hyperparameters.

3. Using the selected hyperparameters from step 2 and still only the train cells, perform grouped 5-fold per cell again to create the feature ranking and quantify its stability:
   - repeat the following procedure for each chosen model random seed;
   - fit the model on the training folds;
   - predict the corresponding validation fold and compute the baseline RMSE;
   - shuffle one feature at a time only in the validation fold;
   - predict again and compute the new RMSE;
   - compute the RMSE increase caused by shuffling that feature;
   - store the RMSE increase of each feature for that `(seed, fold)` pair;
   - also store the model intrinsic importance of each feature for that `(seed, fold)` pair only as supporting material;
   - repeat this for all features and all 5 folds.
   After all seeds and folds are executed, aggregate all stored observations of each feature across every `(seed, fold)` pair and compute the final mean and standard deviation of:
   - the RMSE increase caused by shuffling the feature;
   - the intrinsic importance of the feature.
   Use the final mean RMSE increase to define the permutation feature ranking and the final standard deviation to quantify stability. Make the same rank with respect to the intrinsic importance just for reference and discussion.

4. Build subgroups of top-12, 8, 6, 4 and 2 features following the permutation ranking. For each subgroup:
   - keep the same hyperparameters selected at step 2;
   - run grouped 5-fold per cell using only the train cells;
   - average the RMSE of the 5 validation folds to be the subgroup score.

5. Plot the complexity (k) vs. performance (subgroup score) curve including the full 16-feature configuration and choose the best k value balancing complexity and performance.

6. For each feature from the selected top-k subset:
   - remove that one feature from the selected top-k subset;
   - keep the same hyperparameters selected at step 2;
   - run grouped 5-fold per cell on the train cells only;
   - average the RMSE of the 5 validation folds to be the score of leaving that feature out.
   Use this to quantify the individual relevance of each feature within the selected compact subset.

7. Remove all the temperature features from the full-cycle feature set and:
   - keep the same hyperparameters selected at step 2;
   - run grouped 5-fold per cell on the train cells only;
   - average the RMSE of the 5 validation folds to be the score of the model without temperature features.
   Use this score to compare the no-temperature variant against the full-feature baseline before the final held-out test evaluation and decide whether a temperature-free practical variant remains competitive enough to keep as a final reported configuration.

8. Retrain the final selected full-cycle configurations on the whole train set and evaluate once on the held-out test set. The final selected configurations should include at least:
   - the full 16-feature baseline;
   - the selected top-k subset;
   - the no-temperature variant.

### 2. Charge-only feature analysis track

1. Using the same train/test cell split defined in the full-cycle track, recompute all features using only the charge process part of the voltage, current and temperature signals, including the complete charging event and any small internal rest between charge steps, but excluding discharge and the post-charge rest before discharge.

2. Using only the train cells, optimize an ExtraTrees with all charge-only features. The optimization should be done with grouped 5-fold per cell, averaging the RMSE of the 5 validation folds and informing the TPE sampler. Save the selected charge-only hyperparameters.

3. Using the selected charge-only hyperparameters from step 2 and still only the train cells, perform grouped 5-fold per cell again to create the charge-only feature ranking and quantify its stability:
   - repeat the following procedure for each chosen model random seed;
   - fit the model on the training folds;
   - predict the corresponding validation fold and compute the baseline RMSE;
   - shuffle one feature at a time only in the validation fold;
   - predict again and compute the new RMSE;
   - compute the RMSE increase caused by shuffling that feature;
   - store the RMSE increase of each charge-only feature for that `(seed, fold)` pair;
   - also store the model intrinsic importance of each charge-only feature for that `(seed, fold)` pair only as supporting material;
   - repeat this for all charge-only features and all 5 folds.
   After all seeds and folds are executed, aggregate all stored observations of each charge-only feature across every `(seed, fold)` pair and compute the final mean and standard deviation of:
   - the RMSE increase caused by shuffling the feature;
   - the intrinsic importance of the feature.
   Use the final mean RMSE increase to define the charge-only permutation ranking and the final standard deviation to quantify stability.

4. Build subgroups of top-12, 8, 6, 4 and 2 charge-only features following the charge-only permutation ranking. For each subgroup:
   - keep the same hyperparameters selected at step 2;
   - run grouped 5-fold per cell using only the train cells;
   - average the RMSE of the 5 validation folds to be the subgroup score.

5. Plot the charge-only complexity (k) vs. performance curve and compare it against the full-cycle curve.

6. Remove all the charge-only temperature features from the charge-only feature set and:
   - keep the same hyperparameters selected at step 2;
   - run grouped 5-fold per cell on the train cells only;
   - average the RMSE of the 5 validation folds to be the score of the charge-only model without temperature features.
   Use this score to compare the no-temperature variant against the full charge-only baseline before the final held-out test evaluation and decide whether a temperature-free charge-only variant remains competitive enough to keep as a final reported configuration.

7. Retrain the final selected charge-only configurations on the whole train set and evaluate once on the held-out test set. The final selected configurations should include at least:
   - the full charge-only baseline;
   - the selected charge-only top-k subset;
   - the charge-only no-temperature variant, if temperature is still present in the selected feature space.

### 3. Uncertainty track

1. For each final selected configuration, retrain the same ExtraTrees model multiple times on the whole train set using the same feature set, the same train/test split and the same optimized hyperparameters, changing only the random seed at each repetition.

2. For each repetition:
   - fit the model on the whole train set;
   - predict the same held-out test samples.
   Aggregate the repeated predictions of each sample into:
   - prediction mean;
   - prediction standard deviation;
   - percentile interval when useful.

3. Run this repeated-seed uncertainty analysis for at least:
   - the full-cycle 16-feature baseline;
   - the full-cycle selected top-k subset;
   - the full-cycle no-temperature variant;
   - the charge-only baseline.

4. Aggregate the uncertainty results overall and also analyze them in more difficult regions such as near end-of-life and on long-life cells.

### 4. Targeted diagnostic analysis track

1. Using the final held-out test predictions from the selected full-cycle and charge-only configurations, identify the cells with the largest prediction deviations for SoH and RUL, especially the longest-life cells.

2. For each difficult cell, analyze:
   - their cycle life;
   - their charge protocol metadata;
   - their temperature behavior;
   - whether the model error is concentrated in specific life regions.

3. Compare the difficult cells with the rest of the test set to understand whether the largest errors are associated with protocol distribution, lifespan imbalance or feature behavior not well represented in training.

4. Use this diagnostic analysis to support the manuscript discussion on long-life cells, generalization limits and the practical meaning of the prediction errors.

5. Add a protocol-family robustness experiment based on the average charging C-rate of each cell, computed over the charge process only and excluding the zero-current internal rest samples:
   - compute the average charging C-rate of each cell;
   - bin the cells into a small number of protocol families according to this average C-rate;
   - for this protocol-family experiment, use all cells of the dataset rather than the fixed train/test split adopted in the main tracks;
   - using the final selected full-cycle configuration and fixed hyperparameters, hold out one protocol family at a time;
   - train on the cells from the remaining protocol families;
   - evaluate on the cells from the held-out protocol family;
   - keep all cycles from each cell together in either train or evaluation within each run.
   Use this to test whether the final model generalizes similarly across charge-aggressiveness families.

6. Repeat the same protocol-family robustness experiment for the final selected charge-only configuration using the same protocol-family labels.

### 5. Complementary analytical material for the manuscript

1. Compute the feature-feature correlation matrix for the full-cycle features and use it only as analysis material to interpret redundancy, not as a preprocessing step.

2. Repeat the same correlation analysis for the charge-only features when useful for interpreting changes in the ranking or compact subsets.

3. Use the final selected features and the ablation results to build the manuscript discussion linking the main statistical features to expected degradation mechanisms such as impedance rise, loss of available capacity and changes in voltage distribution during aging.

### 6. Reviewer-point coverage with the planned experiments

#### Reviewer 1

1. **Q5**
   Reviewer comment: "Authors should explain how the 4 features resulted in predictable outcomes compared to the 16. Explain in detail, along with the physics behind it."
   Coverage: addressed by the full-cycle feature analysis track.
   Planned evidence:
   - permutation feature ranking;
   - top-k complexity vs. performance curve;
   - leave-one-feature-out inside the selected compact subset;
   - correlation analysis and physical discussion of the selected features.
   Expected response path:
   - the revised paper will no longer defend an arbitrary 4-feature choice;
   - it will instead show how predictive performance evolves as the feature set is reduced and explain the final compact subset based on ranking, redundancy and physical interpretation.

2. **Q8**
   Reviewer comment: "How to find the cycle errors in SoH and RUL prediction? Confirm whether the error predictions are significant for electric vehicle applications. Explain in detail."
   Coverage: partially addressed by the planned experiments.
   Planned evidence:
   - final held-out test predictions for the selected configurations;
   - targeted diagnostic analysis of difficult cells and long-life cells;
   - uncertainty analysis on final configurations.
   Justification path for the remaining part:
   - the experiments will strengthen the error characterization, but the practical significance for EV applications still requires discussion rather than a new dataset or field validation;
   - the response should explain that the present study remains a laboratory-dataset study and that EV-level significance must be interpreted with that scope limitation in mind.
   - What can be done is to discuss the relevance of the achieved error to the industry, whether is a sufficient value or if it still a high error.

#### Reviewer 2

1. **Q3**
   Reviewer comment: "In the data preprocessing stage, the Savitzky-Golay filter was used to smooth the features. Could you please explain how the 10-cycle in the filter window size were determined? Have you analyzed the impact of different window sizes on the model performance?"
   Coverage: not addressed by new experiments on purpose.
   Justification path:
   - the revised paper scope removes smoothing from the core methodology;
   - previous tests indicated negligible impact for the selected ExtraTrees model;
   - because smoothing was not part of the intended inference-time workflow and did not materially change the tree-based results, it was removed rather than expanded into a larger sensitivity study;
   - the manuscript and reviewer response should state that the final method uses raw cycle-level statistical features without Savitzky-Golay smoothing.

2. **Q4**
   Reviewer comment: "Figure 15-18 shows that the prediction deviations for certain test cells, especially those with long lifespans, are relatively large. Could you please explain whether the characteristics of these abnormal cells, such as charging protocols and temperature anomalies, have been analyzed? Have you considered developing specialized modeling for such cells?"
   Coverage: addressed by the targeted diagnostic analysis track.
   Planned evidence:
   - identification of the largest-error cells on the held-out test set;
   - comparison of their cycle life, charge protocol metadata, temperature behavior and error concentration along life;
   - comparison against the rest of the test population.
   Justification path for specialized modeling:
   - the revised scope prioritizes feature-centered analysis over adding a separate specialized-model branch;
   - therefore the paper will diagnose these cases and discuss the limitation rather than introduce a new tailored model family.

3. **Q5**
   Reviewer comment: "The dataset only contains the data of LFP batteries under fast charging conditions in the laboratory. Have you considered the generalization ability of this model for other battery chemistries such as NMC and NCA?"
   Coverage: not addressed by the planned experiments.
   Justification path:
   - the revised paper remains intentionally limited to the Severson dataset and does not introduce a new dataset;
   - cross-chemistry validation would require additional data and would significantly expand the scope;
   - the manuscript and reviewer response should state this explicitly as a limitation and frame cross-chemistry validation as future work.

4. **Q6**
   Reviewer comment: 'Section 9.7 states that "using a 10% subset for hyperparameter optimization may not fully capture the data distribution". Does this imply that the optimization results might be unstable?'
   Coverage: addressed by the new experimental design.
   Planned evidence:
   - hyperparameter optimization is now based on grouped 5-fold per cell on the training cells rather than a lightweight subset shortcut;
   - the revised procedure is more stable and better aligned with the intended cross-cell generalization claim.
   Expected response path:
   - the concern is resolved by the revised optimization protocol itself and should be described in the manuscript methodology.

#### Reviewer 3

1. **Q1**
   Reviewer comment: "How sensitive are the SoH and RUL predictions to the exclusion of temperature features, given the acknowledged measurement uncertainty?"
   Coverage: addressed by both the full-cycle and charge-only no-temperature comparisons.
   Planned evidence:
   - grouped 5-fold performance comparison of baseline versus no-temperature variants;
   - final held-out test comparison for selected configurations;
   - observation of whether temperature remains in the compact selected subset.

2. **Q2**
   Reviewer comment: "Can partial cycles (e.g., charge-only or discharge-only) provide comparable performance, and how would this affect feature design?"
   Coverage: addressed by the charge-only feature analysis track.
   Planned evidence:
   - recomputation of features on `charge_process_only`;
   - charge-only optimization, ranking, top-k sweep and final held-out test evaluation;
   - comparison of the charge-only complexity/performance curve against the full-cycle curve.

3. **Q3**
   Reviewer comment: "How does model accuracy vary when predictions are made using early-life cycles only (e.g., first 50-100 cycles)?"
   Coverage: not addressed by the planned experiments.
   Justification path:
   - this would require a dedicated early-life study with a different problem formulation and additional experiment branch;
   - the revised scope is instead centered on full-lifespan cross-cell generalization from a single diagnostic cycle representation;
   - the manuscript and reviewer response should explain that early-life-only prediction is a distinct extension and is outside the targeted revision scope.

4. **Q4**
   Reviewer comment: "Were models trained on specific fast-charge protocols tested against cells cycled under different protocols to assess robustness?"
   Coverage: addressed by the planned experiments.
   Planned evidence:
   - targeted diagnostic analysis will inspect whether large errors concentrate in specific protocol groups;
   - a dedicated protocol-family robustness experiment will hold out one protocol family at a time, where protocol families are defined by average charging C-rate excluding zero-current rest samples;
   - this experiment will be run for both the final selected full-cycle configuration and the final selected charge-only configuration.
   Expected response path:
   - the revised paper will not claim robustness to arbitrary unseen operational regimes, but it will provide a structured within-Severson robustness analysis across coarse charge-aggressiveness families.

5. **Q5**
   Reviewer comment: "Have prediction uncertainties (e.g., confidence intervals) been considered, especially for RUL estimation near end-of-life?"
   Coverage: addressed by the uncertainty track.
   Planned evidence:
   - repeated-seed retraining with fixed train/test split and fixed hyperparameters;
   - prediction mean, standard deviation and interval summaries on the held-out test set;
   - dedicated inspection of more difficult regions such as near end-of-life.

6. **Q6**
   Reviewer comment: "How stable are Random-Forest-derived feature rankings across different random seeds or subsets of training cells?"
   Coverage: addressed by the ranking-and-stability procedure in both full-cycle and charge-only tracks.
   Planned evidence:
   - repeated permutation ranking across grouped 5-fold splits and multiple model seeds;
   - final mean and standard deviation of the ranking statistics across all `(seed, fold)` pairs.
   Expected response path:
   - the revised paper will replace the previous single-run importance narrative with an explicit ranking stability analysis.

7. **Q7**
   Reviewer comment: "Recommend including at least one additional public dataset or a cross-chemistry discussion would significantly strengthen the manuscript's impact."
   Coverage: partially addressed.
   Planned evidence:
   - no new experiment is planned for an additional dataset.
   Justification path:
   - the revised scope intentionally remains within the Severson dataset;
   - therefore this point will be addressed through an expanded limitation and future-work discussion rather than empirical multi-dataset validation.

8. **Q8**
   Reviewer comment: "Recommend performing ablation studies on temperature features and smoothing parameters to quantify their influence on model performance."
   Coverage: partially addressed.
   Planned evidence:
   - temperature ablation is explicitly included in both main tracks.
   Justification path for smoothing:
   - smoothing has been removed from the revised core method because it was training-only, not part of the intended single-cycle inference workflow, and previous tests suggested negligible impact for ExtraTrees;
   - therefore the revision addresses the temperature-ablation portion empirically and addresses the smoothing portion by removing the step from the final methodology.

9. **Q9**
   Reviewer comment: "Recommend investigate whether reduced diagnostic cycles (e.g., charge-only segments) can maintain acceptable accuracy."
   Coverage: addressed by the charge-only feature analysis track.
   Planned evidence:
   - recomputation of charge-only features;
   - charge-only optimization and final held-out evaluation;
   - direct comparison against full-cycle results.

10. **Q10**
   Reviewer comment: "Recommend link selected statistical features more explicitly to known degradation mechanisms such as SEI growth, lithium plating, or impedance rise."
   Coverage: partially addressed by the planned experiments and mainly completed in the manuscript discussion.
   Planned evidence:
   - feature ranking, top-k selection, leave-one-feature-out and correlation analysis provide the empirical basis for identifying the relevant features.
   Justification path for the remaining part:
   - the explicit link to degradation mechanisms is primarily a literature-supported interpretation task rather than a new experiment;
   - the manuscript should use the final selected features and their observed behavior to structure that mechanism discussion.

11. **Q11**
   Reviewer comment: "Consider resampling strategies or weighted loss functions to improve prediction accuracy for long-life cells."
   Coverage: not addressed by the planned experiments.
   Justification path:
   - the revised scope prioritizes feature analysis, charge-only feasibility and uncertainty over adding a new imbalance-mitigation modeling branch;
   - the long-life-cell issue will instead be addressed diagnostically by characterizing where and why those errors occur;
   - the manuscript and reviewer response should state that resampling/weighted-loss strategies are reasonable future extensions but are outside the streamlined revision scope.

12. **Q12**
   Reviewer comment: "Recommend incorporating probabilistic or ensemble-based uncertainty estimates would improve practical deployment relevance."
   Coverage: addressed by the uncertainty track.
   Planned evidence:
   - repeated-seed ensemble of the same model on the same train/test split;
   - per-sample and aggregated uncertainty summaries for the final selected configurations.
