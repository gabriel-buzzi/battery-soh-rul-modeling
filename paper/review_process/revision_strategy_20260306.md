# Revision Strategy and Proposed Changes

This document summarizes the proposed changes for the revised manuscript before the final implementation and experiment cycle. Its purpose is to make explicit what will be changed in the paper, why these changes are being made, and how they respond to reviewer feedback while remaining within the scope of the Severson dataset and the available computational resources.

## 1. High-level shift in research focus

The main proposed change is a shift in emphasis:

- from a paper primarily centered on **model comparison and predictive performance**
- to a paper primarily centered on **single-cycle feature relevance, interpretability, and robustness**

### Reasoning

This shift is motivated by three factors:

1. The reviewer comments are much more focused on the meaning, usefulness, stability, and physical interpretation of the extracted features than on ranking different machine learning models.
2. The current multi-model comparison broadens the scope of the work but does not directly strengthen the main scientific contribution.
3. A feature-centered framing is more coherent with the actual novelty of the work: extracting useful degradation information from a single cycle with a lightweight and interpretable representation.

Under this revised framing, the model is no longer the primary contribution. Instead, the model becomes an analysis tool that allows the study of which features matter, how robust they are, and what their limitations are.

---

## 2. Reduction from multiple models to one main reference model

The proposed manuscript revision will reduce the importance of multi-model comparison and adopt **one main reference model**.

### Proposed main model

- **ExtraTreesRegressor** as the primary reference model

### Reasoning

Extra Trees is a strong fit for the revised paper because:

- it performs well on tabular data;
- it captures nonlinear relationships and feature interactions;
- it is computationally efficient and practical for repeated experiments;
- it is simpler to justify in a feature-focused paper than a heavily tuned boosting pipeline;
- it integrates naturally with feature-importance and ablation analysis.

### Why not keep model comparison as a main section

If the paper continues to give equal emphasis to multiple regressors, the revised manuscript will remain partly framed as a model selection study. That weakens the clarity of the contribution and makes reviewer-driven analyses harder to prioritize. In particular, it diverts effort away from:

- feature ablation;
- temperature sensitivity;
- uncertainty quantification;
- feature ranking stability;
- interpretation of physical meaning.

For these reasons, the current recommendation is:

- keep **ExtraTreesRegressor** as the main model;
- avoid broad model comparison in the main results;
- avoid introducing a secondary linear baseline as part of the revised core scope.

---

## 3. Hyperparameter optimization strategy for feature analysis

The proposed revision will **not** perform a full hyperparameter optimization for every feature ablation run.

### Proposed strategy

- optimize the selected reference model once for the baseline feature configuration;
- optionally optimize once more for a compact reference configuration if a reduced feature set is later selected by a formal criterion;
- for most feature ablation studies, keep the hyperparameters fixed and retrain/evaluate with the altered feature set.

### Reasoning

This is the preferred strategy because it isolates the effect of the feature change itself. If hyperparameters are re-optimized for every ablation, then any observed difference mixes two effects:

- the effect of removing or modifying the feature set;
- the effect of the optimizer adapting the model to that new subset.

For a feature-centered paper, this is undesirable because it makes the interpretation less direct. Fixed-hyperparameter ablations provide a cleaner measure of feature contribution.

### Exceptions

Re-optimization may still be justified for a very small number of substantially different feature sets, for example:

- all 16 features;
- a compact top-`k` reference set selected from a feature-count sensitivity analysis.

But it is not recommended for every leave-one-feature-out experiment.

---

## 4. Removal of smoothing from the core methodology

The proposed revision is to **remove the Savitzky-Golay smoothing step from the main method** and treat it, at most, as a supporting sensitivity result.

### Reasoning

The current smoothing step has the following characteristics:

- it is applied only to the training data;
- it is not part of the intended inference-time workflow;
- for Extra Trees, previous experiments suggest it has little or no practical effect on performance;
- it increases methodological complexity without clearly improving the scientific contribution.

Because of this, keeping smoothing as a central part of the pipeline is difficult to justify in a revised paper focused on practical and interpretable single-cycle features.

### Proposed manuscript treatment

- the main pipeline will use the raw cycle-level statistical features without smoothing;
- the manuscript may briefly mention that smoothing was evaluated as a training-only denoising step;
- a compact sensitivity note may be retained to show that the chosen tree model was not materially affected by smoothing.

This strengthens the paper by making the method simpler, more realistic, and easier to interpret.

---

## 5. Feature analysis as a central analysis

The revised manuscript will prioritize feature analysis over model comparison.

### Proposed analyses

- model-based feature ranking using the same Extra Trees reference model used for prediction;
- permutation importance to quantify predictive reliance of the fitted model on each feature;
- feature-count sensitivity analysis using top-`k` ranked features, for example:
  - 2;
  - 4;
  - 6;
  - 8;
  - 12;
  - 16;
- grouped ablations, especially:
  - without temperature features;
  - voltage-only or voltage-dominant subsets where useful;
- limited leave-one-feature-out analysis applied only to the top-ranked subset of features rather than the entire 16-feature set;
- feature ranking stability under repeated seeds/subsamples;
- correlation analysis among features to identify redundancy and collinearity;
- physical interpretation of the most important features.

### Reasoning

These analyses directly answer reviewer concerns about:

- why a reduced feature set works and how compact it can be made without substantial loss;
- whether temperature features are truly necessary;
- whether the selected features are stable;
- whether the model really depends on a feature or can replace it with correlated alternatives;
- how the chosen features connect to known battery degradation mechanisms.

This is a much better use of the revision effort than broadening the model comparison.

### Why not rely only on impurity-based feature importance

Impurity-based feature importance from a tree ensemble is useful, but it is not sufficient on its own for interpretability because:

- correlated features can split importance across redundant variables;
- a feature may appear important in ranking while having limited unique contribution when correlated alternatives are present;
- reviewer concerns are not only about ranking but also about robustness and necessity.

For this reason, the revised scope should combine ranking with permutation importance, grouped ablation, and a top-`k` sensitivity analysis.

### Why not use full leave-one-feature-out as the main analysis

Although leave-one-feature-out ablation is informative, applying it to all 16 features is not the most efficient or interpretable primary analysis in the presence of correlated tabular features. The preferred strategy is:

- use ranking and stability analysis to identify the most relevant subset;
- apply leave-one-feature-out only within that smaller high-value subset;
- use grouped ablations for broader signal-family questions such as temperature sensitivity.

This reduces redundant experiments while preserving interpretability.

### Treatment of the previous 4-feature setting

The previous 4-feature result was based on an arbitrary cutoff and should no longer be presented as a specially meaningful number by itself. Instead:

- the revised manuscript should present a feature-count sensitivity analysis;
- a compact subset should be selected only after observing where performance stabilizes;
- the chosen compact subset should be justified by a stated criterion, for example minimal feature count within a small performance tolerance of the full-feature baseline.

---

## 6. Temperature-feature sensitivity

Temperature-feature sensitivity remains relevant, even if smoothing is removed.

### Proposed treatment

- explicitly compare the baseline feature set with a no-temperature variant;
- assess whether performance changes materially when temperature features are excluded;
- discuss the result in light of known thermocouple-contact uncertainties in the Severson dataset.

### Reasoning

This directly addresses reviewer concerns about:

- the reliability of temperature measurements;
- whether temperature contributes meaningfully to prediction;
- whether the model relies excessively on a potentially noisy signal family.

If the no-temperature model performs similarly, that becomes a useful scientific finding in itself.

---

## 7. Correlation analysis and dimensionality reduction

The revised manuscript should use correlation analysis among features as supporting analysis material, but should not adopt unsupervised dimensionality reduction as part of the main predictive pipeline.

### Proposed treatment

- compute a feature-feature correlation matrix;
- identify strongly redundant feature groups;
- use this information to interpret feature rankings, grouped ablations, and leave-one-feature-out results;
- avoid PCA or other latent transformations in the core method.

### Reasoning

This choice is preferred because the revised paper aims to strengthen interpretability. Latent-space dimensionality reduction would make the final representation harder to explain physically, while correlation analysis provides useful information about redundancy without sacrificing interpretability.

At this stage, correlation analysis should be treated as analytical support rather than as a mandatory preprocessing or feature-selection step.

---

## 8. Uncertainty quantification

The revised manuscript should add an uncertainty analysis, but in a way that remains computationally realistic.

### Proposed method

- train the same reference model multiple times using different random seeds;
- keep the same data split and the same hyperparameters;
- generate multiple predictions for each test sample;
- summarize uncertainty with statistics such as:
  - prediction mean;
  - prediction standard deviation;
  - percentile intervals.

### Reasoning

This is the most practical uncertainty strategy under the current constraints:

- no new dataset;
- limited computational resources;
- tabular tree-based model;
- reviewer request for confidence or uncertainty information.

This method is straightforward to explain, feasible to implement, and scientifically useful. It can show whether uncertainty increases in the most difficult regions of the problem, such as:

- long-life cells;
- near-EoL predictions;
- reduced-feature conditions.

---

## 9. Full-cycle versus charge-process-only comparison

The revised manuscript should include a direct comparison between two diagnostic definitions:

- **full-cycle** features, computed from the complete charge-discharge cycle;
- **charge-process-only** features, computed only from the charging event.

### Reasoning

This comparison is worth pursuing because it directly addresses one of the reviewer concerns about the practical applicability of the method when full-cycle information is unavailable or inconvenient to obtain. It also fits well with the revised feature-centered focus of the paper, since it tests how much useful degradation information is preserved when the signal window is reduced.

Even if the charge-only setting performs worse than the full-cycle setting, the comparison remains scientifically valuable because it quantifies the tradeoff between practicality and predictive information.

### Proposed definition of charge-process-only

For the Severson dataset, the charge process should be defined as:

- starting at the first charging-current sample after the preceding discharge/rest;
- ending at the first return to zero current after the end of the CV charge;
- including:
  - the initial CC pulse;
  - the second CC pulse when present;
  - the later charge segment that completes charging;
  - the CV charge tail;
  - any short rest segment that occurs inside the charging procedure;
- excluding:
  - the discharge stage;
  - the post-charge rest that belongs to the transition to discharge.

This definition is preferred because it represents the full charging event as experienced by the cell and avoids introducing artificial discontinuities by removing short internal rest segments.

### Proposed experimental treatment

This comparison should be implemented as two parallel diagnostic settings:

- `full_cycle`
- `charge_process_only`

The pipeline should be repeated from feature extraction onward for both settings, because the features are defined over different signal windows and therefore represent different physical summaries of the cycle. However, this should not be treated as a leave-one-feature-out ablation. Instead, it should be treated as a comparison between two diagnostic formulations.

### Hyperparameter strategy

For this specific comparison, one baseline optimization per diagnostic setting is justified:

- one optimization for `full_cycle`;
- one optimization for `charge_process_only`.

This is different from the recommendation for leave-one-feature-out ablations, where fixed hyperparameters are preferred. Here the input representation itself changes in a substantial way, so a separate baseline optimization for each scenario is methodologically defensible.

---

## 10. What will not be pursued in the revised paper

At this stage, the following directions are not recommended as core contributions:

- broad multi-model benchmarking as a main result;
- full hyperparameter optimization for every ablation variant;
- unsupervised dimensionality reduction as part of the core predictive method;
- a separate simpler linear-regression baseline as part of the revised core scope;
- cross-chemistry validation using a new dataset;
- complex probabilistic modeling approaches that significantly change the methodological scope.

### Reasoning

These directions either:

- move the paper away from its strongest contribution;
- require substantial additional implementation or computation;
- or do not directly answer the main reviewer concerns within the existing dataset scope.

They may be mentioned as future work, but they should not define the revised manuscript.

---

## 11. Interaction between the three main analysis blocks

The revised scope now has three main analysis blocks:

- feature importance and ablation;
- uncertainty quantification;
- full-cycle versus charge-process-only comparison.

These blocks should not all be expanded over one another indiscriminately. Doing so would produce a large and partially redundant experiment matrix that would be difficult to run, interpret, and explain. The revision should instead define a clear hierarchy.

### 11.1 Primary reference setting

The main reference setting for the paper should be:

- diagnostic setting: `full_cycle`;
- model: `ExtraTreesRegressor`;
- preprocessing: no smoothing;
- feature space: full interpretable feature set;
- one baseline hyperparameter optimization.

This is the setting on which the deepest feature analysis should be performed. It should serve as the main source for:

- impurity-based feature ranking;
- permutation importance;
- correlation analysis;
- top-`k` feature-count sensitivity;
- grouped ablations such as no-temperature;
- limited leave-one-feature-out within the top-ranked subset.

### 11.2 How charge-process-only should interact with feature analysis

Feature analysis should be performed in both diagnostic settings, but not at the same depth.

#### Full-cycle

`full_cycle` should carry the full feature-analysis burden because:

- it is the original formulation of the method;
- it is the richest signal setting;
- it provides the clearest basis for interpreting feature relevance and redundancy.

#### Charge-process-only

`charge_process_only` should be used as a secondary diagnostic setting intended to answer a more specific question:

- how much feature utility and predictive quality are retained when only the charging event is available?

For this reason, the recommended scope for `charge_process_only` is:

- one baseline optimization for the charge-process-only feature set;
- one baseline evaluation with the full charge-process-only feature set;
- feature ranking and permutation importance within this setting;
- one no-temperature grouped ablation if temperature remains present and relevant;
- one compact top-`k` sensitivity analysis.

The following are **not** recommended as first-pass requirements for `charge_process_only`:

- a full leave-one-feature-out study;
- a full replication of every ablation run already done for `full_cycle`;
- a large combinatorial comparison of all feature subsets across both diagnostic settings.

This keeps the charge-only study scientifically useful without letting it dominate the revision.

### 11.3 How uncertainty should interact with feature analysis

Uncertainty should be treated as a characterization layer on top of selected final configurations, not as an extra dimension applied to every intermediate ablation.

The main purpose of the uncertainty analysis is to show:

- how stable the final predictions are;
- whether uncertainty increases in more difficult regions;
- whether the practical alternatives introduced in the revision lead to less reliable predictions.

Accordingly, uncertainty should be computed for a small set of headline configurations:

1. `full_cycle` baseline with the full feature set;
2. `full_cycle` no-temperature variant;
3. `full_cycle` compact top-`k` variant selected after the feature-count sensitivity analysis;
4. `charge_process_only` baseline.

This is sufficient to answer the reviewer concern about confidence and deployment relevance while keeping runtime under control.

Uncertainty is **not** required for:

- every value of `k` in the top-`k` sweep;
- every leave-one-feature-out run;
- every correlation or ranking analysis;
- every exploratory intermediate feature subset.

### 11.4 Practical decision rules for the revision

To avoid ambiguity, the revision should follow these rules:

1. Use `full_cycle` as the main setting for interpreting which features matter and why.
2. Use `charge_process_only` as a practicality comparison, not as a second full paper inside the paper.
3. Use uncertainty only on the final and most decision-relevant configurations.
4. Do not propagate uncertainty estimation into the entire ablation matrix unless a very specific result later justifies it.
5. Do not repeat full leave-one-feature-out analysis in both diagnostic settings.

### 11.5 Recommended experiment order

The three blocks should be executed in this order:

1. Build the `full_cycle` baseline and perform the main feature analyses.
2. From that baseline, determine:
   - the top-ranked features;
   - the compact top-`k` candidate;
   - the no-temperature comparison.
3. Build the `charge_process_only` baseline and compare it against the `full_cycle` reference.
4. Run repeated-seed uncertainty only on the final selected configurations.

This order is important because the uncertainty study should be informed by the feature-analysis conclusions rather than run blindly across all possible variants.

### 11.6 Final scope interpretation

Under this plan, the three main analyses play different roles:

- feature importance and ablation: the core scientific analysis;
- charge-process-only: the practicality and applicability analysis;
- uncertainty: the reliability analysis.

This separation keeps the revised paper coherent and prevents one reviewer-driven addition from multiplying all the others into an unmanageable experiment set.

---

## 12. Expected revised contribution statement

Under the proposed revision, the paper contribution is expected to be reframed along the following lines:

- demonstrate that single-cycle statistical features contain sufficient degradation information for SoH and RUL estimation across unseen cells;
- identify which features are most informative, physically meaningful, and robust across analysis methods;
- assess how robust the feature representation is to preprocessing and temperature exclusion;
- quantify redundancy and compactness of the feature set through correlation and top-`k` sensitivity analyses;
- compare full-cycle and charge-process-only diagnostic definitions;
- quantify uncertainty of predictions using repeated-seed ensembles on a lightweight tree-based reference model.

This revised contribution is more coherent with the reviewer feedback and more scientifically focused than the original broader model-comparison framing.

---

## 13. Summary of proposed methodological changes

The proposed revision consists of the following main changes:

1. Shift the paper focus from model comparison to feature analysis.
2. Use ExtraTreesRegressor as the main reference model.
3. Remove smoothing from the core methodology.
4. Use Extra Trees itself, together with permutation importance, as the main feature-ranking framework.
5. Replace the arbitrary 4-feature emphasis with a top-`k` feature-count sensitivity analysis.
6. Keep hyperparameters fixed for most feature ablation studies.
7. Add temperature-feature sensitivity analysis.
8. Use correlation analysis as supporting evidence for redundancy, not as latent feature preprocessing.
9. Compare full-cycle and charge-process-only diagnostic settings.
10. Add repeated-seed uncertainty estimation.
11. Emphasize feature interpretation and stability over marginal performance gains.

Taken together, these changes are intended to produce a revised manuscript that is:

- more focused;
- more interpretable;
- more defensible in light of the reviewer comments;
- and more feasible to complete with the available dataset and compute budget.
