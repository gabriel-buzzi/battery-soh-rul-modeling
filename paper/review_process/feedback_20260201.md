## Workload Classification Scale

- `L0` Already addressed / no additional work.
- `L1` Text-only revision (clarification, structure, wording, citations).
- `L2` Targeted analysis using existing results/code (no full retraining campaign).
- `L3` New experiments on current dataset (retraining, ablations, repeated runs).
- `L4` Major extension (new dataset, new feature pipeline, or substantially new study scope).

### Reviewer 1
Overall:  
In this manuscript, the authors reported that the investigating Lightweight Single-Cycle Prognostics for Li-ion Batteries: Feature Extraction and Cross-Cell Generalization. The aim of the work meets requirement of Results in Engineering, but there will be some insufficient and questionable points in the manuscript. The manuscript needs some improvement in technical content, as it does not meet the standards of the Results in Engineering. Hence, a Major revision is required.

Queries:  
Q1: Authors should change the title of the manuscript properly.  
A1: Did not figure this out yet  
Classification: `L0` (already addressed in baseline; title updated in `paper/20260316_review/___main.tex`).

Q2: Authors should include the nomenclature section.  
A2: This was at the end of the document, may it's the case of moving to the beginning.  
Classification: `L0` (already addressed in baseline; acronym glossary table appears in Introduction).

Q3: What is the reason to choose the single diagnostic cycle for the battery life prediction? Why is it not applicable for continuous operation? Explain in detail.  
A3: In the dataset we used, batteries are tested under repeated and standardized cycles, differently of what would happen in real-world operating batteries. The fact that the load profile in our data is uniform throughout all battery testing implies that the only changes observed in the signal from a new to an old cycle are exclusively due to aging.

Although, the features we've extracted are interpretable, and we could clearly observe their evolution over time and correlate with expected physical behavior like voltage increase and voltage rate-of-change increase due to higher internal resistance and current decrease due to less available capacity. So even though our approach was only validated in well-defined repeated cycle profiles, the features adopted could be translated and somehow extracted from non-uniform cycling.  
Classification: `L1` (mostly text strengthening; baseline already partially addresses this).

Q4: What is the reason for the tree-based models selected? How are the trade-offs for the deep learning approaches beneficial compared to the existing model?  
A4: The tabular characteristic of our feature dataset naturally indicates the use of classical statistical learning models. Deep learning models tend to be more beneficial for unstructured data like raw time-series and images. However, during the work we did test multi-layer perceptron network, but this model didn't achieve comparable performance on test probably due to the fact it has too many parameters which commonly tend to overfitting the training data.  
Classification: `L1` (text clarification with existing evidence; already partially covered).

Q5: Authors should explain how the 4 features resulted in predictable outcomes compared to the 16. Explain in detail, along with the physics behind it.  
A5: Four most important features for SOH:

V_iqr : 0.07, V_std : 0.09, I_iqr : 0.18, V_entropy : 0.51

Four most important features for RUL:

V_median : 0.10, I_std : 0.20, V_iqr : 0.21, V_kurtosis : 0.34

The reduced set of features was better because it retained only features that express clear correlation with the target values across the majority of the cells, since some features might work better or worse for some cells, using all features can help in some cases but can be worse in others, while using only the features that worked for most cells on average benefits more the learning process, i.e. it confuses less the model while training.

We observed that for RUL estimation using the simpler models (KNeighbors and Tweedie) the performance was better using fewer features, that's exactly why these models might suffer more when the input feature set is more complex and has more confounding factor, while when the input set of features has already being treated somehow and has less useless information, they are able to perform better.  
Classification: `L2` (deeper quantitative/physics discussion from existing outputs; no mandatory rerun).
Dataset feasibility (no new dataset): Yes (can be addressed with additional analysis on Severson outputs; optional reruns on same dataset only).

Q6: Why was the differential entropy a valuable feature in analyzing system's behavior? Explain the detailed mechanism in the manuscript.

A6: The differential entropy measures the degree of uncertainty of a signal, i.e. how regular its amplitude is, in this work we've computed the differential entropy of voltage over cycles and observed that it drops as the cell ages, this makes total sense once the internal resistance increase makes the voltage more rapidly reach the upper and lower thresholds when positive and negative current is applied, and once the voltage reaches the thresholds it stays fixed until the cell completely charges or completely discharges. The fact that the voltage stays constant for more time reduces the uncertainty about its value, which reflects a reduced differential entropy.  
Classification: `L1` (text-level expansion; baseline still has TODO marker).

Q7: Authors should include a validation section in the manuscript.  
A7: What do you mean by validation? Our validation is the performance of the models in the test data.  
Classification: `L1` (section structuring/terminology fix; content largely exists already).

Q8: How to find the cycle errors in SoH and RUL prediction? Confirm whether the error predictions are significant for electric vehicle applications. Explain in detail.  
A8: We compute the error metrics by comparing the ground truth values, known in our case because the dataset tests the cells from the start to the end of their lives, with the value outputted from our models.

For the SOH the ground-truth value of each cycle is determined by dividing the total capacity extracted from the cell in each cycle divided by the cell rated capacity offered by the cell manufacturer.

For the RUL the ground-truth value is computed using the known cycle number when each cell reaches the end-of-life, i.e. 80% SOH, this cycle number is than subtracted by each previous cycle number resulting in the number of cycles remaining until the cell reaches the end-of-life.

In real-world scenario only the error of the SOH can be computed, the battery would need to undergo a complete uniform discharge process and the current would need to be integrated, i.e. coulombic count method, this way the total capacity extracted on that test cycle could be divided by the manufacturer rated capacity and the SOH value could be determined and compared with the model output.

Otherwise, for RUL, it wouldn't be possible to compute the error of the estimate until the battery undergoes its full useful life, since we would only be able to know the number of cycles it went through at that moment.

Additionally, the cycle definition it cell would need to be redesigned for real world application, since in real-world there are no well-defined cycles representing a time-frame of the battery life, so counting the remaining useful life in number of cycles might be unfeasible once we don't exactly know what a cycle is. In this case, we would probably need to approximate a cycle with equivalent cycle measures.  
Classification: `L2` (requires stronger benchmarking argument for EV relevance, mostly analysis/discussion).
Dataset feasibility (no new dataset): Yes (error analysis + discussion can be done entirely on Severson test results).

### Reviewer 2
Overall:  
This article proposes a single-cycle prediction method for the State of Health (SOH) and Remaining Useful Life (RUL) of lithium-ion batteries based on a lightweight machine learning framework. The study utilized an open dataset and extracted 16 statistical features from the voltage, current, and temperature signals of a single charging and discharging cycle. These features were then reduced to 4 key features through feature selection. The study evaluated various tree ensemble models including LightGBM and Extra Trees, and used TPE for hyperparameter optimization.

Queries:  
Q1: Figure 3 shows the correlation between capacity and temperature. But why is there no discussion about this in the text?  
A1: Eighter remove this figure or write some discussion. The temperature curve follows the capacity when it's changing; this is due to the fact that when capacity is changing higher currents are flowing through the cell, heating it up.  
Classification: `L1` (text cleanup / figure consistency check).

Q2: The text states that the cycle life distribution in Figure 5 shows that most batteries reach End of Life (EoL) after approximately 700 cycles. However, Figure 5 actually indicates that most batteries reach EoL after only 500 cycles. Please carefully review and make the necessary corrections.  
A2: Update the text to 500 cycles as in the image.  
Classification: `L0` (already corrected in baseline: mode around 500, mean around 700).

Q3: In the data preprocessing stage, the Savitzky-Golay filter was used to smooth the features. Could you please explain how the 10-cycle in the filter window size were determined? Have you analyzed the impact of different window sizes on the model performance?  
A3: This value was arbitrary, and we did not consistently evaluate the effect of different values. The consequence of that values would be that if one would like to obtain more accurate estimates when applying the training model, it could obtain data from 10 cycles and apply the filtering, as the goal of our solution is to allow inference with a single cycle we didn't apply the filter when testing the models, but this could be done upon data availability in real-world scenarios.  
Classification: `L3` (needs ablation runs across window sizes and updated results/tables).
Dataset feasibility (no new dataset): Yes (ablation can be run on Severson only).

Q4: Figure 15-18 shows that the prediction deviations for certain test cells, especially those with long lifespans, are relatively large. Could you please explain whether the characteristics of these abnormal cells, such as charging protocols and temperature anomalies, have been analyzed? Have you considered developing specialized modeling for such cells?  
A4: It would be good to check the protocol adopted on those cells, I believe they should have lower current rates which reflected in longer lives, but this must be double-checked.  
Performance could be increased for those cases if we have more data on longer life cells in the training set.  
Classification: `L2` (targeted data/profile analysis required; specialized models would be `L3`).
Dataset feasibility (no new dataset): Yes (cell-level/protocol-level analysis and optional mitigation can be done on Severson only).

Q5: The dataset only contains the data of LFP batteries under fast charging conditions in the laboratory. Have you considered the generalization ability of this model for other battery chemistries such as NMC and NCA?  
A5: We didn't test this.  
Classification: `L1` (if discussion-only response) or `L4` (if adding cross-chemistry validation experiments). Baseline currently covers discussion only.
Dataset feasibility (no new dataset): Partial (discussion/limitations: yes; empirical cross-chemistry validation: no).

Q6: Section 9.7 states that "using a 10% subset for hyperparameter optimization may not fully capture the data distribution". Does this imply that the optimization results might be unstable?  
A6:  
Classification: `L2` (analysis response plus optional sensitivity checks; full reruns across subsets/seeds would move to `L3`).
Dataset feasibility (no new dataset): Yes (sensitivity checks can be run on Severson only).

### Reviewer 3

Overall:  
The manuscript addresses a critical challenge in battery management systems by proposing an accurate and practical approach for SoH and RUL estimation under realistic data constraints. Its key contribution lies in demonstrating reliable predictions from a single diagnostic cycle, significantly improving real-world applicability. The use of a large, well-characterized public dataset ensures statistical robustness, while the methodology—from preprocessing to model evaluation—is systematic, transparent, and reproducible. The comparative analysis of linear and non-linear models, combined with rigorous cell-wise validation, strengthens the technical credibility. Overall, the feature-based approach offers a good balance between interpretability, computational efficiency, and predictive performance, supporting practical deployment in embedded BMS.
My Observation:  
1. The study is limited to LFP/graphite cells under aggressive fast-charging and full DoD conditions, raising concerns about generalization to other chemistries (NMC, NCA, LTO).  
2. Although reduced to a single cycle, the requirement of a complete charge-discharge cycle may still be impractical for many in-service battery systems.
3. Known inconsistencies in thermocouple contact introduce uncertainty in temperature-based features, yet temperature remains an important predictor without sufficient sensitivity analysis.
4. Defining RUL strictly in terms of remaining cycles limits applicability to irregular usage patterns where time-based or energy-throughput-based metrics may be more appropriate.
5. While feature trends are discussed qualitatively, the connection between selected statistical features and underlying electrochemical degradation mechanisms remains largely speculative.
6. The models underperform for cells with unusually long cycle life (~2000 cycles), highlighting dataset imbalance and limited extrapolation capability.

Queries:  
Q1: How sensitive are the SoH and RUL predictions to the exclusion of temperature features, given the acknowledged measurement uncertainty?  
A1: The 4-features selected does not account for any temperature features; the impact was not significant.  
Classification: `L2` (explicit ablation summary from existing outputs; no full new study required).
Dataset feasibility (no new dataset): Yes (temperature ablation can be run on Severson only).

Q2: Can partial cycles (e.g., charge-only or discharge-only) provide comparable performance, and how would this affect feature design?  
A2: We did not do this experiments.  
Classification: `L4` (new feature engineering + full retraining/evaluation pipeline).
Dataset feasibility (no new dataset): Yes (partial-cycle experiments can be implemented by slicing Severson cycles; no external dataset required).

Q3: How does model accuracy vary when predictions are made using early-life cycles only (e.g., first 50-100 cycles)?  
A3: If we use only early life cycles to train the model, the trained model only works for early life test cycles; it cannot generalize late cycles when trained on early life examples.
In our case, we use cycles from the whole lifespan of a set of cells and evaluate cycles from the whole lifespan of another disjoint set of cells; the generalization is from one set of cells to another, not from one step of life to another.
If we train the model as we did, using cycles from the whole life of cells, the model will later perform well for any point of a test cell life.  
Classification: `L3` (needs dedicated early-life split experiments and new result tables).
Dataset feasibility (no new dataset): Yes (early-life restrictions can be evaluated on Severson only).

Q4: Were models trained on specific fast-charge protocols tested against cells cycled under different protocols to assess robustness?  
A4: I don't believe we can test this since our dataset only contains fast-charge cycles.  
Classification: `L3` (protocol-wise split robustness experiments within existing data, or else explicit limitation text only as `L1`).
Dataset feasibility (no new dataset): Partial (policy-wise robustness within Severson: yes; robustness to substantially different operational profiles: no).

Q5: Have prediction uncertainties (e.g., confidence intervals) been considered, especially for RUL estimation near end-of-life?  
A5: No, for that we would have to have multiple estimates for each test sample, eighter by having multiple models trained with different random seeds or by having multiple models from different architectures, in both cases each set of models would have to be trained on the exact same training dataset.

As in our study the goal was to compare different model architectures, it would make much sense to ensemble them afterwards. I believe in our case if we wanted to obtain these uncertainties we would have to train the same model repeated times with different random seeds.

I'm not sure if we would have to train multiple models on both validation and test steps. In the validation, multiple trainings are done, for each set of hyperparameters and for each fold of the training data, after that to each set of hyperparameters is given a average performance across each fold of data and with that average performance we select the best set of hyperparameters. If we were willing to compute the uncertainty of the performance here, we would need to train multiple models we different random seeds for each fold on each set of hyperparameters. This way the performance of one set of hyperparameters on each fold would have an uncertainty associated, and we would have to account for that uncertainty for aggregating the result of each hyperparameter set across the folds and to select the best hyperparameter set.

However, if we choose to consider this uncertainty only for the test set, we will have to, once the best model is defined with the current hyperparamter tuning strategy, retrain and evaluate that best model with the whole training dataset multiple times with different random seeds instead of only once as we are doing now. This would result in multiple predictions for each test sample, and we would be able to estimate the uncertainty of each prediction, further being able to aggregate this uncertainty into the final overall metrics.

Both strategies would take some time to implement and mainly to run the experiments, but the second can be easier.  
Classification: `L3` (new repeated-train experiments required).
Dataset feasibility (no new dataset): Yes (repeat-seed/ensemble uncertainty can be computed on Severson only).

Q6: How stable are Random-Forest-derived feature rankings across different random seeds or subsets of training cells?  
A6: Although section 6.5 states that 10-fold cross validation was performed to generate the feature importance scores, looking at the code, I couldn't find that cross-validation, the code seems to implement a single train using all training cells to obtain the feature importance from the trained random forest model. Furthermore, in the code the number of estimators used is 50, not 10 as stated in the text. We need to update the text to account for that.
Answering the question, as we noticed there is no 10-fold cross validation for computing the feature importance, there is no evidence that the Random-Forest-derived feature rankings are stable across different subsets of training cells. Moreover, we did not perform any tests with different random seeds.  
Classification: `L3` (requires stability experiments; immediate text/code consistency correction is `L1`).
Dataset feasibility (no new dataset): Yes (seed/subset stability can be tested on Severson only).

Q7: Recommend including at least one additional public dataset or a cross-chemistry discussion would significantly strengthen the manuscript's impact.  
A7: If we manage to process this dataset to make it follow the same data format/schema of the Severson, testing the model with it should be difficult using the existing codebase.  
Classification: `L4` for additional dataset validation; `L1` for expanded cross-chemistry discussion only.
Dataset feasibility (no new dataset): Partial (discussion only: yes; adding another dataset: no).

Q8: Recommend performing ablation studies on temperature features and smoothing parameters to quantify their influence on model performance.  
A8: I believe this would take us more time and effort. From the coding point of view there should be ease, but running more experiments and aggregating the results can take more time.  
Classification: `L3` (ablation experiments and aggregated reporting).
Dataset feasibility (no new dataset): Yes (temperature/smoothing ablations can be run on Severson only).

Q9: Recommend investigate whether reduced diagnostic cycles (e.g., charge-only segments) can maintain acceptable accuracy.  
A9:  
Classification: `L4` (new data slicing strategy + feature redesign + retraining).
Dataset feasibility (no new dataset): Yes (reduced diagnostic segments can be tested by slicing Severson cycles).

Q10: Recommend link selected statistical features more explicitly to known degradation mechanisms such as SEI growth, lithium plating, or impedance rise.  
A10: We could do this by adding a subsection near fig. 11.  
Classification: `L2` (targeted technical discussion with literature support; usually no reruns).
Dataset feasibility (no new dataset): Yes (text + references; no new data required).

Q11: Consider resampling strategies or weighted loss functions to improve prediction accuracy for long-life cells.  
A11: I didn't fully understand this comment. What does he mean by resampling strategies? Moreover, how would weighting the loss based on the total cell lifespan wouldn't configure data leakage?  
Classification: `L3` (modeling changes + retraining + comparison against baseline).
Dataset feasibility (no new dataset): Yes (reweighting/resampling can be evaluated on Severson only).

Q12: Recommend incorporating probabilistic or ensemble-based uncertainty estimates would improve practical deployment relevance.  
A12: As far as I understood, this would involve making inference for the same cycle with multiple models to derive an uncertainty metric based on the set of estimates. For that, we would have to train multiple models for each train/test to split either the same architecture with different random seed or different architectures and use them to make multiple inferences of each test sample.  
Classification: `L3` (new multi-run/ensemble pipeline and additional reporting).
Dataset feasibility (no new dataset): Yes (uncertainty estimates can be produced via repeated training on Severson only).
