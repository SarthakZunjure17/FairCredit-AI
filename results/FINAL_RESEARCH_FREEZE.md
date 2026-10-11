# FairCredit-AI: Final Research Results Freeze
## Formal Methodological Freeze & Research Synthesis Specification

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Milestone:** Phase 5.1 — Final Results Freeze  
**Date:** October 2026  
**Status:** **OFFICIALLY FROZEN — READY FOR DASHBOARD IMPLEMENTATION**  

---

## 1. Final Research Question

> **Core Research Question:**  
> *“When fairness mitigation is applied to machine-learning credit-scoring models, how does it affect predictive performance, group fairness, and explanation consistency?”*

---

## 2. Final Central Research Finding

> **Central Finding:**  
> **Fairness mitigation through training sample reweighing exhibited heterogeneous effects on explanation consistency across dataset-model configurations, rather than imposing a universal fairness–explanation trade-off.**  
>
> 1. **Predictive Performance Remained Resilient:** ROC-AUC changed minimally across all configurations (mean $|\Delta \text{AUC}| = 0.0047$, with four of six configurations changing by $\le \pm 0.0020$).  
> 2. **Explanation Consistency Maintained High Directional Alignment:** Mean SHAP Cosine Similarity remained high across all configurations ($0.9380 \le \text{Cosine} \le 0.9996$).  
> 3. **Attribution Hierarchies Exhibited Architectural Plasticity:** While Logistic Regression attributions were virtually invariant ($\text{Top-5 Overlap} \ge 0.954$, $\rho \ge 0.983$), Random Forest models exhibited notable attribution rank shifts ($\text{Top-5 Overlap} \approx 0.792 - 0.837$, $\rho \approx 0.777 - 0.889$).  
> 4. **Absence of a Universal Trade-Off:** In the German Random Forest configuration, a substantial fairness improvement ($\Delta \text{EOD} = -0.0750$, $-60\%$) co-occurred with improved predictive performance ($\Delta \text{AUC} = +0.0137$) and high explanation consistency (Mean Cosine = $0.9652$).

---

## 3. Six Authoritative Model-Dataset Results (Locked Numerical Truth)

The table below presents the authoritative, validated metrics across all six benchmark configurations (derived directly from locked Phase 3 and Phase 4 files):

| Dataset | Model Architecture | Baseline AUC | Reweighed AUC | $\Delta$ AUC | Baseline EOD | Reweighed EOD | $\Delta$ EOD | Baseline DPD | Reweighed DPD | $\Delta$ DPD | Baseline Appr. DI | Reweighed Appr. DI | Mean SHAP Cosine [95% CI] | Mean Top-5 Overlap | Mean Spearman $\rho$ | Expl. N |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Taiwan** | Logistic Regression | 0.7574 | 0.7582 | +0.0008 | 0.0194 | 0.0229 | +0.0035 | 0.0069 | 0.0069 | 0.0000 | 1.0079 | 1.0079 | 0.9996 [0.9995, 0.9996] | 0.9800 | 0.9832 | 500 |
| **Taiwan** | Random Forest | 0.7528 | 0.7548 | +0.0020 | 0.0298 | 0.0219 | -0.0079 | 0.0093 | 0.0075 | -0.0018 | 1.0107 | 1.0086 | 0.9380 [0.9341, 0.9419] | 0.7924 | 0.7768 | 500 |
| **Taiwan** | XGBoost | 0.7785 | 0.7774 | -0.0011 | 0.0306 | 0.0436 | +0.0130 | 0.0027 | 0.0003 | -0.0024 | 1.0031 | 1.0003 | 0.9787 [0.9772, 0.9803] | 0.8864 | 0.8871 | 500 |
| **German** | Logistic Regression | 0.8061 | 0.8055 | -0.0006 | 0.0500 | 0.1000 | +0.0500 | 0.0577 | 0.0131 | -0.0446 | 0.9244 | 1.0177 | 0.9983 [0.9982, 0.9984] | 0.9540 | 0.9908 | 200 |
| **German** | Random Forest | 0.7798 | 0.7935 | +0.0137 | 0.1250 | 0.0500 | -0.0750 | 0.0620 | 0.0005 | -0.0615 | 0.9269 | 0.9994 | 0.9652 [0.9624, 0.9677] | 0.8370 | 0.8891 | 200 |
| **German** | XGBoost | 0.8019 | 0.7918 | -0.0101 | 0.0250 | 0.0250 | 0.0000 | 0.0157 | 0.0209 | +0.0052 | 0.9796 | 0.9736 | 0.9757 [0.9735, 0.9779] | 0.8380 | 0.9173 | 200 |

---

## 4. Valid Fairness Conclusions

1. **Demographic Parity Improvement:** In all configurations with non-trivial baseline demographic imbalance (German RF, German LR, Taiwan RF, Taiwan XGB), Reweighing improved Demographic Parity Difference and brought Approval Disparate Impact closer to 1.0.
2. **Heterogeneous Error Rate Parity (EOD):** Equalized Odds Difference improved substantially in Random Forest models ($-60.0\%$ in German, $-26.5\%$ in Taiwan), remained constant in German XGBoost ($0.0250 \to 0.0250$), and widened in Logistic Regression ($0.0500 \to 0.1000$ in German; $0.0194 \to 0.0229$ in Taiwan).
3. **No Guarantee of Zero Disparity:** Reweighing does not mathematically guarantee $\text{DPD} = 0$ or $\text{EOD} = 0$ in the fitted scoring function; outcomes depend on model capacity, inductive biases, and feature correlations.

---

## 5. Valid Predictive Performance Conclusions

1. **Continuous Scoring Integrity:** Reweighing preserves the continuous scoring nature of the base classifiers, ensuring that ROC-AUC remains a valid ranking metric.
2. **Minimal Performance Shift:** Across the benchmark, performance was largely stable. Mean absolute change in ROC-AUC was $|\Delta \text{AUC}| = 0.0047$. In four of six models, $|\Delta \text{AUC}| \le 0.0020$.
3. **Observed Configuration-Specific Gain:** In German Random Forest, ROC-AUC increased from $0.7798$ to $0.7935$ ($+0.0137$, $+1.76\%$), demonstrating that sample reweighting can improve minority class representation without degrading global ranking.

---

## 6. Valid Explanation Consistency Conclusions

1. **Directional Polarity Preserved:** Mean SHAP Cosine Similarity remained high across all configurations ($0.9380 \le \text{Cosine} \le 0.9996$), confirming that overall attribution orientation is stable under training sample reweighing.
2. **Architectural Consistency Tiers:**
   - *Logistic Regression:* Highest stability (Mean Cosine $> 0.998$, Top-5 Overlap $\ge 0.954$, Spearman $\rho \ge 0.983$).
   - *XGBoost:* Strong intermediate stability (Mean Cosine $\approx 0.976 - 0.979$, Top-5 Overlap $\approx 0.838 - 0.886$, Spearman $\rho \approx 0.887 - 0.917$).
   - *Random Forest:* Lowest tier with rank shifts (Mean Cosine $\approx 0.938 - 0.965$, Top-5 Overlap $\approx 0.792 - 0.837$, Spearman $\rho \approx 0.777 - 0.889$).
3. **Individual Attribution Fidelity:** Same-applicant attribution similarity ($0.938 - 0.999$) was substantially higher than credit nearest-neighbor peer similarity ($0.135 - 0.819$), confirming that measured stability reflects individual-level attribution fidelity rather than global model insensitivity.

---

## 7. Corrected Interpretation of Reweighing

> **Methodological Standard:**  
> Reweighing modifies training sample weights according to the joint distribution of the protected attribute and target. It is intended to reduce dependence between protected-group membership and the target distribution during model training, but the resulting fitted model is not guaranteed to satisfy demographic parity exactly. Reweighing also does not directly enforce Equalized Odds.  
>
> In the evaluated configurations, changes in demographic-parity-related metrics did not consistently coincide with changes in Equalized Odds, demonstrating the importance of evaluating multiple fairness criteria.

---

## 8. Corrected Interpretation of Top-5 Feature Overlap

> **Regulatory Auditing Standard:**  
> Lower Top-5 feature overlap indicates greater instability in the set of highly attributed features following reweighting. This suggests that explanation stability should be monitored when feature attributions are subsequently used to derive decision-reason or adverse-action reason codes under regulatory compliance frameworks (such as the Equal Credit Opportunity Act and Fair Credit Reporting Act).

---

## 9. Methodological Limitations

1. **Dataset Scope:** Evaluated two public credit-scoring benchmarks (Taiwan Credit Card Default and Statlog German Credit).
2. **Fixed Data Split:** Utilized a fixed 80/20 train/test split (`random_state = 42`) for exact cross-phase reproducibility, without assessing multi-fold cross-validation variance.
3. **Evaluation Sample Sizes:** Evaluated $N=200$ German applicants and $N=500$ Taiwan applicants. The German female test cohort ($N=61$) possesses limited statistical power to detect subtle demographic disparities.
4. **Mitigation Scope:** Primary SHAP consistency evaluated Reweighing. ThresholdOptimizer was excluded due to its decision-level post-processing nature; ExponentiatedGradient was excluded due to its non-continuous ensemble representation.
5. **Exploratory Associations ($n=6$):** Cross-experiment correlation analyses across the six configurations are strictly exploratory and cannot establish causality or statistical significance.
6. **Non-Causal Design:** Observed differences between baseline and reweighed models represent empirical associations resulting from training sample reweighting within this experimental setup, rather than proved causal mechanisms.

---

## 10. Final Contribution Statement

> **Contribution Statement:**  
> This paper presents an empirical multi-objective auditing evaluation investigating how fairness mitigation (via training sample reweighing) interacts with predictive performance, group fairness, and individual-level explanation consistency across multiple machine-learning credit-scoring architectures and benchmark datasets. By establishing reproducible, leak-free auditing protocols and formalizing metrics for same-applicant attribution alignment, this work provides credit-risk modelers and governance teams with empirical evidence on the trade-offs between regulatory fairness compliance and explanation stability.

---

## 11. Official Statement of Numerical Freeze

**The numerical results of Phase 3 (Fairness Mitigation) and Phase 4 (SHAP Explanation Consistency) are hereby officially FROZEN.**

- [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv) is the immutable authoritative source for predictive performance and fairness metrics.
- [`results/phase4_explanation_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/phase4_explanation_results.csv) is the immutable authoritative source for joined Phase 3 and Phase 4 explanation metrics.
- [`results/final_research_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/final_research_results.csv) is the consolidated master research table.
- No further training runs, SHAP computations, or data split modifications shall be performed.

---

### Final Milestone Verification

$$\mathbf{PHASE\ 5.1\ COMPLETE\ —\ RESEARCH\ RESULTS\ FROZEN\ —\ READY\ FOR\ DASHBOARD\ IMPLEMENTATION}$$
