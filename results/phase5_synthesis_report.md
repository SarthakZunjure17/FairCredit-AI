# FairCredit-AI: Phase 5 Research Synthesis Report
## Multi-Objective Evaluation of Fairness Mitigation, Predictive Performance, and Explanation Consistency

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Phase:** Phase 5 — Research Synthesis  
**Date:** October 2026  
**Status:** Complete & Validated  

---

## 1. Central Research Finding

Based on the empirical evidence across the six locked baseline-versus-reweighing credit-scoring configurations:

> **Central Finding:**  
> **Fairness mitigation through training sample reweighing does not universally degrade predictive accuracy or cause explanation collapse, but it exerts heterogeneous effects on feature attribution hierarchies that are governed primarily by model architecture.**  
>
> 1. Predictive performance remained remarkably resilient across all configurations (mean $|\Delta \text{AUC}| = 0.0047$).  
> 2. Explanation consistency remained high directionally across all models (Mean SHAP Cosine $\ge 0.9380$).  
> 3. However, feature attribution hierarchies exhibited distinct architectural vulnerability: Logistic Regression attributions were virtually invariant ($\text{Top-5 Overlap} \ge 0.954$, $\rho \ge 0.983$), while Random Forest models exhibited notable attribution rank shifts ($\text{Top-5 Overlap} \approx 0.792 - 0.837$, $\rho \approx 0.777 - 0.889$).  
> 4. A universal fairness–explanation trade-off was **not** observed: the largest fairness improvement in the benchmark (German Random Forest, $\Delta \text{EOD} = -0.0750$) was achieved alongside a gain in predictive performance ($\Delta \text{AUC} = +0.0137$) and high explanation consistency (Mean Cosine = $0.9652$).

---

## 2. Six-Experiment Comprehensive Synthesis

The complete empirical results across all six benchmark configurations are summarized below (authoritative data from [`results/final_research_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/final_research_results.csv)):

| Dataset | Model Architecture | Baseline AUC | Reweighed AUC | $\Delta$ AUC | Baseline EOD | Reweighed EOD | $\Delta$ EOD | Mean SHAP Cosine | Mean Top-5 Overlap | Mean Spearman $\rho$ | Observed Profile |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Taiwan** | Logistic Regression | 0.7574 | 0.7582 | +0.0008 | 0.0194 | 0.0229 | +0.0035 | **0.9996** | **0.9800** | **0.9832** | Stable AUC, Invariant Attributions |
| **Taiwan** | Random Forest | 0.7528 | 0.7548 | +0.0020 | 0.0298 | 0.0219 | -0.0079 | **0.9380** | **0.7924** | **0.7768** | Fairness Gain, Rank Plasticity |
| **Taiwan** | XGBoost | 0.7785 | 0.7774 | -0.0011 | 0.0306 | 0.0436 | +0.0130 | **0.9787** | **0.8864** | **0.8871** | Stable AUC, High Consistency |
| **German** | Logistic Regression | 0.8061 | 0.8055 | -0.0006 | 0.0500 | 0.1000 | +0.0500 | **0.9983** | **0.9540** | **0.9908** | Metric Conflict, Invariant Attributions |
| **German** | Random Forest | 0.7798 | 0.7935 | +0.0137 | 0.1250 | 0.0500 | -0.0750 | **0.9652** | **0.8370** | **0.8891** | **Triple Win (Fairness, AUC, Consistency)** |
| **German** | XGBoost | 0.8019 | 0.7918 | -0.0101 | 0.0250 | 0.0250 | 0.0000 | **0.9757** | **0.8380** | **0.9173** | Invariant Fairness, High Consistency |

---

## 3. Predictive Performance Findings

1. **Scoring Validity:** Because Kamiran & Calders Reweighing alters instance weights during training, the resulting estimators yield continuous prediction scores. ROC-AUC is mathematically valid for both baseline and reweighed models.
2. **Minimal Performance Degradation:** The mean absolute change across all six models was $|\Delta \text{AUC}| = 0.0047$. 
3. **Model Resilience:** In four of the six configurations (Taiwan LR, Taiwan RF, Taiwan XGB, and German LR), $|\Delta \text{AUC}| \le 0.0020$. 
4. **Performance Gain:** In German Random Forest, Reweighing increased ROC-AUC from $0.7798$ to $0.7935$ ($+0.0137$, a $+1.76\%$ relative gain), likely because reweighting mitigated class imbalance within underrepresented demographic partitions.
5. **Minor Decline:** In German XGBoost, ROC-AUC decreased from $0.8019$ to $0.7918$ ($-0.0101$, a $-1.26\%$ relative change).

---

## 4. Group Fairness Findings

1. **Demographic Parity Success:** Reweighing universally improved Demographic Parity Difference (DPD) and Approval Disparate Impact (Approval DI) in all configurations with baseline demographic disparity:
   - German Random Forest: DPD dropped from $0.0620$ to $0.0005$ ($\Delta \text{DPD} = -0.0615$), and Approval DI moved from $0.9269$ to $0.9994$.
   - German Logistic Regression: DPD dropped from $0.0577$ to $0.0131$ ($\Delta \text{DPD} = -0.0446$), and Approval DI moved from $0.9244$ to $1.0177$.
   - Taiwan Random Forest: DPD dropped from $0.0093$ to $0.0075$.
   - Taiwan XGBoost: DPD dropped from $0.0027$ to $0.0003$.
2. **Equalized Odds Divergence:** Equalized Odds Difference (EOD) exhibited mixed outcomes:
   - Improved in Random Forest: $-60.0\%$ in German ($\Delta \text{EOD} = -0.0750$) and $-26.5\%$ in Taiwan ($\Delta \text{EOD} = -0.0079$).
   - Invariant in German XGBoost: $\Delta \text{EOD} = 0.0000$.
   - Worsened in Logistic Regression: doubled from $0.0500$ to $0.1000$ in German and increased from $0.0194$ to $0.0229$ in Taiwan.
3. **Mechanistic Explanation:** Reweighing modifies training sample weights according to the joint distribution of the protected attribute and target. It is intended to reduce dependence between protected-group membership and the target distribution during model training, but the resulting fitted model is not guaranteed to satisfy demographic parity exactly. Reweighing also does not directly enforce Equalized Odds. In the evaluated configurations, changes in demographic-parity-related metrics did not consistently coincide with changes in Equalized Odds, demonstrating the importance of evaluating multiple fairness criteria.

---

## 5. Explanation Consistency Findings

1. **High Directional Stability:** Across all six configurations, mean SHAP Cosine Similarity remained between $0.9380$ and $0.9996$, and median Cosine remained between $0.9458$ and $0.9997$. Reweighing preserved overall attribution polarity for identical applicants.
2. **Top-5 Salience Set Retention:** Mean Top-5 Feature Overlap ranged from $0.7924$ to $0.9800$. In tree ensembles, feature attributions exhibited lower overlap, indicating rank shifts among the top-5 most influential credit features following reweighting.
3. **Rank Ordering Alignment:** Mean Spearman Rank Correlation ranged from $0.7768$ (Taiwan RF) to $0.9908$ (German LR), tracking the architectural tiers.
4. **Peer Comparison Validation:** Same-applicant cosine similarity ($0.938 - 0.999$) was substantially higher than credit nearest-neighbor attribution similarity ($0.135 - 0.819$), supporting the conclusion that the metrics capture individual applicant-level attribution fidelity rather than global model insensitivity.

---

## 6. Fairness–Performance Trade-offs

The empirical evidence demonstrates that fairness improvements do **not** require predictive performance degradation:
- **German Random Forest:** $\Delta \text{EOD} = -0.0750$ with $\Delta \text{AUC} = +0.0137$ (Dual Improvement).
- **Taiwan Random Forest:** $\Delta \text{EOD} = -0.0079$ with $\Delta \text{AUC} = +0.0020$ (Dual Improvement).
- **Taiwan XGBoost:** $\Delta \text{EOD} = +0.0130$ with $\Delta \text{AUC} = -0.0011$ (Stable AUC, worsened EOD).
- **Exploratory Correlation ($n=6$):** Spearman rank correlation between $\Delta \text{AUC}$ and $\Delta \text{EOD}$ was $\rho = -0.6000$ ($p = 0.2080$). While not statistically significant due to $n=6$, the negative sign descriptively reflects that fairness gains were associated with stable or improved performance in this benchmark.

---

## 7. Fairness–Explanation Relationship

We investigated whether larger fairness improvements were associated with lower explanation consistency:
- In this sample of $n=6$ experiments:
  - $\Delta \text{EOD}$ vs. Mean Cosine: Spearman $\rho = 0.7714$ ($p = 0.0724$).
  - $\Delta \text{EOD}$ vs. Mean Top-5 Overlap: Spearman $\rho = 0.7714$ ($p = 0.0724$).
  - $\Delta \text{EOD}$ vs. Mean Rank Correlation: Spearman $\rho = 0.5429$ ($p = 0.2657$).
- **Scientific Interpretation:** Because negative $\Delta \text{EOD}$ represents fairness improvement, the positive correlation indicates that models with larger EOD reductions exhibited comparatively lower explanation consistency. However, this association is driven by model architecture: Random Forest models exhibited both the largest fairness improvements and the greatest attribution rank plasticity. There is no evidence of an intrinsic mathematical barrier preventing high explanation consistency in fair models.

---

## 8. Model Architecture Differences

Under the evaluated configurations, model families exhibited consistent behavioral tiers across both datasets:

| Model Architecture | Output Prediction Space | Explanation Consistency Tier | Top-5 Overlap Retention | Feature Rank Retention | EOD Responsiveness |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Margin (log-odds) | **Tier 1: Very High** | $95.4\% - 98.0\%$ | $\rho \ge 0.983$ | Poor / Worsened |
| **XGBoost** | Margin (log-odds) | **Tier 2: High** | $83.8\% - 88.6\%$ | $\rho \approx 0.887 - 0.917$ | Intermediate / Mixed |
| **Random Forest** | Class 1 Probability | **Tier 3: Moderate-High** | $79.2\% - 83.7\%$ | $\rho \approx 0.777 - 0.889$ | **Strongest Improvement** |

*Methodological Rule:* Raw SHAP magnitudes are not directly comparable across architectures because LR and XGB explain margin space while RF explains probability space. Comparisons are valid strictly within each baseline/reweighed model pair.

---

## 9. Dataset Differences

1. **Baseline Disparities:** The German dataset ($N_{\text{total}} = 1,000$) exhibited substantially higher baseline disparity (EOD up to $0.1250$) than the Taiwan dataset ($N_{\text{total}} = 30,000$, EOD $\le 0.0306$). Reweighing produced much larger absolute fairness shifts in German credit scoring.
2. **Evaluation Sample Sizes & Confidence Intervals:**
   - German ($N_{\text{eval}} = 200$, 61 females): 95% bootstrap CI half-widths were $\pm 0.016$ for Top-5 overlap and $\pm 0.008$ for Spearman rank.
   - Taiwan ($N_{\text{eval}} = 500$, 311 females): 95% bootstrap CI half-widths were $\pm 0.011$ for Top-5 overlap and $\pm 0.004$ for Spearman rank.
3. **Consistency Invariance:** Despite substantial demographic differences, the relative ranking of model architectures by explanation consistency was identical across both datasets ($\text{LR} > \text{XGB} > \text{RF}$).

---

## 10. Selected Informative Case Studies

### Case A: The "Triple Win" (German Random Forest)
- **Profile:** Strong fairness improvement ($\Delta \text{EOD} = -0.0750$, $-60\%$; $\Delta \text{DPD} = -0.0615$) + Performance gain ($\Delta \text{AUC} = +0.0137$, $+1.76\%$) + High explanation consistency (Mean Cosine = $0.9652$, Top-5 Overlap = $0.8370$).
- **Significance:** Supports the empirical observation that under specific configurations, fairness mitigation does not inherently require sacrificing accuracy or stability.

### Case B: Fairness Gain with Attribution Rank Plasticity (Taiwan Random Forest)
- **Profile:** Substantial fairness gain ($\Delta \text{EOD} = -0.0079$, $-26.5\%$) + Stable AUC ($\Delta \text{AUC} = +0.0020$) + Moderate explanation consistency (Mean Cosine = $0.9380$, Top-5 Overlap = $0.7924$, Rank Correlation = $0.7768$).
- **Significance:** Highlights that unconstrained bagging tree ensembles can exhibit noticeable top-5 feature shifts (Mean Top-5 Overlap of 0.7924) despite maintaining high directional cosine alignment.

### Case C: Invariant Fairness with High Attribution Stability (German XGBoost)
- **Profile:** Invariant fairness ($\Delta \text{EOD} = 0.0000$) + Minor AUC drop ($\Delta \text{AUC} = -0.0101$) + High explanation consistency (Mean Cosine = $0.9757$, Top-5 Overlap = $0.8380$).
- **Significance:** Demonstrates that sample reweighting can leave error rate parity completely untouched while preserving model explanations.

### Case D: Metric Conflict with Invariant Attributions (German Logistic Regression)
- **Profile:** Fairness divergence (DPD improved by $-0.0446$, but EOD worsened by $+0.0500$) + Stable AUC ($\Delta \text{AUC} = -0.0006$) + Near-perfect explanation consistency (Mean Cosine = $0.9983$, Top-5 Overlap = $0.9540$, Rank Correlation = $0.9908$).
- **Significance:** Illustrates the fundamental mathematical tension between unconditional demographic parity and conditional error-rate balance in linear models.

---

## 11. Practical Implications for Credit Model Governance

1. **Regulatory Adverse Action Notices (FCRA & ECOA):**
   Lenders are legally mandated to provide rejected applicants with principal reasons for adverse action. These reasons are typically derived from top negative feature attributions. Lower Top-5 feature overlap indicates greater instability in the set of highly attributed features following reweighting. This suggests that explanation stability should be monitored when feature attributions are subsequently used to derive decision-reason or adverse-action reason codes. Model risk teams must audit explanation consistency to evaluate attribution stability.
2. **Model Risk Management (SR 11-7 / OCC 2011-12):**
   Under supervisory guidance, model validation must evaluate conceptual soundness and sensitivity to data perturbations. Quantifying attribution consistency (Cosine Similarity and Top-5 Overlap) provides risk committees with a direct metric of attribution robustness.
3. **Framework Positioning:**
   FairCredit-AI is strictly framed as an **auditing and governance decision-support framework** for risk analysts, compliance officers, and model validators, rather than an automated decisioning tool.

---

## 12. Methodological Limitations

1. **Two Datasets:** Evaluated public datasets (Taiwan and German) representing specific historical lending environments.
2. **Fixed Train/Test Split:** Used a fixed 80/20 split (`random_state = 42`) ensuring reproducibility across project phases, but not assessing multi-fold variance.
3. **Evaluation Sample Sizes:** Evaluated $N=200$ German applicants and $N=500$ Taiwan applicants. Small female sample size in German ($N=61$) limits statistical power.
4. **Mitigation Scope:** Primary SHAP consistency evaluated Reweighing. ThresholdOptimizer was excluded due to post-processing decision nature; ExponentiatedGradient was excluded due to non-continuous mixture representations.
5. **Exploratory Cross-Experiment Statistics ($n=6$):** Associations across the six configurations cannot support causal inference or claims of statistical significance.

---

## 13. Paper Contribution Statement

> **Contribution Statement:**  
> This paper presents an empirical multi-objective auditing evaluation investigating how fairness mitigation (via training sample reweighing) interacts with predictive performance, group fairness, and individual-level explanation consistency across multiple machine-learning credit-scoring architectures and benchmark datasets. By establishing reproducible, leak-free auditing protocols and formalizing metrics for same-applicant attribution alignment, this work provides credit-risk modelers and governance teams with empirical evidence on the trade-offs between regulatory fairness compliance and explanation stability.

---

## 14. Final Research Conclusion

Fairness mitigation does not inherently force a trade-off between predictive accuracy and explanation stability. As demonstrated by the German Random Forest configuration, models can simultaneously improve group fairness, enhance predictive performance, and maintain high explanation fidelity. 

However, because fairness interventions were associated with lower Top-5 feature overlap (79% to 84%) in non-linear tree models, credit risk modelers should incorporate explanation consistency auditing into standard model risk management workflows to monitor potential instability in derived decision reasons or adverse-action disclosures.

---

### Phase 5 Synthesis Status

**PHASE 5 COMPLETE — RESEARCH SYNTHESIS READY FOR REVIEW**
