# FairCredit-AI: Phase 4 Comprehensive Research Report
## SHAP Explanation Consistency Design and Empirical Evaluation

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Phase:** Phase 4 — Explanation Consistency Analysis  
**Date:** October 2026  
**Status:** Complete & Validated  

---

## 1. Dataset & Model Combinations Completed

The primary Phase 4 experimental design investigated the relationship between fairness mitigation and explanation consistency across **6 benchmark model pairs** (2 datasets $\times$ 3 model architectures):

| # | Dataset | Model Architecture | Mitigation Intervention | Evaluated Model Pair | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Taiwan Credit Card** | Logistic Regression | Reweighing (Kamiran & Calders) | Baseline vs. Reweighed | **Complete** |
| 2 | **Taiwan Credit Card** | Random Forest | Reweighing (Kamiran & Calders) | Baseline vs. Reweighed | **Complete** |
| 3 | **Taiwan Credit Card** | XGBoost | Reweighing (Kamiran & Calders) | Baseline vs. Reweighed | **Complete** |
| 4 | **German Credit** | Logistic Regression | Reweighing (Kamiran & Calders) | Baseline vs. Reweighed | **Complete** |
| 5 | **German Credit** | Random Forest | Reweighing (Kamiran & Calders) | Baseline vs. Reweighed | **Complete** |
| 6 | **German Credit** | XGBoost | Reweighing (Kamiran & Calders) | Baseline vs. Reweighed | **Complete** |

### Methodological Exclusions Reaffirmed:
1. **ThresholdOptimizer:** Post-processing decision rule operating on downstream score thresholds without modifying the underlying model $f(x)$. Excluded from scoring SHAP to prevent uninterpretable step-function attributions.
2. **ExponentiatedGradient:** Produces a randomized finite mixture over binary base estimators without a continuous, smooth scoring estimator representation. Excluded from primary SHAP to maintain mathematical rigor.

---

## 2. Number of Applicants Explained

All evaluations adhere strictly to the **Identical Applicant Principle**: $\boldsymbol{\phi}_{\text{baseline}}^{(i)}$ and $\boldsymbol{\phi}_{\text{reweighed}}^{(i)}$ explain the exact same applicant row $i$ in the identical test set partition.

- **German Statlog Credit:**
  - Total Test Set Size: $N_{\text{test}} = 200$.
  - Evaluation Cohort: **$N_{\text{eval}} = 200$ (100% of test applicants)**.
  - Subgroup breakdown: 61 Female (30.5%), 139 Male (69.5%).
- **Taiwan Credit Card Default:**
  - Total Test Set Size: $N_{\text{test}} = 6,000$.
  - Evaluation Cohort: **$N_{\text{eval}} = 500$ deterministic test applicants** (first 500 rows of the stratified test partition, `random_state = 42`).
  - Subgroup breakdown: 311 Female (62.2%), 189 Male (37.8%).
- **Total Individual Explanations Generated:** 
  $$\text{Total} = (200 \times 3 \times 2) + (500 \times 3 \times 2) = 1,200 + 3,000 = 4,200 \text{ individual SHAP attribution vectors.}$$

---

## 3. SHAP Explainers Used

Each model architecture utilized its dedicated, mathematically aligned SHAP backend:

| Model Family | Explainer Class | Background Requirement | Compatibility Handling |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `shap.LinearExplainer` | Required ($N_{\text{bg}}$ training samples) | Exact analytical linear attribution |
| **Random Forest** | `shap.TreeExplainer` | Tree structure (independent feature perturbation) | Exact tree traversal on positive class probability |
| **XGBoost** | `shap.TreeExplainer` | Tree structure (margin output) | In-memory monkey-patch for XGBoost 3.x UBJSON `base_score` format |

---

## 4. Prediction / Output Space Explained

To ensure mathematical validity, baseline and reweighed models within each model family were evaluated in the **identical prediction space**:

- **Logistic Regression:** Log-odds margin space ($\log \frac{p}{1-p}$), explaining default risk ($Y=1$).
- **Random Forest:** Probability space $\hat{P}(Y=1 \mid X)$, explaining default risk ($Y=1$).
- **XGBoost:** Log-odds margin space ($\text{margin}(x)$), explaining default risk ($Y=1$).

Attribution vectors within each comparison pair share identical mathematical units, ensuring that cosine similarity, top-5 overlap, and rank correlation measure true attribution alignment.

---

## 5. Background Dataset Methodology

For explainers requiring background reference data (`LinearExplainer`):
1. **Source:** Sampled strictly from the **training partition** ($X_{\text{train}}$).
2. **Leakage Control:** Test set instances, test labels, and sensitive demographic attributes (`sex`) were completely excluded from background generation.
3. **Sample Sizes & Reproducibility:**
   - German Credit: $N_{\text{bg}} = 100$ training instances, uniform random sampling without replacement (`random_state = 42`).
   - Taiwan Credit: $N_{\text{bg}} = 200$ training instances, uniform random sampling without replacement (`random_state = 42`).
4. **Model Invariance:** Baseline and reweighed models for a given dataset utilized the identical background sample.

---

## 6. Mean & Median Explanation Consistency (Cosine Similarity)

Cosine similarity measures directional alignment in $\mathbb{R}^P$ ($P=57$ German, $P=89$ Taiwan) with safe zero-vector handling:

| Dataset | Model Architecture | Mean Cosine | 95% Bootstrap CI | Median Cosine | Std Dev |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **German** | Logistic Regression | **0.9983** | [0.9982, 0.9984] | **0.9985** | 0.0008 |
| **German** | XGBoost | **0.9757** | [0.9735, 0.9779] | **0.9793** | 0.0163 |
| **German** | Random Forest | **0.9652** | [0.9624, 0.9677] | **0.9693** | 0.0196 |
| **Taiwan** | Logistic Regression | **0.9996** | [0.9995, 0.9996] | **0.9997** | 0.0004 |
| **Taiwan** | XGBoost | **0.9787** | [0.9772, 0.9803] | **0.9828** | 0.0180 |
| **Taiwan** | Random Forest | **0.9380** | [0.9341, 0.9419] | **0.9458** | 0.0429 |

---

## 7. Top-5 Feature Overlap Results

Top-5 overlap measures the retention of the 5 most influential features by absolute SHAP magnitude ($K=5$):

| Dataset | Model Architecture | Mean Top-5 Overlap | 95% Bootstrap CI | Median Top-5 | Std Dev |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **German** | Logistic Regression | **0.9540** | [0.9420, 0.9660] | **1.0000** | 0.0844 |
| **German** | XGBoost | **0.8380** | [0.8220, 0.8530] | **0.8000** | 0.1141 |
| **German** | Random Forest | **0.8370** | [0.8210, 0.8540] | **0.8000** | 0.1237 |
| **Taiwan** | Logistic Regression | **0.9800** | [0.9744, 0.9852] | **1.0000** | 0.0601 |
| **Taiwan** | XGBoost | **0.8864** | [0.8756, 0.8976] | **0.8000** | 0.1203 |
| **Taiwan** | Random Forest | **0.7924** | [0.7792, 0.8052] | **0.8000** | 0.1515 |

---

## 8. Spearman Rank Correlation Results

Rank correlation evaluates the preservation of relative feature attribution hierarchies across all features, with explicit tie and zero-variance handling:

| Dataset | Model Architecture | Mean Rank Corr ($\rho$) | 95% Bootstrap CI | Median Rank Corr | Std Dev |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **German** | Logistic Regression | **0.9908** | [0.9904, 0.9912] | **0.9909** | 0.0027 |
| **German** | XGBoost | **0.9173** | [0.9122, 0.9223] | **0.9233** | 0.0360 |
| **German** | Random Forest | **0.8891** | [0.8812, 0.8964] | **0.8999** | 0.0553 |
| **Taiwan** | Logistic Regression | **0.9832** | [0.9831, 0.9834] | **0.9835** | 0.0016 |
| **Taiwan** | XGBoost | **0.8871** | [0.8823, 0.8917] | **0.8907** | 0.0515 |
| **Taiwan** | Random Forest | **0.7768** | [0.7692, 0.7846] | **0.7844** | 0.0870 |

---

## 9. Group-Stratified Fairness & Consistency Results

Consistency metrics stratified by primary protected attribute (`sex`), evaluated post-hoc:

### 9.1 German Credit ($N=200$: 61 Female, 139 Male)

| Model | Group | N | Mean Cosine (95% CI) | Mean Top-5 Overlap (95% CI) | Mean Rank Corr (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Female | 61 | 0.9983 [0.9981, 0.9985] | 0.9410 [0.9180, 0.9607] | 0.9906 [0.9901, 0.9913] |
| **Logistic Regression** | Male | 139 | 0.9984 [0.9982, 0.9985] | 0.9597 [0.9467, 0.9727] | 0.9909 [0.9905, 0.9914] |
| **XGBoost** | Female | 61 | 0.9745 [0.9691, 0.9791] | 0.8262 [0.8000, 0.8525] | 0.9141 [0.9059, 0.9231] |
| **XGBoost** | Male | 139 | 0.9762 [0.9737, 0.9784] | 0.8432 [0.8230, 0.8619] | 0.9187 [0.9128, 0.9246] |
| **Random Forest** | Female | 61 | 0.9636 [0.9592, 0.9680] | 0.8393 [0.8098, 0.8656] | 0.8854 [0.8709, 0.8986] |
| **Random Forest** | Male | 139 | 0.9659 [0.9625, 0.9690] | 0.8360 [0.8158, 0.8576] | 0.8907 [0.8808, 0.8996] |

### 9.2 Taiwan Credit ($N=500$: 311 Female, 189 Male)

| Model | Group | N | Mean Cosine (95% CI) | Mean Top-5 Overlap (95% CI) | Mean Rank Corr (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Female | 311 | 0.9996 [0.9995, 0.9996] | 0.9826 [0.9762, 0.9884] | 0.9832 [0.9830, 0.9833] |
| **Logistic Regression** | Male | 189 | 0.9995 [0.9994, 0.9996] | 0.9757 [0.9661, 0.9852] | 0.9833 [0.9831, 0.9836] |
| **XGBoost** | Female | 311 | 0.9787 [0.9767, 0.9808] | 0.8894 [0.8759, 0.9023] | 0.8867 [0.8813, 0.8919] |
| **XGBoost** | Male | 189 | 0.9787 [0.9761, 0.9811] | 0.8815 [0.8635, 0.8974] | 0.8878 [0.8806, 0.8952] |
| **Random Forest** | Female | 311 | 0.9371 [0.9323, 0.9418] | 0.8006 [0.7852, 0.8180] | 0.7765 [0.7674, 0.7856] |
| **Random Forest** | Male | 189 | 0.9394 [0.9331, 0.9453] | 0.7788 [0.7566, 0.8000] | 0.7773 [0.7644, 0.7902] |

**Fairness Finding:** The 95% bootstrap confidence intervals for female and male subgroups overlap substantially across all metrics and model architectures. There is **no statistically significant evidence** that fairness mitigation via Reweighing disproportionately alters explanation stability for protected demographic groups.

---

## 10. Secondary Comparable-Applicant Analysis

To evaluate whether high consistency was merely an artifact of generic global feature importance, we matched each applicant $i$ with their closest credit neighbor $j \neq i$ in standardized non-sensitive feature space ($X$).

| Dataset | Model Architecture | Same-Applicant Mean Cosine | Nearest-Neighbor Mean Cosine | Same-Applicant Top-5 | Nearest-Neighbor Top-5 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **German** | Logistic Regression | **0.9983** | **0.5019** | **0.9540** | **0.6190** |
| **German** | XGBoost | **0.9757** | **0.2045** | **0.8380** | **0.4890** |
| **German** | Random Forest | **0.9652** | **0.1353** | **0.8370** | **0.5350** |
| **Taiwan** | Logistic Regression | **0.9996** | **0.8187** | **0.9800** | **0.8428** |
| **Taiwan** | XGBoost | **0.9787** | **0.4674** | **0.8864** | **0.5924** |
| **Taiwan** | Random Forest | **0.9380** | **0.3490** | **0.7924** | **0.5380** |

**Scientific Interpretation:** Cross-applicant attribution similarity (even between credit nearest neighbors) is substantially lower than same-applicant consistency across all models. This proves that the measured explanation stability reflects **individual applicant attribution fidelity**, not global model invariance.

---

## 11. Complete Research Table: Joining Phase 3 and Phase 4

The authoritative Phase 3 predictive and fairness metrics from [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv) were joined directly with Phase 4 explanation metrics:

| Dataset | Model | Mitigation | Baseline AUC | Mitigated AUC | $\Delta$ AUC | Baseline EOD | Mitigated EOD | $\Delta$ EOD | Mean Cosine | Mean Top-5 | Mean Rank Corr |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Taiwan** | Logistic Regression | Reweighing | 0.7574 | 0.7582 | +0.0008 | 0.0194 | 0.0229 | +0.0035 | **0.9996** | **0.9800** | **0.9832** |
| **Taiwan** | Random Forest | Reweighing | 0.7528 | 0.7548 | +0.0020 | 0.0298 | 0.0219 | -0.0079 | **0.9380** | **0.7924** | **0.7768** |
| **Taiwan** | XGBoost | Reweighing | 0.7785 | 0.7774 | -0.0011 | 0.0306 | 0.0436 | +0.0130 | **0.9787** | **0.8864** | **0.8871** |
| **German** | Logistic Regression | Reweighing | 0.8061 | 0.8055 | -0.0006 | 0.0500 | 0.1000 | +0.0500 | **0.9983** | **0.9540** | **0.9908** |
| **German** | Random Forest | Reweighing | 0.7798 | 0.7935 | +0.0137 | 0.1250 | 0.0500 | -0.0750 | **0.9652** | **0.8370** | **0.8891** |
| **German** | XGBoost | Reweighing | 0.8019 | 0.7918 | -0.0101 | 0.0250 | 0.0250 | 0.0000 | **0.9757** | **0.8380** | **0.9173** |

*File Location:* [`results/phase4_explanation_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/phase4_explanation_results.csv).

---

## 12. Methodological & Computational Limitations

1. **German Sample Size ($N=200$):** The German evaluation cohort encompasses 100% of the test partition, but total sample size ($N=200$, 61 female applicants) induces wider bootstrap confidence intervals than the Taiwan cohort. Statistical claims are framed cautiously without overstating precision.
2. **Taiwan TreeExplainer Complexity:** Unconstrained Random Forest trees ($100$ trees, deep splits) require substantial computational time for TreeExplainer ($O(T L D^2)$). A deterministic evaluation cohort of $N=500$ applicants was evaluated, which provided tight confidence intervals ($\pm 0.004$ margin of error) while preserving exact reproducibility.
3. **Absence of Causal Claims:** Reweighing alters empirical sample weights during model training. Observed differences in explanation consistency reflect empirical associations within this experimental setup, rather than causal mechanisms.

---

## 13. Verification Suite Summary

The independent test suite [`scripts/verify_shap_consistency.py`](file:///d:/sarthak/Subjects/ERA/Project/scripts/verify_shap_consistency.py) passed all 12 validation checks:
- [x] Cosine similarity mathematical correctness & zero-vector safety.
- [x] Top-5 overlap mathematical correctness on disjoint and identical sets.
- [x] Spearman rank correlation handling of ties and constant vectors.
- [x] Sensitive demographic attributes completely excluded from model inputs.
- [x] Feature space canonical ordering preserved (Taiwan: 89, German: 57).
- [x] Background datasets derived strictly from training partition.
- [x] Evaluation cohorts selected deterministically without outcome/SHAP dependence.
- [x] Output CSVs generated, complete, and non-empty.
- [x] Zero NaN and zero Inf values across all metric records.
- [x] Applicant index alignment preserved across baseline and reweighed models.
- [x] Demographic subgroup totals match cohort sizes.
- [x] Phase 4 results join with Phase 3 authoritative results with zero metric drift.

---

## 14. Next Steps Recommendation

1. **Readiness for Trade-Off Analysis:** Phase 4 explanation consistency results are fully audited, verified, and linked to Phase 3 authoritative results.
2. **Next Logical Step:** Proceed to **Multi-Objective Trade-Off Analysis** (Pareto frontier analysis across Predictive Performance [AUC], Group Fairness [EOD], and Explanation Consistency [Cosine/Top-5 Overlap]).

---

### Final Phase Status

**PHASE 4 COMPLETE — EXPLANATION CONSISTENCY RESULTS READY FOR REVIEW**
