# FairCredit-AI: SHAP Explanation Consistency Methodology

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Phase:** Phase 4 — Explainability & Consistency Analysis  
**Date:** October 2026  
**Status:** Approved Methodology  

---

## 1. Executive Summary & Research Framework

Phase 4 investigates the foundational research question:
> *"When fairness mitigation is applied to machine-learning credit-scoring models, how does it affect predictive performance, group fairness, and explanation consistency?"*

While Phase 2 established locked baseline models and Phase 3 evaluated fairness interventions, Phase 4 measures whether fairness interventions alter **feature attributions** for identical loan applicants. 

### 1.1 Primary Comparison Pair
The **primary explanation consistency comparison** is strictly defined as:
$$\boldsymbol{\phi}_{\text{baseline}}^{(i)} \quad \text{vs.} \quad \boldsymbol{\phi}_{\text{reweighed}}^{(i)}$$
across the **6 benchmark configurations**:
- **2 Datasets:** Taiwan Credit Card Default ($N_{\text{total}} = 30,000$), Statlog German Credit ($N_{\text{total}} = 1,000$).
- **3 Model Architectures:** Logistic Regression (linear), Random Forest (bagging ensemble), XGBoost (gradient boosted trees).

Reweighing (Kamiran & Calders, 2012) is the primary mitigation method because it applies statistical independence sample weights $W(A, Y)$ strictly during training, producing a genuinely refitted scoring model $f_{\text{reweighed}}(X)$ with continuous prediction capability over the exact same feature space.

---

## 2. Methodological Exclusions & Scientific Rationale

### 2.1 Exclusion of ThresholdOptimizer
- **Nature of Method:** Post-processing decision rule (Hardt et al., 2016).
- **Rationale for Exclusion:** ThresholdOptimizer operates downstream of scoring. It learns group-specific decision thresholds $\tau_a$ applied to baseline probabilities $\hat{P}(Y=1|X) \gtrless \tau_a$. The underlying scoring function $f(X)$ and its gradient/tree structure are completely unmodified. Computing SHAP on ThresholdOptimizer would either redundantly re-explain the unchanged baseline scoring function or attempt to differentiate a discontinuous step-function. Therefore, ThresholdOptimizer is strictly preserved in Phase 3 decision/fairness analysis and excluded from Phase 4 scoring SHAP.

### 2.2 Exclusion of ExponentiatedGradient from Primary SHAP
- **Nature of Method:** Reduction framework (Agarwal et al., 2018).
- **Rationale for Exclusion:** In our Fairlearn implementation with `EqualizedOdds()`, ExponentiatedGradient produces a randomized mixture/distribution over multiple binary base classifiers ($\sum_t \beta_t h_t(X)$) rather than a single smooth scoring estimator. Standard TreeExplainer or LinearExplainer backends do not natively represent randomized mixtures of binary threshold estimators without heuristic ensemble averaging. To maintain methodological rigor and prevent unverified mathematical approximations, ExponentiatedGradient is excluded from the primary SHAP consistency evaluation.

---

## 3. Prediction Output Space Specification

A critical methodological rule is that **baseline and mitigated models must be compared in the exact same mathematical prediction space**. Comparing attributions from probability space against attributions from log-odds margin space produces spurious directional mismatches.

| Model Architecture | Explainer Backend | Output Prediction Space | Explanation Target | Mathematical Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `shap.LinearExplainer` | **Log-odds (margin space)** | Default risk ($Y=1$) | LinearExplainer computes exact additive attributions in margin space: $\sum_j \phi_j(x) = \mathbf{w}^T \mathbf{x} - \mathbb{E}[\mathbf{w}^T X]$. |
| **Random Forest** | `shap.TreeExplainer` | **Probability space $P(Y=1\|X)$** | Default risk ($Y=1$) | Scikit-learn Random Forest outputs class vote probabilities. TreeExplainer computes attributions directly on class 1 probability: $\sum_j \phi_j(x) = \hat{P}(Y=1\|x) - \mathbb{E}[\hat{P}(Y=1)]$. |
| **XGBoost** | `shap.TreeExplainer` | **Log-odds (margin space)** | Default risk ($Y=1$) | TreeExplainer on XGBoost trees explains the raw margin/logit score: $\sum_j \phi_j(x) = \text{margin}(x) - \mathbb{E}[\text{margin}(X)]$. |

**Consistency Guarantee:** For every dataset/model configuration, $\boldsymbol{\phi}_{\text{baseline}}^{(i)}$ and $\boldsymbol{\phi}_{\text{reweighed}}^{(i)}$ are computed using the **identical explainer configuration and identical output space**.

---

## 4. Feature Space & Canonical Alignment

To ensure that feature attribution vectors $\boldsymbol{\phi} \in \mathbb{R}^P$ represent identical feature dimensions in the exact same ordering:
1. **Canonical ColumnTransformer:** Preprocessing pipelines for baseline and reweighed models share identical categorical OneHotEncoder configurations and numerical transformers.
2. **Feature Dimensions:**
   - **Taiwan:** $P = 89$ features (expanded one-hot categorical features for education, marriage, payment history + continuous financial features).
   - **German:** $P = 57$ features (expanded one-hot categorical features for credit history, purpose, savings, employment, etc. + continuous financial features).
3. **Protected Demographic Exclusion:** Sensitive sex (`sex` in Taiwan, `derived_sex` in German) is **strictly excluded** from feature matrix $X$ and preprocessor inputs. It never enters the model or explainer.

---

## 5. Background / Reference Dataset Policy

Certain SHAP explainers (notably `LinearExplainer`) require a background/reference dataset to estimate expected baseline feature values $\mathbb{E}[X]$:
1. **Source:** Background samples are drawn **strictly from training data** ($X_{\text{train}}$). Test set instances and test labels are never accessed.
2. **Demographic Blindness:** Background sampling is performed without knowledge of sensitive attributes.
3. **Sampling Policy & Reproducibility:**
   - **German:** $N_{\text{bg}} = 100$ training instances, sampled uniformly without replacement using fixed `random_state = 42`.
   - **Taiwan:** $N_{\text{bg}} = 200$ training instances, sampled uniformly without replacement using fixed `random_state = 42`.
4. **Model Invariance:** Baseline and reweighed models for a given dataset share the exact same background dataset.

---

## 6. Evaluation Cohort Selection

1. **Applicant Identity Principle:** The primary explanation stability experiment compares $\boldsymbol{\phi}_{\text{baseline}}^{(i)}$ against $\boldsymbol{\phi}_{\text{reweighed}}^{(i)}$ for the **exact same applicant $i$**.
2. **Cohort Sizes:**
   - **German Statlog Credit:** All $N_{\text{eval}} = 200$ test applicants (100% of the test partition).
   - **Taiwan Credit:** A deterministic cohort of the first $N_{\text{eval}} = 500$ test applicants from the stratified test split (`random_state = 42`).
3. **Deterministic Selection:** Selection is performed purely on row indexing prior to explanation generation, completely independent of model predictions, labels, or SHAP outcomes.
4. **Statistical Power:** $N_{\text{eval}} = 500$ yields an estimated standard error of the mean for cosine similarity of $\le 0.008$, providing a 95% bootstrap confidence interval width of less than $\pm 0.016$.

---

## 7. Explanation Consistency Metrics

### 7.1 Cosine Similarity (Attribution Orientation)
Measures the directional alignment of the attribution vectors in $\mathbb{R}^P$:
$$\text{Cosine}(\boldsymbol{\phi}_b, \boldsymbol{\phi}_m) = \frac{\boldsymbol{\phi}_b \cdot \boldsymbol{\phi}_m}{\|\boldsymbol{\phi}_b\|_2 \|\boldsymbol{\phi}_m\|_2}$$
**Zero-Vector Protection:**
- If $\|\boldsymbol{\phi}_b\|_2 = 0$ and $\|\boldsymbol{\phi}_m\|_2 = 0$: $\text{Cosine} = 1.0$ (identical zero attribution).
- If exactly one vector has zero norm: $\text{Cosine} = 0.0$ (orthogonal).
- Numerical values are strictly clipped to $[-1.0, 1.0]$.

### 7.2 Top-5 Feature Overlap (Salience Set Stability)
Identifies whether the top-5 most influential features by absolute magnitude are preserved after mitigation:
$$\text{Top-5 Overlap} = \frac{|\text{Top5}(\boldsymbol{\phi}_b) \cap \text{Top5}(\boldsymbol{\phi}_m)|}{5}$$
where $\text{Top5}(\boldsymbol{\phi})$ denotes the indices of the 5 features with largest $|\phi_j|$. Metric lies in $\{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$.

### 7.3 Spearman Rank Correlation (Attribution Ordering)
Measures the preservation of relative feature importance rankings across all $P$ features:
$$\rho = \text{SpearmanRank}(\boldsymbol{\phi}_b, \boldsymbol{\phi}_m)$$
- Handles ties explicitly via average ranks.
- If both vectors have zero variance: $\rho = 1.0$.
- If exactly one vector has zero variance: $\rho = 0.0$.

---

## 8. Group-Stratified Fairness & Uncertainty Analysis

1. **Post-Hoc Stratification:** Sensitive demographic attributes (`sex`) are joined with the individual consistency metrics **after** SHAP values are calculated.
2. **Subgroups Analyzed:**
   - Taiwan: Female vs. Male.
   - German: Female vs. Male (derived from personal status).
3. **Statistical Uncertainty:** For all summary metrics, non-parametric bootstrap resampling ($B = 1,000$ iterations, seed 42) provides empirical 95% confidence intervals.
4. **Scientific Claim Policy:** Differences between subgroups are reported as empirical associations. Causal claims regarding mitigation-induced disparity in explanation stability are strictly avoided.
