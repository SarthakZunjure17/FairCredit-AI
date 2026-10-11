# Methodological Audit: Critical ROC-AUC Analysis of Post-Processing & In-Processing Mitigations

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Audit Component:** Section 2 — Critical ROC-AUC Audit  
**Auditor:** Lead ML Data Engineer & Research Methodology Auditor  
**Date:** October 2026  
**Status:** **CRITICAL METHODOLOGICAL ISSUE IDENTIFIED & FLAGGED**

---

## 1. Executive Summary & Audit Objective

The primary objective of this audit is to inspect every mitigation experiment in Phase 3—specifically Fairlearn's `ThresholdOptimizer` and `ExponentiatedGradient`—and determine precisely what data type and value distribution are passed into:
```python
roc_auc_score(y_true, y_score)
```
Specifically, we evaluated whether `y_score` is:
- **(A)** A valid continuous probability/ranking score $\in [0, 1]$, OR
- **(B)** A discrete/binary postprocessed prediction $\in \{0, 1\}$.

### Audit Verdict on ROC-AUC Calculation
- **ThresholdOptimizer:** **100% BINARY PREDICTIONS PASSED TO `roc_auc_score`.** All 6 experiments passed `y_pred.astype(float)` into `roc_auc_score`.
- **Exponentiated Gradient:** **4 OUT OF 6 EXPERIMENTS PASSED 100% BINARY PREDICTIONS.** In German RF, German XGB, Taiwan RF, and Taiwan XGB, Fairlearn's solver converged to a single active base predictor ($w_0 = 1.0$), causing `_pmf_predict` to return degenerate binary predictions $\{0.0, 1.0\}$. The remaining 2 experiments (Logistic Regression) returned a 4-value discrete step function.
- **Scientific Impact:** **Comparing these values against baseline continuous ROC-AUC is methodologically invalid.** It generates an artificial, illusory performance drop of $\Delta\text{AUC} \approx -0.07$ to $-0.12$.

---

## 2. ThresholdOptimizer Implementation Trace

### 2.1 The Code Path
In [`src/modeling/train_mitigation.py`](file:///d:/sarthak/Subjects/ERA/Project/src/modeling/train_mitigation.py), lines 191–201:
```python
elif mit_name == "threshold_optimizer":
    model_obj = fit_threshold_optimizer(
        fitted_baseline_pipeline=base_pipeline,
        X_train=X_train,
        y_train=y_train,
        sens_train=sens_train["sex"],
        constraints="equalized_odds"
    )
    y_pred = model_obj.predict(X_test, sensitive_features=sens_test["sex"])
    # ThresholdOptimizer applies a discrete thresholding rule; AUC evaluated on decision rule
    y_prob = y_pred.astype(float)
```
And in lines 220–225:
```python
perf = evaluate_performance(y_test, y_pred, y_prob)
```
Inside [`src/modeling/evaluation.py`](file:///d:/sarthak/Subjects/ERA/Project/src/modeling/evaluation.py), line 66:
```python
auc = float(roc_auc_score(y_true_arr, y_prob_arr))
```

### 2.2 Mathematical Identity: Binary ROC-AUC = Balanced Accuracy
In classification theory, the Receiver Operating Characteristic (ROC) curve evaluates true positive rate (TPR) against false positive rate (FPR) across an infinite or large continuum of discrimination thresholds $\theta \in (-\infty, \infty)$.

When `y_score` is a binary vector $\{0, 1\}$, there is only a single interior threshold $\theta \in (0, 1)$. The ROC curve collapses to three points:
1. $(0, 0)$ corresponding to threshold $\theta > 1.0$
2. $(\text{FPR}, \text{TPR})$ corresponding to threshold $\theta \in (0, 1)$
3. $(1, 1)$ corresponding to threshold $\theta \le 0.0$

The trapezoidal integration of the area under this two-segment piecewise curve is algebraically:
$$\text{Area} = \frac{1}{2}\cdot\text{FPR}\cdot\text{TPR} + \frac{1}{2}(1 - \text{FPR})(1 + \text{TPR}) - \frac{1}{2}\text{FPR}\cdot\text{TPR}$$
$$\text{ROC-AUC}_{\text{binary}} = \frac{\text{TPR} + (1 - \text{FPR})}{2} = \frac{\text{TPR} + \text{TNR}}{2} = \text{Balanced Accuracy}$$

### 2.3 Empirical Proof across All 6 ThresholdOptimizer Experiments
The table below compares the reported `mitigated_roc_auc` in [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv) against the exact Balanced Accuracy computed from the subgroup confusion matrices in [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv):

| Dataset | Model | Reported Mitigated AUC | Test TPR | Test FPR | Calculated Balanced Acc: $\frac{\text{TPR} + (1-\text{FPR})}{2}$ | Exact Match? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Taiwan** | Logistic Regression | **0.6598** | 0.3828 | 0.0631 | $\frac{0.3828 + 0.9369}{2} = \mathbf{0.6598}$ | **YES ($< 10^{-4}$)** |
| **Taiwan** | Random Forest | **0.6821** | 0.4484 | 0.0841 | $\frac{0.4484 + 0.9159}{2} = \mathbf{0.6821}$ | **YES ($< 10^{-4}$)** |
| **Taiwan** | XGBoost | **0.6597** | 0.3753 | 0.0559 | $\frac{0.3753 + 0.9441}{2} = \mathbf{0.6597}$ | **YES ($< 10^{-4}$)** |
| **German** | Logistic Regression | **0.6988** | 0.5333 | 0.1357 | $\frac{0.5333 + 0.8643}{2} = \mathbf{0.6988}$ | **YES ($< 10^{-4}$)** |
| **German** | Random Forest | **0.6774** | 0.4833 | 0.1286 | $\frac{0.4833 + 0.8714}{2} = \mathbf{0.6774}$ | **YES ($< 10^{-4}$)** |
| **German** | XGBoost | **0.7119** | 0.5167 | 0.0929 | $\frac{0.5167 + 0.9071}{2} = \mathbf{0.7119}$ | **YES ($< 10^{-4}$)** |

**Conclusion:** In every single ThresholdOptimizer experiment, the value entered into the master table as `mitigated_roc_auc` is **Balanced Accuracy**, not continuous ROC-AUC.

### 2.4 Why this Comparison with Baseline is Methodologically Invalid
1. **Apples-to-Oranges Comparison:**
   - In baseline experiments, `roc_auc_score` evaluated continuous predicted probabilities `pipeline.predict_proba(X_test)[:, 1]` spanning thousands of thresholds (e.g. Baseline German LR AUC = 0.8061).
   - In ThresholdOptimizer, `roc_auc_score` evaluated a binary step vector, yielding Balanced Accuracy (0.6988).
   - Computing $\Delta\text{AUC} = 0.6988 - 0.8061 = -0.1073$ creates the false impression that ThresholdOptimizer destroyed the model's ranking ability by 10.7 percentage points.
2. **Post-Processing Invariance of Scoring Function:**
   - ThresholdOptimizer is a post-processing decision rule. It does **not** alter the underlying continuous scoring function of the baseline model.
   - Within each protected group, applicant ranking is completely preserved.
   - ThresholdOptimizer does not possess a continuous `predict_proba` method by design; it outputs decisions, not scores.
   - Reporting a continuous vs. discrete delta misrepresents post-processing mechanics.

---

## 3. Exponentiated Gradient ROC-AUC Audit

### 3.1 The Code Path
In [`src/modeling/train_mitigation.py`](file:///d:/sarthak/Subjects/ERA/Project/src/modeling/train_mitigation.py), lines 185–188:
```python
y_pred = model_obj.predict(X_test_proc)
# Extract continuous probability from ensemble PMF
pmf = model_obj._pmf_predict(X_test_proc)
y_prob = pmf[:, 1] if pmf.shape[1] > 1 else y_pred.astype(float)
```

### 3.2 Diagnostic Investigation of `_pmf_predict`
Fairlearn's `ExponentiatedGradient` algorithm solves a constrained optimization problem via a zero-sum game between a primal player (choosing a distribution over base estimators) and a dual player (choosing Lagrange multipliers).

The predictor returned is a **randomized ensemble** of deterministic base classifiers $h_1, \dots, h_T$ with weights $w_1, \dots, w_T \ge 0$, $\sum w_t = 1$.
The internal method `_pmf_predict` computes:
$$P(\hat{Y} = 1 \mid X) = \sum_{t=1}^T w_t h_t(X)$$
where each $h_t(X) \in \{0, 1\}$ is a **discrete binary prediction**.

### 3.3 Diagnostic Empirical Results
Our independent diagnostic run on the fitted models revealed:

| Dataset | Base Model | Iterations ($T$) | Active Estimators ($w_t > 0$) | Unique Output Scores in `pmf[:, 1]` | Actual Score Range | Nature of Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Taiwan** | Random Forest | 4 | **1** ($w_0 = 1.0$) | **2** | $\{0.0, 1.0\}$ | **Pure Binary** |
| **Taiwan** | XGBoost | 7 | **1** ($w_0 = 1.0$) | **2** | $\{0.0, 1.0\}$ | **Pure Binary** |
| **Taiwan** | Logistic Regression | 7 | 2 ($w_0=0.08, w_1=0.92$) | 4 | $\{0.0, 0.080, 0.920, 1.0\}$ | **Coarse 4-step** |
| **German** | Random Forest | 1 | **1** ($w_0 = 1.0$) | **2** | $\{0.0, 1.0\}$ | **Pure Binary** |
| **German** | XGBoost | 6 | **1** ($w_0 = 1.0$) | **2** | $\{0.0, 1.0\}$ | **Pure Binary** |
| **German** | Logistic Regression | 7 | 2 ($w_0=0.26, w_1=0.74$) | 4 | $\{0.0, 0.260, 0.740, 1.0\}$ | **Coarse 4-step** |

### 3.4 Findings on Exponentiated Gradient AUC
1. **Tree-Based Models (4 of 6):**
   - In Taiwan RF, Taiwan XGB, German RF, and German XGB, the Fairlearn reduction converged with non-zero weight assigned to **only one binary classifier** ($w_0 = 1.0$).
   - `_pmf_predict` output was identical to `y_pred.astype(float)` $\in \{0.0, 1.0\}$.
   - Consequently, the reported `mitigated_roc_auc` was **again exactly Balanced Accuracy**:
     - German RF ExpGrad: reported AUC = 0.6762 $\equiv$ Balanced Acc = 0.6762
     - German XGB ExpGrad: reported AUC = 0.7012 $\equiv$ Balanced Acc = 0.7012
     - Taiwan RF ExpGrad: reported AUC = 0.6499 $\equiv$ Balanced Acc = 0.6499
     - Taiwan XGB ExpGrad: reported AUC = 0.6528 $\equiv$ Balanced Acc = 0.6528
2. **Logistic Regression Models (2 of 6):**
   - In Taiwan LR and German LR, the ensemble combined 2 binary classifiers, producing a coarse 4-level step function.
   - The resulting ROC curve has only 3 interior operating points, resulting in a coarse-step approximation of AUC (0.6546 in Taiwan, 0.7396 in German).
3. **Core Methodological Takeaway:**
   - Exponentiated Gradient does **not** provide a smooth calibrated ranking function.
   - It is designed to find a randomized policy satisfying group-fairness parity on **binary decisions**.
   - Comparing its coarse-step AUC to the fine-grained continuous baseline ROC-AUC is misleading.

---

## 4. Methodological Comparison Summary

| Dimension | Baseline Models | Pre-Processing: Reweighing | In-Processing: Exponentiated Gradient | Post-Processing: ThresholdOptimizer |
| :--- | :--- | :--- | :--- | :--- |
| **Model Type** | Continuous Scoring Model | Continuous Scoring Model | Randomized Ensemble of Binary Classifiers | Group-Specific Decision Cutoffs on Prefit Model |
| **Output Type** | Continuous $P(Y=1\|X) \in [0, 1]$ | Continuous $P(Y=1\|X) \in [0, 1]$ | Coarse Discrete Distribution ($\le 4$ values or binary) | Discrete Binary Decisions $\hat{Y} \in \{0, 1\}$ |
| **Valid for Continuous ROC-AUC?** | **YES** | **YES** | **NO** (Discrete step-approximation or Balanced Acc) | **NO** (Mathematically equal to Balanced Acc) |
| **Valid for SHAP Local Explanations?** | **YES** | **YES** (Standard scoring function) | Complex (Ensemble of binary estimators) | **NO** (Underlying scoring function is unchanged) |

---

## 5. Required Methodological Resolution Before Paper / Phase 4

To ensure scientific defensibility:
1. **Explicit Designation:** In all tables and publications, clearly mark ThresholdOptimizer ROC-AUC as `N/A (Decision-Level Rule)` or report it explicitly in a dedicated column: `balanced_accuracy`.
2. **Underlying Model Scoring Invariance:** Report that ThresholdOptimizer maintains the underlying model's ROC-AUC unchanged because post-processing alters the decision threshold, not applicant risk scores.
3. **Exponentiated Gradient Score Qualification:** Clearly disclose that Exponentiated Gradient returns a distribution over binary base classifiers rather than a continuous score.
4. **Halt Condition:** **Do not proceed to SHAP analysis until this metric reporting policy is formally ratified.**
