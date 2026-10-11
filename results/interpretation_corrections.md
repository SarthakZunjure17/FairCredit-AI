# Scientific Interpretation Audit & Corrections Report

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Audit Component:** Sections 3, 4, 5, 12, 13 — Research Claims, Terminology, Causal Language & Case Studies  
**Auditor:** Lead ML Data Engineer & Research Methodology Auditor  
**Date:** October 2026  
**Status:** **MANDATORY TEXTUAL & CONCEPTUAL CORRECTIONS IDENTIFIED**

---

## 1. Executive Summary

Scientific validity requires that research conclusions be strictly grounded in empirical evidence, free of unsupported causal claims, and precise in technical terminology.

This audit reviews all narrative interpretations in Phase 3 documentation and enforces necessary corrections regarding:
1. Decision-level vs. representation-level mitigation (ThresholdOptimizer).
2. Optimization constraints vs. secondary observed effects (Exponentiated Gradient).
3. Elimination of unsupported causal claims (e.g., attributing AUC drops to calibration).
4. Correction of over-generalized claims to configuration-specific empirical facts.
5. In-depth analysis of three critical empirical case studies.

---

## 2. ThresholdOptimizer: Decision-Level vs. Scoring-Level Mitigation

### 2.1 The Distinction
Fairlearn's `ThresholdOptimizer` (Hardt et al., 2016) is a **post-processing, decision-level intervention**.

- **Mechanics:** It takes the predicted probability scores from a prefit baseline model and selects group-specific classification thresholds $\tau_a$ (potentially with randomized interpolation between two thresholds) to satisfy equalized odds or demographic parity constraints.
- **Invariance of the Underlying Model:** ThresholdOptimizer does **not** modify the underlying model's weights, tree structures, decision boundaries, or learned risk score function $f(x) = P(Y=1 \mid X=x)$.
- **Applicant Ranking Preservation:** Within any given demographic group $a$, applicant ranking is identical to the baseline model.

### 2.2 Crucial Implication for Later SHAP Analysis
- **DO NOT claim ThresholdOptimizer alters model explanations or SHAP values.**
- If SHAP values are computed on the underlying model's scoring function, the explanations before and after ThresholdOptimizer are **trivially identical** because the model function $f(x)$ was not retrained.
- ThresholdOptimizer changes only the final binary grant/deny decision $\hat{Y} \in \{0, 1\}$.
- **Research Protocol Decision:** Later SHAP explanation consistency analysis must focus on mitigation paradigms that actually modify the underlying predictive scoring function: **Pre-processing (Reweighing)** and **In-processing (Exponentiated Gradient)**.

---

## 3. Exponentiated Gradient: Terminology & Constraint Verification

### 3.1 Optimization Constraint
In [`src/modeling/mitigation.py`](file:///d:/sarthak/Subjects/ERA/Project/src/modeling/mitigation.py), line 118:
```python
exp_grad = ExponentiatedGradient(
    estimator=base_model,
    constraints=EqualizedOdds(),
    eps=eps,
    max_iter=max_iter
)
```
- **Constraint Used:** The algorithm is configured strictly with `EqualizedOdds()`.
- **Correction:** In earlier drafts of `results/mitigation_summary.md` (e.g., section 3.2), the text stated: *"Exponentiated Gradient reduced demographic parity disparity across models..."*
- **Required Revision:** Do **not** describe Exponentiated Gradient as optimizing Demographic Parity. Its objective constraint was `EqualizedOdds()`. Any observed changes in Demographic Parity Difference (DPD) or selection rate are secondary observed empirical side-effects, not the optimization objective.

---

## 4. Removal of Unsupported Causal Claims

### 4.1 "Calibration Loss Caused the AUC Drop"
- **Scientific Flaw:** ROC-AUC is a pure rank-order discrimination metric (evaluating whether a randomly selected positive instance is ranked above a randomly selected negative instance). It is **invariant** to strictly monotonic probability calibration transformations. A model can be severely uncalibrated while retaining a perfect ROC-AUC of 1.0.
- **Audit Rule:** Unless explicit calibration curve analysis (e.g. Expected Calibration Error / Brier score decomposition) is conducted, attributing an AUC reduction to "calibration loss" is scientifically invalid.
- **Approved Replacement Language:**
  > *"Exponentiated Gradient and ThresholdOptimizer exhibited substantially lower calculated ROC-AUC values in these experiments; investigation confirms that this reduction is primarily an artifact of evaluating coarse-step and binary decision rules using `roc_auc_score`, which evaluates Balanced Accuracy rather than continuous ranking discrimination."*

---

## 5. Review & Correction of Over-Generalized Statements

| Disallowed Generalization | Scientific Reality from 18 Experiments | Correct Configuration-Specific Statement |
| :--- | :--- | :--- |
| **"Reweighing improved fairness."** | In Taiwan XGBoost, Reweighing **worsened** EOD from $0.0306 \to 0.0436$. In German LR, it **worsened** EOD from $0.0500 \to 0.1000$. EOD improved in only 2 of 6 configurations. | *"Reweighing improved Equalized Odds in German Random Forest and Taiwan Random Forest, but worsened Equalized Odds in German Logistic Regression and Taiwan XGBoost."* |
| **"ThresholdOptimizer improved fairness."** | ThresholdOptimizer **worsened** test EOD in 3 of 6 configurations (Taiwan LR, Taiwan XGB, German XGB) due to calibration shift on low-disparity baselines. | *"ThresholdOptimizer substantially reduced EOD on German Random Forest (where baseline disparity was high), but increased test-set EOD on configurations with low baseline disparity."* |
| **"XGBoost is fair."** | In German baseline, XGBoost had low sex disparity (EOD = 0.0250) but high age disparity (EOD = 0.1000). In Taiwan, XGBoost had age EOD = 0.1383. | *"In these experiments, XGBoost exhibited low baseline disparity on sex across both datasets, but demonstrated substantial baseline disparity across age groups."* |
| **"German RF + Reweighing is the best method."** | Reweighing worked well for German RF, but performed poorly on German LR and Taiwan XGB. | *"For German Random Forest, Reweighing achieved joint improvements across fairness metrics without loss of accuracy in this test partition."* |
| **"Fairness mitigation reduces performance."** | Performance effects were heterogeneous: accuracy improved in 5 runs, decreased in 7 runs, and remained unchanged in 6 runs. | *"Fairness mitigation produced mixed, configuration-dependent performance effects across the evaluated models."* |

---

## 6. Deep-Dive Audit of Specific Case Studies

### 6.1 Case Study 1: German Random Forest + Reweighing
- **The Discrepancy Found:**
  - In earlier markdown summary text, it was claimed that Accuracy increased from $0.7800 \to 0.7950$.
  - **Independent Verification:** The verified master results in [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv) and subgroup confusion matrices in [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv) confirm:
    $$\text{TP} + \text{TN} = (8 + 38) + (18 + 92) = 156 \implies \text{Accuracy} = \frac{156}{200} = \mathbf{0.7800} \quad (\Delta\text{Acc} = \mathbf{0.0000})$$
  - The true empirical values are:
    - **Baseline:** $\text{Accuracy} = 0.7800$, $\text{ROC-AUC} = 0.7798$, $\text{EOD} = 0.1250$, $\text{DPD} = 0.0620$, $\text{Approval DI} = 0.9269$ ($\text{dist} = 0.0731$).
    - **Mitigated:** $\text{Accuracy} = 0.7800$ ($\Delta = 0.0000$), $\text{ROC-AUC} = 0.7935$ ($\Delta = +0.0137$), $\text{EOD} = 0.0500$ ($\Delta = -0.0750$), $\text{DPD} = 0.0005$ ($\Delta = -0.0615$), $\text{Approval DI} = 0.9994$ ($\text{dist} = 0.0006$).
  - **Scientific Interpretation:** Even with Accuracy remaining unchanged at $0.7800$, this remains a remarkable result: continuous ROC-AUC increased by $0.0137$ while EOD was cut by $60\%$ and Approval DI reached near-perfect parity ($0.9994$).

### 6.2 Case Study 2: German Random Forest + ThresholdOptimizer
- **Empirical Metrics:**
  - **Baseline:** $\text{Acc} = 0.7800$, $\text{EOD} = 0.1250$, $\text{DPD} = 0.0620$, $\text{Approval DI} = 0.9269$.
  - **Mitigated:** $\text{Acc} = 0.7550$ ($\Delta\text{Acc} = -0.0250$), $\text{EOD} = 0.0251$ ($\Delta\text{EOD} = -0.0999$, a **$79.9\%$ drop in disparity**), $\text{DPD} = 0.0393$ ($\Delta\text{DPD} = -0.0227$), $\text{Approval DI} = 0.9495$.
  - **Reported AUC:** $0.6774$ ($\Delta = -0.1024$).
- **Scientific Interpretation:**
  - Demonstrates a genuine, stark fairness-performance trade-off in accuracy: achieving near-parity in EOD required accepting a $2.5\%$ drop in classification accuracy ($0.780 \to 0.755$).
  - However, the $-0.1024$ drop in ROC-AUC is an artifact of computing Balanced Accuracy on binary predictions. The underlying model's ranking capability was unchanged.

### 6.3 Case Study 3: Taiwan XGBoost + Reweighing (The Fairness Criterion Conflict)
- **Empirical Metrics:**
  - **Baseline:** $\text{Acc} = 0.8193$, $\text{ROC-AUC} = 0.7785$, $\text{EOD} = 0.0306$, $\text{DPD} = 0.0027$, $\text{Approval DI} = 1.0031$.
  - **Mitigated:** $\text{Acc} = 0.8190$ ($\Delta = -0.0003$), $\text{ROC-AUC} = 0.7774$ ($\Delta = -0.0011$).
  - **Fairness Metrics:**
    - $\text{DPD}: 0.0027 \to 0.0003$ ($\Delta = -0.0024$, **88.9% improvement toward parity**).
    - $\text{Approval DI}: 1.0031 \to 1.0003$ ($\Delta\text{dist} = -0.0028$, **moved to within 0.0003 of 1.0**).
    - $\text{EOD}: 0.0306 \to 0.0436$ ($\Delta = \mathbf{+0.0130}$, **worsened by 42.5%**).
    - $\text{EOppD}: 0.0306 \to 0.0436$ ($\Delta = \mathbf{+0.0130}$, **worsened by 42.5%**).
- **Subgroup Mechanism:**
  - Reweighing equalized selection rates between Female ($11.67\%$) and Male ($11.70\%$), driving Demographic Parity to near zero ($0.0003$).
  - To achieve this, Male True Positives dropped from $193 \to 185$, lowering Male TPR from $0.3440 \to 0.3298$, while Female TPR remained virtually constant ($0.3747 \to 0.3734$).
  - This widened the TPR gap from $0.0306 \to 0.0436$, directly inflating Equalized Odds Difference!
- **Scientific Significance:**
  - This is an empirical illustration of the **Impossibility Theorem of Machine Learning Fairness** (Kleinberg et al., 2016; Chouldechova, 2017). When base rates and conditional feature distributions differ across demographic groups, optimizing Demographic Parity (via reweighing) mathematically conflicts with Equalized Odds / Predictive Parity.
  - This finding must be highlighted in the research paper as empirical proof of the trade-off between demographic parity and error-rate parity.
