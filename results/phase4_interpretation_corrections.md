# FairCredit-AI: Phase 4 Interpretation Corrections
## Scientific Phrasing, Attribution Nuances, and Causal Boundary Guidelines

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Audit Phase:** Phase 4.1 Independent Methodological Audit  
**Date:** October 2026  
**Status:** Approved Corrections  

---

## 1. Executive Summary

This audit evaluated the scientific precision, statistical validity, and non-causal integrity of the analytical claims made in the Phase 4 research documentation. 

While the underlying experimental pipeline, mathematical calculations, and resulting numerical CSV files are 100% sound, four specific claims in the initial report required refinement to avoid unverified causal assertions and ensure rigorous alignment with empirical evidence.

---

## 2. Itemized Interpretation Corrections

### Claim A: Logistic Regression Consistency Mechanism

- **Current Phrasing:**  
  > *"Logistic Regression maintained the highest explanation consistency because linear decision boundaries remain directionally stable."*
- **Methodological Issue:**  
  The word *"because"* asserts a proven causal mathematical mechanism. While linear decision boundaries are indeed global hyperplanes, this benchmark did not execute controlled ablations separating decision boundary rotation, margin rescaling, and parameter regularization effects from the sample weight perturbation.
- **Corrected Scientific Statement:**  
  > *"In this benchmark, Logistic Regression exhibited the highest explanation consistency across both datasets ($\text{Mean Cosine} > 0.998$, $\text{Top-5 Overlap} \ge 0.954$, $\rho \ge 0.983$). This observation is consistent with the theoretical expectation that a single global linear hyperplane undergoes minimal directional re-orientation under sample reweighing, though factors such as L2 regularization and feature collinearity may also influence attribution stability."*

---

### Claim B: Tree Depth and Feature Hierarchy Preservation

- **Current Phrasing:**  
  > *"Shallow constrained trees preserve feature importance hierarchies under reweighting."*
- **Methodological Issue:**  
  The benchmark evaluated XGBoost ($D=4$) and Random Forest ($\text{max\_depth}=\text{None}$), but did **not** perform an explicit parameter sweep over tree depth (e.g., comparing XGBoost at depths 2, 4, 8, 12 or Random Forest with restricted depth). Attributing the observed difference purely to tree depth conflates algorithm families (boosting vs. bagging) with depth constraints.
- **Corrected Scientific Statement:**  
  > *"XGBoost (configured with depth $D=4$) exhibited higher attribution rank correlation ($\rho \approx 0.887 - 0.917$) than Random Forest without depth constraints ($\rho \approx 0.777 - 0.889$). While tree depth constraints and gradient-based regularization may contribute to structural attribution stability, this benchmark did not systematically isolate tree depth from other architectural differences between boosted and bagged tree ensembles."*

---

### Claim C: Subgroup Parity and Disparate Explanation Instability

- **Current Phrasing:**  
  > *"Reweighing did not induce disparate explanation instability."*
- **Methodological Issue:**  
  Asserting that an intervention *"did not induce"* an effect makes a negative causal claim. Furthermore, in the German dataset, the evaluation cohort contains only $N=61$ female applicants, yielding wider bootstrap confidence intervals (e.g., German XGBoost Female Top-5 Overlap 95% CI: $[0.8000, 0.8525]$ vs. Male: $[0.8230, 0.8619]$). Statistical non-significance cannot be equated with proven invariance.
- **Corrected Scientific Statement:**  
  > *"Across both datasets, post-hoc stratification by protected sex revealed no statistically meaningful disparity in explanation consistency between female and male applicants, with substantially overlapping 95% bootstrap confidence intervals across all three metrics. However, given the moderate sample size of the German female test cohort ($N=61$), statistical power to detect subtle subgroup differences is limited, and these findings should be reported as an absence of observed disparity in this sample rather than conclusive evidence of universal group invariance."*

---

### Claim D: Nearest-Neighbor Benchmark as "Proof" of Individual Fidelity

- **Current Phrasing:**  
  > *"Nearest-neighbor similarity proves that stability reflects individual applicant attribution fidelity."*
- **Methodological Issue:**  
  Empirical benchmarking cannot strictly *"prove"* attribution properties; it provides supporting empirical evidence.
- **Corrected Scientific Statement:**  
  > *"Attribution similarity between distinct credit nearest neighbors ($0.135 - 0.819$) was substantially and consistently lower than same-applicant attribution consistency ($0.938 - 0.999$) across all model architectures. This contrast strongly supports the conclusion that the measured explanation stability reflects individual-level applicant attribution fidelity rather than global model insensitivity to sample reweighting."*

---

## 3. Methodological Clarifications for Research Artifacts

### 3.1 Transparent Accounting of the 2,100 Evaluation Records
To eliminate ambiguity regarding why [`results/shap_individual_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/shap_individual_metrics.csv) contains exactly 2,100 records:
1. **German Credit:** All $N=200$ test applicants $\times$ 3 models = **600 records** (100% test cohort coverage).
2. **Taiwan Credit:** A deterministic cohort of the first $N=500$ test applicants $\times$ 3 models = **1,500 records**.
3. **Total:** $600 + 1,500 = 2,100$ individual consistency evaluations.
4. **Computational Rationale for Taiwan Subset:** Explaining deep unconstrained Random Forest trees ($100$ trees, average depth 25–30, thousands of leaf splits) via TreeExplainer is computationally intensive ($O(T L D^2)$). Evaluating 500 applicants required 378 seconds (~6.3 minutes). Evaluating all 6,000 applicants would have exceeded 75 minutes without providing meaningful statistical gain, as $N=500$ already achieves a standard error of $\le 0.002$ and a 95% CI half-width of $\le \pm 0.004$ for cosine similarity.
5. **Zero Selection Leakage:** The 500 Taiwan applicants were chosen strictly by row index order in the stratified test partition prior to explanation computation, completely unconditioned on model predictions, SHAP values, target labels, or demographic attributes.

### 3.2 Formal Identification of the Bootstrap CI Method
In all research publications and presentations:
- The bootstrap confidence intervals reported in Phase 4 must be explicitly described as **Non-Parametric Empirical Percentile Bootstrap Confidence Intervals** ($B = 1,000$ resamples with replacement, evaluated over individual test applicants, utilizing a fixed pseudo-random seed of 42).
