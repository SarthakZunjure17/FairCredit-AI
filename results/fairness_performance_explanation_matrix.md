# FairCredit-AI: Fairness × Performance × Explanation Consistency Matrix

**Research Framework:** Empirical synthesis across the 6 baseline-vs-reweighed credit-scoring configurations.

| Dataset | Model Architecture | Group Fairness (EOD) | Predictive Performance (AUC) | Explanation Consistency (SHAP) | Observed Case Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Taiwan** | Logistic Regression | Worsened (Delta: +0.0035) | Stable (Delta: +0.0008) | Very High (Cosine: 0.9996), Top-5: 0.9800, Rank: 0.9832 | Fairness Divergence (EOD Worsened, DPD Stable, Very High Consistency) |
| **Taiwan** | Random Forest | Improved (Delta: -0.0079) | Stable (Delta: +0.0020) | Moderate (Cosine: 0.9380), Top-5: 0.7924, Rank: 0.7768 | Fairness Improved, Stable AUC, Moderate Consistency |
| **Taiwan** | Xgboost | Worsened (Delta: +0.0130) | Stable (Delta: -0.0011) | High (Cosine: 0.9787), Top-5: 0.8864, Rank: 0.8871 | Fairness Worsened, Stable AUC, High Consistency |
| **German** | Logistic Regression | Worsened (Delta: +0.0500) | Stable (Delta: -0.0006) | Very High (Cosine: 0.9983), Top-5: 0.9540, Rank: 0.9908 | Fairness Divergence (EOD Worsened, DPD Improved, Very High Consistency) |
| **German** | Random Forest | Improved (Delta: -0.0750) | Improved (Delta: +0.0137) | High (Cosine: 0.9652), Top-5: 0.8370, Rank: 0.8891 | Configuration-Specific Gain (Fairness Improved, AUC Improved, High Consistency) |
| **German** | Xgboost | Stable (Delta: +0.0000) | Degraded (Delta: -0.0101) | High (Cosine: 0.9757), Top-5: 0.8380, Rank: 0.9173 | Invariant Fairness, Slight AUC Drop, High Consistency |

*Note: Categorical directions are derived directly from empirical metrics without subjective thresholding. Explanations within linear models operate in margin log-odds space; XGBoost operates in margin log-odds space; Random Forest operates in probability space.*
