# Section 5: Research Discussion

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Framework:** FairCredit-AI Research Benchmark  

---

### 5.1 Substantive Interpretation of Empirical Findings

This benchmark investigated the empirical interaction between fairness mitigation (Reweighing), predictive performance (ROC-AUC), and explanation consistency (SHAP) across six credit-scoring configurations. 

The empirical findings challenge two common assumptions in the Responsible AI literature:
1. **The Inevitability of Performance Loss:** Fairness mitigation did not systematically degrade credit-scoring accuracy. In four of six configurations, ROC-AUC changed by less than $\pm 0.0020$, and in German Random Forest, fairness improved substantially alongside a $+0.0137$ gain in AUC.
2. **The Inevitability of Explanation Collapse:** Reweighing did not induce severe attribution instability. Across all six configurations, mean SHAP Cosine Similarity remained above $0.938$, indicating that the primary directional drivers of credit risk attribution were preserved after mitigation.

However, the degree of explanation consistency varied substantially across model families, demonstrating that fairness interventions exert **heterogeneous effects** that depend heavily on model architecture.

---

### 5.2 Architectural Determinants of Explanation Consistency

Under the evaluated configurations, explanation consistency followed a strict, identical ordering across both datasets: $\text{Logistic Regression} > \text{XGBoost} > \text{Random Forest}$.

- **Observed Result:** Logistic Regression maintained near-perfect attribution consistency ($\text{Mean Cosine} > 0.998$, $\text{Top-5 Overlap} \ge 0.954$, $\rho \ge 0.983$). Random Forest exhibited the lowest consistency ($\text{Mean Cosine} \approx 0.938 - 0.965$, $\text{Top-5 Overlap} \approx 0.792 - 0.837$, $\rho \approx 0.777 - 0.889$). XGBoost occupied an intermediate tier ($\text{Mean Cosine} \approx 0.976 - 0.979$, $\text{Top-5 Overlap} \approx 0.838 - 0.886$, $\rho \approx 0.887 - 0.917$).
- **Possible Interpretation:** 
  - *Linear Models:* In Logistic Regression, the scoring function is parameterized by a single global weight vector $\mathbf{w}$. Reweighing alters the loss surface via instance weights, but because the decision boundary is constrained to a global linear hyperplane, the resulting coefficient vector undergoes minor proportional adjustment, leaving attribution rankings and angles virtually unchanged.
  - *Tree Ensembles & Splitting Plasticity:* In unconstrained Random Forest models ($\text{max\_depth}=\text{None}$), individual decision trees possess high capacity and sensitivity to training data perturbations. Reweighing alters node split criteria across hundreds of deep decision paths, resulting in structural tree modifications. While aggregate predictions remain stable, the localized attributions assigned to correlated financial features undergo rank shifts.
  - *Gradient Boosting Regularization:* XGBoost models were configured with shallow depth constraints ($D=4$) and shrinkage ($\eta=0.1$). Constrained tree depth and gradient-based regularized objective functions may limit splitting variance relative to fully grown bagging trees.
- **Future Hypothesis:** Systematically varying tree depth constraints and shrinkage parameters within tree ensembles will reveal a continuous trade-off between architectural flexibility and attribution stability under data reweighting.

---

### 5.3 Fairness Metric Divergence: Independence vs. Separation

A pivotal empirical finding is the divergence between Demographic Parity (Statistical Independence) and Equalized Odds (Error Rate Parity):

- **Observed Result:** In German Logistic Regression, Reweighing improved Demographic Parity Difference dramatically ($0.0577 \to 0.0131$, a $-0.0446$ reduction) and brought Approval Disparate Impact to near-parity ($0.9244 \to 1.0177$). However, Equalized Odds Difference doubled ($0.0500 \to 0.1000$). Similarly, in Taiwan XGBoost, DPD improved from $0.0027$ to $0.0003$, while EOD widened from $0.0306$ to $0.0436$.
- **Possible Interpretation:** Reweighing modifies training sample weights according to the joint distribution of the protected attribute and target. It is intended to reduce dependence between protected-group membership and the target distribution during model training, but the resulting fitted model is not guaranteed to satisfy demographic parity exactly. Reweighing also does not directly enforce Equalized Odds. In the evaluated configurations, changes in demographic-parity-related metrics did not consistently coincide with changes in Equalized Odds, demonstrating the importance of evaluating multiple fairness criteria.
- **Auditing Implication:** Credit model auditors must never evaluate fairness using a single aggregate metric. A mitigation technique that improves demographic parity metrics may simultaneously worsen equal opportunity or equalized odds.

---

### 5.4 Practical Implications for Credit-Risk Model Auditing

The findings of this benchmark have direct operational relevance for financial institutions, credit-risk modelers, and regulatory compliance teams:

1. **Regulatory Adverse Action Disclosures (FCRA & ECOA):**
   Under the Fair Credit Reporting Act (FCRA) and Equal Credit Opportunity Act (ECOA), lenders must provide credit applicants with specific key principal reasons when adverse action is taken. These reasons are typically derived from the top-4 or top-5 negative feature attributions. Lower Top-5 feature overlap indicates greater instability in the set of highly attributed features following reweighting. This suggests that explanation stability should be monitored when feature attributions are subsequently used to derive decision-reason or adverse-action reason codes. For credit risk analysts, evaluating explanation stability is therefore an essential dimension of model validation.
2. **Model Risk Management Frameworks (SR 11-7 / OCC 2011-12):**
   Supervisory guidance on model risk management requires rigorous validation of model conceptual soundness, outcomes analysis, and sensitivity testing. Explanation consistency metrics (Cosine Similarity and Top-5 Overlap) provide banks with a quantitative measure of **attribution robustness** under sample perturbations.
3. **Decision-Support Framing:**
   FairCredit-AI is designed as an **auditing and governance framework**, not an automated underwriting engine. It equips model risk committees with a multi-objective view: comparing candidate mitigated models across predictive performance, regulatory fairness metrics, and explanation stability before deployment.

---

### 5.5 Study Limitations

To maintain scientific integrity, the limitations of this benchmark must be explicitly recognized:

1. **Dataset Scope:** The benchmark evaluated two public credit-scoring datasets (Taiwan Credit Card Default and Statlog German Credit). While standard in the algorithmic fairness literature, these datasets represent specific historical lending regimes that may not reflect contemporary digital credit portfolios.
2. **Evaluation Sample Sizes:** Explanation consistency was evaluated on $N=200$ test applicants for German and $N=500$ deterministic test applicants for Taiwan. While $N=500$ yielded narrow bootstrap confidence intervals ($\pm 0.004$), the German female test cohort ($N=61$) possesses limited statistical power to detect small demographic differences.
3. **Fixed Experimental Splitting:** Experiments utilized a fixed 80/20 train/test split (`random_state = 42`). While this guarantees exact index reproducibility across phases, it does not assess variance across multiple cross-validation folds.
4. **Mitigation Scope:** Primary SHAP consistency analysis was restricted to pre-processing Reweighing. Post-processing ThresholdOptimizer was excluded because it does not modify the underlying scoring function; in-processing ExponentiatedGradient was excluded because its randomized mixture of binary base estimators lacks a continuous scoring explainer representation.
5. **Sample Size for Cross-Experiment Correlations ($n=6$):** Associations between fairness changes, performance changes, and explanation consistency were calculated across six configurations. These statistics are strictly descriptive and exploratory; they cannot support causal inference or claims of statistical significance.
6. **Non-Causal Design:** Observed differences between baseline and reweighed models represent empirical associations resulting from training sample reweighting within this experimental setup, rather than proved causal mechanisms.

---

### 5.6 Future Research Directions

1. **Differentiable In-Processing Explanations:** Developing specialized SHAP explainer representations for in-processing reduction algorithms (such as ExponentiatedGradient) to evaluate attribution consistency under explicit Equalized Odds constraints.
2. **Systematic Hyperparameter & Depth Sweeps:** Executing controlled depth-ablation studies within tree ensembles (e.g., varying Random Forest `max_depth` from 2 to 20) to isolate the exact relationship between tree capacity and attribution stability.
3. **Multi-Cohort & Multi-Fold Cross-Validation:** Expanding the auditing framework across repeated stratified k-fold splits and additional international credit datasets (such as Home Credit Default Risk).
