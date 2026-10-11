# Section 6: Research Conclusion

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Framework:** FairCredit-AI Research Benchmark  

---

### Direct Answer to the Core Research Question

> **Research Question:** *“When fairness mitigation is applied to machine-learning credit-scoring models, how does it affect predictive performance, group fairness, and explanation consistency?”*

Based on the empirical findings across the six locked benchmark configurations (Taiwan Credit Card Default and German Statlog Credit datasets evaluated across Logistic Regression, Random Forest, and XGBoost models under Kamiran & Calders Reweighing), the evidence demonstrates that:

1. **Predictive Performance Remained Resilient:**
   Fairness mitigation via Reweighing did **not** induce a severe predictive penalty. In four of the six configurations, ROC-AUC changed by less than $\pm 0.0020$. In German Random Forest, performance improved by $+0.0137$ ($+1.76\%$), while in German XGBoost, performance declined by $-0.0101$ ($-1.26\%$). Across the benchmark, mean absolute change was $|\Delta \text{AUC}| = 0.0047$.
2. **Group Fairness Changes Were Metric-Dependent:**
   Reweighing systematically improved **Demographic Parity** and **Approval Disparate Impact** across all configurations with baseline demographic imbalance (e.g., German Random Forest DPD dropped from $0.0620$ to $0.0005$, and Disparate Impact moved from $0.9269$ to $0.9994$). However, its effect on **Equalized Odds (EOD)** was heterogeneous: EOD improved substantially in Random Forest models ($-60.0\%$ in German, $-26.5\%$ in Taiwan), remained constant in German XGBoost, but widened in Logistic Regression ($0.0500 \to 0.1000$ in German) due to mathematical misalignment between unconditional demographic independence and conditional error-rate separation.
3. **Explanation Consistency Was Preserved Directionally but Varied Structurally:**
   Reweighing did **not** cause an explanation breakdown. Across all six configurations, mean SHAP Cosine Similarity remained high, ranging from $0.9380$ to $0.9996$, confirming that the directional polarity of credit risk attributions was largely preserved for identical applicants. However, attribution rank hierarchies and salience sets exhibited clear architectural dependence:
   - *Logistic Regression:* Near-perfect stability (Mean Cosine $> 0.998$, Top-5 Overlap $\ge 0.954$, Rank Correlation $\ge 0.983$).
   - *XGBoost:* Strong intermediate stability (Mean Cosine $\approx 0.976 - 0.979$, Top-5 Overlap $\approx 0.838 - 0.886$, Rank Correlation $\approx 0.887 - 0.917$).
   - *Random Forest:* Noticeable attribution rank plasticity (Mean Cosine $\approx 0.938 - 0.965$, Top-5 Overlap $\approx 0.792 - 0.837$, Rank Correlation $\approx 0.777 - 0.889$).

---

### Absence of a Universal Fairness–Explanation Trade-Off

The empirical results challenge the hypothesis that fairness mitigation universally or inevitably compromises explanation consistency:
- In **German Random Forest**, the largest fairness improvement in the benchmark ($\Delta \text{EOD} = -0.0750$, $-60\%$ disparity reduction; $\Delta \text{DPD} = -0.0615$) was achieved simultaneously with an **increase in predictive performance** ($\Delta \text{AUC} = +0.0137$) and **high explanation consistency** (Mean Cosine = $0.9652$, Top-5 Overlap = $0.8370$). This empirical configuration demonstrates that fairness, performance, and explanation fidelity can co-exist within non-linear tree models under specific settings.
- While configurations with larger fairness gains in this benchmark exhibited comparatively lower rank correlation (exploratory Spearman $\rho = 0.5429$, $n=6$), this association was driven by architectural characteristics (unconstrained Random Forest models possessing higher splitting plasticity) rather than an intrinsic mathematical trade-off.

---

### Strongest Evidence-Based Takeaway

> **Core Research Finding:**  
> **Fairness mitigation through Reweighing does not inherently degrade predictive accuracy or cause explanation collapse, but its impact on attribution hierarchies is heterogeneous and governed primarily by model architecture.**  
>
> In linear models, attributions remain essentially invariant under reweighting, but Equalized Odds may worsen. In unconstrained tree ensembles, Reweighing can achieve substantial fairness improvements while preserving directional attribution alignment, but lower Top-5 feature overlap indicates greater instability in the set of highly attributed features.  
>
> For financial institutions and credit model auditors, evaluating explanation consistency alongside fairness and accuracy is essential: lower Top-5 feature overlap indicates greater instability in the set of highly attributed features following reweighting, suggesting that explanation stability should be monitored when feature attributions are subsequently used to derive decision-reason or adverse-action reason codes under regulatory frameworks (FCRA/ECOA).
