# Section 4: Experimental Results

**Project:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Framework:** FairCredit-AI Research Benchmark  

---

### 4.1 Baseline vs. Reweighing Predictive Performance

Predictive performance was evaluated using receiver operating characteristic area under the curve (ROC-AUC) across the six benchmark configurations. Because Reweighing (Kamiran & Calders, 2012) refits continuous base estimators under modified training instance weights, ROC-AUC provides a valid scoring metric for both baseline and mitigated models.

Across all six configurations, predictive performance remained stable following fairness reweighing:

- **German Statlog Credit ($N_{\text{test}} = 200$):**
  - **Logistic Regression:** Baseline AUC was $0.8061$, and Reweighed AUC was $0.8055$ ($\Delta \text{AUC} = -0.0006$, a relative change of $-0.07\%$).
  - **Random Forest:** Baseline AUC was $0.7798$, and Reweighed AUC was $0.7935$ ($\Delta \text{AUC} = +0.0137$, a relative gain of $+1.76\%$).
  - **XGBoost:** Baseline AUC was $0.8019$, and Reweighed AUC was $0.7918$ ($\Delta \text{AUC} = -0.0101$, a relative change of $-1.26\%$).
- **Taiwan Credit Card Default ($N_{\text{test}} = 6,000$):**
  - **Logistic Regression:** Baseline AUC was $0.7574$, and Reweighed AUC was $0.7582$ ($\Delta \text{AUC} = +0.0008$, a relative change of $+0.11\%$).
  - **Random Forest:** Baseline AUC was $0.7528$, and Reweighed AUC was $0.7548$ ($\Delta \text{AUC} = +0.0020$, a relative change of $+0.27\%$).
  - **XGBoost:** Baseline AUC was $0.7785$, and Reweighed AUC was $0.7774$ ($\Delta \text{AUC} = -0.0011$, a relative change of $-0.14\%$).

Across the benchmark, the mean absolute change in ROC-AUC was $|\Delta \text{AUC}| = 0.0047$. Performance shifts exceeded $\pm 0.01$ in only two configurations (German Random Forest and German XGBoost), while the remaining four configurations exhibited shifts of $\le \pm 0.0020$.

---

### 4.2 Group Fairness Changes

Group fairness was evaluated on the primary protected attribute (`sex` in Taiwan; derived `sex` in German). Equalized Odds Difference (EOD) served as the primary equalized error rate metric, supplemented by Demographic Parity Difference (DPD) and Approval Disparate Impact (Approval DI). In accordance with definition standards, lower absolute EOD and DPD values represent greater fairness:

- **German Statlog Credit:**
  - **Random Forest:** Exhibited the largest fairness improvement in the benchmark. Baseline EOD decreased from $0.1250$ to $0.0500$ ($\Delta \text{EOD} = -0.0750$, a relative reduction of $-60.0\%$). Simultaneously, DPD dropped from $0.0620$ to $0.0005$ ($\Delta \text{DPD} = -0.0615$), and Approval DI shifted from $0.9269$ to $0.9994$, achieving near-perfect demographic parity.
  - **XGBoost:** Maintained constant EOD at $0.0250$ for both baseline and reweighed models ($\Delta \text{EOD} = 0.0000$). DPD shifted slightly from $0.0157$ to $0.0209$ ($\Delta \text{DPD} = +0.0052$).
  - **Logistic Regression:** Exhibited metric divergence. While DPD improved substantially from $0.0577$ to $0.0131$ ($\Delta \text{DPD} = -0.0446$) and Approval DI moved closer to parity ($0.9244 \to 1.0177$), EOD doubled from $0.0500$ to $0.1000$ ($\Delta \text{EOD} = +0.0500$).
- **Taiwan Credit Card Default:**
  - **Random Forest:** EOD decreased from $0.0298$ to $0.0219$ ($\Delta \text{EOD} = -0.0079$, a relative reduction of $-26.51\%$). DPD improved from $0.0093$ to $0.0075$ ($\Delta \text{DPD} = -0.0018$).
  - **Logistic Regression:** EOD changed from $0.0194$ to $0.0229$ ($\Delta \text{EOD} = +0.0035$), while DPD remained identical at $0.0069$ ($\Delta \text{DPD} = 0.0000$) and Approval DI remained constant at $1.0079$.
  - **XGBoost:** EOD widened from $0.0306$ to $0.0436$ ($\Delta \text{EOD} = +0.0130$), whereas DPD improved from $0.0027$ to $0.0003$ ($\Delta \text{DPD} = -0.0024$).

---

### 4.3 Explanation Consistency

Explanation consistency between baseline and reweighed models was evaluated on identical test applicants ($N=200$ for German; deterministic first $N=500$ for Taiwan; $N_{\text{total}} = 2,100$ individual evaluations). Three complementary metrics were measured: Cosine Similarity (directional orientation), Top-5 Feature Overlap ($K=5$ salience set stability), and Spearman Rank Correlation ($\rho$, feature hierarchy alignment).

| Dataset | Model Architecture | Mean Cosine (95% CI) | Median Cosine | Mean Top-5 Overlap (95% CI) | Median Top-5 | Mean Spearman $\rho$ (95% CI) | Median Spearman |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **German** | Logistic Regression | 0.9983 [0.9982, 0.9984] | 0.9985 | 0.9540 [0.9420, 0.9660] | 1.0000 | 0.9908 [0.9904, 0.9912] | 0.9909 |
| **German** | XGBoost | 0.9757 [0.9735, 0.9779] | 0.9793 | 0.8380 [0.8220, 0.8530] | 0.8000 | 0.9173 [0.9122, 0.9223] | 0.9233 |
| **German** | Random Forest | 0.9652 [0.9624, 0.9677] | 0.9693 | 0.8370 [0.8210, 0.8540] | 0.8000 | 0.8891 [0.8812, 0.8964] | 0.8999 |
| **Taiwan** | Logistic Regression | 0.9996 [0.9995, 0.9996] | 0.9997 | 0.9800 [0.9744, 0.9852] | 1.0000 | 0.9832 [0.9831, 0.9834] | 0.9835 |
| **Taiwan** | XGBoost | 0.9787 [0.9772, 0.9803] | 0.9828 | 0.8864 [0.8756, 0.8976] | 0.8000 | 0.8871 [0.8823, 0.8917] | 0.8907 |
| **Taiwan** | Random Forest | 0.9380 [0.9341, 0.9419] | 0.9458 | 0.7924 [0.7792, 0.8052] | 0.8000 | 0.7768 [0.7692, 0.7846] | 0.7844 |

Across all six configurations, directional attribution consistency remained high, with mean Cosine Similarity consistently exceeding $0.938$. However, feature hierarchy retention varied markedly across model families: Logistic Regression preserved over $95\%$ of top-5 features, whereas Random Forest exhibited greater attribution plasticity, with top-5 overlap dropping to $79.24\%$ in the Taiwan cohort.

---

### 4.4 Fairness–Performance Trade-offs

The empirical relationship between group fairness change ($\Delta \text{EOD}$) and predictive performance change ($\Delta \text{AUC}$) did not conform to a universal Pareto trade-off:

1. **Simultaneous Fairness and Performance Gain:** In German Random Forest, fairness improved substantially ($\Delta \text{EOD} = -0.0750$) alongside an improvement in predictive performance ($\Delta \text{AUC} = +0.0137$).
2. **Stable Trade-Offs:** In Taiwan Random Forest, fairness improved ($\Delta \text{EOD} = -0.0079$) with stable predictive performance ($\Delta \text{AUC} = +0.0020$). In German XGBoost, fairness remained invariant ($\Delta \text{EOD} = 0.0000$) while AUC declined slightly ($\Delta \text{AUC} = -0.0101$).
3. **Exploratory Correlation ($n=6$):** The exploratory Spearman rank correlation between $\Delta \text{EOD}$ and $\Delta \text{AUC}$ across the six configurations was $\rho = -0.6000$ ($p = 0.2080$). Because $n=6$, this negative association is descriptive rather than statistically conclusive, but it challenges the assumption that fairness improvement must always incur predictive degradation.

---

### 4.5 Fairness–Explanation Relationship

We examined whether configurations with greater fairness improvement experienced reduced explanation consistency:

- In German Random Forest, the largest fairness improvement ($\Delta \text{EOD} = -0.0750$) was accompanied by high explanation consistency (Mean Cosine = $0.9652$, Top-5 Overlap = $0.8370$).
- In Taiwan Random Forest, fairness improvement ($\Delta \text{EOD} = -0.0079$) co-occurred with moderate explanation consistency (Mean Cosine = $0.9380$, Top-5 Overlap = $0.7924$).
- Conversely, in Taiwan Logistic Regression, fairness worsened slightly ($\Delta \text{EOD} = +0.0035$), yet explanation consistency was near-perfect (Mean Cosine = $0.9996$, Top-5 Overlap = $0.9800$).
- **Exploratory Correlation ($n=6$):**
  - $\Delta \text{EOD}$ vs. Mean Cosine: Spearman $\rho = 0.7714$ ($p = 0.0724$).
  - $\Delta \text{EOD}$ vs. Mean Top-5 Overlap: Spearman $\rho = 0.7714$ ($p = 0.0724$).
  - $\Delta \text{EOD}$ vs. Mean Rank Correlation: Spearman $\rho = 0.5429$ ($p = 0.2657$).

Because negative $\Delta \text{EOD}$ indicates fairness improvement, the positive correlation indicates that in this sample, configurations with larger fairness gains (more negative $\Delta \text{EOD}$) tended to exhibit comparatively lower explanation consistency. However, this trend is driven by model architecture differences (specifically Random Forest exhibiting both larger EOD improvements and lower consistency) rather than a general mathematical law.

---

### 4.6 Model Architecture Differences

Under the evaluated configurations, model architectures exhibited distinct consistency tiers:

1. **Logistic Regression:** Consistently occupied the highest tier across both datasets (Mean Cosine $> 0.998$, Top-5 Overlap $\ge 0.954$, Rank Correlation $\ge 0.983$).
2. **XGBoost:** Displayed strong intermediate stability (Mean Cosine $\approx 0.976 - 0.979$, Top-5 Overlap $\approx 0.838 - 0.886$, Rank Correlation $\approx 0.887 - 0.917$).
3. **Random Forest:** Exhibited the lowest consistency tier in both datasets (Mean Cosine $\approx 0.938 - 0.965$, Top-5 Overlap $\approx 0.792 - 0.837$, Rank Correlation $\approx 0.777 - 0.889$).

*Important Methodological Caveat:* These models operate in different explanation output spaces (margin log-odds for LR and XGB; probability space for RF). Comparisons are strictly valid *within* each architecture across baseline and reweighed models.

---

### 4.7 Cross-Dataset Observations

- **Baseline Disparities:** The German dataset exhibited higher baseline unfairness (Baseline EOD up to $0.1250$) than the Taiwan dataset (Baseline EOD $\le 0.0306$). Consequently, Reweighing produced larger absolute metric shifts in the German dataset.
- **Statistical Uncertainty:** The German evaluation cohort ($N=200$, 61 female applicants) yielded wider 95% bootstrap confidence intervals (e.g., German RF Top-5 Overlap CI width $= 0.0330$) than the Taiwan cohort ($N=500$, CI width $= 0.0260$).
- **Consistency Preservation:** Despite substantial demographic differences, the relative ordering of model architectures by explanation consistency was identical across both datasets: $\text{Logistic Regression} > \text{XGBoost} > \text{Random Forest}$.
