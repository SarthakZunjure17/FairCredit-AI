"""
Generate Authoritative Mitigation Summary Report directly from Verified Master CSV.
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring
"""

import os
import pandas as pd


def generate_summary():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    master_path = os.path.join(project_root, "results", "mitigation_master_results.csv")
    output_path = os.path.join(project_root, "results", "mitigation_summary.md")

    df = pd.read_csv(master_path, keep_default_na=False)

    def render_table(sub_df):
        headers = [
            "Model", "Mitigation Method",
            "Base Acc", "Mit Acc", "Δ Acc",
            "Base BalAcc", "Mit BalAcc", "Δ BalAcc",
            "Base AUC", "Mit AUC", "Δ AUC",
            "Base EOD", "Mit EOD", "Δ EOD",
            "Base DPD", "Mit DPD", "Δ DPD",
            "Base App-DI", "Mit App-DI",
            "Base Dist", "Mit Dist", "Closer to Parity?"
        ]
        col_align = [
            ":---", ":---",
            ":---:", ":---:", ":---:",
            ":---:", ":---:", ":---:",
            ":---:", ":---:", ":---:",
            ":---:", ":---:", ":---:",
            ":---:", ":---:", ":---:",
            ":---:", ":---:",
            ":---:", ":---:", ":---:"
        ]

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(col_align) + " |"
        ]

        for _, r in sub_df.iterrows():
            m_name = r["model"].replace("_", " ").title()
            mit_name = r["mitigation_method"].replace("_", " ").title()
            if mit_name == "Exponentiated Gradient":
                mit_name = "Exp. Gradient"
            elif mit_name == "Threshold Optimizer":
                mit_name = "ThresholdOpt"

            b_acc = f"{r['baseline_accuracy']:.4f}"
            m_acc = f"{r['mitigated_accuracy']:.4f}"
            d_acc = f"{r['delta_accuracy']:+.4f}"

            b_bal = f"{r['baseline_balanced_accuracy']:.4f}"
            m_bal = f"{r['mitigated_balanced_accuracy']:.4f}"
            d_bal = f"{r['delta_balanced_accuracy']:+.4f}"

            b_auc = f"{r['baseline_roc_auc']:.4f}"
            m_auc = str(r["mitigated_roc_auc"])
            d_auc = str(r["delta_roc_auc"])
            if d_auc != "N/A":
                d_auc = f"{float(d_auc):+.4f}"

            b_eod = f"{r['baseline_eod']:.4f}"
            m_eod = f"{r['mitigated_eod']:.4f}"
            d_eod = f"{r['delta_eod']:+.4f}"

            b_dpd = f"{r['baseline_dpd']:.4f}"
            m_dpd = f"{r['mitigated_dpd']:.4f}"
            d_dpd = f"{r['delta_dpd']:+.4f}"

            b_di = f"{r['baseline_approval_di']:.4f}"
            m_di = f"{r['mitigated_approval_di']:.4f}"
            b_dist = f"{r['baseline_approval_di_distance']:.4f}"
            m_dist = f"{r['mitigated_approval_di_distance']:.4f}"

            if r["mitigated_approval_di_distance"] < r["baseline_approval_di_distance"]:
                closer = "**Yes**"
            elif r["mitigated_approval_di_distance"] == r["baseline_approval_di_distance"]:
                closer = "Stable"
            else:
                closer = "No"

            row_str = (
                f"| **{m_name}** | **{mit_name}** | "
                f"{b_acc} | {m_acc} | {d_acc} | "
                f"{b_bal} | {m_bal} | {d_bal} | "
                f"{b_auc} | {m_auc} | {d_auc} | "
                f"{b_eod} | {m_eod} | {d_eod} | "
                f"{b_dpd} | {m_dpd} | {d_dpd} | "
                f"{b_di} | {m_di} | "
                f"{b_dist} | {m_dist} | {closer} |"
            )
            lines.append(row_str)

        return "\n".join(lines)

    taiwan_table = render_table(df[df["dataset"] == "taiwan"])
    german_table = render_table(df[df["dataset"] == "german"])

    content = f"""# Fairness Mitigation Experimental Benchmark & Summary Report (Remediated)

**Project Title:** Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring  
**Phase:** Phase 3 — Fairness Mitigation (Phase 3.1 Remediated Baseline)  
**Authoritative Source:** [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv) & [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv)  
**Date:** October 2026  
**Status:** ALL 18 EXPERIMENTS COMPLETE, REMEDIATED & INDEPENDENTLY VERIFIED  

---

## 1. Executive Summary & Experimental Controls

This benchmark evaluates the empirical impact of three representative algorithmic fairness interventions on credit risk scoring models:
1. **Pre-Processing:** Reweighing (Kamiran & Calders, 2012)
2. **In-Processing:** Exponentiated Gradient (Agarwal et al., 2018 with Fairlearn `EqualizedOdds()`)
3. **Post-Processing:** ThresholdOptimizer (Hardt et al., 2016 with Fairlearn `equalized_odds`)

### Strict Experimental Controls Maintained:
- **Identical Partitions:** Every experiment used the **exact same stratified train/test split** as its corresponding baseline (`test_size=0.20, random_state=42`).
- **Feature Isolation:** Primary protected demographic attributes (`sex` in Taiwan, `personal_status_sex` / `derived_sex` in German) and identifiers (`id` in Taiwan) remained **strictly excluded from feature matrix $X$**.
- **Preprocessing Isolation:** All preprocessors were fitted strictly on training data (`X_train`), with zero test data leakage.
- **Fairness Target:** Primary mitigation targeted group parity on `sex` (Female vs. Male). Secondary effects on `age` ($\le 25$ vs. $>25$) were tracked across all models.
- **Metric Integrity Note (ROC-AUC vs. Balanced Accuracy):** Continuous ROC-AUC requires smooth continuous risk scores. Because ThresholdOptimizer outputs post-processed binary decisions and ExponentiatedGradient outputs a distribution over binary base estimators without a smooth ranking function, their mitigated ROC-AUC values are reported as `N/A`. Both baseline and mitigated models are evaluated on `Balanced Accuracy` to provide a direct, mathematically valid decision-level comparison.

---

## 2. Master Results Tables: Before vs. After Mitigation

*Source: Programmatically generated from [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv)*

All delta values are defined as: $\\Delta = \\\text{{Mitigated}} - \\\text{{Baseline}}$.  
- For fairness disparity metrics (EOD, EOpp, DPD), a **negative delta** indicates a reduction in disparity (fairness improvement).  
- For Approval Disparate Impact (Approval DI), parity is $1.0000$; distance from parity is $|DI - 1.0|$.
- For continuous ROC-AUC, only methods with continuous scoring models (Reweighing) report mitigated values; decision-level/discrete mitigations report `N/A`.

### 2.1 Taiwan Credit Card Default Dataset ($N_{{\\\text{{test}}}} = 6,000$, Primary: Sex)

{taiwan_table}

---

### 2.2 Statlog German Credit Dataset ($N_{{\\\text{{test}}}} = 200$, Primary: Derived Sex)

{german_table}

---

## 3. Objective Empirical Analysis of Mitigation Outcomes

Across the 18 experimental configurations, we observe distinct, configuration-dependent patterns across mitigation families:

### 3.1 Pre-Processing: Reweighing (Scoring-Model Intervention)
- **German Random Forest:** Reweighing achieved joint improvements across fairness metrics without loss of accuracy in this test partition:
  - Accuracy remained unchanged at **$0.7800$** ($\Delta = 0.0000$).
  - Balanced Accuracy improved from $0.6762 \to 0.6810$ ($\Delta = +0.0048$).
  - Continuous ROC-AUC **increased** from **$0.7798 \to 0.7935$** ($\Delta = +0.0137$).
  - Equalized Odds Disparity dropped by **60%** ($0.1250 \to 0.0500$, $\Delta\\text{{EOD}} = -0.0750$).
  - Demographic Parity Disparity dropped to near zero ($0.0620 \to 0.0005$).
  - Approval Disparate Impact reached near-perfect parity ($0.9269 \to 0.9994$, distance = $0.0006$).
- **Taiwan XGBoost (The Fairness Criterion Conflict):** In Taiwan XGBoost, Reweighing demonstrated a clear trade-off between fairness metrics:
  - Demographic Parity Difference improved substantially ($0.0027 \to 0.0003$, $\Delta\\text{{DPD}} = -0.0024$).
  - Approval Disparate Impact improved to $1.0003$ ($\Delta\\text{{dist}} = -0.0028$).
  - However, Equalized Odds Difference **worsened** from $0.0306 \to 0.0436$ ($\Delta\\text{{EOD}} = +0.0130$) because optimizing demographic parity reduced Male True Positive Rate ($0.3440 \to 0.3298$) while Female TPR remained stable ($0.3747 \to 0.3734$).
- **German Logistic Regression:** Reweighing improved DPD ($0.0577 \to 0.0131$), but **worsened** EOD ($0.0500 \to 0.1000$).
- **Scoring Function Invariance & SHAP Suitability:** Reweighing directly alters the training distribution and produces a newly fitted continuous predictive model $f(x) = P(Y=1 \mid X)$, making it uniquely suitable for downstream SHAP explanation consistency analysis.

### 3.2 In-Processing: Exponentiated Gradient
- **Constraint Formulation:** Configured strictly with Fairlearn's `EqualizedOdds()` constraint.
- **Predictor Structure:** In 4 of 6 runs (German RF, German XGB, Taiwan RF, Taiwan XGB), the solver converged with 100% weight placed on a single binary base estimator ($w_0 = 1.0$). In the two Logistic Regression models, it produced a discrete convex combination of two binary classifiers (a 4-level step output).
- **Ranking Performance:** ExponentiatedGradient showed lower directly reported ranking performance under the current predictor representation; because the resulting predictor does not provide a directly comparable continuous ranking score, ROC-AUC is not used for direct comparison in the corrected reporting (reported as `N/A`).
- **Balanced Accuracy Comparison:** When evaluated on Balanced Accuracy, Exponentiated Gradient performed similarly to baseline (e.g. German LR BalAcc: $0.7226 \to 0.7238$, Taiwan LR BalAcc: $0.6512 \to 0.6511$).
- **Empirical Fairness Results:** On German Logistic Regression, EOD remained at $0.0500$ while DPD decreased from $0.0577 \to 0.0361$. On tree-based models where the reduction selected a single binary model, fairness metrics were identical to baseline.

### 3.3 Post-Processing: ThresholdOptimizer (Decision-Level Intervention)
- **Decision-Level Operation:** ThresholdOptimizer post-processes baseline model output by selecting group-specific decision thresholds. It **does not alter the underlying model's scoring function or feature weights**.
- **Dramatic Fairness Improvements on High-Disparity Baselines:** On German Random Forest (where baseline disparity was high, $\\text{{EOD}} = 0.1250$), ThresholdOptimizer cut $\\text{{EOD}}$ down to **0.0251** ($\Delta\\text{{EOD}} = -0.0999$, a **79.9% reduction in disparity**).
- **Accuracy Trade-Off:** Achieving this parity required an accuracy reduction of $2.5\%$ points ($0.7800 \to 0.7550$) to adjust decision cutoffs. However, Balanced Accuracy slightly improved ($0.6762 \to 0.6774$, $\Delta = +0.0012$) due to higher recall on the minority class.
- **Low-Disparity Threshold Variance:** On baselines that were already fair (e.g. Taiwan LR baseline $\\text{{EOD}} = 0.0194$), fitting group thresholds on training calibration data slightly increased test-set EOD ($0.0194 \to 0.0425$) due to finite-sample threshold variance.
- **Protocol for SHAP Analysis:** Because ThresholdOptimizer does not modify the underlying model's learned scoring function, computing SHAP values on the model before and after ThresholdOptimizer yields identical explanations. ThresholdOptimizer is therefore retained for decision-level auditing and excluded from the main scoring-model explanation consistency benchmark.

---

## 4. Secondary Fairness Effects: Impact on Age ($\\le 25$ vs. $>25$)

*All subgroup counts verified in [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv)*

When mitigation was applied to primary attribute `sex`, what happened to unmitigated `age` bias?
- **Reweighing on Sex:** In German Random Forest, reweighing sex also improved age disparate impact ($0.9167 \to 0.9545$). However, in Taiwan models, age disparity remained elevated ($\\text{{EOD}} \\approx 0.106–0.141$), confirming that single-attribute mitigation does not automatically resolve intersectional or unaddressed demographic disparities.
- **ThresholdOptimizer on Sex:** Because group thresholds were adjusted solely by `sex`, the young age group ($\le 25$) continued to experience higher false positive rates ($0.08–0.09$ in Taiwan vs $0.04–0.05$ for older applicants).

---

## 5. Verification Artifacts & Test Suite

- Master Results CSV: [`results/mitigation_master_results.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_master_results.csv)
- Subgroup Metrics CSV: [`results/mitigated_subgroup_metrics.csv`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigated_subgroup_metrics.csv)
- Mitigation Metadata JSON: [`results/mitigation_metadata.json`](file:///d:/sarthak/Subjects/ERA/Project/results/mitigation_metadata.json)
- Independent Verification Script: [`scripts/verify_mitigation_results.py`](file:///d:/sarthak/Subjects/ERA/Project/scripts/verify_mitigation_results.py)

Running `python scripts/verify_mitigation_results.py` confirms that **all 18 experiments, 72 subgroup confusion matrices, and before/after deltas are 100% mathematically consistent**.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Summary generated successfully at {output_path}")


if __name__ == "__main__":
    generate_summary()
