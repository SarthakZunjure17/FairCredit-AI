"use client";

import React, { useState, useEffect } from "react";
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  Table,
  CheckCircle2,
  AlertTriangle,
  Info,
  Scale,
  BarChart3,
  ExternalLink
} from "lucide-react";
import { api } from "@/lib/api";
import { ResearchMatrixRow, ShapSummaryRow } from "@/types";
import { formatMetric } from "@/lib/utils";

interface SectionProps {
  id: string;
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function ExpandableSection({ id, title, isOpen, onToggle, children }: SectionProps) {
  return (
    <div className="rounded-xl bg-white border border-slate-200 shadow-xs overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full px-5 py-4 flex items-center justify-between text-left text-sm font-bold text-slate-900 hover:bg-slate-50 transition-colors"
      >
        <span>{title}</span>
        {isOpen ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
      </button>
      {isOpen && <div className="px-5 pb-5 pt-1 text-xs text-slate-600 leading-relaxed border-t border-slate-100 space-y-3">{children}</div>}
    </div>
  );
}

export default function ResearchPage() {
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    s1: true,
    s7: true,
    s10: true
  });
  const [matrixData, setMatrixData] = useState<ResearchMatrixRow[]>([]);
  const [shapSummaries, setShapSummaries] = useState<ShapSummaryRow[]>([]);

  useEffect(() => {
    async function loadData() {
      try {
        const [matrixRes, shapRes] = await Promise.all([
          api.getResearchMatrix(),
          api.getShapSummary()
        ]);
        setMatrixData(matrixRes);
        setShapSummaries(shapRes);
      } catch (e) {
        console.error("Research page data load error:", e);
      }
    }
    loadData();
  }, []);

  const toggleSection = (id: string) => {
    setOpenSections((prev) => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-semibold text-teal-800 uppercase tracking-wider">
          <BookOpen className="w-4 h-4 text-teal-600" />
          <span>Research Synthesis & Documentation</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-1">
          Empirical Research & Methodology Workspace
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
          Evaluating the impact of fairness mitigation on predictive performance, group fairness, and explanation consistency in credit scoring models.
        </p>
      </div>

      <div className="space-y-4">
        {/* Section 1: Research Question & Motivation */}
        <ExpandableSection
          id="s1"
          title="1. Research Question & Motivation"
          isOpen={!!openSections["s1"]}
          onToggle={() => toggleSection("s1")}
        >
          <div className="p-3.5 rounded-lg bg-teal-50/70 border border-teal-200/60 text-teal-900 font-medium">
            <strong>Core Research Question:</strong> “When fairness mitigation is applied to machine-learning credit-scoring models, how does it affect predictive performance, group fairness, and explanation consistency?”
          </div>
          <p>
            Credit scoring systems rely heavily on algorithmic classification (e.g. loan approvals, interest rates). Under regulatory frameworks such as the U.S. Equal Credit Opportunity Act (ECOA) and EU AI Act, lenders face dual requirements: models must demonstrate demographic non-discrimination and must provide individualized adverse action notices (explanations).
          </p>
          <p>
            While substantial literature investigates the Pareto trade-off between accuracy and group fairness, critical questions remain regarding whether fairness interventions destabilize feature attribution profiles (SHAP explanations) for individual applicants.
          </p>
        </ExpandableSection>

        {/* Section 2: Datasets */}
        <ExpandableSection
          id="s2"
          title="2. Canonical Datasets & Protected Attributes"
          isOpen={!!openSections["s2"]}
          onToggle={() => toggleSection("s2")}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50 space-y-1">
              <h4 className="font-bold text-slate-900">Taiwan Credit Card Default (2005)</h4>
              <p>• <strong>Sample Size:</strong> 30,000 observations (6,000 test set, stratified 80/20 split).</p>
              <p>• <strong>Target:</strong> Default payment next month (22.1% prevalence).</p>
              <p>• <strong>Primary Protected Attribute:</strong> Sex (female 60.4%, male 39.6%).</p>
              <p>• <strong>Predictive Features:</strong> 23 features (limit balance, bill amounts, repayment records). Sensitive sex is strictly excluded from feature matrix X.</p>
            </div>
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50 space-y-1">
              <h4 className="font-bold text-slate-900">Statlog German Credit (1994)</h4>
              <p>• <strong>Sample Size:</strong> 1,000 observations (200 test set, stratified 80/20 split).</p>
              <p>• <strong>Target:</strong> Credit risk (1=bad credit, 0=good credit; 30.0% default rate).</p>
              <p>• <strong>Primary Protected Attribute:</strong> Derived sex from personal status (A92=female, A91/A93/A94=male).</p>
              <p>• <strong>Predictive Features:</strong> 20 credit attributes. Derived sex is strictly excluded from feature matrix X.</p>
            </div>
          </div>
        </ExpandableSection>

        {/* Section 3: Models & Training Protocol */}
        <ExpandableSection
          id="s3"
          title="3. Model Architectures & Training Protocol"
          isOpen={!!openSections["s3"]}
          onToggle={() => toggleSection("s3")}
        >
          <p>
            Three canonical architectures spanning linear, bagging, and boosting paradigms were evaluated:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-slate-600">
            <li><strong>Logistic Regression:</strong> L-BFGS solver, max_iter=1000, standardized numerical features.</li>
            <li><strong>Random Forest:</strong> 100 trees, unconstrained depth, parallelized ensemble.</li>
            <li><strong>XGBoost:</strong> 100 boosting rounds, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8.</li>
          </ul>
          <p>
            All splits and initializations used fixed <code>random_state=42</code>. Preprocessors (OneHotEncoder with handle_unknown=&apos;ignore&apos; and StandardScaler) were fitted strictly on training partitions to prevent data leakage.
          </p>
        </ExpandableSection>

        {/* Section 4: Fairness Mitigation Methods */}
        <ExpandableSection
          id="s4"
          title="4. Evaluated Fairness Mitigation Interventions"
          isOpen={!!openSections["s4"]}
          onToggle={() => toggleSection("s4")}
        >
          <div className="space-y-2">
            <div className="p-2.5 rounded border border-slate-200 bg-slate-50">
              <strong>1. Pre-Processing — Reweighing (Kamiran & Calders, 2012):</strong>
              <p className="mt-0.5">Assigns statistical independence weights to training samples based on the joint distribution of target and protected attribute, training an adjusted model under weighted empirical risk minimization.</p>
            </div>
            <div className="p-2.5 rounded border border-slate-200 bg-slate-50">
              <strong>2. In-Processing — Exponentiated Gradient (Agarwal et al., 2018):</strong>
              <p className="mt-0.5">Formulates fair classification as a two-player zero-sum game subject to an EqualizedOdds constraint, returning a randomized mixture of base estimators.</p>
            </div>
            <div className="p-2.5 rounded border border-slate-200 bg-slate-50">
              <strong>3. Post-Processing — ThresholdOptimizer (Hardt et al., 2016):</strong>
              <p className="mt-0.5">Calibrates group-specific classification decision thresholds on a pre-fit model to equalize true positive and false positive rates without altering scoring weights.</p>
            </div>
          </div>
        </ExpandableSection>

        {/* Section 5: Metrics & Auditing Standards */}
        <ExpandableSection
          id="s5"
          title="5. Metrics & Auditing Standards"
          isOpen={!!openSections["s5"]}
          onToggle={() => toggleSection("s5")}
        >
          <p><strong>Predictive Performance:</strong> Accuracy, Balanced Accuracy, and ROC-AUC. Following strict auditing standards, ROC-AUC is reported as N/A for step/binary decision rules.</p>
          <p><strong>Group Fairness:</strong></p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Equalized Odds Difference (EOD):</strong> Maximum difference in TPR and FPR between protected groups. Target: 0.</li>
            <li><strong>Demographic Parity Difference (DPD):</strong> Difference in positive selection rates between groups. Target: 0.</li>
            <li><strong>Approval Disparate Impact (Approval-DI):</strong> Ratio of unprivileged to privileged positive prediction rates. Target: 1.0 (Four-Fifths Rule &gt;= 0.80).</li>
          </ul>
        </ExpandableSection>

        {/* Section 6: SHAP Methodology */}
        <ExpandableSection
          id="s6"
          title="6. SHAP Methodology & Explanation Consistency"
          isOpen={!!openSections["s6"]}
          onToggle={() => toggleSection("s6")}
        >
          <p>
            To evaluate explanation consistency, local SHAP attributions explaining Default Risk (Y=1) were calculated for identical held-out test cohorts (Taiwan N=500, German N=200) under Baseline vs. Reweighed models:
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>LinearExplainer:</strong> For Logistic Regression using training-derived background dataset.</li>
            <li><strong>TreeExplainer:</strong> For Random Forest (probability space) and XGBoost (margin log-odds space).</li>
          </ul>
          <p>
            <strong>Consistency Metrics:</strong>
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Cosine Similarity:</strong> Measures direction and angle alignment of attribution vectors in feature space.</li>
            <li><strong>Top-5 Feature Overlap:</strong> Jaccard set intersection of the 5 highest-magnitude features.</li>
            <li><strong>Spearman Rank Correlation:</strong> Preservation of relative feature importance rankings.</li>
          </ul>
        </ExpandableSection>

        {/* Section 7: Key Empirical Findings */}
        <ExpandableSection
          id="s7"
          title="7. Key Empirical Findings (Heterogeneous Effects)"
          isOpen={!!openSections["s7"]}
          onToggle={() => toggleSection("s7")}
        >
          <div className="p-4 rounded-lg bg-teal-50 border border-teal-200 text-teal-950 font-medium space-y-2">
            <h4 className="font-bold flex items-center space-x-1.5 text-sm">
              <CheckCircle2 className="w-4 h-4 text-teal-700" />
              <span>Central Empirical Finding</span>
            </h4>
            <p className="text-xs leading-relaxed">
              “Fairness mitigation through Reweighing exhibited heterogeneous effects on explanation consistency across dataset-model configurations rather than a universal fairness–explanation trade-off.”
            </p>
          </div>

          <div className="space-y-2 pt-1">
            <p><strong>Detailed Observations:</strong></p>
            <p>1. <strong>Architecture Sensitivity:</strong> Tree-based ensembles displayed distinct stability profiles compared to linear models. For German Random Forest, Reweighing achieved a 60% reduction in Equalized Odds disparity with high cosine consistency (0.9652), whereas other configurations experienced substantial feature ranking shifts.</p>
            <p>2. <strong>No Universal Guarantee:</strong> Applying Reweighing did not uniformly guarantee improved demographic parity or preserved explanations across all configurations.</p>
            <p>3. <strong>Feature Redistribution:</strong> In configurations with lower consistency, Reweighing caused credit history and payment timing features to trade relative importance, impacting adverse action explanation reliability.</p>
          </div>
        </ExpandableSection>

        {/* Section 8: Limitations */}
        <ExpandableSection
          id="s8"
          title="8. Limitations & Scope Boundaries"
          isOpen={!!openSections["s8"]}
          onToggle={() => toggleSection("s8")}
        >
          <div className="p-3.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 space-y-1">
            <h4 className="font-bold text-xs flex items-center space-x-1.5">
              <AlertTriangle className="w-4 h-4 text-amber-700" />
              <span>Scope & Research Boundaries</span>
            </h4>
            <ul className="list-disc pl-5 space-y-1 text-xs">
              <li>• <strong>Public Benchmark Datasets:</strong> Findings are derived from Taiwan (2005) and German (1994) public benchmarks and may not generalize to production credit portfolios with modern regulatory data.</li>
              <li>• <strong>Observational Disparities:</strong> Statistical disparities reflect observed historical dataset biases and do not establish causal discrimination.</li>
              <li>• <strong>Explanation Proxy:</strong> SHAP represents model feature attribution, not ground-truth causal mechanisms in borrower financial outcomes.</li>
              <li>• <strong>Prototype Boundary:</strong> This system is strictly an academic model-auditing prototype, not an automated lending underwriting engine.</li>
            </ul>
          </div>
        </ExpandableSection>

        {/* Section 10: Complete Verified Research Matrix Table */}
        <ExpandableSection
          id="s10"
          title="9. Complete Verified Research Results Table"
          isOpen={!!openSections["s10"]}
          onToggle={() => toggleSection("s10")}
        >
          <p className="text-[11px] text-slate-500">
            Authoritative results joined from <code>results/fairness_performance_explanation_matrix.csv</code> and <code>results/shap_summary.csv</code>.
          </p>

          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="w-full text-left text-[11px] border-collapse">
              <thead>
                <tr className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                  <th className="p-2.5">Dataset</th>
                  <th className="p-2.5">Model</th>
                  <th className="p-2.5">Mitigation</th>
                  <th className="p-2.5 text-right">Acc (Base)</th>
                  <th className="p-2.5 text-right">Acc (Mit)</th>
                  <th className="p-2.5 text-right">AUC (Base)</th>
                  <th className="p-2.5 text-right">AUC (Mit)</th>
                  <th className="p-2.5 text-right">EOD (Base)</th>
                  <th className="p-2.5 text-right">EOD (Mit)</th>
                  <th className="p-2.5 text-right">SHAP Cosine</th>
                  <th className="p-2.5 text-right">Top-5 Overlap</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {matrixData.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50/70 font-mono">
                    <td className="p-2.5 font-sans font-medium capitalize text-slate-900">{row.dataset}</td>
                    <td className="p-2.5 font-sans text-slate-700 capitalize">{row.model.replace(/_/g, " ")}</td>
                    <td className="p-2.5 font-sans text-teal-800 font-semibold capitalize">{row.mitigation_method.replace(/_/g, " ")}</td>
                    <td className="p-2.5 text-right">{formatMetric(row.baseline_acc)}</td>
                    <td className="p-2.5 text-right">{formatMetric(row.mitigated_acc)}</td>
                    <td className="p-2.5 text-right">{formatMetric(row.baseline_auc)}</td>
                    <td className="p-2.5 text-right">{row.mitigated_auc || "N/A"}</td>
                    <td className="p-2.5 text-right">{formatMetric(row.baseline_eod)}</td>
                    <td className="p-2.5 text-right font-bold text-teal-700">{formatMetric(row.mitigated_eod)}</td>
                    <td className="p-2.5 text-right text-indigo-700">{row.shap_cosine_sim ? formatMetric(row.shap_cosine_sim) : "N/A"}</td>
                    <td className="p-2.5 text-right text-slate-600">{row.shap_top5_overlap ? formatMetric(row.shap_top5_overlap) : "N/A"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ExpandableSection>
      </div>
    </div>
  );
}
