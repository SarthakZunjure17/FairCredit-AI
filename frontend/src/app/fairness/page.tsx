"use client";

import React, { useState, useEffect } from "react";
import {
  Scale,
  BarChart2,
  FileText,
  AlertCircle,
  HelpCircle,
  TrendingDown,
  TrendingUp,
  Minus,
  CheckCircle2,
  UserCheck
} from "lucide-react";
import { api } from "@/lib/api";
import {
  ResearchMatrixRow,
  ShapSummaryRow,
  DatasetKey,
  ModelKey,
  MitigationKey,
  IndividualShapResponse
} from "@/types";
import { MetricCard } from "@/components/MetricCard";
import { ShapBarChart } from "@/components/ShapBarChart";
import {
  DATASET_LABELS,
  MODEL_LABELS,
  MITIGATION_LABELS,
  formatMetric
} from "@/lib/utils";

export default function FairnessPage() {
  const [matrixData, setMatrixData] = useState<ResearchMatrixRow[]>([]);
  const [shapSummaries, setShapSummaries] = useState<ShapSummaryRow[]>([]);
  
  const [selectedDataset, setSelectedDataset] = useState<DatasetKey>("german");
  const [selectedModel, setSelectedModel] = useState<ModelKey>("random_forest");
  const [selectedMitigation, setSelectedMitigation] = useState<MitigationKey>("reweighing");
  
  const [individualShap, setIndividualShap] = useState<IndividualShapResponse | null>(null);
  const [applicantIdx, setApplicantIdx] = useState<number>(0);
  
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    async function loadResearch() {
      try {
        setIsLoading(true);
        const [matrixRes, shapRes] = await Promise.all([
          api.getResearchMatrix(),
          api.getShapSummary()
        ]);
        setMatrixData(matrixRes);
        setShapSummaries(shapRes);
      } catch (err: any) {
        setErrorMessage(`Failed to load research tables: ${err.message}`);
      } finally {
        setIsLoading(false);
      }
    }
    loadResearch();
  }, []);

  // Fetch individual held-out applicant SHAP attribution
  useEffect(() => {
    async function loadApplicantShap() {
      try {
        const res = await api.getIndividualShap(selectedDataset, selectedModel, applicantIdx);
        setIndividualShap(res);
      } catch (err: any) {
        console.error("Individual SHAP load error:", err);
      }
    }
    loadApplicantShap();
  }, [selectedDataset, selectedModel, applicantIdx]);

  // Find matching row in matrix
  const currentExp = matrixData.find(
    (r) =>
      r.dataset.toLowerCase() === selectedDataset &&
      r.model.toLowerCase() === selectedModel &&
      r.mitigation_method.toLowerCase() === selectedMitigation
  );

  // Find matching SHAP summary row
  const currentShap = shapSummaries.find(
    (s) =>
      s.dataset.toLowerCase() === selectedDataset &&
      s.model.toLowerCase() === selectedModel
  );

  return (
    <div className="space-y-10 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-semibold text-teal-800 uppercase tracking-wider">
          <Scale className="w-4 h-4 text-teal-600" />
          <span>Experience B</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-1">
          Fairness Audit & Explanation Consistency
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
          Examine empirical before-and-after fairness metrics across audited mitigation methods,
          and evaluate whether fairness adjustments preserved or altered individual SHAP explanation stability.
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Selectors Bar */}
      <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs flex flex-wrap gap-4 items-center justify-between">
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">Dataset</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value as DatasetKey)}
              className="text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700"
            >
              <option value="german">Statlog German Credit</option>
              <option value="taiwan">Taiwan Credit Card Default</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">Model Architecture</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
              className="text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700"
            >
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
              <option value="logistic_regression">Logistic Regression</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-bold uppercase text-slate-400 block mb-1">Fairness Mitigation</label>
            <select
              value={selectedMitigation}
              onChange={(e) => setSelectedMitigation(e.target.value as MitigationKey)}
              className="text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700"
            >
              <option value="reweighing">Reweighing (Pre-processing)</option>
              <option value="exponentiated_gradient">Exponentiated Gradient (In-processing)</option>
              <option value="threshold_optimizer">ThresholdOptimizer (Post-processing)</option>
            </select>
          </div>
        </div>

        <div className="text-[11px] text-slate-500 font-medium">
          Target Configuration: <span className="text-slate-900 font-semibold">{DATASET_LABELS[selectedDataset]} · {MODEL_LABELS[selectedModel]}</span>
        </div>
      </div>

      {/* SECTION A: MODEL & FAIRNESS AUDIT */}
      <div className="space-y-4">
        <div className="border-b border-slate-200 pb-2 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              A. Model Performance & Group Fairness Audit
            </h2>
            <p className="text-xs text-slate-500">
              Auditing empirical changes from baseline to {MITIGATION_LABELS[selectedMitigation]}.
            </p>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            {selectedMitigation === "threshold_optimizer" || selectedMitigation === "exponentiated_gradient"
              ? "ROC-AUC: N/A (Step/Binary Output)"
              : "Continuous Output Space"}
          </span>
        </div>

        {/* Two-Column Comparison: Performance vs Group Fairness */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Predictive Performance Card */}
          <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-teal-700" />
              <span>Predictive Performance Metrics</span>
            </h3>

            <div className="space-y-3">
              {/* Classification Accuracy */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                <div>
                  <span className="font-semibold text-slate-800">Test Accuracy</span>
                  <p className="text-[11px] text-slate-500">Higher is favorable (↑)</p>
                </div>
                <div className="text-right">
                  <div className="font-mono font-bold text-slate-900">
                    {formatMetric(currentExp?.mitigated_acc)}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Baseline: {formatMetric(currentExp?.baseline_acc)}
                  </div>
                </div>
              </div>

              {/* ROC-AUC */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                <div>
                  <span className="font-semibold text-slate-800">ROC-AUC Discrimination</span>
                  <p className="text-[11px] text-slate-500">Continuous rank ordering (↑)</p>
                </div>
                <div className="text-right">
                  <div className="font-mono font-bold text-slate-900">
                    {currentExp?.mitigated_auc || "N/A"}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Baseline: {formatMetric(currentExp?.baseline_auc)}
                  </div>
                </div>
              </div>
            </div>

            <p className="text-[11px] text-slate-500 leading-normal">
              * Note: In compliance with strict auditing guidelines, ROC-AUC is reported as strictly <strong>N/A</strong> for ThresholdOptimizer and step predictions where binary thresholds discard score rank order.
            </p>
          </div>

          {/* Group Fairness Disparities Card */}
          <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
              <Scale className="w-4 h-4 text-teal-700" />
              <span>Group Fairness Disparity Metrics</span>
            </h3>

            <div className="space-y-3">
              {/* Equalized Odds Difference */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                <div>
                  <span className="font-semibold text-slate-800">Equalized Odds Difference (EOD)</span>
                  <p className="text-[11px] text-slate-500">TPR/FPR gap between groups. Lower is favorable (↓)</p>
                </div>
                <div className="text-right">
                  <div className="font-mono font-bold text-slate-900">
                    {formatMetric(currentExp?.mitigated_eod)}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Baseline: {formatMetric(currentExp?.baseline_eod)}
                  </div>
                </div>
              </div>

              {/* Demographic Parity Difference */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                <div>
                  <span className="font-semibold text-slate-800">Demographic Parity Diff (DPD)</span>
                  <p className="text-[11px] text-slate-500">Approval rate gap. Lower is favorable (↓)</p>
                </div>
                <div className="text-right">
                  <div className="font-mono font-bold text-slate-900">
                    {formatMetric(currentExp?.mitigated_dpd)}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Baseline: {formatMetric(currentExp?.baseline_dpd)}
                  </div>
                </div>
              </div>
            </div>

            <p className="text-[11px] text-slate-500 leading-normal">
              A reduction in disparity metrics (EOD / DPD) indicates progress toward parity between demographic groups, but must be evaluated alongside predictive utility.
            </p>
          </div>
        </div>
      </div>

      {/* SECTION B: EXPLANATION STABILITY */}
      <div className="space-y-4 pt-4 border-t border-slate-200">
        <div>
          <h2 className="text-lg font-bold text-slate-900">
            B. Explanation Stability (SHAP Consistency)
          </h2>
          <p className="text-xs text-slate-500">
            Measuring how similar local SHAP feature attributions are for the same held-out applicants before and after Reweighing.
          </p>
        </div>

        {/* Aggregate Consistency Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetricCard
            label="Mean Cosine Similarity"
            value={currentShap?.mean_cosine ? currentShap.mean_cosine.toFixed(4) : "N/A"}
            subtext={`95% CI: ${currentShap?.ci95_cosine || "N/A"}`}
            badge="Vector Alignment"
          />
          <MetricCard
            label="Mean Top-5 Overlap"
            value={currentShap?.mean_top5 ? currentShap.mean_top5.toFixed(4) : "N/A"}
            subtext={`95% CI: ${currentShap?.ci95_top5 || "N/A"}`}
            badge="Key Feature Set"
          />
          <MetricCard
            label="Mean Spearman Rank Correlation"
            value={currentShap?.mean_rank ? currentShap.mean_rank.toFixed(4) : "N/A"}
            subtext={`95% CI: ${currentShap?.ci95_rank || "N/A"}`}
            badge="Rank Order"
          />
        </div>

        {/* Individual Explanation Comparison from Frozen Research Parquet */}
        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
                <UserCheck className="w-4 h-4 text-teal-700" />
                <span>Individual Held-Out Applicant Explanation (Frozen Cohort)</span>
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Inspect how baseline vs. reweighed explanations differed for specific applicants evaluated during Phase 4.
              </p>
            </div>

            {/* Applicant Index Selector */}
            <div className="flex items-center space-x-2 text-xs">
              <label className="text-slate-500 font-medium">Applicant Index:</label>
              <select
                value={applicantIdx}
                onChange={(e) => setApplicantIdx(parseInt(e.target.value))}
                className="rounded border border-slate-300 bg-white px-2 py-1 text-slate-800 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-teal-700"
              >
                {[0, 1, 2, 5, 10, 15, 20, 50].map((idx) => (
                  <option key={idx} value={idx}>
                    #{idx}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {individualShap?.attributions && individualShap.attributions.length > 0 ? (
            <ShapBarChart
              data={individualShap.attributions.map((a) => ({
                feature: a.feature,
                feature_raw: a.feature,
                baseline_shap: a.baseline_shap,
                reweighed_shap: a.reweighed_shap,
                diff_shap: a.diff_shap,
                direction: a.baseline_shap > 0 ? "increases_risk" : "reduces_risk"
              }))}
              title={`Held-Out Applicant #${applicantIdx} — Baseline vs. Reweighed SHAP Attributions`}
            />
          ) : (
            <div className="h-40 flex items-center justify-center text-slate-400 text-xs italic">
              Loading individual held-out applicant SHAP data…
            </div>
          )}

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-teal-700 shrink-0 mt-0.5" />
            <p className="text-[11px] leading-relaxed">
              <strong>Distinction Note:</strong> These records reflect genuine held-out test cohort applicants from the frozen Phase 4 evaluation (`dashboard/data/individual_feature_shap.parquet`), distinct from interactive synthetic applicant predictions generated on Page 2.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
