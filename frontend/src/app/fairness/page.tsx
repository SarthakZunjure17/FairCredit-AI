"use client";

import React, { useState, useEffect } from "react";
import {
  Scale,
  BarChart2,
  AlertCircle,
  TrendingDown,
  TrendingUp,
  Minus,
  CheckCircle2,
  UserCheck,
  ShieldAlert,
  Info,
  Layers,
  ArrowRight,
  Database,
  HelpCircle,
  Activity
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from "recharts";
import { api } from "@/lib/api";
import {
  FairnessAuditResponse,
  ShapSummaryRow,
  DatasetKey,
  ModelKey,
  MitigationKey,
  ProtectedAttributeKey,
  IndividualShapResponse
} from "@/types";
import { MetricCard } from "@/components/MetricCard";
import { ShapBarChart } from "@/components/ShapBarChart";
import {
  DATASET_LABELS,
  MODEL_LABELS,
  MITIGATION_LABELS,
  formatMetric,
  formatPercent,
  formatModelLabel,
  formatDatasetLabel,
  formatMitigationLabel,
  formatFeatureLabel
} from "@/lib/utils";

export default function FairnessPage() {
  const [selectedDataset, setSelectedDataset] = useState<DatasetKey>("german");
  const [selectedModel, setSelectedModel] = useState<ModelKey>("random_forest");
  const [selectedMitigation, setSelectedMitigation] = useState<MitigationKey>("reweighing");
  const [selectedProtectedAttr, setSelectedProtectedAttr] = useState<ProtectedAttributeKey>("sex");

  const [auditData, setAuditData] = useState<FairnessAuditResponse | null>(null);
  const [shapSummaries, setShapSummaries] = useState<ShapSummaryRow[]>([]);
  const [individualShap, setIndividualShap] = useState<IndividualShapResponse | null>(null);
  const [applicantIdx, setApplicantIdx] = useState<number>(0);

  const [isAuditLoading, setIsAuditLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Fetch SHAP summaries once on mount
  useEffect(() => {
    async function loadShap() {
      try {
        const shapRes = await api.getShapSummary();
        if (Array.isArray(shapRes)) {
          setShapSummaries(shapRes);
        }
      } catch (err: any) {
        console.error("SHAP summary load error:", err);
      }
    }
    loadShap();
  }, []);

  // Fetch fairness audit whenever selection changes
  useEffect(() => {
    async function loadAudit() {
      try {
        setIsAuditLoading(true);
        setErrorMessage(null);
        const data = await api.getFairnessAudit(
          selectedDataset,
          selectedModel,
          selectedMitigation,
          selectedProtectedAttr
        );
        setAuditData(data);
      } catch (err: any) {
        setErrorMessage(`Failed to load fairness audit: ${err.message}`);
      } finally {
        setIsAuditLoading(false);
      }
    }
    loadAudit();
  }, [selectedDataset, selectedModel, selectedMitigation, selectedProtectedAttr]);

  // Fetch individual applicant SHAP attribution (evaluated for Reweighing held-out cohort)
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

  // Find matching SHAP summary row
  const currentShap = shapSummaries.find(
    (s) => s.dataset === selectedDataset && s.model === selectedModel
  );

  // Prepare chart data for Approval Rates
  const approvalChartData = React.useMemo(() => {
    if (!auditData) return [];
    const bU = auditData.baseline_subgroups.find((g) => g.group_role === "unprivileged");
    const bP = auditData.baseline_subgroups.find((g) => g.group_role === "privileged");
    const mU = auditData.mitigated_subgroups.find((g) => g.group_role === "unprivileged");
    const mP = auditData.mitigated_subgroups.find((g) => g.group_role === "privileged");

    return [
      {
        category: `Unprivileged (${bU?.group || "u"})`,
        role: "unprivileged",
        baselineApproval: bU ? Math.round(bU.approval_rate * 10000) / 100 : 0,
        mitigatedApproval: mU ? Math.round(mU.approval_rate * 10000) / 100 : 0,
        sampleN: bU?.sample_count || 0
      },
      {
        category: `Privileged (${bP?.group || "p"})`,
        role: "privileged",
        baselineApproval: bP ? Math.round(bP.approval_rate * 10000) / 100 : 0,
        mitigatedApproval: mP ? Math.round(mP.approval_rate * 10000) / 100 : 0,
        sampleN: bP?.sample_count || 0
      }
    ];
  }, [auditData]);

  // Prepare chart data for Equalized Odds components (TPR & FPR)
  const eqOddsChartData = React.useMemo(() => {
    if (!auditData) return [];
    const bU = auditData.baseline_subgroups.find((g) => g.group_role === "unprivileged");
    const bP = auditData.baseline_subgroups.find((g) => g.group_role === "privileged");
    const mU = auditData.mitigated_subgroups.find((g) => g.group_role === "unprivileged");
    const mP = auditData.mitigated_subgroups.find((g) => g.group_role === "privileged");

    return [
      {
        name: `Baseline: ${bU?.group || "u"} (Unprivileged)`,
        tpr: bU ? Math.round(bU.true_positive_rate * 10000) / 100 : 0,
        fpr: bU ? Math.round(bU.false_positive_rate * 10000) / 100 : 0
      },
      {
        name: `Mitigated: ${mU?.group || "u"} (Unprivileged)`,
        tpr: mU ? Math.round(mU.true_positive_rate * 10000) / 100 : 0,
        fpr: mU ? Math.round(mU.false_positive_rate * 10000) / 100 : 0
      },
      {
        name: `Baseline: ${bP?.group || "p"} (Privileged)`,
        tpr: bP ? Math.round(bP.true_positive_rate * 10000) / 100 : 0,
        fpr: bP ? Math.round(bP.false_positive_rate * 10000) / 100 : 0
      },
      {
        name: `Mitigated: ${mP?.group || "p"} (Privileged)`,
        tpr: mP ? Math.round(mP.true_positive_rate * 10000) / 100 : 0,
        fpr: mP ? Math.round(mP.false_positive_rate * 10000) / 100 : 0
      }
    ];
  }, [auditData]);

  const eodItem = auditData?.metric_comparisons.find((c) => c.metric_key === "eod");
  const dpdItem = auditData?.metric_comparisons.find((c) => c.metric_key === "dpd");
  const diItem = auditData?.metric_comparisons.find((c) => c.metric_key === "approval_di");
  const accItem = auditData?.metric_comparisons.find((c) => c.metric_key === "accuracy");
  const aucItem = auditData?.metric_comparisons.find((c) => c.metric_key === "roc_auc");

  return (
    <div className="space-y-10 max-w-6xl mx-auto pb-12">
      {/* Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-semibold text-teal-800 uppercase tracking-wider">
          <Scale className="w-4 h-4 text-teal-600" />
          <span>Stage 3 Interactive Fairness Workbench</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-1">
          Group Fairness Audit & Disparity Workbench
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
          Audit group-level demographic fairness before and after algorithmic mitigation.
          Inspect subgroup confusion matrices, verify Equalized Odds and Demographic Parity differences,
          and examine how mitigation policies impact explanation stability.
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Selectors Bar */}
      <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-700 uppercase tracking-wide">
            <Layers className="w-4 h-4 text-teal-700" />
            <span>Interactive Audit Configuration</span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            {DATASET_LABELS[selectedDataset]} · {MODEL_LABELS[selectedModel]}
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="text-[11px] font-bold uppercase text-slate-500 block mb-1">
              Dataset
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value as DatasetKey)}
              className="w-full text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700 shadow-2xs"
            >
              <option value="german">Statlog German Credit (N=1,000 / Test N=200)</option>
              <option value="taiwan">Taiwan Credit Card Default (N=30,000 / Test N=6,000)</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-bold uppercase text-slate-500 block mb-1">
              Model Architecture
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
              className="w-full text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700 shadow-2xs"
            >
              <option value="random_forest">Random Forest (Ensemble Trees)</option>
              <option value="xgboost">XGBoost (Gradient Boosted Trees)</option>
              <option value="logistic_regression">Logistic Regression (Linear Margin)</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-bold uppercase text-slate-500 block mb-1">
              Mitigation Method
            </label>
            <select
              value={selectedMitigation}
              onChange={(e) => setSelectedMitigation(e.target.value as MitigationKey)}
              className="w-full text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700 shadow-2xs"
            >
              <option value="reweighing">Reweighing (Pre-processing · Weights)</option>
              <option value="exponentiated_gradient">Exponentiated Gradient (In-processing · Fairlearn)</option>
              <option value="threshold_optimizer">ThresholdOptimizer (Post-processing · Thresholds)</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-bold uppercase text-slate-500 block mb-1">
              Protected Attribute
            </label>
            <select
              value={selectedProtectedAttr}
              onChange={(e) => setSelectedProtectedAttr(e.target.value as ProtectedAttributeKey)}
              className="w-full text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700 shadow-2xs"
            >
              <option value="sex">Gender / Sex (Primary: Female vs. Male)</option>
              <option value="age">Age Cohort (Secondary: &le;25 vs. &gt;25)</option>
            </select>
          </div>
        </div>

        {/* Evaluation Context Callout */}
        {auditData?.evaluation_context && (
          <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between text-[11px] text-slate-500 gap-2">
            <div className="flex items-center space-x-2">
              <Database className="w-3.5 h-3.5 text-teal-700 shrink-0" />
              <span>
                <strong>Evaluation Cohort:</strong> {auditData.evaluation_context.dataset_display} Held-Out Test Split (
                {auditData.evaluation_context.total_test_samples.toLocaleString()} applicants:{" "}
                <span className="font-semibold text-rose-700">
                  {auditData.evaluation_context.unprivileged_group_definition}
                </span>{" "}
                unprivileged vs.{" "}
                <span className="font-semibold text-sky-700">
                  {auditData.evaluation_context.privileged_group_definition}
                </span>{" "}
                privileged)
              </span>
            </div>
            <div className="flex items-center space-x-1.5 bg-slate-100 px-2 py-0.5 rounded text-[10px] text-slate-600 font-mono">
              <span>Artifact: results/mitigated_subgroup_metrics.csv</span>
            </div>
          </div>
        )}
      </div>

      {isAuditLoading ? (
        <div className="p-12 rounded-xl bg-white border border-slate-200 text-center text-xs text-slate-500 space-y-2">
          <Activity className="w-6 h-6 text-teal-600 animate-spin mx-auto" />
          <p>Loading verified fairness audit data…</p>
        </div>
      ) : auditData ? (
        <>
          {/* SECTION 1: FAIRNESS DISPARITY SUMMARY CARDS */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                  <span>1. Disparity Metrics (Baseline vs. {formatMitigationLabel(selectedMitigation)})</span>
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Comparison of group-fairness gap metrics and classification utility. Lower absolute EOD and DPD reflect smaller disparities.
                </p>
              </div>
              <span className="text-[11px] font-semibold text-teal-800 bg-teal-50 px-2.5 py-1 rounded-md border border-teal-200/60">
                Screening Rule: 80% (4/5ths)
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Card 1: EOD */}
              <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
                  <span>Equalized Odds Diff (EOD)</span>
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 font-mono">
                    Parity = 0.00
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">
                    {formatMetric(eodItem?.mitigated_value)}
                  </span>
                  {eodItem?.direction && (
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center space-x-1 ${
                        eodItem.direction === "improved"
                          ? "bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                          : eodItem.direction === "degraded"
                          ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                          : "bg-slate-100 text-slate-700"
                      }`}
                    >
                      {eodItem.direction === "improved" ? (
                        <>
                          <TrendingDown className="w-3 h-3" />
                          <span>Reduced {formatMetric(eodItem.absolute_change)}</span>
                        </>
                      ) : eodItem.direction === "degraded" ? (
                        <>
                          <TrendingUp className="w-3 h-3" />
                          <span>Widened +{formatMetric(eodItem.absolute_change)}</span>
                        </>
                      ) : (
                        <span>Unchanged</span>
                      )}
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-slate-500 flex justify-between pt-1 border-t border-slate-100">
                  <span>Baseline: {formatMetric(eodItem?.baseline_value)}</span>
                  <span>Gap in TPR/FPR</span>
                </div>
              </div>

              {/* Card 2: DPD */}
              <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
                  <span>Demographic Parity (DPD)</span>
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 font-mono">
                    Parity = 0.00
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">
                    {formatMetric(dpdItem?.mitigated_value)}
                  </span>
                  {dpdItem?.direction && (
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center space-x-1 ${
                        dpdItem.direction === "improved"
                          ? "bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                          : dpdItem.direction === "degraded"
                          ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                          : "bg-slate-100 text-slate-700"
                      }`}
                    >
                      {dpdItem.direction === "improved" ? (
                        <>
                          <TrendingDown className="w-3 h-3" />
                          <span>Reduced {formatMetric(dpdItem.absolute_change)}</span>
                        </>
                      ) : dpdItem.direction === "degraded" ? (
                        <>
                          <TrendingUp className="w-3 h-3" />
                          <span>Widened +{formatMetric(dpdItem.absolute_change)}</span>
                        </>
                      ) : (
                        <span>Unchanged</span>
                      )}
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-slate-500 flex justify-between pt-1 border-t border-slate-100">
                  <span>Baseline: {formatMetric(dpdItem?.baseline_value)}</span>
                  <span>Gap in Default Rate</span>
                </div>
              </div>

              {/* Card 3: Approval Disparate Impact */}
              <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
                  <span>Approval Disparate Impact</span>
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 font-mono">
                    Parity = 1.00
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">
                    {formatMetric(diItem?.mitigated_value)}
                  </span>
                  {diItem?.four_fifths_status && (
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                        diItem.four_fifths_status.includes("Passes")
                          ? "bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                          : "bg-amber-50 text-amber-800 border border-amber-200/60"
                      }`}
                    >
                      {diItem.four_fifths_status.includes("Passes") ? "80% Screening Met" : "Outside 80%"}
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-slate-500 flex justify-between pt-1 border-t border-slate-100">
                  <span>Baseline: {formatMetric(diItem?.baseline_value)}</span>
                  <span>Ratio: AR_u / AR_p</span>
                </div>
              </div>

              {/* Card 4: Utility (Accuracy & AUC) */}
              <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
                  <span>Predictive Utility</span>
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 font-mono">
                    Held-out Test
                  </span>
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-xl font-bold text-slate-900 tracking-tight">
                    Acc: {formatMetric(accItem?.mitigated_value, 3)}
                  </span>
                  <span className="text-xs font-semibold text-slate-600">
                    AUC: {aucItem?.mitigated_value !== null && aucItem?.mitigated_value !== undefined ? formatMetric(aucItem.mitigated_value, 3) : "N/A"}
                  </span>
                </div>
                <div className="text-[11px] text-slate-500 flex justify-between pt-1 border-t border-slate-100">
                  <span>
                    ΔAcc: {accItem?.absolute_change !== null && accItem?.absolute_change !== undefined ? `${accItem.absolute_change > 0 ? "+" : ""}${formatMetric(accItem.absolute_change, 3)}` : "N/A"}
                  </span>
                  <span>
                    ΔAUC: {aucItem?.absolute_change !== null && aucItem?.absolute_change !== undefined ? `${aucItem.absolute_change > 0 ? "+" : ""}${formatMetric(aucItem.absolute_change, 3)}` : "N/A"}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* SECTION 2: CHARTS */}
          <div className="space-y-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">
                2. Subgroup Distribution & Error Rate Visualizations
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Visualizing empirical approval rates and equalized-odds components (TPR vs. FPR) across sensitive demographic partitions.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Chart A: Approval Rates */}
              <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wide flex items-center space-x-1.5">
                      <BarChart2 className="w-3.5 h-3.5 text-teal-700" />
                      <span>Credit Approval Rates (Favorable Outcome)</span>
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      P(Y_hat=0 | A=g) for Unprivileged vs. Privileged subgroups
                    </p>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">Higher = More Approvals</span>
                </div>

                <div className="h-64 w-full">
                  {isMounted ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={approvalChartData}
                        margin={{ top: 15, right: 20, left: 0, bottom: 25 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                          dataKey="category"
                          tick={{ fontSize: 11, fill: "#475569" }}
                          interval={0}
                        />
                        <YAxis
                          domain={[50, 100]}
                          tick={{ fontSize: 10, fill: "#64748b" }}
                          unit="%"
                        />
                        <Tooltip
                          formatter={(value: any, name: any) => [
                            `${value}%`,
                            name === "baselineApproval" ? "Baseline Approval Rate" : "Mitigated Approval Rate"
                          ]}
                          contentStyle={{ fontSize: 11, borderRadius: 8, borderColor: "#cbd5e1" }}
                        />
                        <Legend
                          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                          formatter={(val) => (val === "baselineApproval" ? "Baseline Approval Rate" : `Mitigated (${formatMitigationLabel(selectedMitigation)})`)}
                        />
                        <Bar
                          dataKey="baselineApproval"
                          name="baselineApproval"
                          fill="#94a3b8"
                          radius={[4, 4, 0, 0]}
                        />
                        <Bar
                          dataKey="mitigatedApproval"
                          name="mitigatedApproval"
                          fill="#0f766e"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-slate-400">
                      Rendering chart…
                    </div>
                  )}
                </div>

                <p className="text-[11px] text-slate-500 leading-relaxed border-t border-slate-100 pt-2">
                  <strong>Approval Parity Note:</strong> Under demographic parity, both groups receive identical approval rates.
                  The current Disparate Impact ratio is <strong>{formatMetric(diItem?.mitigated_value)}</strong> (Baseline was {formatMetric(diItem?.baseline_value)}).
                </p>
              </div>

              {/* Chart B: Equalized Odds TPR & FPR */}
              <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wide flex items-center space-x-1.5">
                      <Scale className="w-3.5 h-3.5 text-teal-700" />
                      <span>Equalized Odds: Defaulter Recall (TPR) & False Alarms (FPR)</span>
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Comparing true default detection vs. erroneous denial rates
                    </p>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">Equalized Odds</span>
                </div>

                <div className="h-64 w-full">
                  {isMounted ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={eqOddsChartData}
                        margin={{ top: 15, right: 20, left: 0, bottom: 25 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                          dataKey="name"
                          tick={{ fontSize: 9, fill: "#475569" }}
                          interval={0}
                          angle={-10}
                          textAnchor="end"
                        />
                        <YAxis
                          domain={[0, 70]}
                          tick={{ fontSize: 10, fill: "#64748b" }}
                          unit="%"
                        />
                        <Tooltip
                          formatter={(value: any, name: any) => [
                            `${value}%`,
                            name === "tpr" ? "Defaulter Recall (TPR)" : "False Denial Rate (FPR)"
                          ]}
                          contentStyle={{ fontSize: 11, borderRadius: 8, borderColor: "#cbd5e1" }}
                        />
                        <Legend
                          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
                          formatter={(val) => (val === "tpr" ? "Defaulter Recall (TPR)" : "False Alarm (FPR)")}
                        />
                        <Bar
                          dataKey="tpr"
                          name="tpr"
                          fill="#0284c7"
                          radius={[4, 4, 0, 0]}
                        />
                        <Bar
                          dataKey="fpr"
                          name="fpr"
                          fill="#e11d48"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-xs text-slate-400">
                      Rendering chart…
                    </div>
                  )}
                </div>

                <p className="text-[11px] text-slate-500 leading-relaxed border-t border-slate-100 pt-2">
                  <strong>Equalized Odds Principle:</strong> Equalized Odds requires both Defaulter Recall (TPR) and False Denial Rate (FPR) to be equal between groups:
                  Current maximum gap (EOD) = <strong>{formatMetric(eodItem?.mitigated_value)}</strong>.
                </p>
              </div>
            </div>
          </div>

          {/* SECTION 3: SUBGROUP OPERATIONAL TABLE */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                  <span>3. Subgroup Confusion Matrix & Operational Rates</span>
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Complete demographic subgroup counts and rates evaluated on the held-out test partition.
                </p>
              </div>
              <span className="text-[11px] text-slate-500">
                Ground Truth: Y=1 (Default), Y=0 (Non-default)
              </span>
            </div>

            <div className="rounded-xl bg-white border border-slate-200 shadow-xs overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50/80 text-slate-600 border-b border-slate-200 font-semibold text-[11px] uppercase tracking-wider">
                    <tr>
                      <th className="py-2.5 px-3">Pipeline</th>
                      <th className="py-2.5 px-3">Subgroup</th>
                      <th className="py-2.5 px-2">Role</th>
                      <th className="py-2.5 px-2 text-right">Sample N</th>
                      <th className="py-2.5 px-2 text-right">TP</th>
                      <th className="py-2.5 px-2 text-right">TN</th>
                      <th className="py-2.5 px-2 text-right">FP</th>
                      <th className="py-2.5 px-2 text-right">FN</th>
                      <th className="py-2.5 px-3 text-right">Obs. Default Rate</th>
                      <th className="py-2.5 px-3 text-right">Defaulter Recall (TPR)</th>
                      <th className="py-2.5 px-3 text-right">False Alarm (FPR)</th>
                      <th className="py-2.5 px-3 text-right font-bold text-slate-800">Approval Rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {/* Baseline Rows */}
                    {auditData.baseline_subgroups.map((row) => (
                      <tr key={`b-${row.group}`} className="hover:bg-slate-50/60 transition-colors">
                        <td className="py-2 px-3 font-sans font-medium text-slate-600">Baseline</td>
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900 capitalize">
                          {row.group}
                        </td>
                        <td className="py-2 px-2">
                          <span
                            className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                              row.group_role === "unprivileged"
                                ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                                : "bg-sky-50 text-sky-700 border border-sky-200/60"
                            }`}
                          >
                            {row.group_role === "unprivileged" ? "Unprivileged (u)" : "Privileged (p)"}
                          </span>
                        </td>
                        <td className="py-2 px-2 text-right text-slate-800">{row.sample_count}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.tp}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.tn}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.fp}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.fn}</td>
                        <td className="py-2 px-3 text-right text-slate-700">{formatPercent(row.observed_default_rate)}</td>
                        <td className="py-2 px-3 text-right text-slate-700">{formatPercent(row.true_positive_rate)}</td>
                        <td className="py-2 px-3 text-right text-slate-700">{formatPercent(row.false_positive_rate)}</td>
                        <td className="py-2 px-3 text-right font-bold text-slate-900 bg-slate-50/40">
                          {formatPercent(row.approval_rate)}
                        </td>
                      </tr>
                    ))}

                    {/* Mitigated Rows */}
                    {auditData.mitigated_subgroups.map((row) => (
                      <tr key={`m-${row.group}`} className="hover:bg-teal-50/30 transition-colors bg-teal-50/10">
                        <td className="py-2 px-3 font-sans font-medium text-teal-800">
                          Mitigated ({formatMitigationLabel(selectedMitigation).split(" ")[0]})
                        </td>
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900 capitalize">
                          {row.group}
                        </td>
                        <td className="py-2 px-2">
                          <span
                            className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                              row.group_role === "unprivileged"
                                ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                                : "bg-sky-50 text-sky-700 border border-sky-200/60"
                            }`}
                          >
                            {row.group_role === "unprivileged" ? "Unprivileged (u)" : "Privileged (p)"}
                          </span>
                        </td>
                        <td className="py-2 px-2 text-right text-slate-800">{row.sample_count}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.tp}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.tn}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.fp}</td>
                        <td className="py-2 px-2 text-right text-slate-600">{row.fn}</td>
                        <td className="py-2 px-3 text-right text-slate-700">{formatPercent(row.observed_default_rate)}</td>
                        <td className="py-2 px-3 text-right text-slate-700">{formatPercent(row.true_positive_rate)}</td>
                        <td className="py-2 px-3 text-right text-slate-700">{formatPercent(row.false_positive_rate)}</td>
                        <td className="py-2 px-3 text-right font-bold text-teal-900 bg-teal-50/40">
                          {formatPercent(row.approval_rate)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Table Footnotes and Operational Conventions */}
              <div className="p-3.5 bg-slate-50 border-t border-slate-200 text-[11px] text-slate-500 space-y-1">
                <div className="flex items-center space-x-1.5 font-semibold text-slate-700">
                  <Info className="w-3.5 h-3.5 text-teal-700 shrink-0" />
                  <span>Metric Formulas & Target Conventions:</span>
                </div>
                <p>
                  • <strong>Target Variable:</strong> Y=1 represents default / credit risk (unfavorable ground truth); Y=0 represents non-default (favorable ground truth).
                </p>
                <p>
                  • <strong>Prediction Convention:</strong> Positive prediction Y_hat=1 corresponds to default prediction (loan denial). Credit approval event is Y_hat=0.
                </p>
                <p>
                  • <strong>Operational Rates:</strong> Approval Rate = (TN + FN) / N = 1 - Selection Rate. Defaulter Recall (TPR) = TP / (TP + FN). False Alarm (FPR) = FP / (FP + TN).
                </p>
              </div>
            </div>
          </div>

          {/* SECTION 4: EVIDENCE-BASED AUDIT FINDINGS */}
          <div className="space-y-4">
            <div>
              <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
                <span>4. Evidence-Based Audit Findings & Interpretations</span>
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Objective observations derived from empirical test results, including metric trade-offs and sample-size caveats.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {auditData.audit_findings.map((finding, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-xl border text-xs leading-relaxed space-y-1.5 ${
                    finding.includes("conflict")
                      ? "bg-amber-50/70 border-amber-200 text-amber-900"
                      : finding.includes("Sample-size")
                      ? "bg-sky-50/70 border-sky-200 text-sky-900"
                      : finding.includes("Screening threshold")
                      ? "bg-purple-50/70 border-purple-200 text-purple-900"
                      : "bg-white border-slate-200 text-slate-700"
                  }`}
                >
                  <div className="flex items-center space-x-2 font-bold">
                    {finding.includes("conflict") ? (
                      <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
                    ) : finding.includes("Sample-size") ? (
                      <Database className="w-4 h-4 text-sky-600 shrink-0" />
                    ) : finding.includes("Screening threshold") ? (
                      <ShieldAlert className="w-4 h-4 text-purple-600 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-teal-600 shrink-0" />
                    )}
                    <span>
                      {finding.includes("conflict")
                        ? "Fairness Metric Trade-Off"
                        : finding.includes("Sample-size")
                        ? "Sample-Size Sensitivity Notice"
                        : finding.includes("Screening threshold")
                        ? "Regulatory Screening Disclaimer"
                        : "Disparity Audit Summary"}
                    </span>
                  </div>
                  <p className="text-slate-600">{finding}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="p-8 rounded-xl bg-slate-50 border border-dashed border-slate-200 text-center text-xs text-slate-500">
          No empirical audit found for this configuration.
        </div>
      )}

      {/* SECTION 5: EXPLANATION STABILITY (SHAP CONSISTENCY) */}
      <div className="space-y-4 pt-6 border-t border-slate-200">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
            <span>5. Explanation Stability & Local SHAP Consistency (Phase 4 Freeze)</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluating whether fairness mitigation preserves or destabilizes local feature attribution explanations across held-out applicants.
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
            label="Mean Spearman Rank Corr"
            value={currentShap?.mean_rank ? currentShap.mean_rank.toFixed(4) : "N/A"}
            subtext={`95% CI: ${currentShap?.ci95_rank || "N/A"}`}
            badge="Rank Order"
          />
        </div>

        {/* Individual Explanation Comparison from Frozen Research Parquet */}
        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wide flex items-center space-x-2">
                <UserCheck className="w-4 h-4 text-teal-700" />
                <span>Individual Held-Out Applicant Explanation (Frozen Cohort)</span>
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Inspect how baseline vs. reweighed explanations differed for specific held-out applicants evaluated during Phase 4.
              </p>
            </div>

            {/* Applicant Index Selector */}
            <div className="flex items-center space-x-2 text-xs">
              <label className="text-slate-500 font-medium">Applicant Index:</label>
              <select
                value={applicantIdx}
                onChange={(e) => setApplicantIdx(parseInt(e.target.value))}
                className="rounded border border-slate-300 bg-white px-2 py-1 text-slate-800 font-mono text-xs focus:outline-none focus:ring-1 focus:ring-teal-700 shadow-2xs"
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
                feature: formatFeatureLabel(a.feature),
                feature_raw: a.feature_raw || a.feature || "",
                baseline_shap: typeof a.baseline_shap === "number" ? a.baseline_shap : 0,
                reweighed_shap: typeof a.reweighed_shap === "number" ? a.reweighed_shap : 0,
                diff_shap: typeof a.diff_shap === "number" ? a.diff_shap : 0,
                direction: (a.baseline_shap ?? 0) > 0 ? "increases_risk" : "reduces_risk"
              }))}
              title={`Held-Out Applicant #${applicantIdx} — Baseline vs. Reweighed SHAP Attributions`}
              outputSpace={
                selectedModel === "random_forest"
                  ? "Probability Space P(Y=1)"
                  : selectedModel === "logistic_regression"
                  ? "Log-Odds Margin Space (Linear)"
                  : "Margin Space (Tree Ensembles)"
              }
              explainerMethod={
                selectedModel === "random_forest" || selectedModel === "xgboost"
                  ? "TreeExplainer (Exact Tree Path Attribution)"
                  : "LinearExplainer (Analytical Weight Attribution)"
              }
            />
          ) : (
            <div className="h-40 flex items-center justify-center text-slate-400 text-xs italic">
              Loading individual held-out applicant SHAP data…
            </div>
          )}

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-teal-700 shrink-0 mt-0.5" />
            <p className="text-[11px] leading-relaxed">
              <strong>Evaluation Context:</strong> Local explanations reflect genuine held-out test cohort applicants from the frozen Phase 4 evaluation (`dashboard/data/individual_feature_shap.parquet`), allowing direct verification of attribution changes alongside group fairness gains.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
