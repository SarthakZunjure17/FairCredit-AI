"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  BookOpen,
  Scale,
  BarChart2,
  TrendingDown,
  TrendingUp,
  Minus,
  CheckCircle2,
  AlertTriangle,
  Download,
  Filter,
  ArrowUpDown,
  ChevronDown,
  ChevronRight,
  Info,
  ShieldCheck,
  Layers,
  FileSpreadsheet,
  Activity,
  SlidersHorizontal
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Cell
} from "recharts";
import { api } from "@/lib/api";
import {
  FinalResearchResultRow,
  ResearchMatrixRow,
  DatasetKey,
  ModelKey
} from "@/types";
import { MetricCard } from "@/components/MetricCard";
import {
  formatMetric,
  formatPercent,
  formatDelta,
  formatDatasetLabel,
  formatModelLabel,
  formatMitigationLabel
} from "@/lib/utils";

// Custom Collapsible Accordion Section
function ExpandableSection({
  id,
  title,
  isOpen,
  onToggle,
  badge,
  children
}: {
  id: string;
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  badge?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-xs overflow-hidden transition-all duration-200">
      <button
        type="button"
        onClick={onToggle}
        className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-slate-50/80 transition-colors"
        aria-expanded={isOpen}
      >
        <div className="flex items-center space-x-2.5">
          <span className="text-sm font-bold text-slate-900 tracking-tight">{title}</span>
          {badge && (
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 border border-teal-200/60">
              {badge}
            </span>
          )}
        </div>
        <div className="text-slate-400">
          {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </div>
      </button>
      {isOpen && (
        <div className="px-5 pb-5 pt-1 border-t border-slate-100 text-xs sm:text-sm text-slate-600 leading-relaxed space-y-3">
          {children}
        </div>
      )}
    </div>
  );
}

// Custom Tooltip for Recharts
function CustomChartTooltip({ active, payload, label }: any) {
  if (active && payload && payload.length) {
    return (
      <div className="p-3 bg-white border border-slate-200 shadow-md rounded-lg text-xs space-y-1">
        <p className="font-bold text-slate-900">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={`item-${index}`} className="flex items-center justify-between space-x-4">
            <span className="flex items-center space-x-1.5" style={{ color: entry.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              <span>{entry.name}:</span>
            </span>
            <span className="font-mono font-bold text-slate-900">
              {typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}

export default function ResearchPage() {
  const [isMounted, setIsMounted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Authoritative data stores
  const [finalResults, setFinalResults] = useState<FinalResearchResultRow[]>([]);
  const [matrixData, setMatrixData] = useState<ResearchMatrixRow[]>([]);

  // Feature 1: Experiment selection states
  const [selectedDataset, setSelectedDataset] = useState<DatasetKey>("taiwan");
  const [selectedModel, setSelectedModel] = useState<ModelKey>("logistic_regression");
  const [comparisonMode, setComparisonMode] = useState<"reweighing" | "all_mitigations">("reweighing");

  // Chart selection tab
  const [activeChartMetric, setActiveChartMetric] = useState<"auc" | "eod" | "dpd" | "di" | "shap">("auc");

  // Feature 4: Table view mode & filter/sort states
  const [tableViewScope, setTableViewScope] = useState<"six_configs" | "all_mitigations">("six_configs");
  const [tableDatasetFilter, setTableDatasetFilter] = useState<string>("all");
  const [tableModelFilter, setTableModelFilter] = useState<string>("all");
  const [sortColumn, setSortColumn] = useState<string>("Dataset");
  const [sortAsc, setSortAsc] = useState<boolean>(true);

  // Methodology accordion states
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    s1: false,
    s2: false,
    s3: false,
    s4: false,
    s5: false,
    s6: false,
    s7: true,
    s8: false
  });

  useEffect(() => {
    setIsMounted(true);
    async function loadData() {
      try {
        setIsLoading(true);
        setErrorMessage(null);
        const [finalRes, matrixRes] = await Promise.all([
          api.getFinalResearch(),
          api.getResearchMatrix()
        ]);

        if (Array.isArray(finalRes)) {
          setFinalResults(finalRes);
        }
        if (Array.isArray(matrixRes)) {
          setMatrixData(matrixRes);
        }
      } catch (err: any) {
        console.error("Failed to load research datasets:", err);
        setErrorMessage(`Could not load authoritative research artifacts: ${err.message}`);
      } finally {
        setIsLoading(false);
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

  // Find currently selected configuration in the authoritative 6-config results
  const currentConfig = useMemo(() => {
    return finalResults.find(
      (r) =>
        r.Dataset.toLowerCase() === selectedDataset.toLowerCase() &&
        r.Model.toLowerCase() === selectedModel.toLowerCase()
    );
  }, [finalResults, selectedDataset, selectedModel]);

  // Find extended mitigation records for the selected dataset and model (3 rows: Reweighing, Exponentiated Gradient, ThresholdOptimizer)
  const currentMatrixRecords = useMemo(() => {
    return matrixData.filter(
      (r) =>
        r.dataset.toLowerCase() === selectedDataset.toLowerCase() &&
        r.model.toLowerCase() === selectedModel.toLowerCase()
    );
  }, [matrixData, selectedDataset, selectedModel]);

  // Prepare chart data for the selected dataset across all 3 models
  const datasetChartData = useMemo(() => {
    const datasetRows = finalResults.filter(
      (r) => r.Dataset.toLowerCase() === selectedDataset.toLowerCase()
    );

    return datasetRows.map((r) => {
      const modelLabel = formatModelLabel(r.Model);
      const isSelected = r.Model.toLowerCase() === selectedModel.toLowerCase();
      return {
        modelKey: r.Model,
        modelLabel,
        isSelected,
        // Predictive AUC
        baselineAuc: r.Baseline_AUC,
        reweighedAuc: r.Reweighed_AUC,
        deltaAuc: r.Delta_AUC,
        // Fairness Disparity
        baselineEod: r.Baseline_EOD,
        reweighedEod: r.Reweighed_EOD,
        baselineDpd: r.Baseline_DPD,
        reweighedDpd: r.Reweighed_DPD,
        // Disparate Impact
        baselineDi: r.Baseline_Approval_DI ?? 1.0,
        reweighedDi: r.Reweighed_Approval_DI ?? 1.0,
        // SHAP Consistency
        shapCosine: r.Mean_SHAP_Cosine ?? 0,
        shapTop5: r.Mean_Top5_Overlap ?? 0,
        shapSpearman: r.Mean_Spearman ?? 0
      };
    });
  }, [finalResults, selectedDataset, selectedModel]);

  // Filtered and sorted table records (6-config mode)
  const filteredSixConfigs = useMemo(() => {
    let rows = [...finalResults];

    if (tableDatasetFilter !== "all") {
      rows = rows.filter((r) => r.Dataset.toLowerCase() === tableDatasetFilter.toLowerCase());
    }
    if (tableModelFilter !== "all") {
      rows = rows.filter((r) => r.Model.toLowerCase() === tableModelFilter.toLowerCase());
    }

    rows.sort((a: any, b: any) => {
      const valA = a[sortColumn];
      const valB = b[sortColumn];
      if (valA === undefined || valA === null) return 1;
      if (valB === undefined || valB === null) return -1;

      if (typeof valA === "number" && typeof valB === "number") {
        return sortAsc ? valA - valB : valB - valA;
      }
      return sortAsc
        ? String(valA).localeCompare(String(valB))
        : String(valB).localeCompare(String(valA));
    });

    return rows;
  }, [finalResults, tableDatasetFilter, tableModelFilter, sortColumn, sortAsc]);

  // Filtered and sorted table records (18-mitigation matrix mode)
  const filteredMatrixRows = useMemo(() => {
    let rows = [...matrixData];

    if (tableDatasetFilter !== "all") {
      rows = rows.filter((r) => r.dataset.toLowerCase() === tableDatasetFilter.toLowerCase());
    }
    if (tableModelFilter !== "all") {
      rows = rows.filter((r) => r.model.toLowerCase() === tableModelFilter.toLowerCase());
    }

    rows.sort((a: any, b: any) => {
      const keyMap: Record<string, string> = {
        Dataset: "dataset",
        Model: "model",
        Baseline_AUC: "baseline_auc",
        Reweighed_AUC: "mitigated_auc",
        Baseline_EOD: "baseline_eod",
        Reweighed_EOD: "mitigated_eod"
      };
      const mappedCol = keyMap[sortColumn] || sortColumn.toLowerCase();
      const valA = a[mappedCol];
      const valB = b[mappedCol];
      if (valA === undefined || valA === null) return 1;
      if (valB === undefined || valB === null) return -1;

      if (typeof valA === "number" && typeof valB === "number") {
        return sortAsc ? valA - valB : valB - valA;
      }
      return sortAsc
        ? String(valA).localeCompare(String(valB))
        : String(valB).localeCompare(String(valA));
    });

    return rows;
  }, [matrixData, tableDatasetFilter, tableModelFilter, sortColumn, sortAsc]);

  // Sorting handler
  const handleSort = (col: string) => {
    if (sortColumn === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortColumn(col);
      setSortAsc(true);
    }
  };

  // CSV Export Function
  const exportTableToCsv = () => {
    if (tableViewScope === "six_configs") {
      if (!filteredSixConfigs.length) return;

      const headers = [
        "Dataset",
        "Model",
        "Baseline_ROC_AUC",
        "Reweighed_ROC_AUC",
        "Delta_AUC",
        "Baseline_EOD",
        "Reweighed_EOD",
        "Delta_EOD",
        "Baseline_DPD",
        "Reweighed_DPD",
        "Baseline_Approval_DI",
        "Reweighed_Approval_DI",
        "SHAP_Cosine_Similarity",
        "SHAP_Top5_Overlap",
        "SHAP_Spearman_Rank",
        "Explanation_Sample_Size"
      ];

      const csvLines = [headers.join(",")];

      filteredSixConfigs.forEach((r) => {
        const line = [
          r.Dataset,
          r.Model,
          r.Baseline_AUC.toFixed(4),
          r.Reweighed_AUC.toFixed(4),
          r.Delta_AUC.toFixed(4),
          r.Baseline_EOD.toFixed(4),
          r.Reweighed_EOD.toFixed(4),
          r.Delta_EOD.toFixed(4),
          r.Baseline_DPD.toFixed(4),
          r.Reweighed_DPD.toFixed(4),
          r.Baseline_Approval_DI !== undefined ? r.Baseline_Approval_DI.toFixed(4) : "N/A",
          r.Reweighed_Approval_DI !== undefined ? r.Reweighed_Approval_DI.toFixed(4) : "N/A",
          r.Mean_SHAP_Cosine !== undefined ? r.Mean_SHAP_Cosine.toFixed(4) : "N/A",
          r.Mean_Top5_Overlap !== undefined ? r.Mean_Top5_Overlap.toFixed(4) : "N/A",
          r.Mean_Spearman !== undefined ? r.Mean_Spearman.toFixed(4) : "N/A",
          r.Explanation_Sample_Size ?? "N/A"
        ];
        csvLines.push(line.join(","));
      });

      const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute(
        "download",
        `faircredit_model_comparison_benchmark_${tableDatasetFilter}_${tableModelFilter}.csv`
      );
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      if (!filteredMatrixRows.length) return;

      const headers = [
        "Dataset",
        "Model",
        "Mitigation_Method",
        "Baseline_Accuracy",
        "Mitigated_Accuracy",
        "Delta_Accuracy",
        "Baseline_ROC_AUC",
        "Mitigated_ROC_AUC",
        "Delta_ROC_AUC",
        "Baseline_EOD",
        "Mitigated_EOD",
        "Delta_EOD",
        "Baseline_DPD",
        "Mitigated_DPD",
        "Delta_DPD",
        "SHAP_Cosine_Similarity",
        "SHAP_Top5_Overlap",
        "SHAP_Spearman_Rank"
      ];

      const csvLines = [headers.join(",")];

      filteredMatrixRows.forEach((r) => {
        const line = [
          r.dataset,
          r.model,
          r.mitigation_method,
          r.baseline_acc !== undefined ? r.baseline_acc.toFixed(4) : "N/A",
          r.mitigated_acc !== undefined ? r.mitigated_acc.toFixed(4) : "N/A",
          r.delta_acc !== undefined ? r.delta_acc.toFixed(4) : "N/A",
          r.baseline_auc !== undefined ? r.baseline_auc.toFixed(4) : "N/A",
          r.mitigated_auc || "N/A",
          r.delta_auc || "N/A",
          r.baseline_eod !== undefined ? r.baseline_eod.toFixed(4) : "N/A",
          r.mitigated_eod !== undefined ? r.mitigated_eod.toFixed(4) : "N/A",
          r.delta_eod !== undefined ? r.delta_eod.toFixed(4) : "N/A",
          r.baseline_dpd !== undefined ? r.baseline_dpd.toFixed(4) : "N/A",
          r.mitigated_dpd !== undefined ? r.mitigated_dpd.toFixed(4) : "N/A",
          r.delta_dpd !== undefined ? r.delta_dpd.toFixed(4) : "N/A",
          r.shap_cosine_sim !== undefined ? r.shap_cosine_sim.toFixed(4) : "N/A",
          r.shap_top5_overlap !== undefined ? r.shap_top5_overlap.toFixed(4) : "N/A",
          r.shap_spearman_rank !== undefined ? r.shap_spearman_rank.toFixed(4) : "N/A"
        ];
        csvLines.push(line.join(","));
      });

      const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute(
        "download",
        `faircredit_extended_mitigations_matrix_${tableDatasetFilter}_${tableModelFilter}.csv`
      );
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="space-y-10 max-w-6xl mx-auto pb-12">
      {/* Page Header */}
      <div className="border-b border-slate-200 pb-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-xs font-semibold text-teal-800 uppercase tracking-wider">
            <BookOpen className="w-4 h-4 text-teal-600" />
            <span>Research Repository · Model Comparison Lab</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
              <span>Frozen Benchmark Artifacts</span>
            </span>
            <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-100 text-slate-700">
              <span>Seed 42</span>
            </span>
          </div>
        </div>

        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-2">
          Model Comparison Lab & Research Explorer
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1.5 max-w-3xl leading-relaxed">
          Interactive evaluation of empirical trade-offs between predictive performance, group fairness metrics,
          and SHAP explanation consistency across canonical credit-scoring architectures and fairness interventions.
        </p>

        {/* Scope distinction alert */}
        <div className="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 text-xs flex items-start space-x-2.5">
          <Info className="w-4 h-4 text-teal-700 mt-0.5 shrink-0" />
          <div className="space-y-0.5">
            <span className="font-semibold text-slate-900">Research Benchmark Scope:</span>{" "}
            This lab visualizes verified, frozen research experiments executed across held-out test partitions.
            Metrics are derived directly from <code>results/final_research_results.csv</code> and{" "}
            <code>results/mitigation_master_results.csv</code>. It is strictly separate from live applicant scoring.
          </div>
        </div>
      </div>

      {errorMessage && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* FEATURE 1: EXPERIMENT SELECTION BAR */}
      <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-teal-700" />
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              1. Experiment Selector & Configuration
            </h2>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            Active: <strong className="text-slate-800">{formatDatasetLabel(selectedDataset)}</strong> ·{" "}
            <strong className="text-slate-800">{formatModelLabel(selectedModel)}</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Dataset Selector */}
          <div>
            <label className="text-[11px] font-bold uppercase text-slate-500 block mb-1.5">
              Target Dataset
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value as DatasetKey)}
              className="w-full text-xs font-medium rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-700"
            >
              <option value="taiwan">Taiwan Credit Card Default (30,000 obs)</option>
              <option value="german">Statlog German Credit (1,000 obs)</option>
            </select>
            <p className="text-[10px] text-slate-500 mt-1">
              {selectedDataset === "taiwan"
                ? "Protected attribute: Sex (Female/Male) · N_test=6,000"
                : "Protected attribute: Derived Sex (A92 vs A91/3/4) · N_test=200"}
            </p>
          </div>

          {/* Model Selector */}
          <div>
            <label className="text-[11px] font-bold uppercase text-slate-500 block mb-1.5">
              Model Architecture
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
              className="w-full text-xs font-medium rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-700"
            >
              <option value="logistic_regression">Logistic Regression (Linear)</option>
              <option value="random_forest">Random Forest (Bagged Ensembles)</option>
              <option value="xgboost">XGBoost (Gradient Boosted Trees)</option>
            </select>
            <p className="text-[10px] text-slate-500 mt-1">
              Fixed random seed: 42 · Stratified 80/20 train/test split
            </p>
          </div>

          {/* Comparison Scope Selector */}
          <div>
            <label className="text-[11px] font-bold uppercase text-slate-500 block mb-1.5">
              Mitigation Comparison Scope
            </label>
            <select
              value={comparisonMode}
              onChange={(e) => setComparisonMode(e.target.value as any)}
              className="w-full text-xs font-medium rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-700"
            >
              <option value="reweighing">Baseline vs. Reweighing (Frozen Benchmark)</option>
              <option value="all_mitigations">All Mitigations (Reweighing, Exponentiated Gradient, ThresholdOptimizer)</option>
            </select>
            <p className="text-[10px] text-slate-500 mt-1">
              {comparisonMode === "reweighing"
                ? "Core benchmark evaluated with local SHAP consistency"
                : "Includes pre-, in-, and post-processing interventions (18 total runs)"}
            </p>
          </div>
        </div>
      </div>

      {/* FEATURE 2: COMPARISON SUMMARY CARDS FOR SELECTED CONFIGURATION */}
      {currentConfig ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-slate-900">
                2. Configuration Performance, Fairness & Stability Summary
              </h2>
              <p className="text-xs text-slate-500">
                Authoritative empirical figures for{" "}
                <span className="font-semibold text-slate-800">{formatDatasetLabel(currentConfig.Dataset)}</span> using{" "}
                <span className="font-semibold text-slate-800">{formatModelLabel(currentConfig.Model)}</span>.
              </p>
            </div>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-slate-100 text-slate-700">
              {currentConfig.Explanation_Sample_Size ? `N_shap = ${currentConfig.Explanation_Sample_Size}` : "Authoritative"}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Card 1: Predictive AUC */}
            <MetricCard
              label="ROC-AUC Discrimination"
              value={currentConfig.Reweighed_AUC.toFixed(4)}
              subtext="Higher is favorable (↑). Evaluates rank-order scoring capacity across all thresholds."
              delta={{
                value: formatDelta(currentConfig.Delta_AUC).text,
                isFavorable: currentConfig.Delta_AUC >= 0,
                label: `(${currentConfig.Relative_AUC_Change_Percent ? (currentConfig.Relative_AUC_Change_Percent > 0 ? "+" : "") + currentConfig.Relative_AUC_Change_Percent.toFixed(2) + "%" : "from base"})`
              }}
              badge={`Base: ${currentConfig.Baseline_AUC.toFixed(4)}`}
            />

            {/* Card 2: Equalized Odds Difference */}
            <MetricCard
              label="Equalized Odds Difference (EOD)"
              value={currentConfig.Reweighed_EOD.toFixed(4)}
              subtext="Lower disparity is favorable (↓). Max gap between protected groups in TPR and FPR."
              delta={{
                value: formatDelta(currentConfig.Delta_EOD).text,
                isFavorable: currentConfig.Delta_EOD <= 0,
                label: currentConfig.Relative_EOD_Change_Percent ? `${currentConfig.Relative_EOD_Change_Percent > 0 ? "+" : ""}${currentConfig.Relative_EOD_Change_Percent.toFixed(1)}%` : undefined
              }}
              badge={`Base: ${currentConfig.Baseline_EOD.toFixed(4)}`}
            />

            {/* Card 3: Demographic Parity Difference */}
            <MetricCard
              label="Demographic Parity Difference (DPD)"
              value={currentConfig.Reweighed_DPD.toFixed(4)}
              subtext="Lower disparity is favorable (↓). Absolute difference in positive approval rate between groups."
              delta={{
                value: formatDelta(currentConfig.Delta_DPD).text,
                isFavorable: currentConfig.Delta_DPD <= 0,
                label: "abs change"
              }}
              badge={`Base: ${currentConfig.Baseline_DPD.toFixed(4)}`}
            />

            {/* Card 4: Approval Disparate Impact */}
            <MetricCard
              label="Approval Disparate Impact (DI)"
              value={
                currentConfig.Reweighed_Approval_DI !== undefined
                  ? currentConfig.Reweighed_Approval_DI.toFixed(4)
                  : "N/A"
              }
              subtext="Parity = 1.00. Unprivileged / Privileged approval ratio. Four-Fifths rule threshold: ≥ 0.80."
              badge={`Base: ${
                currentConfig.Baseline_Approval_DI !== undefined
                  ? currentConfig.Baseline_Approval_DI.toFixed(4)
                  : "N/A"
              }`}
            />

            {/* Card 5: SHAP Mean Cosine Similarity */}
            <MetricCard
              label="SHAP Cosine Consistency"
              value={
                currentConfig.Mean_SHAP_Cosine !== undefined
                  ? currentConfig.Mean_SHAP_Cosine.toFixed(4)
                  : "N/A"
              }
              subtext={
                currentConfig.Cosine_CI_Lower !== undefined && currentConfig.Cosine_CI_Upper !== undefined
                  ? `95% Bootstrap CI: [${currentConfig.Cosine_CI_Lower.toFixed(4)}, ${currentConfig.Cosine_CI_Upper.toFixed(4)}]`
                  : "Directional vector attribution alignment"
              }
              badge={`N=${currentConfig.Explanation_Sample_Size || "Cohort"}`}
            />

            {/* Card 6: SHAP Feature Rank & Set Overlap */}
            <MetricCard
              label="Top-5 Overlap & Spearman Rank"
              value={
                currentConfig.Mean_Top5_Overlap !== undefined
                  ? `${(currentConfig.Mean_Top5_Overlap * 100).toFixed(1)}% / ${currentConfig.Mean_Spearman?.toFixed(4) || "N/A"}`
                  : "N/A"
              }
              subtext="Top-5 feature set intersection (Jaccard) / Spearman monotonic rank correlation across all features."
              badge="Attribution Stability"
            />
          </div>

          {/* Extended Multi-Mitigation Panel (When all_mitigations selected) */}
          {comparisonMode === "all_mitigations" && (
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-1.5">
                  <Layers className="w-3.5 h-3.5 text-teal-700" />
                  <span>Extended Mitigation Audit: {formatDatasetLabel(selectedDataset)} · {formatModelLabel(selectedModel)}</span>
                </h3>
                <span className="text-[10px] text-slate-500 font-mono">3 Interventions Evaluated</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {currentMatrixRecords.map((mRow, mIdx) => (
                  <div key={mIdx} className="p-3 bg-white rounded-lg border border-slate-200 shadow-xs space-y-2">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
                      <span className="text-xs font-bold text-teal-900">{formatMitigationLabel(mRow.mitigation_method)}</span>
                      <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 uppercase">
                        {mRow.mitigation_method === "reweighing" ? "Pre-Proc" : mRow.mitigation_method === "exponentiated_gradient" ? "In-Proc" : "Post-Proc"}
                      </span>
                    </div>
                    <div className="space-y-1 text-[11px]">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Test Accuracy:</span>
                        <span className="font-mono font-semibold text-slate-800">{formatMetric(mRow.mitigated_acc)} (Base: {formatMetric(mRow.baseline_acc)})</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">ROC-AUC:</span>
                        <span className="font-mono font-semibold text-slate-800">
                          {mRow.mitigated_auc || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Equalized Odds (EOD):</span>
                        <span className="font-mono font-semibold text-teal-700">{formatMetric(mRow.mitigated_eod)} (Base: {formatMetric(mRow.baseline_eod)})</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Demographic Parity (DPD):</span>
                        <span className="font-mono font-semibold text-slate-800">{formatMetric(mRow.mitigated_dpd)} (Base: {formatMetric(mRow.baseline_dpd)})</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">SHAP Cosine Sim:</span>
                        <span className="font-mono font-semibold text-slate-800">
                          {mRow.shap_cosine_sim !== undefined ? mRow.shap_cosine_sim.toFixed(4) : "N/A*"}
                        </span>
                      </div>
                    </div>
                    {mRow.mitigation_method !== "reweighing" && (
                      <p className="text-[10px] text-slate-400 italic">
                        * Note: Continuous AUC is mathematically undefined for step/randomized decisions. SHAP consistency was evaluated on pre-processing Reweighing.
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metric Interpretation Guide Note */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-[11px] text-slate-600 space-y-1">
            <span className="font-semibold text-slate-800">Interpretation Principles:</span>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1">
              <div>• <strong>Higher ROC-AUC:</strong> Indicates superior discrimination capacity between default and non-default borrowers.</div>
              <div>• <strong>Lower EOD & DPD:</strong> Indicates smaller measured statistical disparity across protected demographic groups.</div>
              <div>• <strong>Approval DI Closer to 1.0:</strong> Demonstrates equal approval rates (values ≥ 0.80 satisfy the EEOC four-fifths standard).</div>
              <div>• <strong>Higher SHAP Metrics:</strong> Demonstrates preserved adverse action explanation fidelity across model mitigation.</div>
            </div>
            <p className="text-[10px] text-slate-500 pt-1 italic">
              Notice: Never combine disparate measures into a single universal fairness score. A model configuration must not be characterized as universally "fair" or "unfair" from an isolated metric.
            </p>
          </div>
        </div>
      ) : (
        <div className="p-6 bg-slate-50 rounded-xl border border-dashed border-slate-300 text-center text-xs text-slate-500">
          {isLoading ? "Loading authoritative comparison metrics…" : "No records found for the selected configuration."}
        </div>
      )}

      {/* FEATURE 3: INTERACTIVE VISUALIZATIONS */}
      <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-teal-700" />
              <span>3. Empirical Research Visualizations</span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Comparing Baseline versus Reweighed outcomes across model architectures for{" "}
              <strong>{formatDatasetLabel(selectedDataset)}</strong>.
            </p>
          </div>

          {/* Chart Metric Tabs */}
          <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-lg bg-slate-100 text-xs font-semibold">
            <button
              onClick={() => setActiveChartMetric("auc")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeChartMetric === "auc"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              ROC-AUC
            </button>
            <button
              onClick={() => setActiveChartMetric("eod")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeChartMetric === "eod"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Equalized Odds (EOD)
            </button>
            <button
              onClick={() => setActiveChartMetric("dpd")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeChartMetric === "dpd"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Demographic Parity (DPD)
            </button>
            <button
              onClick={() => setActiveChartMetric("di")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeChartMetric === "di"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Approval DI
            </button>
            <button
              onClick={() => setActiveChartMetric("shap")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activeChartMetric === "shap"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              SHAP Stability
            </button>
          </div>
        </div>

        {/* Chart Container */}
        {isMounted ? (
          <div className="space-y-3">
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                {activeChartMetric === "auc" ? (
                  <BarChart data={datasetChartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="modelLabel" tick={{ fill: "#475569", fontSize: 11 }} />
                    <YAxis
                      domain={[0.7, 0.85]}
                      tick={{ fill: "#475569", fontSize: 11 }}
                      label={{ value: "ROC-AUC", angle: -90, position: "insideLeft", fill: "#64748B", fontSize: 11 }}
                    />
                    <Tooltip content={<CustomChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                    <Bar name="Baseline Model" dataKey="baselineAuc" fill="#64748B" radius={[4, 4, 0, 0]} />
                    <Bar name="Reweighed Model" dataKey="reweighedAuc" fill="#0D9488" radius={[4, 4, 0, 0]} />
                  </BarChart>
                ) : activeChartMetric === "eod" ? (
                  <BarChart data={datasetChartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="modelLabel" tick={{ fill: "#475569", fontSize: 11 }} />
                    <YAxis
                      domain={[0, "auto"]}
                      tick={{ fill: "#475569", fontSize: 11 }}
                      label={{ value: "EOD (Max TPR/FPR Gap)", angle: -90, position: "insideLeft", fill: "#64748B", fontSize: 11 }}
                    />
                    <Tooltip content={<CustomChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                    <ReferenceLine y={0} stroke="#10B981" strokeWidth={1.5} label={{ value: "Parity (0.0)", fill: "#059669", fontSize: 10, position: "right" }} />
                    <Bar name="Baseline EOD" dataKey="baselineEod" fill="#94A3B8" radius={[4, 4, 0, 0]} />
                    <Bar name="Reweighed EOD" dataKey="reweighedEod" fill="#0F766E" radius={[4, 4, 0, 0]} />
                  </BarChart>
                ) : activeChartMetric === "dpd" ? (
                  <BarChart data={datasetChartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="modelLabel" tick={{ fill: "#475569", fontSize: 11 }} />
                    <YAxis
                      domain={[0, "auto"]}
                      tick={{ fill: "#475569", fontSize: 11 }}
                      label={{ value: "DPD (Approval Gap)", angle: -90, position: "insideLeft", fill: "#64748B", fontSize: 11 }}
                    />
                    <Tooltip content={<CustomChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                    <ReferenceLine y={0} stroke="#10B981" strokeWidth={1.5} label={{ value: "Parity (0.0)", fill: "#059669", fontSize: 10, position: "right" }} />
                    <Bar name="Baseline DPD" dataKey="baselineDpd" fill="#94A3B8" radius={[4, 4, 0, 0]} />
                    <Bar name="Reweighed DPD" dataKey="reweighedDpd" fill="#0D9488" radius={[4, 4, 0, 0]} />
                  </BarChart>
                ) : activeChartMetric === "di" ? (
                  <BarChart data={datasetChartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="modelLabel" tick={{ fill: "#475569", fontSize: 11 }} />
                    <YAxis
                      domain={[0.7, 1.3]}
                      tick={{ fill: "#475569", fontSize: 11 }}
                      label={{ value: "Approval Disparate Impact Ratio", angle: -90, position: "insideLeft", fill: "#64748B", fontSize: 11 }}
                    />
                    <Tooltip content={<CustomChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                    <ReferenceLine y={1.0} stroke="#059669" strokeWidth={2} label={{ value: "Parity (1.0)", fill: "#059669", fontSize: 10, position: "right" }} />
                    <ReferenceLine y={0.8} stroke="#E11D48" strokeDasharray="4 4" strokeWidth={1.5} label={{ value: "80% Four-Fifths Rule (0.80)", fill: "#E11D48", fontSize: 10, position: "right" }} />
                    <Bar name="Baseline Approval DI" dataKey="baselineDi" fill="#64748B" radius={[4, 4, 0, 0]} />
                    <Bar name="Reweighed Approval DI" dataKey="reweighedDi" fill="#0D9488" radius={[4, 4, 0, 0]} />
                  </BarChart>
                ) : (
                  <BarChart data={datasetChartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                    <XAxis dataKey="modelLabel" tick={{ fill: "#475569", fontSize: 11 }} />
                    <YAxis
                      domain={[0.8, 1.0]}
                      tick={{ fill: "#475569", fontSize: 11 }}
                      label={{ value: "Consistency Index", angle: -90, position: "insideLeft", fill: "#64748B", fontSize: 11 }}
                    />
                    <Tooltip content={<CustomChartTooltip />} />
                    <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                    <ReferenceLine y={1.0} stroke="#059669" strokeWidth={1.5} label={{ value: "Perfect Stability (1.0)", fill: "#059669", fontSize: 10, position: "right" }} />
                    <Bar name="Mean Cosine Similarity" dataKey="shapCosine" fill="#0F766E" radius={[4, 4, 0, 0]} />
                    <Bar name="Top-5 Feature Overlap" dataKey="shapTop5" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                    <Bar name="Spearman Rank Corr" dataKey="shapSpearman" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                )}
              </ResponsiveContainer>
            </div>

            {/* Contextual Chart Caption */}
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs text-slate-600">
              {activeChartMetric === "auc" && (
                <p>
                  <strong>ROC-AUC Interpretation:</strong> Evaluates discrimination without selecting an arbitrary classification threshold.
                  Reweighing preserves strong predictive discrimination across all model families (AUC changes are negligible: ΔAUC ≤ 0.014).
                </p>
              )}
              {activeChartMetric === "eod" && (
                <p>
                  <strong>Equalized Odds Interpretation:</strong> Measures the maximum gap in True Positive Rate (TPR) and False Positive Rate (FPR) across protected demographic groups.
                  Lower values indicate smaller disparity. For German Random Forest, Reweighing achieved a 60% disparity reduction (0.1250 → 0.0500).
                </p>
              )}
              {activeChartMetric === "dpd" && (
                <p>
                  <strong>Demographic Parity Interpretation:</strong> Measures the difference in overall approval rates between groups.
                  Target is 0.0. While Reweighing directly adjusts sample weights, its impact on DPD is heterogeneous across architectures.
                </p>
              )}
              {activeChartMetric === "di" && (
                <p>
                  <strong>Approval Disparate Impact Interpretation:</strong> The ratio of unprivileged to privileged group approval rates.
                  Parity is 1.00. Under regulatory guidelines (EEOC 80% rule), values ≥ 0.80 avoid presumptive adverse impact.
                </p>
              )}
              {activeChartMetric === "shap" && (
                <p>
                  <strong>SHAP Consistency Interpretation:</strong> Evaluates whether fairness mitigation destabilizes local feature attribution.
                  Linear models exhibit near-unity vector alignment (cosine 0.9996 / 0.9983), while tree ensembles show feature ranking re-orderings (top-5 overlap 0.90–0.98).
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="h-72 w-full flex items-center justify-center bg-slate-50 text-xs text-slate-400">
            Initializing chart renderer…
          </div>
        )}
      </div>

      {/* FEATURE 5: EVIDENCE-BASED INTERPRETATION PANEL */}
      {currentConfig && (
        <div className="p-5 rounded-xl bg-gradient-to-br from-teal-50/70 to-slate-50 border border-teal-200 shadow-xs space-y-3">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-teal-700" />
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              4. Evidence-Based Research Synthesis: {formatDatasetLabel(currentConfig.Dataset)} · {formatModelLabel(currentConfig.Model)}
            </h2>
          </div>

          <div className="text-xs sm:text-sm text-slate-700 leading-relaxed space-y-2">
            <p>
              Under fixed experimental initialization (seed 42, 80/20 stratified split), applying Reweighing to{" "}
              <strong>{formatModelLabel(currentConfig.Model)}</strong> on{" "}
              <strong>{formatDatasetLabel(currentConfig.Dataset)}</strong> yielded the following empirical shifts:
            </p>

            <ul className="list-disc pl-5 space-y-1 text-xs">
              <li>
                <strong>Predictive Utility:</strong> ROC-AUC shifted from{" "}
                <span className="font-mono font-semibold">{currentConfig.Baseline_AUC.toFixed(4)}</span> to{" "}
                <span className="font-mono font-semibold">{currentConfig.Reweighed_AUC.toFixed(4)}</span> (Δ ={" "}
                <span className="font-mono">{formatDelta(currentConfig.Delta_AUC).text}</span>, {currentConfig.Relative_AUC_Change_Percent ? `${currentConfig.Relative_AUC_Change_Percent > 0 ? "+" : ""}${currentConfig.Relative_AUC_Change_Percent.toFixed(2)}%` : "stable"}),
                demonstrating that predictive discrimination capacity remained {Math.abs(currentConfig.Delta_AUC) < 0.005 ? "essentially intact" : "minimally perturbed"}.
              </li>
              <li>
                <strong>Group Fairness:</strong> Equalized Odds Difference moved from{" "}
                <span className="font-mono font-semibold">{currentConfig.Baseline_EOD.toFixed(4)}</span> to{" "}
                <span className="font-mono font-semibold">{currentConfig.Reweighed_EOD.toFixed(4)}</span> (Δ ={" "}
                <span className="font-mono">{formatDelta(currentConfig.Delta_EOD).text}</span>).
                Demographic Parity Difference moved from{" "}
                <span className="font-mono font-semibold">{currentConfig.Baseline_DPD.toFixed(4)}</span> to{" "}
                <span className="font-mono font-semibold">{currentConfig.Reweighed_DPD.toFixed(4)}</span>.
                {currentConfig.Delta_EOD < 0 ? (
                  <span className="text-emerald-700 font-medium"> Disparity narrowed under the Equalized Odds criterion.</span>
                ) : currentConfig.Delta_EOD > 0 ? (
                  <span className="text-amber-800 font-medium"> Measured disparity widened slightly under this criterion, illustrating metric divergence.</span>
                ) : (
                  <span> Disparity remained unchanged.</span>
                )}
              </li>
              <li>
                <strong>Approval Disparate Impact:</strong> Baseline DI of{" "}
                <span className="font-mono font-semibold">{currentConfig.Baseline_Approval_DI?.toFixed(4) || "N/A"}</span> vs. Reweighed DI of{" "}
                <span className="font-mono font-semibold">{currentConfig.Reweighed_Approval_DI?.toFixed(4) || "N/A"}</span>.
                {currentConfig.Reweighed_Approval_DI && currentConfig.Reweighed_Approval_DI >= 0.8 ? (
                  <span className="text-emerald-700 font-medium"> Satisfies the EEOC Four-Fifths parity threshold (≥ 0.80).</span>
                ) : (
                  <span className="text-rose-700 font-medium"> Falls below the 0.80 reference threshold.</span>
                )}
              </li>
              <li>
                <strong>Explanation Consistency:</strong> Over the held-out test cohort (N={currentConfig.Explanation_Sample_Size || "Cohort"}),
                mean SHAP cosine similarity was{" "}
                <span className="font-mono font-semibold">{currentConfig.Mean_SHAP_Cosine?.toFixed(4) || "N/A"}</span>{" "}
                (95% CI: [{currentConfig.Cosine_CI_Lower?.toFixed(4) || "N/A"}, {currentConfig.Cosine_CI_Upper?.toFixed(4) || "N/A"}]),
                with Top-5 feature overlap of{" "}
                <span className="font-mono font-semibold">{currentConfig.Mean_Top5_Overlap ? `${(currentConfig.Mean_Top5_Overlap * 100).toFixed(1)}%` : "N/A"}</span>{" "}
                and Spearman rank correlation of{" "}
                <span className="font-mono font-semibold">{currentConfig.Mean_Spearman?.toFixed(4) || "N/A"}</span>.
              </li>
            </ul>

            <div className="pt-2 border-t border-teal-200/60 text-xs font-medium text-slate-800">
              <strong>Key Scientific Conclusion:</strong> Fairness mitigation exhibited heterogeneous effects across configurations rather than a universal trade-off. Improving one metric does not guarantee universal improvement in all fairness dimensions or preserved explanation stability.
            </div>
          </div>
        </div>
      )}

      {/* FEATURE 4: ALL-CONFIGURATION FILTERABLE COMPARISON TABLE */}
      <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
              <FileSpreadsheet className="w-4 h-4 text-teal-700" />
              <span>5. Master Configuration Comparison Table</span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Verified historical benchmark results. Unavailable metrics are explicitly denoted as N/A.
            </p>
          </div>

          <div className="flex items-center space-x-3">
            {/* View Scope Segmented Button */}
            <div className="flex items-center p-1 rounded-lg bg-slate-100 text-xs font-semibold">
              <button
                type="button"
                onClick={() => setTableViewScope("six_configs")}
                className={`px-3 py-1 rounded-md transition-colors ${
                  tableViewScope === "six_configs"
                    ? "bg-white text-slate-900 shadow-xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Benchmark (6 Configs)
              </button>
              <button
                type="button"
                onClick={() => setTableViewScope("all_mitigations")}
                className={`px-3 py-1 rounded-md transition-colors ${
                  tableViewScope === "all_mitigations"
                    ? "bg-white text-slate-900 shadow-xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Extended Matrix (18 Runs)
              </button>
            </div>

            <button
              type="button"
              onClick={exportTableToCsv}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Filtered CSV</span>
            </button>
          </div>
        </div>

        {/* Table Filters Bar */}
        <div className="flex flex-wrap items-center gap-3 text-xs bg-slate-50 p-3 rounded-lg border border-slate-200">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-700">Filter Dataset:</span>
            <select
              value={tableDatasetFilter}
              onChange={(e) => setTableDatasetFilter(e.target.value)}
              className="px-2 py-1 bg-white border border-slate-300 rounded text-slate-800 font-medium"
            >
              <option value="all">All Datasets</option>
              <option value="taiwan">Taiwan Only</option>
              <option value="german">German Only</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-700">Filter Model:</span>
            <select
              value={tableModelFilter}
              onChange={(e) => setTableModelFilter(e.target.value)}
              className="px-2 py-1 bg-white border border-slate-300 rounded text-slate-800 font-medium"
            >
              <option value="all">All Models</option>
              <option value="logistic_regression">Logistic Regression</option>
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
            </select>
          </div>

          <div className="ml-auto text-[11px] text-slate-500">
            Showing{" "}
            <strong>
              {tableViewScope === "six_configs" ? filteredSixConfigs.length : filteredMatrixRows.length}
            </strong>{" "}
            records
          </div>
        </div>

        {/* Comparison Table */}
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          {tableViewScope === "six_configs" ? (
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-100/90 text-slate-700 font-semibold border-b border-slate-200 select-none">
                  <th
                    onClick={() => handleSort("Dataset")}
                    className="p-3 cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center space-x-1">
                      <span>Dataset</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Model")}
                    className="p-3 cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center space-x-1">
                      <span>Model</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Baseline_AUC")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Base AUC</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Reweighed_AUC")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Rew AUC</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Delta_AUC")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Δ AUC</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Baseline_EOD")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Base EOD</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Reweighed_EOD")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Rew EOD</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Baseline_DPD")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Base DPD</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Reweighed_DPD")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Rew DPD</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Reweighed_Approval_DI")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Rew DI</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Mean_SHAP_Cosine")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>SHAP Cosine</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Mean_Top5_Overlap")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Top-5</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                  <th
                    onClick={() => handleSort("Mean_Spearman")}
                    className="p-3 text-right cursor-pointer hover:bg-slate-200/70 transition-colors whitespace-nowrap"
                  >
                    <div className="flex items-center justify-end space-x-1">
                      <span>Spearman</span>
                      <ArrowUpDown className="w-3 h-3 text-slate-400" />
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                {filteredSixConfigs.map((r, idx) => {
                  const isSelected =
                    r.Dataset.toLowerCase() === selectedDataset.toLowerCase() &&
                    r.Model.toLowerCase() === selectedModel.toLowerCase();

                  return (
                    <tr
                      key={idx}
                      onClick={() => {
                        setSelectedDataset(r.Dataset as DatasetKey);
                        setSelectedModel(r.Model as ModelKey);
                      }}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-teal-50/80 font-semibold text-teal-950 border-l-4 border-l-teal-700"
                          : "hover:bg-slate-50 text-slate-800"
                      }`}
                    >
                      <td className="p-3 font-sans font-medium capitalize">{formatDatasetLabel(r.Dataset)}</td>
                      <td className="p-3 font-sans capitalize">{formatModelLabel(r.Model)}</td>
                      <td className="p-3 text-right text-slate-600">{r.Baseline_AUC.toFixed(4)}</td>
                      <td className="p-3 text-right font-bold text-slate-900">{r.Reweighed_AUC.toFixed(4)}</td>
                      <td
                        className={`p-3 text-right font-bold ${
                          r.Delta_AUC >= 0 ? "text-emerald-700" : "text-rose-700"
                        }`}
                      >
                        {formatDelta(r.Delta_AUC).text}
                      </td>
                      <td className="p-3 text-right text-slate-600">{r.Baseline_EOD.toFixed(4)}</td>
                      <td
                        className={`p-3 text-right font-bold ${
                          r.Delta_EOD <= 0 ? "text-emerald-700" : "text-amber-800"
                        }`}
                      >
                        {r.Reweighed_EOD.toFixed(4)}
                      </td>
                      <td className="p-3 text-right text-slate-600">{r.Baseline_DPD.toFixed(4)}</td>
                      <td className="p-3 text-right text-slate-900">{r.Reweighed_DPD.toFixed(4)}</td>
                      <td className="p-3 text-right text-slate-900">
                        {r.Reweighed_Approval_DI !== undefined ? r.Reweighed_Approval_DI.toFixed(4) : "N/A"}
                      </td>
                      <td className="p-3 text-right font-bold text-teal-800">
                        {r.Mean_SHAP_Cosine !== undefined ? r.Mean_SHAP_Cosine.toFixed(4) : "N/A"}
                      </td>
                      <td className="p-3 text-right text-slate-700">
                        {r.Mean_Top5_Overlap !== undefined ? `${(r.Mean_Top5_Overlap * 100).toFixed(0)}%` : "N/A"}
                      </td>
                      <td className="p-3 text-right text-slate-700">
                        {r.Mean_Spearman !== undefined ? r.Mean_Spearman.toFixed(4) : "N/A"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-100/90 text-slate-700 font-semibold border-b border-slate-200 select-none">
                  <th className="p-3">Dataset</th>
                  <th className="p-3">Model</th>
                  <th className="p-3">Mitigation</th>
                  <th className="p-3 text-right">Acc (Base)</th>
                  <th className="p-3 text-right">Acc (Mit)</th>
                  <th className="p-3 text-right">AUC (Base)</th>
                  <th className="p-3 text-right">AUC (Mit)</th>
                  <th className="p-3 text-right">EOD (Base)</th>
                  <th className="p-3 text-right">EOD (Mit)</th>
                  <th className="p-3 text-right">DPD (Mit)</th>
                  <th className="p-3 text-right">SHAP Cosine</th>
                  <th className="p-3 text-right">Top-5</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                {filteredMatrixRows.map((r, idx) => (
                  <tr key={idx} className="hover:bg-slate-50 text-slate-800">
                    <td className="p-3 font-sans font-medium capitalize">{formatDatasetLabel(r.dataset)}</td>
                    <td className="p-3 font-sans capitalize">{formatModelLabel(r.model)}</td>
                    <td className="p-3 font-sans text-teal-800 font-semibold capitalize">
                      {formatMitigationLabel(r.mitigation_method)}
                    </td>
                    <td className="p-3 text-right text-slate-600">{formatMetric(r.baseline_acc)}</td>
                    <td className="p-3 text-right text-slate-900 font-bold">{formatMetric(r.mitigated_acc)}</td>
                    <td className="p-3 text-right text-slate-600">{formatMetric(r.baseline_auc)}</td>
                    <td className="p-3 text-right text-slate-900">
                      {r.mitigated_auc && r.mitigated_auc !== "N/A" ? (
                        formatMetric(r.mitigated_auc)
                      ) : (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">N/A</span>
                      )}
                    </td>
                    <td className="p-3 text-right text-slate-600">{formatMetric(r.baseline_eod)}</td>
                    <td className="p-3 text-right font-bold text-teal-700">{formatMetric(r.mitigated_eod)}</td>
                    <td className="p-3 text-right text-slate-700">{formatMetric(r.mitigated_dpd)}</td>
                    <td className="p-3 text-right text-indigo-700 font-semibold">
                      {r.shap_cosine_sim !== undefined && r.shap_cosine_sim !== null ? (
                        formatMetric(r.shap_cosine_sim)
                      ) : (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-400">N/A</span>
                      )}
                    </td>
                    <td className="p-3 text-right text-slate-600">
                      {r.shap_top5_overlap !== undefined && r.shap_top5_overlap !== null ? (
                        `${(r.shap_top5_overlap * 100).toFixed(0)}%`
                      ) : (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-400">N/A</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* FEATURE 6: RESEARCH METHODOLOGY & INTEGRITY ACCORDION */}
      <div className="space-y-4">
        <div className="border-b border-slate-200 pb-2">
          <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
            <BookOpen className="w-4 h-4 text-teal-700" />
            <span>6. Research Methodology, Integrity & Protocols</span>
          </h2>
          <p className="text-xs text-slate-500">
            Rigorous documentation of experimental design, data partitions, metric formulas, and architectural boundaries.
          </p>
        </div>

        {/* Section 1: Research Question & Motivation */}
        <ExpandableSection
          id="s1"
          title="1. Core Research Question & Motivation"
          isOpen={!!openSections["s1"]}
          onToggle={() => toggleSection("s1")}
        >
          <div className="p-3.5 rounded-lg bg-teal-50/70 border border-teal-200/60 text-teal-950 font-medium">
            <strong>Core Research Question:</strong> “When fairness mitigation is applied to machine-learning credit-scoring models, how does it affect predictive performance, group fairness, and explanation consistency?”
          </div>
          <p>
            Credit scoring systems rely heavily on algorithmic classification (e.g. loan approvals, risk-tiered interest rates). Under regulatory frameworks such as the U.S. Equal Credit Opportunity Act (ECOA) and EU AI Act, lenders face dual requirements: models must demonstrate non-discrimination across protected groups and must provide individualized adverse action notices (explanations).
          </p>
          <p>
            While substantial literature investigates the Pareto trade-off between predictive accuracy and group fairness, critical questions remain regarding whether fairness interventions destabilize feature attribution profiles (SHAP explanations) for individual applicants.
          </p>
        </ExpandableSection>

        {/* Section 2: Canonical Datasets & Protected Attributes */}
        <ExpandableSection
          id="s2"
          title="2. Canonical Datasets & Protected Attributes"
          isOpen={!!openSections["s2"]}
          onToggle={() => toggleSection("s2")}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 space-y-1.5">
              <h4 className="font-bold text-slate-900">Taiwan Credit Card Default (2005)</h4>
              <p>• <strong>Sample Size:</strong> 30,000 observations (6,000 test set, stratified 80/20 split).</p>
              <p>• <strong>Target Variable:</strong> Default payment next month (22.1% prevalence).</p>
              <p>• <strong>Primary Protected Attribute:</strong> Sex (female 60.4%, male 39.6%).</p>
              <p>• <strong>Predictive Features:</strong> 23 features (limit balance, bill amounts, repayment records). Sensitive sex is strictly excluded from feature matrix X.</p>
            </div>
            <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 space-y-1.5">
              <h4 className="font-bold text-slate-900">Statlog German Credit (1994)</h4>
              <p>• <strong>Sample Size:</strong> 1,000 observations (200 test set, stratified 80/20 split).</p>
              <p>• <strong>Target Variable:</strong> Credit risk (1=bad credit, 0=good credit; 30.0% default rate).</p>
              <p>• <strong>Primary Protected Attribute:</strong> Derived sex from personal status (A92=female, A91/A93/A94=male).</p>
              <p>• <strong>Predictive Features:</strong> 20 credit attributes. Derived sex is strictly excluded from feature matrix X.</p>
            </div>
          </div>
        </ExpandableSection>

        {/* Section 3: Model Architectures & Training Protocol */}
        <ExpandableSection
          id="s3"
          title="3. Model Architectures & Training Protocol"
          isOpen={!!openSections["s3"]}
          onToggle={() => toggleSection("s3")}
        >
          <p>
            Three canonical architectures spanning linear, bagging, and boosting paradigms were evaluated:
          </p>
          <ul className="list-disc pl-5 space-y-1.5 text-slate-600">
            <li><strong>Logistic Regression:</strong> L-BFGS solver, max_iter=1000, standardized numerical features.</li>
            <li><strong>Random Forest:</strong> 100 trees, unconstrained depth, parallelized bootstrap ensemble.</li>
            <li><strong>XGBoost:</strong> 100 boosting rounds, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8.</li>
          </ul>
          <p>
            All splits and initializations used fixed <code>random_state=42</code>. Preprocessors (OneHotEncoder with handle_unknown=&apos;ignore&apos; and StandardScaler) were fitted strictly on training partitions to prevent data leakage.
          </p>
        </ExpandableSection>

        {/* Section 4: Fairness Mitigation Interventions */}
        <ExpandableSection
          id="s4"
          title="4. Evaluated Fairness Mitigation Interventions"
          isOpen={!!openSections["s4"]}
          onToggle={() => toggleSection("s4")}
        >
          <div className="space-y-2.5">
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
              <strong className="text-slate-900">1. Pre-Processing — Reweighing (Kamiran & Calders, 2012):</strong>
              <p className="mt-1">
                Assigns statistical independence weights to training samples based on the joint distribution of target and protected attribute, training an adjusted model under weighted empirical risk minimization.
              </p>
            </div>
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
              <strong className="text-slate-900">2. In-Processing — Exponentiated Gradient (Agarwal et al., 2018):</strong>
              <p className="mt-1">
                Formulates fair classification as a two-player zero-sum game subject to an EqualizedOdds constraint, returning a randomized mixture of base estimators.
              </p>
            </div>
            <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
              <strong className="text-slate-900">3. Post-Processing — ThresholdOptimizer (Hardt et al., 2016):</strong>
              <p className="mt-1">
                Calibrates group-specific classification decision thresholds on a pre-fit model to equalize true positive and false positive rates without altering scoring weights.
              </p>
            </div>
          </div>
        </ExpandableSection>

        {/* Section 5: Metrics & Auditing Standards */}
        <ExpandableSection
          id="s5"
          title="5. Auditing Standards & Metric Definitions"
          isOpen={!!openSections["s5"]}
          onToggle={() => toggleSection("s5")}
        >
          <p><strong>Predictive Performance:</strong> Accuracy, Balanced Accuracy, and ROC-AUC. Following strict auditing standards, ROC-AUC is reported as N/A for step/binary decision rules.</p>
          <div className="space-y-1.5 pt-1">
            <p><strong>Group Fairness Criteria:</strong></p>
            <ul className="list-disc pl-5 space-y-1 text-slate-600">
              <li><strong>Equalized Odds Difference (EOD):</strong> Maximum difference in TPR and FPR between protected groups: max(|TPR_unprivileged - TPR_privileged|, |FPR_unprivileged - FPR_privileged|). Target: 0.</li>
              <li><strong>Demographic Parity Difference (DPD):</strong> Difference in positive selection rates between groups: |P(Y_hat=1 | unprivileged) - P(Y_hat=1 | privileged)|. Target: 0.</li>
              <li><strong>Approval Disparate Impact (DI):</strong> Ratio of unprivileged to privileged positive prediction rates: P(Y_hat=0 | unprivileged) / P(Y_hat=0 | privileged). Target: 1.0 (Four-Fifths Rule ≥ 0.80).</li>
            </ul>
          </div>
        </ExpandableSection>

        {/* Section 6: SHAP Methodology & Architectural Spaces */}
        <ExpandableSection
          id="s6"
          title="6. SHAP Methodology & Architectural Differences"
          isOpen={!!openSections["s6"]}
          onToggle={() => toggleSection("s6")}
        >
          <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-xs space-y-1">
            <h4 className="font-bold flex items-center space-x-1.5">
              <AlertTriangle className="w-4 h-4 text-amber-700" />
              <span>Architectural Output Space Distinction:</span>
            </h4>
            <p>
              SHAP output spaces differ fundamentally across model families: LinearExplainer outputs margin space (log-odds), TreeExplainer for Random Forest outputs probability space, and TreeExplainer for XGBoost outputs margin log-odds.
              <strong> Raw SHAP magnitudes must not be directly compared across different model architectures without appropriate justification.</strong>
            </p>
          </div>

          <p className="pt-1">
            Local SHAP attributions explaining Default Risk (Y=1) were calculated for identical held-out test cohorts (Taiwan N=500, German N=200) under Baseline vs. Reweighed models:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-slate-600">
            <li><strong>Cosine Similarity:</strong> Measures directional attribution alignment in multidimensional space: (s_base · s_rew) / (||s_base|| ||s_rew||).</li>
            <li><strong>Top-5 Feature Overlap:</strong> Jaccard set intersection of the 5 highest-magnitude features.</li>
            <li><strong>Spearman Rank Correlation:</strong> Monotonic rank preservation across all features.</li>
            <li><strong>Confidence Intervals:</strong> Derived via 1,000 bootstrap iterations over the test explanation cohorts.</li>
          </ul>
        </ExpandableSection>

        {/* Section 7: Key Empirical Findings */}
        <ExpandableSection
          id="s7"
          title="7. Key Empirical Findings (Heterogeneous Effects)"
          isOpen={!!openSections["s7"]}
          onToggle={() => toggleSection("s7")}
          badge="Core Research Result"
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

          <div className="space-y-2 pt-1 text-xs sm:text-sm">
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
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 space-y-1.5 text-xs">
            <h4 className="font-bold text-slate-900 flex items-center space-x-1.5">
              <ShieldCheck className="w-4 h-4 text-teal-700" />
              <span>Research Scope Boundaries</span>
            </h4>
            <ul className="list-disc pl-5 space-y-1">
              <li>• <strong>Public Benchmark Datasets:</strong> Findings are derived from Taiwan (2005) and German (1994) public benchmarks and may not generalize to production credit portfolios with modern regulatory data.</li>
              <li>• <strong>Observational Disparities:</strong> Statistical disparities reflect observed historical dataset biases and do not establish causal discrimination.</li>
              <li>• <strong>Explanation Proxy:</strong> SHAP represents model feature attribution, not ground-truth causal mechanisms in borrower financial outcomes.</li>
              <li>• <strong>Prototype Boundary:</strong> This system is strictly an academic model-auditing prototype, not an automated lending underwriting engine.</li>
            </ul>
          </div>
        </ExpandableSection>
      </div>
    </div>
  );
}
