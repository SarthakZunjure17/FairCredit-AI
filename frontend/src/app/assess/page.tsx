"use client";

import React, { useState, useEffect } from "react";
import {
  BarChart3,
  Sparkles,
  RefreshCw,
  AlertCircle,
  TrendingDown,
  TrendingUp,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2
} from "lucide-react";
import { api } from "@/lib/api";
import {
  ConfigurationsResponse,
  AssessmentResponse,
  ExplanationResponse,
  DatasetKey,
  ModelKey
} from "@/types";
import { MetricCard } from "@/components/MetricCard";
import { ShapBarChart } from "@/components/ShapBarChart";
import { DATASET_LABELS, MODEL_LABELS, formatPercent } from "@/lib/utils";

export default function AssessPage() {
  const [configData, setConfigData] = useState<ConfigurationsResponse | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetKey>("german");
  const [selectedModel, setSelectedModel] = useState<ModelKey>("random_forest");
  const [features, setFeatures] = useState<Record<string, any>>({});
  
  const [isLoading, setIsLoading] = useState(false);
  const [assessmentResult, setAssessmentResult] = useState<AssessmentResponse | null>(null);
  const [explanationResult, setExplanationResult] = useState<ExplanationResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Load backend configurations & metadata on mount
  useEffect(() => {
    async function loadConfig() {
      try {
        const res = await api.getConfigurations();
        setConfigData(res);
        // Set initial features from default sample
        const schema = res.dataset_schemas["german"];
        if (schema && schema.sample_applicants?.applicant_a_low_risk) {
          setFeatures(schema.sample_applicants.applicant_a_low_risk);
        }
      } catch (err: any) {
        setErrorMessage(`Backend connection error: ${err.message}. Ensure FastAPI is running on port 8000.`);
      }
    }
    loadConfig();
  }, []);

  // Update feature form when dataset changes
  const handleDatasetChange = (newDs: DatasetKey) => {
    setSelectedDataset(newDs);
    setAssessmentResult(null);
    setExplanationResult(null);
    if (configData?.dataset_schemas[newDs]) {
      const sample = configData.dataset_schemas[newDs].sample_applicants?.applicant_a_low_risk || {};
      setFeatures(sample);
    }
  };

  // Load a demonstration preset applicant
  const handleLoadPreset = (presetKey: "applicant_a_low_risk" | "applicant_b_higher_risk") => {
    if (configData?.dataset_schemas[selectedDataset]) {
      const sample = configData.dataset_schemas[selectedDataset].sample_applicants?.[presetKey];
      if (sample) {
        setFeatures(sample);
        setAssessmentResult(null);
        setExplanationResult(null);
      }
    }
  };

  // Update individual feature input
  const handleFeatureChange = (key: string, val: any) => {
    setFeatures((prev) => ({
      ...prev,
      [key]: val
    }));
  };

  // Run assessment & local SHAP explanation
  const handleRunAssessment = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage(null);

    try {
      // 1. Live model prediction (Baseline vs Reweighed)
      const predRes = await api.predictApplicant(selectedDataset, selectedModel, features);
      setAssessmentResult(predRes);

      // 2. Live local SHAP explanation computation
      const expRes = await api.explainApplicant(selectedDataset, selectedModel, features);
      setExplanationResult(expRes);
    } catch (err: any) {
      setErrorMessage(`Inference failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const schemaInfo = configData?.dataset_schemas[selectedDataset];

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Page Header */}
      <div>
        <div className="flex items-center space-x-2 text-xs font-semibold text-teal-800 uppercase tracking-wider">
          <BarChart3 className="w-4 h-4 text-teal-600" />
          <span>Experience A</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-1">
          Interactive Credit Risk Assessment
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-3xl">
          Score credit default risk on verified saved models and observe how pre-processing fairness Reweighing alters
          the applicant’s estimated default probability and local SHAP feature attributions.
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start space-x-2.5">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Assessment Error</p>
            <p className="mt-0.5 leading-normal">{errorMessage}</p>
          </div>
        </div>
      )}

      {/* Grid: Left Column = Form / Selectors, Right Column = Real Output */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Form Column (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <form onSubmit={handleRunAssessment} className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-5">
            {/* Model & Dataset Selectors */}
            <div className="space-y-3 border-b border-slate-100 pb-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                1. Assessment Configuration
              </h3>

              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-700">Benchmark Dataset</label>
                <select
                  value={selectedDataset}
                  onChange={(e) => handleDatasetChange(e.target.value as DatasetKey)}
                  className="w-full text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700"
                >
                  <option value="german">Statlog German Credit (N=1,000, 20 features)</option>
                  <option value="taiwan">Taiwan Credit Card Default (N=30,000, 23 features)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-700">Model Architecture</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
                  className="w-full text-xs font-medium rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-700"
                >
                  <option value="random_forest">Random Forest Classifier (Ensemble)</option>
                  <option value="xgboost">XGBoost Classifier (Gradient Boosting)</option>
                  <option value="logistic_regression">Logistic Regression (Linear Baseline)</option>
                </select>
              </div>

              {/* Sample Profile Presets */}
              <div className="pt-2">
                <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1.5">
                  <span className="font-medium">Quick Synthetic Presets:</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => handleLoadPreset("applicant_a_low_risk")}
                    className="px-2.5 py-1.5 text-xs font-medium rounded border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 text-center transition-colors"
                  >
                    Applicant A (Low Risk)
                  </button>
                  <button
                    type="button"
                    onClick={() => handleLoadPreset("applicant_b_higher_risk")}
                    className="px-2.5 py-1.5 text-xs font-medium rounded border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 text-center transition-colors"
                  >
                    Applicant B (High Risk)
                  </button>
                </div>
              </div>
            </div>

            {/* Applicant Features Form Fields */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  2. Financial Profile Features
                </h3>
                <span className="text-[10px] text-teal-800 bg-teal-50 px-2 py-0.5 rounded font-medium border border-teal-200/60">
                  Sex Excluded from X
                </span>
              </div>

              {selectedDataset === "german" ? (
                /* German Credit Feature Fields */
                <div className="space-y-3 text-xs max-h-96 overflow-y-auto pr-1">
                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Duration (Months)</label>
                      <input
                        type="number"
                        min="4"
                        max="72"
                        value={features["duration_months"] ?? 24}
                        onChange={(e) => handleFeatureChange("duration_months", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Credit Amount (DM)</label>
                      <input
                        type="number"
                        min="250"
                        max="20000"
                        value={features["credit_amount"] ?? 2500}
                        onChange={(e) => handleFeatureChange("credit_amount", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Checking Status</label>
                      <select
                        value={features["checking_status"] ?? "A14"}
                        onChange={(e) => handleFeatureChange("checking_status", e.target.value)}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value="A11">&lt; 0 DM (A11)</option>
                        <option value="A12">0 to 200 DM (A12)</option>
                        <option value="A13">&gt;= 200 DM (A13)</option>
                        <option value="A14">No Checking Account (A14)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Credit History</label>
                      <select
                        value={features["credit_history"] ?? "A32"}
                        onChange={(e) => handleFeatureChange("credit_history", e.target.value)}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value="A30">Delayed in Past (A30)</option>
                        <option value="A31">Critical Account (A31)</option>
                        <option value="A32">Existing Paid Duly (A32)</option>
                        <option value="A33">Delay in Paying Off (A33)</option>
                        <option value="A34">Critical / Other Credits (A34)</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Savings Status</label>
                      <select
                        value={features["savings_status"] ?? "A65"}
                        onChange={(e) => handleFeatureChange("savings_status", e.target.value)}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value="A61">&lt; 100 DM (A61)</option>
                        <option value="A62">100 to 500 DM (A62)</option>
                        <option value="A63">500 to 1000 DM (A63)</option>
                        <option value="A64">&gt;= 1000 DM (A64)</option>
                        <option value="A65">Unknown / No Savings (A65)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Employment Duration</label>
                      <select
                        value={features["employment_duration"] ?? "A73"}
                        onChange={(e) => handleFeatureChange("employment_duration", e.target.value)}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value="A71">Unemployed (A71)</option>
                        <option value="A72">&lt; 1 Year (A72)</option>
                        <option value="A73">1 to 4 Years (A73)</option>
                        <option value="A74">4 to 7 Years (A74)</option>
                        <option value="A75">&gt;= 7 Years (A75)</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Applicant Age (Years)</label>
                      <input
                        type="number"
                        min="18"
                        max="75"
                        value={features["age"] ?? 35}
                        onChange={(e) => handleFeatureChange("age", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Installment Rate (% Inc)</label>
                      <input
                        type="number"
                        min="1"
                        max="4"
                        value={features["installment_rate"] ?? 2}
                        onChange={(e) => handleFeatureChange("installment_rate", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="text-[11px] font-medium text-slate-600">Credit Purpose</label>
                    <select
                      value={features["purpose"] ?? "A40"}
                      onChange={(e) => handleFeatureChange("purpose", e.target.value)}
                      className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                    >
                      <option value="A40">Car (New) (A40)</option>
                      <option value="A41">Car (Used) (A41)</option>
                      <option value="A42">Furniture / Equipment (A42)</option>
                      <option value="A43">Radio / Television (A43)</option>
                      <option value="A44">Domestic Appliances (A44)</option>
                      <option value="A45">Repairs (A45)</option>
                      <option value="A46">Education (A46)</option>
                      <option value="A49">Business (A49)</option>
                    </select>
                  </div>
                </div>
              ) : (
                /* Taiwan Credit Feature Fields */
                <div className="space-y-3 text-xs max-h-96 overflow-y-auto pr-1">
                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Credit Limit (NTD)</label>
                      <input
                        type="number"
                        min="10000"
                        max="1000000"
                        step="10000"
                        value={features["limit_bal"] ?? 100000}
                        onChange={(e) => handleFeatureChange("limit_bal", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Age</label>
                      <input
                        type="number"
                        min="20"
                        max="80"
                        value={features["age"] ?? 30}
                        onChange={(e) => handleFeatureChange("age", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Education</label>
                      <select
                        value={features["education"] ?? 2}
                        onChange={(e) => handleFeatureChange("education", parseInt(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value={1}>Graduate School (1)</option>
                        <option value={2}>University (2)</option>
                        <option value={3}>High School (3)</option>
                        <option value={4}>Others (4)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Marriage Status</label>
                      <select
                        value={features["marriage"] ?? 2}
                        onChange={(e) => handleFeatureChange("marriage", parseInt(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value={1}>Married (1)</option>
                        <option value={2}>Single (2)</option>
                        <option value={3}>Others (3)</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="text-[10px] font-medium text-slate-600">Pay Status Sep (pay_0)</label>
                      <input
                        type="number"
                        min="-2"
                        max="8"
                        value={features["pay_0"] ?? 0}
                        onChange={(e) => handleFeatureChange("pay_0", parseInt(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2 py-1 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] font-medium text-slate-600">Pay Status Aug (pay_2)</label>
                      <input
                        type="number"
                        min="-2"
                        max="8"
                        value={features["pay_2"] ?? 0}
                        onChange={(e) => handleFeatureChange("pay_2", parseInt(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2 py-1 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] font-medium text-slate-600">Pay Status Jul (pay_3)</label>
                      <input
                        type="number"
                        min="-2"
                        max="8"
                        value={features["pay_3"] ?? 0}
                        onChange={(e) => handleFeatureChange("pay_3", parseInt(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2 py-1 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Bill Amount Sep (NTD)</label>
                      <input
                        type="number"
                        value={features["bill_amt1"] ?? 5000}
                        onChange={(e) => handleFeatureChange("bill_amt1", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Payment Amount Sep (NTD)</label>
                      <input
                        type="number"
                        value={features["pay_amt1"] ?? 2000}
                        onChange={(e) => handleFeatureChange("pay_amt1", parseFloat(e.target.value))}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Run Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center space-x-2 py-2.5 px-4 rounded-lg bg-teal-800 hover:bg-teal-900 text-white font-semibold text-xs shadow-xs transition-colors disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Inference & SHAP Values…</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-teal-300" />
                  <span>Run Credit-Risk Assessment</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Output Column (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {assessmentResult ? (
            <div className="space-y-6">
              {/* Prediction Summary Header */}
              <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div>
                    <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                      Model Inference Result
                    </span>
                    <h3 className="text-base font-bold text-slate-900 capitalize">
                      {DATASET_LABELS[assessmentResult.dataset]} — {MODEL_LABELS[assessmentResult.model]}
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-teal-800 bg-teal-50 px-2.5 py-1 rounded border border-teal-200">
                    Live Evaluation
                  </span>
                </div>

                {/* Baseline vs Reweighed Probability Comparison */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {/* Baseline Card */}
                  <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-1">
                    <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
                      <span>Baseline Model</span>
                      <span className="font-semibold text-slate-700">{assessmentResult.baseline.prediction_label}</span>
                    </div>
                    <div className="text-2xl font-bold text-slate-900">
                      {formatPercent(assessmentResult.baseline.default_probability)}
                    </div>
                    <p className="text-[11px] text-slate-500">
                      Estimated Probability of Default P(Y=1)
                    </p>
                  </div>

                  {/* Reweighed Card */}
                  <div className="p-4 rounded-lg bg-teal-50/70 border border-teal-200 space-y-1">
                    <div className="flex items-center justify-between text-xs text-teal-800 font-medium">
                      <span>Reweighed Model (Mitigated)</span>
                      <span className="font-semibold text-teal-900">{assessmentResult.reweighed.prediction_label}</span>
                    </div>
                    <div className="text-2xl font-bold text-teal-950">
                      {formatPercent(assessmentResult.reweighed.default_probability)}
                    </div>
                    <p className="text-[11px] text-teal-700">
                      Fairness-Adjusted Default Risk Estimate
                    </p>
                  </div>
                </div>

                {/* Shift / Delta Interpretation */}
                <div className="p-3.5 rounded-lg bg-white border border-slate-200 text-xs space-y-1">
                  <div className="flex items-center space-x-2 font-semibold text-slate-800">
                    {assessmentResult.delta.direction === "decreased" ? (
                      <TrendingDown className="w-4 h-4 text-emerald-600" />
                    ) : assessmentResult.delta.direction === "increased" ? (
                      <TrendingUp className="w-4 h-4 text-amber-600" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-slate-600" />
                    )}
                    <span>
                      Reweighing Shift: {assessmentResult.delta.percent_difference > 0 ? `+${assessmentResult.delta.percent_difference}%` : `${assessmentResult.delta.percent_difference}%`}
                    </span>
                  </div>
                  <p className="text-slate-600 text-[11px] leading-relaxed">
                    {assessmentResult.delta.interpretation}
                  </p>
                </div>
              </div>

              {/* Local SHAP Explanations Section */}
              <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
                <div className="border-b border-slate-100 pb-3">
                  <h3 className="text-sm font-bold text-slate-900">
                    Local SHAP Feature Attributions
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Measuring the individual marginal contribution of this applicant’s financial features toward default risk.
                  </p>
                </div>

                {explanationResult ? (
                  <div className="space-y-4">
                    <ShapBarChart
                      data={explanationResult.top_contributors}
                      title="Top 10 Feature Contributions to Default Prediction"
                      outputSpace={explanationResult.output_space}
                    />

                    <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
                      <p className="font-semibold text-slate-800">Interpretation Note</p>
                      <p className="text-[11px] leading-relaxed">
                        {explanationResult.summary}
                      </p>
                      <p className="text-[10px] text-slate-400 pt-1 italic">
                        {explanationResult.disclaimer}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="h-48 flex items-center justify-center text-slate-400 text-xs italic">
                    Computing local feature explanations…
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="p-12 rounded-xl bg-white border border-dashed border-slate-300 text-center space-y-3">
              <div className="w-12 h-12 rounded-full bg-slate-100 text-slate-400 mx-auto flex items-center justify-center">
                <BarChart3 className="w-6 h-6" />
              </div>
              <h3 className="font-semibold text-slate-800 text-sm">Ready to Score Synthetic Applicant</h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
                Select your dataset and model architecture, adjust applicant financial parameters or choose a preset,
                then click <strong>Run Credit-Risk Assessment</strong> to initiate real machine-learning inference.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
