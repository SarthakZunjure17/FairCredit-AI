"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  BarChart3,
  Sparkles,
  RefreshCw,
  AlertCircle,
  TrendingDown,
  TrendingUp,
  Minus,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  Info,
  Sliders,
  Scale,
  Activity,
  HelpCircle,
  AlertTriangle,
  ArrowUpDown
} from "lucide-react";
import { api } from "@/lib/api";
import {
  ConfigurationsResponse,
  AssessmentResponse,
  ExplanationResponse,
  DatasetKey,
  ModelKey
} from "@/types";
import { ShapBarChart } from "@/components/ShapBarChart";
import {
  DATASET_LABELS,
  MODEL_LABELS,
  formatPercent,
  formatDatasetLabel,
  formatModelLabel,
  formatFeatureLabel
} from "@/lib/utils";

export default function AssessPage() {
  const [configData, setConfigData] = useState<ConfigurationsResponse | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetKey>("german");
  const [selectedModel, setSelectedModel] = useState<ModelKey>("random_forest");
  const [features, setFeatures] = useState<Record<string, any>>({});
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  
  const [isLoading, setIsLoading] = useState(false);
  const [assessmentResult, setAssessmentResult] = useState<AssessmentResponse | null>(null);
  const [explanationResult, setExplanationResult] = useState<ExplanationResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeExplanationView, setActiveExplanationView] = useState<"chart" | "reranking_table">("chart");

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
    setErrorMessage(null);
    setValidationErrors({});
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
        setErrorMessage(null);
        setValidationErrors({});
      }
    }
  };

  // In-line field validation
  const validateField = (key: string, val: any): string | null => {
    if (selectedDataset === "german") {
      if (key === "duration_months") {
        const num = parseFloat(val);
        if (isNaN(num) || num < 4 || num > 72) return "Duration must be between 4 and 72 months.";
      } else if (key === "credit_amount") {
        const num = parseFloat(val);
        if (isNaN(num) || num <= 0) return "Credit amount must be strictly positive.";
      } else if (key === "age") {
        const num = parseFloat(val);
        if (isNaN(num) || num < 18 || num > 120) return "Age must be between 18 and 120 years.";
      } else if (key === "installment_rate") {
        const num = parseFloat(val);
        if (isNaN(num) || num < 1 || num > 4) return "Installment rate must be between 1 and 4.";
      }
    } else if (selectedDataset === "taiwan") {
      if (key === "limit_bal") {
        const num = parseFloat(val);
        if (isNaN(num) || num <= 0) return "Credit limit must be strictly positive.";
      } else if (key === "age") {
        const num = parseFloat(val);
        if (isNaN(num) || num < 18 || num > 120) return "Age must be between 18 and 120 years.";
      } else if (key.startsWith("pay_") && !key.startsWith("pay_amt")) {
        const num = parseInt(val);
        if (isNaN(num) || num < -2 || num > 8) return "Pay status must be between -2 and 8.";
      }
    }
    return null;
  };

  // Update individual feature input
  const handleFeatureChange = (key: string, val: any) => {
    const error = validateField(key, val);
    setValidationErrors((prev) => {
      const updated = { ...prev };
      if (error) {
        updated[key] = error;
      } else {
        delete updated[key];
      }
      return updated;
    });

    setFeatures((prev) => ({
      ...prev,
      [key]: val
    }));
  };

  // Run assessment & local SHAP explanation
  const handleRunAssessment = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    // Check for existing validation errors
    if (Object.keys(validationErrors).length > 0) {
      setErrorMessage("Please correct the highlighted input validation errors before running assessment.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      // 1. Live model prediction (Baseline vs Reweighed) using identical profile
      const predRes = await api.predictApplicant(selectedDataset, selectedModel, features);
      setAssessmentResult(predRes);

      // 2. Live local SHAP explanation computation
      const expRes = await api.explainApplicant(selectedDataset, selectedModel, features);
      setExplanationResult(expRes);
    } catch (err: any) {
      setAssessmentResult(null);
      setExplanationResult(null);
      setErrorMessage(`Inference failed: ${err.message}. Please verify the backend service is running and retry.`);
    } finally {
      setIsLoading(false);
    }
  };

  // Compute feature re-ranking differences for the explanation table
  const featureRankingDiffs = useMemo(() => {
    if (!explanationResult || !explanationResult.top_contributors) return [];

    return explanationResult.top_contributors.map((item, index) => {
      const bShap = item.baseline_shap;
      const rShap = item.reweighed_shap;
      const diffShap = item.diff_shap;
      const bSign = bShap > 0.001 ? "Increases Risk (+)" : bShap < -0.001 ? "Reduces Risk (-)" : "Neutral";
      const rSign = rShap > 0.001 ? "Increases Risk (+)" : rShap < -0.001 ? "Reduces Risk (-)" : "Neutral";
      const signFlipped = (bShap > 0.001 && rShap < -0.001) || (bShap < -0.001 && rShap > 0.001);

      return {
        rank: index + 1,
        featureName: formatFeatureLabel(item.feature),
        rawFeature: item.feature_raw || item.feature,
        baselineShap: bShap,
        reweighedShap: rShap,
        diffShap: diffShap,
        baselineSign: bSign,
        reweighedSign: rSign,
        signFlipped,
        absShift: Math.abs(diffShap)
      };
    });
  }, [explanationResult]);

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-12">
      {/* Page Header */}
      <div className="border-b border-slate-200 pb-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2 text-xs font-semibold text-teal-800 uppercase tracking-wider">
            <BarChart3 className="w-4 h-4 text-teal-600" />
            <span>Applicant Explanation Studio · Stage 2</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold bg-sky-50 text-sky-800 border border-sky-200">
              <Activity className="w-3.5 h-3.5 text-sky-600" />
              <span>Live Inference &amp; Local SHAP</span>
            </span>
            <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-100 text-slate-700">
              <span>Seed 42 Pipeline</span>
            </span>
          </div>
        </div>

        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 mt-2">
          Applicant Credit Risk &amp; Explanation Studio
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 mt-1.5 max-w-3xl leading-relaxed">
          Audit how pre-processing <strong>Reweighing</strong> fairness mitigation alters the estimated risk score,
          classification decision, and local SHAP feature attributions for an identical validated applicant profile.
        </p>

        {/* Audit scope banner */}
        <div className="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200 text-slate-700 text-xs flex items-start space-x-2.5">
          <ShieldCheck className="w-4 h-4 text-teal-700 mt-0.5 shrink-0" />
          <div className="space-y-0.5">
            <span className="font-semibold text-slate-900">Protected Attribute Protocol:</span>{" "}
            Sensitive demographic attributes (e.g. Sex / Derived Sex) are strictly excluded from predictive feature input matrix <code>X</code> in compliance with ECOA anti-discrimination auditing protocols. They are retained strictly for aggregate group fairness evaluation.
          </div>
        </div>
      </div>

      {errorMessage && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start justify-between gap-3">
          <div className="flex items-start space-x-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Assessment Request Failed</p>
              <p className="mt-0.5 leading-normal">{errorMessage}</p>
              <p className="mt-1 text-[11px] text-rose-700 italic">No fabricated or estimated values are displayed.</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => handleRunAssessment()}
            className="shrink-0 px-3 py-1.5 rounded bg-rose-600 hover:bg-rose-700 text-white font-medium text-xs transition-colors flex items-center space-x-1"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Retry</span>
          </button>
        </div>
      )}

      {/* Grid: Left Column = Form / Selectors, Right Column = Assessment & SHAP Outputs */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Form Column (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <form onSubmit={handleRunAssessment} className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-5">
            {/* Step 1: Model & Dataset Selectors */}
            <div className="space-y-3 border-b border-slate-100 pb-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                  <Sliders className="w-3.5 h-3.5 text-teal-700" />
                  <span>1. Configuration &amp; Model</span>
                </h3>
                <span className="text-[10px] text-slate-400 font-mono">Step 1 of 2</span>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-700">Benchmark Dataset</label>
                <select
                  value={selectedDataset}
                  onChange={(e) => handleDatasetChange(e.target.value as DatasetKey)}
                  className="w-full text-xs font-medium rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-700"
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
                  className="w-full text-xs font-medium rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-700"
                >
                  <option value="random_forest">Random Forest Classifier (Bagged Trees)</option>
                  <option value="xgboost">XGBoost Classifier (Gradient Boosted Trees)</option>
                  <option value="logistic_regression">Logistic Regression (Linear Baseline)</option>
                </select>
              </div>

              {/* Sample Profile Presets */}
              <div className="pt-2">
                <div className="flex items-center justify-between text-[11px] text-slate-500 mb-1.5">
                  <span className="font-medium">Demonstration Profiles:</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => handleLoadPreset("applicant_a_low_risk")}
                    className="px-2.5 py-1.5 text-xs font-medium rounded-lg border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 text-center transition-colors"
                  >
                    Applicant A (Low Risk)
                  </button>
                  <button
                    type="button"
                    onClick={() => handleLoadPreset("applicant_b_higher_risk")}
                    className="px-2.5 py-1.5 text-xs font-medium rounded-lg border border-slate-200 bg-slate-50 hover:bg-slate-100 text-slate-700 text-center transition-colors"
                  >
                    Applicant B (High Risk)
                  </button>
                </div>
              </div>
            </div>

            {/* Step 2: Applicant Features Form Fields */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                  2. Financial Profile Features
                </h3>
                <span className="text-[10px] text-teal-800 bg-teal-50 px-2 py-0.5 rounded font-medium border border-teal-200">
                  Validated Input Schema
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
                        className={`w-full text-xs rounded border px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700 ${
                          validationErrors["duration_months"] ? "border-rose-400 bg-rose-50" : "border-slate-300"
                        }`}
                      />
                      {validationErrors["duration_months"] && (
                        <p className="text-[10px] text-rose-600 mt-0.5">{validationErrors["duration_months"]}</p>
                      )}
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Credit Amount (DM)</label>
                      <input
                        type="number"
                        min="250"
                        max="20000"
                        value={features["credit_amount"] ?? 2500}
                        onChange={(e) => handleFeatureChange("credit_amount", parseFloat(e.target.value))}
                        className={`w-full text-xs rounded border px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700 ${
                          validationErrors["credit_amount"] ? "border-rose-400 bg-rose-50" : "border-slate-300"
                        }`}
                      />
                      {validationErrors["credit_amount"] && (
                        <p className="text-[10px] text-rose-600 mt-0.5">{validationErrors["credit_amount"]}</p>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Checking Status</label>
                      <select
                        value={features["checking_status"] ?? "A11"}
                        onChange={(e) => handleFeatureChange("checking_status", e.target.value)}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value="A11">&lt; 0 DM (A11)</option>
                        <option value="A12">0 to &lt; 200 DM (A12)</option>
                        <option value="A13">&gt;= 200 DM / Salary (A13)</option>
                        <option value="A14">No Checking Account (A14)</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Savings Status</label>
                      <select
                        value={features["savings_status"] ?? "A21"}
                        onChange={(e) => handleFeatureChange("savings_status", e.target.value)}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value="A21">&lt; 100 DM (A21)</option>
                        <option value="A22">100 to &lt; 500 DM (A22)</option>
                        <option value="A23">500 to &lt; 1000 DM (A23)</option>
                        <option value="A24">&gt;= 1000 DM (A24)</option>
                        <option value="A25">Unknown / No Savings (A25)</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[11px] font-medium text-slate-600">Credit History</label>
                      <select
                        value={features["credit_history"] ?? "A32"}
                        onChange={(e) => handleFeatureChange("credit_history", e.target.value)}
                        className="w-full text-xs rounded border border-slate-300 px-2.5 py-1.5 bg-white focus:outline-none focus:ring-1 focus:ring-teal-700"
                      >
                        <option value="A30">No Credits Taken (A30)</option>
                        <option value="A31">All Credits Paid Back Duly (A31)</option>
                        <option value="A32">Existing Credits Paid Duly (A32)</option>
                        <option value="A33">Delay in Paying in Past (A33)</option>
                        <option value="A34">Critical Account / Other Credits (A34)</option>
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
                        className={`w-full text-xs rounded border px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700 ${
                          validationErrors["age"] ? "border-rose-400 bg-rose-50" : "border-slate-300"
                        }`}
                      />
                      {validationErrors["age"] && (
                        <p className="text-[10px] text-rose-600 mt-0.5">{validationErrors["age"]}</p>
                      )}
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
                        className={`w-full text-xs rounded border px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-teal-700 ${
                          validationErrors["limit_bal"] ? "border-rose-400 bg-rose-50" : "border-slate-300"
                        }`}
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
                      <label className="text-[10px] font-medium text-slate-600">Pay Sep (pay_0)</label>
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
                      <label className="text-[10px] font-medium text-slate-600">Pay Aug (pay_2)</label>
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
                      <label className="text-[10px] font-medium text-slate-600">Pay Jul (pay_3)</label>
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
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading || Object.keys(validationErrors).length > 0}
                className="w-full py-2.5 px-4 rounded-lg bg-teal-800 hover:bg-teal-900 disabled:bg-slate-300 text-white font-semibold text-xs transition-colors flex items-center justify-center space-x-2 shadow-xs"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Computing Predictions &amp; SHAP…</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 text-teal-300" />
                    <span>Run Credit-Risk &amp; Explanation Audit</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Right Output Column (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {assessmentResult ? (
            <div className="space-y-6">
              {/* Step 3: Side-by-Side Risk Assessment Card */}
              <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div>
                    <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                      <Scale className="w-4 h-4 text-teal-700" />
                      <span>3. Baseline vs. Reweighed Risk Assessment</span>
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Evaluated on identical profile inputs across pre-processing mitigation.
                    </p>
                  </div>
                  <span className="text-[11px] font-mono text-slate-600 bg-slate-100 px-2.5 py-1 rounded">
                    {formatModelLabel(assessmentResult.model)}
                  </span>
                </div>

                {/* Side-by-Side Prediction Display */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Baseline Card */}
                  <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200 space-y-2.5">
                    <div className="flex items-center justify-between text-xs text-slate-700 font-medium border-b border-slate-200/60 pb-2">
                      <div>
                        <span className="font-bold text-slate-900 text-sm">Baseline Model</span>
                        <p className="text-[10px] text-slate-500">Unmitigated empirical model</p>
                      </div>
                      <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded ${
                        assessmentResult.baseline.prediction === 1
                          ? "bg-rose-100 text-rose-800"
                          : "bg-emerald-100 text-emerald-800"
                      }`}>
                        {assessmentResult.baseline.prediction === 1 ? "Class 1: Default Risk" : "Class 0: Low Risk"}
                      </span>
                    </div>

                    <div>
                      <span className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider block">
                        Estimated Default Probability
                      </span>
                      <div className="text-3xl font-extrabold tracking-tight text-slate-900 mt-0.5">
                        {formatPercent(assessmentResult.baseline.default_probability)}
                      </div>
                      <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                        P(Default = 1 | X) = {assessmentResult.baseline.default_probability.toFixed(4)}
                      </p>
                    </div>

                    <div className="pt-1 text-[10px] text-slate-500 border-t border-slate-200/50">
                      Threshold: 0.50 · {assessmentResult.baseline.output_space}
                    </div>
                  </div>

                  {/* Reweighed Card */}
                  <div className="p-4 rounded-xl bg-teal-50/70 border border-teal-200 space-y-2.5">
                    <div className="flex items-center justify-between text-xs text-teal-900 font-medium border-b border-teal-200/60 pb-2">
                      <div>
                        <span className="font-bold text-teal-950 text-sm">Reweighed Model</span>
                        <p className="text-[10px] text-teal-700">Pre-processing sample weight mitigation</p>
                      </div>
                      <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded ${
                        assessmentResult.reweighed.prediction === 1
                          ? "bg-rose-100 text-rose-800"
                          : "bg-emerald-100 text-emerald-800"
                      }`}>
                        {assessmentResult.reweighed.prediction === 1 ? "Class 1: Default Risk" : "Class 0: Low Risk"}
                      </span>
                    </div>

                    <div>
                      <span className="text-[11px] font-semibold text-teal-800 uppercase tracking-wider block">
                        Estimated Default Probability
                      </span>
                      <div className="text-3xl font-extrabold tracking-tight text-teal-950 mt-0.5">
                        {formatPercent(assessmentResult.reweighed.default_probability)}
                      </div>
                      <p className="text-[10px] text-teal-700 font-mono mt-0.5">
                        P(Default = 1 | X, weights) = {assessmentResult.reweighed.default_probability.toFixed(4)}
                      </p>
                    </div>

                    <div className="pt-1 text-[10px] text-teal-700 border-t border-teal-200/50">
                      Threshold: 0.50 · {assessmentResult.reweighed.output_space}
                    </div>
                  </div>
                </div>

                {/* Shift / Delta Interpretation */}
                {(() => {
                  const pBase = assessmentResult.baseline.default_probability;
                  const pRew = assessmentResult.reweighed.default_probability;
                  const diff = Number((pRew - pBase).toFixed(4));
                  const diffPctPoints = Number((diff * 100).toFixed(2));
                  const signStr = diffPctPoints > 0 ? `+${diffPctPoints.toFixed(2)}` : diffPctPoints.toFixed(2);

                  return (
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2 font-semibold text-slate-900">
                          {diff < -0.0005 ? (
                            <TrendingDown className="w-4 h-4 text-emerald-600" />
                          ) : diff > 0.0005 ? (
                            <TrendingUp className="w-4 h-4 text-amber-600" />
                          ) : (
                            <Minus className="w-4 h-4 text-slate-500" />
                          )}
                          <span>
                            Reweighing Risk Shift: {signStr} percentage points ({signStr}%)
                          </span>
                        </div>
                        <span className="text-[10px] text-slate-500 font-mono">
                          Reweighed − Baseline
                        </span>
                      </div>

                      <p className="text-slate-600 text-xs leading-relaxed">
                        {assessmentResult.delta.interpretation}
                      </p>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-2 border-t border-slate-200/70 text-[11px]">
                        <div>
                          <strong className="text-slate-800">Score Representation:</strong> Model-estimated default probability under test distribution.
                        </div>
                        <div>
                          <strong className="text-slate-800">Calibration Status:</strong> Uncalibrated raw probability score.
                        </div>
                      </div>
                    </div>
                  );
                })()}

                {/* Scope & Interpretation Callout */}
                <div className="p-3.5 rounded-lg bg-amber-50/70 border border-amber-200 text-amber-900 text-xs space-y-1">
                  <div className="flex items-center space-x-1.5 font-bold">
                    <AlertTriangle className="w-4 h-4 text-amber-700" />
                    <span>Scientific &amp; Regulatory Meaning:</span>
                  </div>
                  <ul className="list-disc pl-5 space-y-0.5 text-[11px] leading-relaxed">
                    <li><strong>Research Model Output:</strong> Both values are mathematical risk scores estimated for algorithmic fairness research. They do <em>not</em> constitute a credit bureau score or an adverse action notice.</li>
                    <li><strong>No Calibrated Guarantee:</strong> Uncalibrated model scores should not be interpreted as certified empirical default frequencies.</li>
                    <li><strong>No Causal Claim:</strong> Altering feature values reflects hypothetical model sensitivity, not real-world causal mechanisms in borrower financial behavior.</li>
                  </ul>
                </div>
              </div>

              {/* Step 4: Local SHAP Explanation Studio */}
              <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
                <div className="border-b border-slate-100 pb-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2">
                      <Sparkles className="w-4 h-4 text-teal-700" />
                      <span>4. Local SHAP Explanation Studio</span>
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Marginal feature attributions toward Default Risk (Y=1) for this specific applicant profile.
                    </p>
                  </div>

                  {/* Toggle between Chart and Re-ranking Table */}
                  <div className="flex items-center p-1 rounded-lg bg-slate-100 text-xs font-semibold self-start sm:self-auto">
                    <button
                      type="button"
                      onClick={() => setActiveExplanationView("chart")}
                      className={`px-3 py-1 rounded-md transition-colors ${
                        activeExplanationView === "chart"
                          ? "bg-white text-slate-900 shadow-xs"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      Contribution Chart
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveExplanationView("reranking_table")}
                      className={`px-3 py-1 rounded-md transition-colors ${
                        activeExplanationView === "reranking_table"
                          ? "bg-white text-slate-900 shadow-xs"
                          : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      Feature Re-ranking Table
                    </button>
                  </div>
                </div>

                {explanationResult ? (
                  <div className="space-y-4">
                    {/* View A: ShapBarChart */}
                    {activeExplanationView === "chart" ? (
                      <ShapBarChart
                        data={explanationResult.top_contributors}
                        title={`Top 10 Feature Contributions (${formatModelLabel(selectedModel)})`}
                        outputSpace={explanationResult.output_space}
                        explainerMethod={explanationResult.explainer_method}
                      />
                    ) : (
                      /* View B: Feature Re-ranking Audit Table */
                      <div className="space-y-3">
                        <div className="flex items-center justify-between text-xs text-slate-500">
                          <span>Comparison of Top 10 Drivers: Baseline vs. Reweighed</span>
                          <span className="font-mono text-[11px]">Explainer: {explanationResult.explainer_method || "TreeExplainer"}</span>
                        </div>
                        <div className="overflow-x-auto rounded-lg border border-slate-200">
                          <table className="w-full text-left text-xs border-collapse">
                            <thead>
                              <tr className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                                <th className="p-2.5">Rank</th>
                                <th className="p-2.5">Feature Name</th>
                                <th className="p-2.5 text-right">Baseline SHAP</th>
                                <th className="p-2.5 text-right">Reweighed SHAP</th>
                                <th className="p-2.5 text-right">Net Shift (Δ)</th>
                                <th className="p-2.5 text-center">Direction Status</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                              {featureRankingDiffs.map((row) => (
                                <tr key={row.rank} className="hover:bg-slate-50/80">
                                  <td className="p-2.5 text-slate-500 font-sans font-bold">#{row.rank}</td>
                                  <td className="p-2.5 font-sans font-medium text-slate-900">
                                    {row.featureName}
                                    <span className="block text-[10px] text-slate-400 font-mono">{row.rawFeature}</span>
                                  </td>
                                  <td className={`p-2.5 text-right font-semibold ${row.baselineShap > 0 ? "text-rose-700" : "text-emerald-700"}`}>
                                    {row.baselineShap > 0 ? `+${row.baselineShap.toFixed(4)}` : row.baselineShap.toFixed(4)}
                                  </td>
                                  <td className={`p-2.5 text-right font-semibold ${row.reweighedShap > 0 ? "text-rose-700" : "text-emerald-700"}`}>
                                    {row.reweighedShap > 0 ? `+${row.reweighedShap.toFixed(4)}` : row.reweighedShap.toFixed(4)}
                                  </td>
                                  <td className="p-2.5 text-right font-bold text-slate-800">
                                    {row.diffShap > 0 ? `+${row.diffShap.toFixed(4)}` : row.diffShap.toFixed(4)}
                                  </td>
                                  <td className="p-2.5 text-center">
                                    {row.signFlipped ? (
                                      <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-800">
                                        Sign Inversion
                                      </span>
                                    ) : row.baselineShap > 0 ? (
                                      <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-medium bg-rose-50 text-rose-700">
                                        Adverse
                                      </span>
                                    ) : (
                                      <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-700">
                                        Protective
                                      </span>
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Output Space & Cross-Architecture Warning */}
                    <div className="p-3.5 rounded-lg bg-blue-50/70 border border-blue-200 text-xs text-slate-700 space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-blue-900 flex items-center space-x-1.5">
                          <Info className="w-4 h-4 text-blue-700" />
                          <span>Output Space &amp; Cross-Architecture Semantics</span>
                        </span>
                        <span className="font-mono text-[10px] bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                          {selectedModel === "random_forest"
                            ? "Probability Space P(Y=1)"
                            : selectedModel === "logistic_regression"
                            ? "Log-Odds Margin Space (Linear)"
                            : "Margin Space (Tree Ensembles)"}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-600 leading-relaxed">
                        {selectedModel === "random_forest"
                          ? "Random Forest TreeExplainer computes attributions directly in probability space P(Y=1): the sum of attributions equals the difference between the applicant prediction and the empirical base rate."
                          : selectedModel === "logistic_regression"
                          ? "Logistic Regression LinearExplainer computes attributions in log-odds margin space: values reflect linear logit contributions relative to the background distribution."
                          : "XGBoost TreeExplainer computes attributions in margin log-odds space (raw tree ensemble sum before logistic sigmoid)."}
                      </p>
                      <p className="text-[11px] text-amber-800 font-medium">
                        * Architectural Warning: Raw SHAP values operate on fundamentally different mathematical scales across architectures. Never compare raw SHAP magnitudes between Random Forest and XGBoost directly without appropriate transformation.
                      </p>
                    </div>

                    {/* Narrative Summary */}
                    <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600 space-y-1">
                      <p className="font-semibold text-slate-800">Explanation Summary</p>
                      <p className="text-[11px] leading-relaxed">
                        {explanationResult.summary}
                      </p>
                      <p className="text-[10px] text-slate-400 pt-1 italic">
                        {explanationResult.disclaimer}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="h-48 flex items-center justify-center text-slate-400 text-xs italic bg-slate-50 rounded-lg border border-slate-200">
                    Run assessment to compute real-time local SHAP feature attributions.
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="p-12 rounded-xl bg-white border border-slate-200 shadow-xs text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-teal-50 border border-teal-100 flex items-center justify-center mx-auto text-teal-700">
                <Layers className="w-6 h-6" />
              </div>
              <div className="max-w-md mx-auto space-y-1.5">
                <h3 className="text-base font-bold text-slate-800">
                  Ready for Applicant Explanation Audit
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Select a dataset, model architecture, or demonstration preset profile, then click <strong>Run Credit-Risk &amp; Explanation Audit</strong> to compute live Baseline vs. Reweighed default predictions and individual SHAP feature attributions.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
