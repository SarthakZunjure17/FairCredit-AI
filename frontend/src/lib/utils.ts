import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatMetric(val: number | string | undefined | null, decimals = 4): string {
  if (val === undefined || val === null || val === "N/A" || val === "nan" || val === "") {
    return "N/A";
  }
  if (typeof val === "string") {
    const num = parseFloat(val);
    if (isNaN(num)) return val;
    return num.toFixed(decimals);
  }
  if (typeof val === "number") {
    if (isNaN(val)) return "N/A";
    return val.toFixed(decimals);
  }
  return String(val);
}

export function formatPercent(val: number | undefined | null, decimals = 2): string {
  if (val === undefined || val === null || isNaN(val)) return "N/A";
  return `${(val * 100).toFixed(decimals)}%`;
}

export function formatDelta(val: number | string | undefined | null, decimals = 4): { text: string; favorable: boolean | null } {
  if (val === undefined || val === null || val === "N/A" || val === "nan" || val === "") {
    return { text: "N/A", favorable: null };
  }
  const num = typeof val === "string" ? parseFloat(val) : val;
  if (isNaN(num)) return { text: String(val), favorable: null };
  const sign = num > 0 ? "+" : "";
  return {
    text: `${sign}${num.toFixed(decimals)}`,
    favorable: Math.abs(num) < 0.0001 ? null : num > 0
  };
}

export const DATASET_LABELS: Record<string, string> = {
  taiwan: "Taiwan Credit Card Default",
  german: "Statlog German Credit"
};

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost"
};

export const MITIGATION_LABELS: Record<string, string> = {
  reweighing: "Reweighing (Pre-processing)",
  exponentiated_gradient: "Exponentiated Gradient (In-processing)",
  threshold_optimizer: "ThresholdOptimizer (Post-processing)"
};

export function formatDatasetLabel(dataset: string | undefined | null): string {
  if (!dataset || typeof dataset !== "string") return "Unavailable";
  const key = dataset.toLowerCase().trim();
  return DATASET_LABELS[key] || (dataset.charAt(0).toUpperCase() + dataset.slice(1));
}

export function formatModelLabel(model: string | undefined | null): string {
  if (!model || typeof model !== "string") return "Unavailable";
  const key = model.toLowerCase().trim().replace(/[\s-]+/g, "_");
  return MODEL_LABELS[key] || model.replace(/_/g, " ");
}

export function formatMitigationLabel(mitigation: string | undefined | null): string {
  if (!mitigation || typeof mitigation !== "string") return "Unavailable";
  const key = mitigation.toLowerCase().trim().replace(/[\s-]+/g, "_");
  return MITIGATION_LABELS[key] || mitigation.replace(/_/g, " ");
}

export { formatFeatureLabel } from "./featureDictionary";
