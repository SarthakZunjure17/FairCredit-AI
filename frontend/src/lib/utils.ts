import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatMetric(val: number | string | undefined | null, decimals = 4): string {
  if (val === undefined || val === null || val === "N/A" || val === "nan") {
    return "N/A";
  }
  if (typeof val === "string") {
    const num = parseFloat(val);
    if (isNaN(num)) return val;
    return num.toFixed(decimals);
  }
  return val.toFixed(decimals);
}

export function formatPercent(val: number | undefined | null, decimals = 2): string {
  if (val === undefined || val === null) return "N/A";
  return `${(val * 100).toFixed(decimals)}%`;
}

export function formatDelta(val: number | string | undefined | null, decimals = 4): { text: string; favorable: boolean | null } {
  if (val === undefined || val === null || val === "N/A" || val === "nan") {
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
