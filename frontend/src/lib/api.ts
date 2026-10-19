import {
  ConfigurationsResponse,
  AssessmentResponse,
  ExplanationResponse,
  ResearchMatrixRow,
  ShapSummaryRow,
  CaseStudyData,
  IndividualShapResponse,
  FinalResearchResultRow,
  FairnessAuditResponse
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {})
    },
    ...options
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`API error (${res.status}): ${errorText}`);
  }
  return res.json();
}

export const api = {
  getHealth: () => fetchJson<{ status: string; service: string; version: string }>("/health"),
  
  getConfigurations: () => fetchJson<ConfigurationsResponse>("/configurations"),
  
  predictApplicant: (dataset: string, model: string, features: Record<string, any>) =>
    fetchJson<AssessmentResponse>("/predict", {
      method: "POST",
      body: JSON.stringify({ dataset, model, features })
    }),
    
  explainApplicant: (dataset: string, model: string, features: Record<string, any>) =>
    fetchJson<ExplanationResponse>("/explain", {
      method: "POST",
      body: JSON.stringify({ dataset, model, features })
    }),
    
  getResearchMatrix: () => fetchJson<ResearchMatrixRow[]>("/research/matrix"),
  
  getMitigationMaster: () => fetchJson<any[]>("/research/mitigations"),
  
  getFinalResearch: (dataset?: string, model?: string) => {
    const params = new URLSearchParams();
    if (dataset) params.append("dataset", dataset);
    if (model) params.append("model", model);
    const q = params.toString() ? `?${params.toString()}` : "";
    return fetchJson<FinalResearchResultRow[]>(`/research/final-results${q}`);
  },
  
  getShapSummary: () => fetchJson<ShapSummaryRow[]>("/research/shap-summary"),
  
  getShapGroups: () => fetchJson<any[]>("/research/shap-groups"),
  
  getCaseStudy: () => fetchJson<CaseStudyData>("/research/case-study"),
  
  getFairnessAudit: (dataset?: string, model?: string, mitigation?: string, protectedAttribute?: string) => {
    const params = new URLSearchParams();
    if (dataset) params.append("dataset", dataset);
    if (model) params.append("model", model);
    if (mitigation) params.append("mitigation", mitigation);
    if (protectedAttribute) params.append("protected_attribute", protectedAttribute);
    const q = params.toString() ? `?${params.toString()}` : "";
    return fetchJson<FairnessAuditResponse>(`/fairness/audit${q}`);
  },

  getIndividualShap: (dataset: string, model: string, applicant_idx = 0) =>
    fetchJson<IndividualShapResponse>(`/research/individual-shap?dataset=${dataset}&model=${model}&applicant_idx=${applicant_idx}`)
};
