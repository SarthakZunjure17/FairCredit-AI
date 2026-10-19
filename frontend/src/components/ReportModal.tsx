"use client";

import React, { useState } from "react";
import {
  FileText,
  Download,
  X,
  CheckCircle2,
  AlertCircle,
  ShieldCheck,
  Activity,
  Layers,
  FileCheck
} from "lucide-react";

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  reportType: "research" | "applicant";
  title: string;
  subtitle: string;
  configuration: Record<string, string | number>;
  sections: string[];
  pageEstimate: string;
  onDownload: () => Promise<void>;
  disclaimer: string;
}

export function ReportModal({
  isOpen,
  onClose,
  reportType,
  title,
  subtitle,
  configuration,
  sections,
  pageEstimate,
  onDownload,
  disclaimer
}: ReportModalProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  if (!isOpen) return null;

  async function handleDownloadClick() {
    try {
      setIsGenerating(true);
      setErrorMessage(null);
      setSuccessMessage(null);
      await onDownload();
      setSuccessMessage("PDF report successfully compiled and downloaded!");
      setTimeout(() => {
        setSuccessMessage(null);
      }, 4000);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to generate report.");
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="relative w-full max-w-xl rounded-2xl bg-white border border-slate-200 shadow-xl overflow-hidden space-y-0">
        {/* Header */}
        <div className="p-5 bg-slate-50 border-b border-slate-200 flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-teal-100 text-teal-800">
                {reportType === "research" ? "Research Audit Report" : "Individual Assessment Audit"}
              </span>
              <span className="text-[11px] font-medium text-slate-500 font-mono">
                {pageEstimate}
              </span>
            </div>
            <h2 className="text-lg font-bold text-slate-900">{title}</h2>
            <p className="text-xs text-slate-500 leading-normal">{subtitle}</p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 space-y-4 max-h-[70vh] overflow-y-auto">
          {errorMessage && (
            <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          {successMessage && (
            <div className="p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center space-x-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{successMessage}</span>
            </div>
          )}

          {/* Configuration Inspector Table */}
          <div className="space-y-2">
            <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider block">
              Audited Configuration Parameters
            </span>
            <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 grid grid-cols-2 gap-2 text-xs">
              {Object.entries(configuration).map(([key, val]) => (
                <div key={key} className="space-y-0.5">
                  <span className="text-[10px] font-bold text-slate-400 uppercase block">{key}</span>
                  <span className="font-semibold text-slate-800 capitalize font-mono text-[11px]">{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Report Sections Checklist */}
          <div className="space-y-2">
            <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider block">
              Sections Included in Generated PDF
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {sections.map((sec, idx) => (
                <div key={idx} className="flex items-center space-x-2 text-xs text-slate-700 bg-slate-50/50 p-2 rounded border border-slate-100">
                  <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0" />
                  <span className="text-[11px] font-medium">{sec}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Safety & Isolation Safeguard Note */}
          <div className="p-3 rounded-lg bg-teal-50/50 border border-teal-200 text-xs text-teal-900 space-y-1">
            <div className="flex items-center space-x-1.5 font-bold">
              <ShieldCheck className="w-4 h-4 text-teal-700 shrink-0" />
              <span>Auditing & Privacy Safeguards</span>
            </div>
            <p className="text-[11px] text-teal-800 leading-relaxed">
              {disclaimer}
            </p>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between">
          <button
            onClick={onClose}
            disabled={isGenerating}
            className="px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-200/50 rounded-lg transition-colors"
          >
            Cancel
          </button>

          <button
            onClick={handleDownloadClick}
            disabled={isGenerating}
            className="flex items-center space-x-2 px-5 py-2 text-xs font-bold text-white bg-teal-700 hover:bg-teal-800 rounded-lg shadow-xs transition-all disabled:opacity-60"
          >
            {isGenerating ? (
              <>
                <Activity className="w-3.5 h-3.5 animate-spin" />
                <span>Compiling PDF Report…</span>
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                <span>Download PDF Audit Report</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
