import Link from "next/link";
import { AlertCircle, ExternalLink, ShieldCheck } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-[#E2E8F0] bg-white mt-16 py-10 text-xs text-slate-500">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <div className="flex items-center space-x-2 text-slate-800 font-semibold text-sm">
              <ShieldCheck className="w-4 h-4 text-teal-700" />
              <span>FairCredit-AI Research Platform</span>
            </div>
            <p className="text-slate-500 leading-relaxed">
              Evaluating the empirical impact of fairness mitigation (Reweighing, Exponentiated Gradient, ThresholdOptimizer) 
              on local SHAP explanation consistency across credit risk scoring models.
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold text-slate-800 text-sm">Authoritative Benchmark Datasets</h4>
            <ul className="space-y-1 text-slate-500">
              <li>• Taiwan Credit Card Default (N=30,000, 23 financial predictors)</li>
              <li>• Statlog German Credit (N=1,000, 20 credit attributes)</li>
              <li>• Demographic sex strictly excluded from predictive features X</li>
            </ul>
          </div>

          <div className="space-y-2">
            <h4 className="font-semibold text-slate-800 text-sm">Regulatory & Compliance Notice</h4>
            <div className="p-2.5 rounded-md bg-amber-50/70 border border-amber-200/60 text-amber-800 text-[11px] leading-relaxed flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-amber-600 mt-0.5 shrink-0" />
              <span>
                <strong>Research Prototype:</strong> This software is strictly for model-risk analysis, academic auditing, and algorithmic accountability. It does not render automated lending decisions or CIBIL scores.
              </span>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-100 pt-4 flex flex-col sm:flex-row items-center justify-between text-slate-400 text-[11px]">
          <p>© 2026 FairCredit-AI Framework. Built with Next.js, FastAPI & SHAP.</p>
          <div className="flex items-center space-x-4 mt-2 sm:mt-0">
            <a
              href="https://github.com/SarthakZunjure17/FairCredit-AI"
              target="_blank"
              rel="noreferrer"
              className="text-slate-600 hover:text-slate-900 inline-flex items-center space-x-1"
            >
              <span>GitHub Repository</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
