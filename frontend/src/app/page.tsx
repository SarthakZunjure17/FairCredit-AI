import Link from "next/link";
import { ArrowRight, BarChart3, Scale, ShieldAlert, CheckCircle2, TrendingUp, Sparkles } from "lucide-react";
import { MetricCard } from "@/components/MetricCard";

export default function OverviewPage() {
  return (
    <div className="space-y-12 max-w-5xl mx-auto">
      {/* Hero Section */}
      <div className="space-y-4">
        <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-teal-50 border border-teal-200/80 text-teal-800 text-xs font-semibold">
          <Sparkles className="w-3.5 h-3.5 text-teal-600" />
          <span>Responsible AI Credit Scoring Framework</span>
        </div>

        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-900 leading-tight">
          Credit-Risk Assessment & Responsible-AI Auditing
        </h1>

        <p className="text-base sm:text-lg text-slate-600 max-w-3xl leading-relaxed">
          FairCredit-AI demonstrates how machine-learning models estimate credit default risk
          and investigates how fairness mitigation affects group fairness and the stability of model explanations.
        </p>

        {/* Primary CTA Buttons */}
        <div className="flex flex-wrap gap-3 pt-2">
          <Link
            href="/assess"
            className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-teal-800 text-white font-medium text-sm hover:bg-teal-900 shadow-xs transition-colors"
          >
            <span>Assess an Applicant</span>
            <ArrowRight className="w-4 h-4" />
          </Link>

          <Link
            href="/fairness"
            className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-white border border-slate-300 text-slate-700 font-medium text-sm hover:bg-slate-50 transition-colors shadow-xs"
          >
            <span>Explore Research Findings</span>
          </Link>
        </div>
      </div>

      {/* Restrained Three-Part Visual Summary */}
      <div className="space-y-4">
        <h2 className="text-xs uppercase tracking-wider font-bold text-slate-400">
          Core Auditing Dimensions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-lg bg-white border border-slate-200 shadow-xs space-y-2">
            <div className="w-8 h-8 rounded-md bg-teal-50 text-teal-700 flex items-center justify-center font-bold text-sm">
              1
            </div>
            <h3 className="font-semibold text-slate-900 text-sm">Predictive Performance</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Evaluating discrimination accuracy, ROC-AUC, and balanced classification error across Logistic Regression, Random Forest, and XGBoost.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-slate-200 shadow-xs space-y-2">
            <div className="w-8 h-8 rounded-md bg-sky-50 text-sky-700 flex items-center justify-center font-bold text-sm">
              2
            </div>
            <h3 className="font-semibold text-slate-900 text-sm">Group Fairness Disparities</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Auditing demographic gaps via Equalized Odds Difference, Demographic Parity Difference, and Disparate Impact under strict leakage controls.
            </p>
          </div>

          <div className="p-5 rounded-lg bg-white border border-slate-200 shadow-xs space-y-2">
            <div className="w-8 h-8 rounded-md bg-indigo-50 text-indigo-700 flex items-center justify-center font-bold text-sm">
              3
            </div>
            <h3 className="font-semibold text-slate-900 text-sm">SHAP Explanation Stability</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Measuring local explanation consistency before and after Reweighing using Cosine Similarity, Top-5 Feature Overlap, and Spearman Rank Correlation.
            </p>
          </div>
        </div>
      </div>

      {/* Evidence-Led Case Study Finding */}
      <div className="p-6 rounded-xl bg-white border border-slate-200 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-100 pb-3">
          <div>
            <span className="text-[11px] uppercase font-bold text-teal-700 tracking-wider">
              Empirical Research Highlight
            </span>
            <h3 className="text-base font-bold text-slate-900">
              Statlog German Credit — Random Forest + Reweighing
            </h3>
          </div>
          <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2.5 py-1 rounded">
            Held-out Test Cohort (N=200)
          </span>
        </div>

        <p className="text-xs text-slate-600 leading-relaxed">
          On the German credit benchmark, Reweighing reduced the Equalized Odds disparity by 60% while simultaneously
          improving ROC-AUC discrimination and preserving a high explanation consistency (cosine similarity 0.9652).
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
          <MetricCard
            label="Predictive ROC-AUC"
            value="0.7935"
            subtext="Baseline was 0.7798"
            delta={{ value: "+0.0137", isFavorable: true, label: "AUC" }}
            badge="Discrimination"
          />
          <MetricCard
            label="Equalized Odds Difference"
            value="0.0500"
            subtext="Baseline was 0.1250 (60% drop)"
            delta={{ value: "-0.0750", isFavorable: true, label: "parity gap" }}
            badge="Fairness"
          />
          <MetricCard
            label="SHAP Cosine Similarity"
            value="0.9652"
            subtext="95% CI: [0.9572, 0.9723]"
            badge="Consistency"
          />
        </div>

        <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-[11px] text-slate-500 flex items-start space-x-2">
          <CheckCircle2 className="w-4 h-4 text-teal-700 shrink-0 mt-0.5" />
          <span>
            <strong>Central Empirical Finding:</strong> Fairness mitigation exhibited heterogeneous effects across configurations rather than a universal fairness–explanation trade-off. Reweighing does not universally preserve explanations or guarantee parity across every model architecture.
          </span>
        </div>
      </div>

      {/* Compact Disclaimer */}
      <div className="p-3.5 rounded-lg bg-amber-50/80 border border-amber-200/70 text-amber-900 text-xs flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-amber-700 shrink-0" />
          <span>
            <strong>Research prototype:</strong> Not intended for automated credit decisions, lending determinations, or production credit underwriting.
          </span>
        </div>
        <Link href="/research" className="underline font-semibold hover:text-amber-950 shrink-0 ml-4">
          Read Methodology & Limitations →
        </Link>
      </div>
    </div>
  );
}
