import React from "react";
import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  delta?: {
    value: string;
    isFavorable?: boolean | null; // true=green, false=red, null=neutral
    label?: string;
  };
  badge?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  subtext,
  delta,
  badge,
  className = ""
}: MetricCardProps) {
  return (
    <div className={`p-4 rounded-lg bg-white border border-[#E2E8F0] shadow-xs space-y-2 ${className}`}>
      <div className="flex items-center justify-between text-xs text-slate-500 font-medium">
        <span>{label}</span>
        {badge && (
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
            {badge}
          </span>
        )}
      </div>

      <div className="flex items-baseline justify-between">
        <span className="text-2xl font-bold text-slate-900 tracking-tight">{value}</span>

        {delta && (
          <div
            className={`flex items-center space-x-0.5 text-xs font-semibold px-2 py-0.5 rounded-full ${
              delta.isFavorable === true
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                : delta.isFavorable === false
                ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                : "bg-slate-100 text-slate-700"
            }`}
          >
            {delta.isFavorable === true ? (
              <ArrowUpRight className="w-3.5 h-3.5" />
            ) : delta.isFavorable === false ? (
              <ArrowDownRight className="w-3.5 h-3.5" />
            ) : (
              <Minus className="w-3.5 h-3.5" />
            )}
            <span>{delta.value}</span>
            {delta.label && <span className="text-[10px] opacity-75 ml-0.5">{delta.label}</span>}
          </div>
        )}
      </div>

      {subtext && <p className="text-[11px] text-slate-500 leading-normal">{subtext}</p>}
    </div>
  );
}
