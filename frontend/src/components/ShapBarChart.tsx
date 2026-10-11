"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine
} from "recharts";
import { FeatureAttribution } from "@/types";

interface ShapBarChartProps {
  data: FeatureAttribution[];
  title?: string;
  outputSpace?: string;
}

export function ShapBarChart({ data, title, outputSpace }: ShapBarChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400 text-xs italic bg-slate-50 rounded-lg border border-slate-200">
        No feature explanation data available for this configuration.
      </div>
    );
  }

  // Format data for Recharts (truncate feature names if too long)
  const chartData = data.slice(0, 10).map((d) => ({
    name: d.feature.length > 25 ? `${d.feature.substring(0, 23)}…` : d.feature,
    fullName: d.feature,
    Baseline: d.baseline_shap,
    Reweighed: d.reweighed_shap,
    diff: d.diff_shap
  })).reverse(); // reverse so highest impact is on top

  return (
    <div className="space-y-2">
      {title && (
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-semibold text-slate-800">{title}</h4>
          {outputSpace && (
            <span className="text-[10px] text-slate-500 font-mono bg-slate-100 px-2 py-0.5 rounded">
              Space: {outputSpace}
            </span>
          )}
        </div>
      )}

      <div className="h-84 w-full bg-white p-3 rounded-lg border border-slate-200/80 shadow-xs">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 25, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "#64748B" }}
              tickFormatter={(v) => v.toFixed(3)}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 11, fill: "#334155" }}
              width={120}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const item = payload[0].payload;
                  return (
                    <div className="bg-slate-900 text-white text-xs p-2.5 rounded shadow-lg space-y-1 z-50">
                      <p className="font-semibold text-slate-100">{item.fullName}</p>
                      <div className="flex justify-between space-x-4 text-teal-300">
                        <span>Baseline SHAP:</span>
                        <span className="font-mono">{item.Baseline > 0 ? `+${item.Baseline.toFixed(4)}` : item.Baseline.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between space-x-4 text-amber-300">
                        <span>Reweighed SHAP:</span>
                        <span className="font-mono">{item.Reweighed > 0 ? `+${item.Reweighed.toFixed(4)}` : item.Reweighed.toFixed(4)}</span>
                      </div>
                      <div className="border-t border-slate-700 pt-1 flex justify-between space-x-4 text-slate-300">
                        <span>Shift (Delta):</span>
                        <span className="font-mono">{item.diff > 0 ? `+${item.diff.toFixed(4)}` : item.diff.toFixed(4)}</span>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 6 }}
              iconSize={10}
            />
            <ReferenceLine x={0} stroke="#94A3B8" strokeWidth={1.5} />
            <Bar dataKey="Baseline" fill="#0D9488" radius={[0, 3, 3, 0]} name="Baseline Model" />
            <Bar dataKey="Reweighed" fill="#0284C7" radius={[0, 3, 3, 0]} name="Reweighed Model" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1 px-1">
        <span>← Negative: Reduces Default Risk (Favorable)</span>
        <span>Positive: Increases Default Risk (Adverse) →</span>
      </div>
    </div>
  );
}
