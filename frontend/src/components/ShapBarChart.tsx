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
import { formatFeatureLabel } from "@/lib/utils";

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

  // Format and safely map data for Recharts
  const chartData = data.slice(0, 10).map((d) => {
    const formatted = formatFeatureLabel(d.feature);
    const raw = d.feature_raw || d.feature;
    const truncatedName = formatted.length > 28 ? `${formatted.substring(0, 26)}…` : formatted;

    return {
      name: truncatedName,
      fullName: formatted,
      rawName: raw,
      Baseline: typeof d.baseline_shap === "number" ? d.baseline_shap : 0,
      Reweighed: typeof d.reweighed_shap === "number" ? d.reweighed_shap : 0,
      diff: typeof d.diff_shap === "number" ? d.diff_shap : 0
    };
  }).reverse(); // Reverse so top impact item renders at the top of the chart

  return (
    <div className="space-y-3">
      {title && (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
          <h4 className="text-xs font-semibold text-slate-800">{title}</h4>
          {outputSpace && (
            <span className="text-[10px] text-slate-500 font-mono bg-slate-100 px-2 py-0.5 rounded self-start sm:self-auto">
              Space: {outputSpace}
            </span>
          )}
        </div>
      )}

      {/* Main Bar Chart Container */}
      <div className="h-96 w-full bg-white p-3 rounded-lg border border-slate-200/80 shadow-xs">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: "#64748B" }}
              tickFormatter={(v) => (v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3))}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 11, fill: "#334155" }}
              width={190}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const item = payload[0].payload;
                  const bVal = item.Baseline;
                  const rVal = item.Reweighed;
                  const dVal = item.diff;

                  return (
                    <div className="bg-slate-900 text-white text-xs p-3 rounded-lg shadow-xl space-y-2 z-50 max-w-xs border border-slate-700">
                      <div>
                        <p className="font-semibold text-slate-100 text-xs leading-snug">
                          {item.fullName}
                        </p>
                        {item.rawName && (
                          <p className="font-mono text-[10px] text-slate-400 mt-0.5">
                            ID: {item.rawName}
                          </p>
                        )}
                      </div>

                      <div className="space-y-1 text-[11px] border-t border-slate-700/80 pt-2">
                        <div className="flex justify-between space-x-3 text-teal-300">
                          <span>Baseline SHAP:</span>
                          <span className="font-mono font-medium">
                            {bVal > 0 ? `+${bVal.toFixed(4)}` : bVal.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex justify-between space-x-3 text-sky-300">
                          <span>Reweighed SHAP:</span>
                          <span className="font-mono font-medium">
                            {rVal > 0 ? `+${rVal.toFixed(4)}` : rVal.toFixed(4)}
                          </span>
                        </div>
                        <div className="flex justify-between space-x-3 text-slate-300 border-t border-slate-800 pt-1">
                          <span>Shift (Reweighed − Baseline):</span>
                          <span className="font-mono font-medium">
                            {dVal > 0 ? `+${dVal.toFixed(4)}` : dVal.toFixed(4)}
                          </span>
                        </div>
                      </div>

                      <div className="text-[10px] text-slate-400 border-t border-slate-800 pt-1.5 leading-tight">
                        {bVal > 0.001 ? (
                          <span className="text-amber-300">Positive: Pushes prediction toward higher default risk.</span>
                        ) : bVal < -0.001 ? (
                          <span className="text-emerald-300">Negative: Pushes prediction toward lower default risk.</span>
                        ) : (
                          <span>Near-zero: Negligible marginal impact.</span>
                        )}
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

      {/* Attribution Direction Guide */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between text-[11px] text-slate-500 bg-slate-50 px-3 py-2 rounded-md border border-slate-200/80 gap-1.5">
        <span className="flex items-center space-x-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block"></span>
          <span><strong>Negative (&lt; 0):</strong> Protective influence (pushes toward lower default risk)</span>
        </span>
        <span className="flex items-center space-x-1.5">
          <span className="w-2 h-2 rounded-full bg-rose-500 inline-block"></span>
          <span><strong>Positive (&gt; 0):</strong> Adverse influence (pushes toward higher default risk)</span>
        </span>
      </div>
    </div>
  );
}
