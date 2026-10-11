"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldAlert, BarChart3, Scale, BookOpen, Layers } from "lucide-react";

export function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { href: "/", label: "Overview", icon: Layers },
    { href: "/assess", label: "Assess Risk", icon: BarChart3 },
    { href: "/fairness", label: "Fairness & Explanations", icon: Scale },
    { href: "/research", label: "Research", icon: BookOpen }
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#FBFBF9]/90 backdrop-blur-md border-b border-[#E2E8F0]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <Link href="/" className="flex items-center space-x-2.5 group">
              <div className="w-9 h-9 rounded-lg bg-teal-800 text-white flex items-center justify-center shadow-xs">
                <Scale className="w-5 h-5 text-teal-200" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-semibold text-base text-slate-900 tracking-tight">FairCredit-AI</span>
                  <span className="text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200/60">
                    Research Prototype
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 font-medium">Credit Risk Scoring + Responsible AI Audit</p>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center space-x-1.5 px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-teal-50/80 text-teal-800 font-semibold border border-teal-200/70"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/70"
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? "text-teal-700" : "text-slate-400"}`} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
}
