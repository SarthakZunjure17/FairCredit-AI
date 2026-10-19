import io
import datetime
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from backend.app.services.research_service import research_service
from backend.app.services.feature_dictionary import format_feature_label
from backend.app.schemas.report import ResearchReportRequest, ApplicantReportRequest

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748B'))
        # Running header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 755, 'FairCredit-AI 2.0 — Responsible AI Credit Model Audit Report')
            self.setStrokeColor(colors.HexColor('#E2E8F0'))
            self.setLineWidth(0.5)
            self.line(40, 747, 612 - 40, 747)
        # Running footer (all pages)
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.5)
        self.line(40, 42, 612 - 40, 42)
        self.drawString(40, 30, 'CONFIDENTIAL & PROPRIETARY — FOR RESEARCH AUDIT & ACADEMIC EVALUATION ONLY')
        self.drawRightString(612 - 40, 30, f'Page {self._pageNumber} of {page_count}')
        self.restoreState()

def _create_report_styles():
    styles = getSampleStyleSheet()
    
    primary_color = colors.HexColor('#0F766E') # Teal 700
    dark_slate = colors.HexColor('#0F172A')
    muted_slate = colors.HexColor('#475569')

    styles.add(ParagraphStyle(
        name='DocTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=dark_slate,
        spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        name='DocSubTitle',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=primary_color,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=dark_slate,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='SubSectionHeading',
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='BodyRegular',
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=muted_slate,
        spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1 # Center
    ))
    styles.add(ParagraphStyle(
        name='TableHeaderLeft',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0 # Left
    ))
    styles.add(ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        textColor=dark_slate,
        alignment=0 # Left
    ))
    styles.add(ParagraphStyle(
        name='TableCellCenter',
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        textColor=dark_slate,
        alignment=1
    ))
    styles.add(ParagraphStyle(
        name='TableFootnote',
        fontName='Helvetica-Oblique',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#64748B'),
        spaceBefore=3
    ))

    return styles

def _build_pdf_buffer(story: List[Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=45
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue()

def _generate_subgroup_figure(audit: Dict[str, Any]) -> io.BytesIO:
    bU = next((g for g in audit['baseline_subgroups'] if g['group_role'] == 'unprivileged'), None)
    bP = next((g for g in audit['baseline_subgroups'] if g['group_role'] == 'privileged'), None)
    mU = next((g for g in audit['mitigated_subgroups'] if g['group_role'] == 'unprivileged'), None)
    mP = next((g for g in audit['mitigated_subgroups'] if g['group_role'] == 'privileged'), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.2), dpi=160)

    # Plot 1: Approval Rates
    groups = [f"Unprivileged\n({bU['group'] if bU else 'u'})", f"Privileged\n({bP['group'] if bP else 'p'})"]
    b_rates = [(bU['approval_rate']*100) if bU else 0, (bP['approval_rate']*100) if bP else 0]
    m_rates = [(mU['approval_rate']*100) if mU else 0, (mP['approval_rate']*100) if mP else 0]

    x = [0, 1]
    ax1.bar([i - 0.18 for i in x], b_rates, width=0.34, label='Baseline', color='#94a3b8', edgecolor='#64748b', linewidth=0.5)
    ax1.bar([i + 0.18 for i in x], m_rates, width=0.34, label=f"Mitigated", color='#0f766e', edgecolor='#115e59', linewidth=0.5)
    ax1.set_ylabel('Approval Rate (%)', fontsize=7.5, color='#334155')
    ax1.set_title('Subgroup Credit Approval Rates', fontsize=8.5, fontweight='bold', color='#0f172a', pad=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=7.5, color='#334155')
    ax1.set_ylim(max(0, min(b_rates + m_rates) - 15), 100)
    ax1.legend(fontsize=7, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='both', which='both', labelsize=7)

    # Plot 2: Equalized Odds Components (TPR vs FPR)
    categories = ['Unpriv (u) Base', 'Unpriv (u) Mit', 'Priv (p) Base', 'Priv (p) Mit']
    tpr_vals = [
        (bU['true_positive_rate']*100) if bU else 0,
        (mU['true_positive_rate']*100) if mU else 0,
        (bP['true_positive_rate']*100) if bP else 0,
        (mP['true_positive_rate']*100) if mP else 0
    ]
    fpr_vals = [
        (bU['false_positive_rate']*100) if bU else 0,
        (mU['false_positive_rate']*100) if mU else 0,
        (bP['false_positive_rate']*100) if bP else 0,
        (mP['false_positive_rate']*100) if mP else 0
    ]

    x2 = range(4)
    ax2.bar([i - 0.18 for i in x2], tpr_vals, width=0.34, label='Defaulter Recall (TPR)', color='#0284c7', edgecolor='#0369a1', linewidth=0.5)
    ax2.bar([i + 0.18 for i in x2], fpr_vals, width=0.34, label='False Alarm (FPR)', color='#e11d48', edgecolor='#be123c', linewidth=0.5)
    ax2.set_ylabel('Operational Rate (%)', fontsize=7.5, color='#334155')
    ax2.set_title('Equalized Odds Error Rates (TPR & FPR)', fontsize=8.5, fontweight='bold', color='#0f172a', pad=4)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=6.5, color='#334155', rotation=10)
    ax2.set_ylim(0, max(tpr_vals + fpr_vals) * 1.25 if max(tpr_vals + fpr_vals) > 0 else 50)
    ax2.legend(fontsize=7, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.tick_params(axis='both', which='both', labelsize=7)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

def _generate_applicant_shap_figure(top_contributors: List[Dict[str, Any]]) -> io.BytesIO:
    top = top_contributors[:8][::-1] # reverse so highest is top
    labels = [format_feature_label(c.get("feature", "")) for c in top]
    base_vals = [c.get("baseline_shap", 0) for c in top]
    mit_vals = [c.get("reweighed_shap", 0) for c in top]

    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=160)
    y = range(len(labels))

    ax.barh([i - 0.18 for i in y], base_vals, height=0.34, label='Baseline Model SHAP', color='#94a3b8', edgecolor='#64748b', linewidth=0.5)
    ax.barh([i + 0.18 for i in y], mit_vals, height=0.34, label='Reweighed Model SHAP', color='#0f766e', edgecolor='#115e59', linewidth=0.5)

    ax.axvline(0, color='#0f172a', linewidth=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, color='#0f172a')
    ax.set_xlabel('SHAP Feature Attribution Value (Contribution to Risk)', fontsize=7.5, color='#334155')
    ax.set_title('Top Local Feature Attributions (Baseline vs. Reweighed)', fontsize=8.5, fontweight='bold', color='#0f172a', pad=4)
    ax.legend(fontsize=7, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', which='both', labelsize=7)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close(fig)
    img_buf.seek(0)
    return img_buf

class ReportService:
    def generate_research_report(self, req: ResearchReportRequest) -> bytes:
        styles = _create_report_styles()
        audit = research_service.get_fairness_audit(req.dataset, req.model, req.mitigation, req.protected_attribute)
        matrix = research_service.get_research_matrix()
        shap_summary = research_service.get_shap_summary()

        exp_row = next(
            (r for r in matrix if r["dataset"] == req.dataset.lower() and r["model"] == req.model.lower() and r["mitigation_method"] == req.mitigation.lower()),
            None
        )
        shap_row = next(
            (s for s in shap_summary if s["dataset"] == req.dataset.lower() and s["model"] == req.model.lower()),
            None
        )

        story = []

        # ================= PAGE 1 =================
        story.append(Paragraph("FairCredit-AI 2.0 — Research Audit Report", styles['DocTitle']))
        story.append(Paragraph(
            f"Empirical Fairness Mitigation & Explanation Stability Audit — {audit['evaluation_context']['dataset_display']} ({audit['evaluation_context']['model_architecture'].replace('_', ' ').title()})",
            styles['DocSubTitle']
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0F766E'), spaceBefore=2, spaceAfter=8))

        gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        meta_data = [
            [
                Paragraph("<b>Dataset:</b>", styles['TableCell']),
                Paragraph(audit['evaluation_context']['dataset_display'], styles['TableCell']),
                Paragraph("<b>Mitigation Method:</b>", styles['TableCell']),
                Paragraph(f"{req.mitigation.replace('_', ' ').title()} ({audit['evaluation_context']['mitigation_method']})", styles['TableCell'])
            ],
            [
                Paragraph("<b>Model Architecture:</b>", styles['TableCell']),
                Paragraph(req.model.replace('_', ' ').title(), styles['TableCell']),
                Paragraph("<b>Protected Attribute:</b>", styles['TableCell']),
                Paragraph(audit['evaluation_context']['protected_attribute_label'], styles['TableCell'])
            ],
            [
                Paragraph("<b>Evaluation Cohort:</b>", styles['TableCell']),
                Paragraph(f"Held-Out Test Set (N={audit['evaluation_context']['total_test_samples']:,})", styles['TableCell']),
                Paragraph("<b>Sensitive Isolation:</b>", styles['TableCell']),
                Paragraph("Protected attributes excluded from model features X", styles['TableCell'])
            ],
            [
                Paragraph("<b>Audit Date:</b>", styles['TableCell']),
                Paragraph(gen_time, styles['TableCell']),
                Paragraph("<b>Source Artifacts:</b>", styles['TableCell']),
                Paragraph("results/mitigation_master_results.csv", styles['TableCell'])
            ]
        ]
        t_meta = Table(meta_data, colWidths=[95, 165, 105, 167])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 8))

        story.append(Paragraph("1. Executive Summary & Auditing Scope", styles['SectionHeading']))
        summary_text = (
            f"This audit report provides a rigorous empirical evaluation of the impact of fairness mitigation on credit risk scoring. "
            f"The evaluation assesses {audit['evaluation_context']['model_architecture'].replace('_', ' ').title()} "
            f"on the {audit['evaluation_context']['dataset_display']} held-out partition ({audit['evaluation_context']['total_test_samples']:,} applicants). "
            f"Fairness interventions were audited for {audit['evaluation_context']['protected_attribute_label']} "
            f"(unprivileged group: <b>{audit['evaluation_context']['unprivileged_group_definition']}</b>; "
            f"privileged group: <b>{audit['evaluation_context']['privileged_group_definition']}</b>). "
            f"All values are derived directly from verified historical benchmark runs without synthetic substitution. "
            f"In accordance with established research protocol, protected demographic attributes are excluded from model training features, "
            f"guaranteeing demographic parity across the predictive input representation."
        )
        story.append(Paragraph(summary_text, styles['BodyRegular']))

        story.append(Paragraph("2. Predictive Utility & Classification Performance Benchmark", styles['SectionHeading']))
        perf_data = [
            [
                Paragraph("Metric", styles['TableHeaderLeft']),
                Paragraph("Baseline", styles['TableHeader']),
                Paragraph(f"Mitigated ({req.mitigation.split('_')[0].title()})", styles['TableHeader']),
                Paragraph("Absolute Change (Δ)", styles['TableHeader']),
                Paragraph("Direction", styles['TableHeader']),
                Paragraph("Parity / Utility Goal", styles['TableHeader'])
            ]
        ]

        for comp in audit['metric_comparisons']:
            b_val = "N/A" if comp["baseline_value"] is None else f"{comp['baseline_value']:.4f}"
            m_val = "N/A" if comp["mitigated_value"] is None else f"{comp['mitigated_value']:.4f}"
            d_val = "N/A" if comp["absolute_change"] is None else f"{comp['absolute_change']:+.4f}"
            dir_label = comp["direction"].replace('_', ' ').title()

            perf_data.append([
                Paragraph(comp["metric_name"], styles['TableCell']),
                Paragraph(b_val, styles['TableCellCenter']),
                Paragraph(m_val, styles['TableCellCenter']),
                Paragraph(d_val, styles['TableCellCenter']),
                Paragraph(dir_label, styles['TableCellCenter']),
                Paragraph(comp["parity_benchmark"].split('(')[0].strip(), styles['TableCell'])
            ])

        t_perf = Table(perf_data, colWidths=[150, 68, 80, 80, 64, 90])
        t_perf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_perf)
        story.append(Paragraph(
            "Note: ThresholdOptimizer ROC-AUC is reported as N/A because post-processing threshold adjustments yield discrete decision classifications, not a continuous risk ranking scoring function.",
            styles['TableFootnote']
        ))

        # ================= PAGE 2 =================
        story.append(PageBreak())
        story.append(Paragraph("3. Demographic Subgroup Confusion Matrix & Operational Rates", styles['SectionHeading']))
        story.append(Paragraph(
            "Evaluation of true positives, false positives, true negatives, and false negatives across sensitive demographic partitions, "
            "evaluating observed default base rates, recall on defaulters (TPR), false alarm rates (FPR), and credit approval rates.",
            styles['BodyRegular']
        ))

        sub_data = [
            [
                Paragraph("Pipeline", styles['TableHeaderLeft']),
                Paragraph("Subgroup", styles['TableHeaderLeft']),
                Paragraph("Role", styles['TableHeader']),
                Paragraph("N", styles['TableHeader']),
                Paragraph("TP", styles['TableHeader']),
                Paragraph("TN", styles['TableHeader']),
                Paragraph("FP", styles['TableHeader']),
                Paragraph("FN", styles['TableHeader']),
                Paragraph("Obs. Def.", styles['TableHeader']),
                Paragraph("Defaulter TPR", styles['TableHeader']),
                Paragraph("False Alarm FPR", styles['TableHeader']),
                Paragraph("Approval Rate", styles['TableHeader'])
            ]
        ]

        all_subgroups = [("Baseline", g) for g in audit['baseline_subgroups']] + [
            (f"Mitigated", g) for g in audit['mitigated_subgroups']
        ]

        for pipe, g in all_subgroups:
            role_tag = "Unprivileged (u)" if g["group_role"] == "unprivileged" else "Privileged (p)"
            sub_data.append([
                Paragraph(pipe, styles['TableCell']),
                Paragraph(g["group"].capitalize(), styles['TableCell']),
                Paragraph(role_tag, styles['TableCellCenter']),
                Paragraph(str(g["sample_count"]), styles['TableCellCenter']),
                Paragraph(str(g["tp"]), styles['TableCellCenter']),
                Paragraph(str(g["tn"]), styles['TableCellCenter']),
                Paragraph(str(g["fp"]), styles['TableCellCenter']),
                Paragraph(str(g["fn"]), styles['TableCellCenter']),
                Paragraph(f"{g['observed_default_rate']*100:.1f}%", styles['TableCellCenter']),
                Paragraph(f"{g['true_positive_rate']*100:.1f}%", styles['TableCellCenter']),
                Paragraph(f"{g['false_positive_rate']*100:.1f}%", styles['TableCellCenter']),
                Paragraph(f"<b>{g['approval_rate']*100:.1f}%</b>", styles['TableCellCenter'])
            ])

        t_sub = Table(sub_data, colWidths=[52, 50, 70, 30, 24, 25, 24, 24, 45, 60, 62, 56])
        t_sub.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ]))
        story.append(t_sub)
        story.append(Paragraph(
            "Conventions: Y=1 Default Risk Event; Y_hat=1 Default Prediction (Loan Denial); Y_hat=0 Credit Approval (Favorable). "
            "Approval Rate = (TN + FN) / N. Defaulter TPR = TP / (TP + FN). False Alarm FPR = FP / (FP + TN).",
            styles['TableFootnote']
        ))
        story.append(Spacer(1, 8))

        # Embedded Chart
        story.append(Paragraph("<b>Empirical Subgroup Distribution & Error Rate Visualizations</b>", styles['SubSectionHeading']))
        fig_buf = _generate_subgroup_figure(audit)
        story.append(Image(fig_buf, width=532, height=162))
        story.append(Paragraph(
            "Figure 1: Side-by-side comparison of credit approval rates (left) and equalized odds error rate components (right) before and after fairness mitigation.",
            styles['TableFootnote']
        ))

        # ================= PAGE 3 =================
        story.append(PageBreak())
        story.append(Paragraph("4. Explanation Stability Audit & Local SHAP Consistency (Phase 4 Freeze)", styles['SectionHeading']))
        story.append(Paragraph(
            "Evaluating whether fairness mitigation preserves or destabilizes local feature attribution explanations across held-out applicants. "
            "Consistency metrics are computed from held-out test splits under frozen Phase 4 experimental seeds.",
            styles['BodyRegular']
        ))

        if shap_row is not None:
            shap_data = [
                [
                    Paragraph("Consistency Metric", styles['TableHeaderLeft']),
                    Paragraph("Mean Value", styles['TableHeader']),
                    Paragraph("95% Confidence Interval", styles['TableHeader']),
                    Paragraph("Evaluation Space", styles['TableHeader']),
                    Paragraph("Methodological Interpretation", styles['TableHeader'])
                ],
                [
                    Paragraph("Vector Cosine Similarity", styles['TableCell']),
                    Paragraph(f"{shap_row['mean_cosine']:.4f}", styles['TableCellCenter']),
                    Paragraph(shap_row['ci95_cosine'], styles['TableCellCenter']),
                    Paragraph("Cosine angle between attribution vectors", styles['TableCell']),
                    Paragraph("Measures overall directional alignment of local feature importance.", styles['TableCell'])
                ],
                [
                    Paragraph("Top-5 Feature Set Overlap", styles['TableCell']),
                    Paragraph(f"{shap_row['mean_top5']:.4f}", styles['TableCellCenter']),
                    Paragraph(shap_row['ci95_top5'], styles['TableCellCenter']),
                    Paragraph("Jaccard / Rank set intersection", styles['TableCell']),
                    Paragraph("Evaluates whether key decision drivers persist under mitigation.", styles['TableCell'])
                ],
                [
                    Paragraph("Spearman Rank Correlation", styles['TableCell']),
                    Paragraph(f"{shap_row['mean_rank']:.4f}", styles['TableCellCenter']),
                    Paragraph(shap_row['ci95_rank'], styles['TableCellCenter']),
                    Paragraph("Monotonic rank preservation", styles['TableCell']),
                    Paragraph("Measures stability of relative feature importance rankings.", styles['TableCell'])
                ]
            ]
            t_shap = Table(shap_data, colWidths=[120, 60, 100, 110, 142])
            t_shap.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(t_shap)
            story.append(Paragraph(
                "Source: results/shap_summary.csv. Explanations computed using exact TreeExplainer for tree ensembles and LinearExplainer for linear models.",
                styles['TableFootnote']
            ))
        else:
            story.append(Paragraph("SHAP consistency metrics are unavailable for this experimental combination.", styles['BodyRegular']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("5. Qualitative Observations & Cross-Method Comparison", styles['SectionHeading']))
        story.append(Paragraph(
            "Empirical synthesis across mitigation families reveals critical architectural nuances: "
            "Tree ensemble architectures (Random Forest and XGBoost) demonstrated notably higher top-5 explanation stability "
            "(top-5 overlap between 0.7924 and 0.8864) compared to linear baselines. "
            "Furthermore, while post-processing threshold adjustments achieve Equalized Odds parity at the decision boundary, "
            "they preserve applicant ranking identically within each group because internal tree splits remain invariant.",
            styles['BodyRegular']
        ))

        # ================= PAGE 4 =================
        story.append(PageBreak())
        story.append(Paragraph("6. Evidence-Based Audit Interpretations & Trade-Offs", styles['SectionHeading']))
        for f in audit["audit_findings"]:
            bullet_text = f"• <b>{f.split(':')[0]}:</b> {':'.join(f.split(':')[1:]).strip()}" if ":" in f else f"• {f}"
            story.append(Paragraph(bullet_text, styles['BodyRegular']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("7. Auditing Governance, Limitations & Research Disclaimers", styles['SectionHeading']))
        disclaimers = [
            "<b>EEOC Four-Fifths (80%) Screening Heuristic:</b> The disparate impact approval ratio is an administrative screening heuristic established by the EEOC Uniform Guidelines on Employee Selection Procedures. It serves as an investigative threshold [0.80, 1.25], not a conclusive legal determination of compliance or non-discrimination.",
            "<b>Statistical Uncertainty & Sample Size:</b> Findings derived from modest sample sizes (such as the German Credit test set with N=200 applicants) carry wide sampling uncertainty. A single applicant's classification shift changes subgroup rates by 1.6% to 2.3%. Absolute metric changes must be interpreted cautiously.",
            "<b>Causality & Invariance Limitations:</b> Statistical fairness mitigation balances observed confusion matrix rates. It does not validate causal mechanisms, eliminate structural historical data biases, or guarantee adverse-action code stability.",
            "<b>Research Prototype Notice:</b> This audit report was automatically generated by FairCredit-AI 2.0 for academic review and responsible-AI auditing demonstrations. It is strictly not certified for live commercial lending, automated adverse action issuance, or underwriting decisions."
        ]
        for d in disclaimers:
            story.append(Paragraph(d, styles['BodyRegular']))
            story.append(Spacer(1, 2))

        return _build_pdf_buffer(story)

    def generate_applicant_report(self, req: ApplicantReportRequest) -> bytes:
        styles = _create_report_styles()
        story = []

        dataset_display = "Statlog German Credit" if req.dataset == "german" else "Taiwan Credit Card Default"
        model_display = req.model.replace('_', ' ').title()

        # ================= PAGE 1 =================
        story.append(Paragraph("FairCredit-AI 2.0 — Applicant Assessment Audit", styles['DocTitle']))
        story.append(Paragraph(
            f"Individual Credit Risk Prediction & Local Explainability Audit — {dataset_display} ({model_display})",
            styles['DocSubTitle']
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0F766E'), spaceBefore=2, spaceAfter=8))

        gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        meta_data = [
            [
                Paragraph("<b>Dataset:</b>", styles['TableCell']),
                Paragraph(dataset_display, styles['TableCell']),
                Paragraph("<b>Model Architecture:</b>", styles['TableCell']),
                Paragraph(model_display, styles['TableCell'])
            ],
            [
                Paragraph("<b>Explainer Method:</b>", styles['TableCell']),
                Paragraph(req.explainer_method or "TreeExplainer", styles['TableCell']),
                Paragraph("<b>Output Space:</b>", styles['TableCell']),
                Paragraph(req.output_space, styles['TableCell'])
            ],
            [
                Paragraph("<b>Assessment Date:</b>", styles['TableCell']),
                Paragraph(gen_time, styles['TableCell']),
                Paragraph("<b>Privacy Isolation:</b>", styles['TableCell']),
                Paragraph("Protected attributes excluded from input features", styles['TableCell'])
            ]
        ]
        t_meta = Table(meta_data, colWidths=[95, 165, 105, 167])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 8))

        story.append(Paragraph("1. Individual Risk Prediction & Uncalibrated Score Breakdown", styles['SectionHeading']))
        story.append(Paragraph(
            "This report documents the individual risk scoring and local feature attribution for a single applicant. "
            "Outputs reflect research prediction models trained on frozen development partitions. "
            "<b>Crucial Distinction:</b> An individual applicant prediction evaluates specific feature inputs and does not establish or imply population-level group fairness.",
            styles['BodyRegular']
        ))

        base_prob = float(req.baseline.get('default_probability', 0))
        rew_prob = float(req.reweighed.get('default_probability', 0))
        prob_diff = float(req.delta.get('probability_difference', rew_prob - base_prob))
        pct_diff = float(req.delta.get('percent_difference', 0))

        pred_data = [
            [
                Paragraph("Evaluation Pipeline", styles['TableHeaderLeft']),
                Paragraph("Predicted Class", styles['TableHeader']),
                Paragraph("Risk Score", styles['TableHeader']),
                Paragraph("Score Meaning & Calibration Status", styles['TableHeaderLeft'])
            ],
            [
                Paragraph("<b>Baseline Model</b> (Unmitigated)", styles['TableCell']),
                Paragraph(f"<b>{req.baseline.get('prediction_label', 'N/A')}</b> (Class {req.baseline.get('prediction', 'N/A')})", styles['TableCellCenter']),
                Paragraph(f"<b>{base_prob:.4f}</b>", styles['TableCellCenter']),
                Paragraph(req.baseline.get('score_meaning', 'Empirical model risk score. Uncalibrated.'), styles['TableCell'])
            ],
            [
                Paragraph("<b>Reweighed Model</b> (Mitigated)", styles['TableCell']),
                Paragraph(f"<b>{req.reweighed.get('prediction_label', 'N/A')}</b> (Class {req.reweighed.get('prediction', 'N/A')})", styles['TableCellCenter']),
                Paragraph(f"<b>{rew_prob:.4f}</b>", styles['TableCellCenter']),
                Paragraph(req.reweighed.get('score_meaning', 'Empirical model risk score. Uncalibrated.'), styles['TableCell'])
            ]
        ]
        t_pred = Table(pred_data, colWidths=[120, 95, 75, 242])
        t_pred.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_pred)

        story.append(Paragraph(
            f"<b>Mitigation Probability Shift:</b> {req.delta.get('interpretation', f'Score shift of {prob_diff:+.4f}')}. "
            f"Score changed by {prob_diff:+.4f} ({pct_diff:+.2f}% relative shift). "
            "Scores represent empirical ranking indices, not verified true posterior probabilities.",
            styles['TableFootnote']
        ))

        # ================= PAGE 2 =================
        story.append(PageBreak())
        story.append(Paragraph("2. Local SHAP Feature Attribution Audit", styles['SectionHeading']))
        story.append(Paragraph(
            f"Local feature attributions quantify how each applicant feature pushed the prediction above or below the baseline model expectation in {req.output_space}. "
            "Positive SHAP values push toward default prediction (increased risk); negative SHAP values push toward favorable credit assessment.",
            styles['BodyRegular']
        ))

        shap_rows = [
            [
                Paragraph("Rank", styles['TableHeader']),
                Paragraph("Feature", styles['TableHeaderLeft']),
                Paragraph("Baseline SHAP", styles['TableHeader']),
                Paragraph("Reweighed SHAP", styles['TableHeader']),
                Paragraph("Difference (Δ)", styles['TableHeader']),
                Paragraph("Impact Direction", styles['TableHeaderLeft'])
            ]
        ]

        for idx, c in enumerate(req.top_contributors[:10]):
            b_s = float(c.get("baseline_shap", 0))
            r_s = float(c.get("reweighed_shap", 0))
            d_s = float(c.get("diff_shap", r_s - b_s))
            dir_lbl = "Increases Risk (Denied ↑)" if b_s > 0 else "Reduces Risk (Approved ↓)"
            feat_name = format_feature_label(c.get("feature", f"Feature {idx+1}"))
            raw_name = c.get("feature_raw", "")
            feat_display = f"<b>{feat_name}</b>" + (f" <font color='#64748B'>({raw_name})</font>" if raw_name and raw_name != feat_name else "")

            shap_rows.append([
                Paragraph(f"#{idx+1}", styles['TableCellCenter']),
                Paragraph(feat_display, styles['TableCell']),
                Paragraph(f"{b_s:+.4f}", styles['TableCellCenter']),
                Paragraph(f"{r_s:+.4f}", styles['TableCellCenter']),
                Paragraph(f"{d_s:+.4f}", styles['TableCellCenter']),
                Paragraph(dir_lbl, styles['TableCell'])
            ])

        t_shap = Table(shap_rows, colWidths=[28, 174, 75, 75, 70, 110])
        t_shap.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ]))
        story.append(t_shap)
        story.append(Spacer(1, 8))

        # Embedded SHAP Figure
        if len(req.top_contributors) > 0:
            story.append(Paragraph("<b>Local Feature Attribution Waterfall Figure</b>", styles['SubSectionHeading']))
            shap_fig_buf = _generate_applicant_shap_figure(req.top_contributors)
            story.append(Image(shap_fig_buf, width=532, height=185))
            story.append(Paragraph(
                "Figure 1: Comparison of local SHAP feature attributions before and after Reweighing. Green bars represent the mitigated model; grey bars represent baseline.",
                styles['TableFootnote']
            ))

        # ================= PAGE 3 =================
        story.append(PageBreak())
        story.append(Paragraph("3. Auditing Interpretation, Privacy & Decision Safeguards", styles['SectionHeading']))
        disclaimers = [
            "<b>No Adverse Action Significance:</b> Local feature attributions indicate statistical importance in the model's learned mapping. They do not constitute formal Adverse Action Reason Codes under the Equal Credit Opportunity Act (ECOA) or Fair Credit Reporting Act (FCRA).",
            "<b>Absence of Causal Proof:</b> Feature attribution changes under Reweighing reflect adjustments in the reweighted training loss function. They do not demonstrate causal impacts or guarantee counterfactual recourse for the applicant.",
            "<b>Applicant Privacy Preservation:</b> Sensitive demographic attributes (such as gender and age) were excluded from model input features under established research protocol. The applicant input vector was processed ephemerally in volatile memory and has not been stored in database tables or persistent logs.",
            "<b>Research Prototype Notice:</b> This report was generated by FairCredit-AI 2.0 strictly for academic and algorithmic auditing research demonstrations. It must not be utilized for production commercial lending or credit decisioning."
        ]
        for d in disclaimers:
            story.append(Paragraph(d, styles['BodyRegular']))
            story.append(Spacer(1, 4))

        return _build_pdf_buffer(story)

report_service = ReportService()
