"""Markdown renderer for ExecutiveReplayReport."""
from app.replay.replay_report import ExecutiveReplayReport


def render_replay_markdown_report(report: ExecutiveReplayReport) -> str:
    lines: list[str] = []

    lines.append("# Gradient Replay Analysis")
    lines.append("")
    lines.append(f"_Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")

    # ── Executive Summary ─────────────────────────────────────────────────────
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.executive_summary)
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total Requests | {report.total_requests:,} |")
    lines.append(f"| Replay Results | {report.total_results:,} |")
    lines.append(f"| Current Annual Spend | ${report.current_annualized_spend:,.2f} |")
    lines.append(f"| Potential Annual Savings | ${report.estimated_annual_savings:,.2f} |")
    lines.append(f"| Avg Quality Impact | {report.avg_quality_impact:+.1%} |")
    lines.append(f"| Overall Confidence | {report.overall_confidence}% |")
    lines.append("")

    # ── Recommended Migrations ────────────────────────────────────────────────
    lines.append("## Recommended Migrations")
    lines.append("")
    if report.recommended_migrations:
        lines.append("| Source Model | Target Model | Task | Annual Savings | Quality Loss | Confidence | Action |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in report.recommended_migrations:
            lines.append(
                f"| {s.source_model} | {s.target_model} | {s.task_type} "
                f"| ${s.estimated_annual_savings:,.2f} "
                f"| {s.quality_loss_pct:.1f}% "
                f"| {s.confidence_pct}% "
                f"| {s.recommendation} |"
            )
        lines.append("")
        for s in report.recommended_migrations:
            lines.append(
                f"**{s.source_model} → {s.target_model} ({s.task_type}):** "
                f"{s.rationale}"
            )
            lines.append("")
    else:
        lines.append("No migration scenarios meet the threshold for a positive recommendation.")
        lines.append("")

    # ── Do Not Migrate ────────────────────────────────────────────────────────
    lines.append("## Do Not Migrate")
    lines.append("")
    if report.do_not_migrate:
        lines.append("| Source Model | Target Model | Task | Quality Loss | Reason |")
        lines.append("|---|---|---|---|---|")
        for s in report.do_not_migrate:
            short_rationale = s.rationale.split(".")[0] if s.rationale else "Quality threshold exceeded"
            lines.append(
                f"| {s.source_model} | {s.target_model} | {s.task_type} "
                f"| {s.quality_loss_pct:.1f}% "
                f"| {short_rationale} |"
            )
        lines.append("")
    else:
        lines.append("No scenarios were flagged for explicit non-migration.")
        lines.append("")

    # ── Risk Notes ────────────────────────────────────────────────────────────
    lines.append("## Risk Notes")
    lines.append("")
    if report.risk_notes:
        for note in report.risk_notes:
            lines.append(f"- {note}")
        lines.append("")
    else:
        lines.append("No specific risk flags identified.")
        lines.append("")

    # ── Next Actions ──────────────────────────────────────────────────────────
    lines.append("## Next Actions")
    lines.append("")
    for i, action in enumerate(report.next_actions, 1):
        lines.append(f"{i}. {action}")
    lines.append("")

    return "\n".join(lines)
