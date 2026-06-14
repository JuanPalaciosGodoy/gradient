"""Markdown renderer for ExecutiveReplayReport."""
from app.replay.replay_report import ExecutiveReplayReport, ScenarioOutcome
from app.schemas import EvidenceLevel


def _render_scenario_tier(
    lines: list[str],
    header: str,
    scenarios: list[ScenarioOutcome],
) -> None:
    """Render one evidence-tier subsection under ## Recommended Migrations."""
    if not scenarios:
        return
    lines.append(header)
    lines.append("")
    lines.append(
        "| Source Model | Target Model | Task | Annual Savings | Quality Loss | Confidence | Action |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in scenarios:
        lines.append(
            f"| {s.source_model} | {s.target_model} | {s.task_type} "
            f"| ${s.estimated_annual_savings:,.2f} "
            f"| {s.quality_loss_pct:.1f}% "
            f"| {s.confidence_pct}% "
            f"| {s.recommendation} |"
        )
    lines.append("")
    for s in scenarios:
        evidence_label = s.evidence_level.value.replace("_", " ").title()
        coverage_note = f" · {s.evidence_coverage_pct:.0%} coverage" if s.evidence_coverage_pct > 0 else ""
        lines.append(
            f"**{s.source_model} → {s.target_model} ({s.task_type}):** "
            f"{s.rationale}"
        )
        lines.append(f"_Evidence: {evidence_label}{coverage_note} · {s.evidence_summary}_")
        lines.append("")


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

    # ── Recommended Migrations — split by evidence tier (highest → lowest) ──────
    lines.append("## Recommended Migrations")
    lines.append("")
    if report.recommended_migrations:
        production_validated = [
            s for s in report.recommended_migrations
            if s.evidence_level == EvidenceLevel.PRODUCTION_VALIDATED
        ]
        human_reviewed = [
            s for s in report.recommended_migrations
            if s.evidence_level == EvidenceLevel.HUMAN_REVIEWED
        ]
        llm_judged = [
            s for s in report.recommended_migrations
            if s.evidence_level == EvidenceLevel.LLM_JUDGE
        ]
        replay_validated = [
            s for s in report.recommended_migrations
            if s.evidence_level == EvidenceLevel.OBSERVED_REPLAY
        ]
        heuristic_estimated = [
            s for s in report.recommended_migrations
            if s.evidence_level in (EvidenceLevel.HEURISTIC, EvidenceLevel.ESTIMATED)
        ]

        _render_scenario_tier(lines, "### Production-Validated Opportunities", production_validated)
        _render_scenario_tier(lines, "### Human-Reviewed Opportunities", human_reviewed)
        _render_scenario_tier(lines, "### LLM-Judged Opportunities", llm_judged)
        _render_scenario_tier(lines, "### Replay-Validated Opportunities", replay_validated)
        _render_scenario_tier(lines, "### Heuristic / Estimated Opportunities", heuristic_estimated)

        rendered_count = (
            len(production_validated) + len(human_reviewed) + len(llm_judged)
            + len(replay_validated) + len(heuristic_estimated)
        )
        if rendered_count == 0:
            lines.append("No migration scenarios meet the threshold for a positive recommendation.")
            lines.append("")
    else:
        lines.append("No migration scenarios meet the threshold for a positive recommendation.")
        lines.append("")

    # ── Needs More Evidence / Investigate ─────────────────────────────────────
    lines.append("## Needs More Evidence / Investigate")
    lines.append("")
    if report.investigate_scenarios:
        lines.append(
            "_These scenarios show potential savings but have insufficient replay "
            "coverage or confidence to recommend migration. Expand the replay sample "
            "or add human review to upgrade evidence._"
        )
        lines.append("")
        lines.append(
            "| Source Model | Target Model | Task | Potential Savings | Confidence | Evidence |"
        )
        lines.append("|---|---|---|---|---|---|")
        for s in report.investigate_scenarios:
            evidence_label = s.evidence_level.value.replace("_", " ").title()
            lines.append(
                f"| {s.source_model} | {s.target_model} | {s.task_type} "
                f"| ${s.estimated_annual_savings:,.2f} "
                f"| {s.confidence_pct}% "
                f"| {evidence_label} |"
            )
        lines.append("")
        for s in report.investigate_scenarios:
            lines.append(
                f"**{s.source_model} → {s.target_model} ({s.task_type}):** {s.rationale}"
            )
            lines.append("")
    else:
        lines.append("No scenarios require further investigation.")
        lines.append("")

    # ── Hold / No Savings ─────────────────────────────────────────────────────
    lines.append("## Hold / No Savings")
    lines.append("")
    if report.hold_scenarios:
        lines.append("| Source Model | Target Model | Task | Savings | Reason |")
        lines.append("|---|---|---|---|---|")
        for s in report.hold_scenarios:
            short_rationale = s.rationale.split(".")[0] if s.rationale else "No cost savings identified"
            lines.append(
                f"| {s.source_model} | {s.target_model} | {s.task_type} "
                f"| ${s.estimated_annual_savings:,.2f} "
                f"| {short_rationale} |"
            )
        lines.append("")
    else:
        lines.append("No scenarios are on hold.")
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
