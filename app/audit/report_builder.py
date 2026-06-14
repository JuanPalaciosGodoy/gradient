import uuid
from datetime import datetime, timezone

from app.audit.confidence import compute_confidence
from app.audit.model_catalog import get_provider, get_quality_tier
from app.audit.opportunity_analyzer import find_opportunities
from app.audit.spend_analyzer import analyze_spend
from app.audit.task_classifier import classify_task
from app.schemas import AuditReport, RiskNote, SpendSummary, TaskType, UsageRecord

_FRONTIER_TIERS = {"frontier"}
_UNCLASSIFIED = {TaskType.OTHER.value, TaskType.UNKNOWN.value}


def build_report(audit_run_id: str, records: list[UsageRecord]) -> AuditReport:
    for record in records:
        if record.task_type is None:
            record.task_type = classify_task(record.prompt)

    spend = analyze_spend(records)
    opportunities = find_opportunities(records, spend.date_range_days)
    confidence = compute_confidence(records)
    risk_notes = _assess_risks(records, spend)

    raw_savings = sum(opp.estimated_annual_savings for opp in opportunities)
    potential_annual_savings = round(min(raw_savings, spend.annualized_cost), 2)
    savings_rate = (
        round(potential_annual_savings / spend.annualized_cost, 4)
        if spend.annualized_cost > 0
        else 0.0
    )

    executive_summary = _build_executive_summary(spend, potential_annual_savings, opportunities)
    next_actions = _generate_next_actions(spend, opportunities, risk_notes)

    return AuditReport(
        report_id=str(uuid.uuid4()),
        audit_run_id=audit_run_id,
        generated_at=datetime.now(timezone.utc),
        executive_summary=executive_summary,
        current_annual_spend=spend.annualized_cost,
        spend_summary=spend,
        top_opportunities=opportunities,
        risk_notes=risk_notes,
        confidence_score=confidence,
        potential_annual_savings=potential_annual_savings,
        savings_rate=savings_rate,
        recommended_next_actions=next_actions,
    )


def _build_executive_summary(
    spend: SpendSummary,
    potential_annual_savings: float,
    opportunities: list,
) -> str:
    spend_str = f"${spend.annualized_cost:,.0f}"
    savings_str = f"${potential_annual_savings:,.0f}"

    if not opportunities:
        return (
            f"Based on the uploaded usage data, Gradient estimates annualized AI spend of {spend_str}. "
            "No optimization opportunities were identified in this dataset. "
            "Upload a broader data sample to improve audit coverage."
        )

    top_tasks = list(dict.fromkeys(opp.task_type.replace("_", " ") for opp in opportunities[:3]))
    if len(top_tasks) == 1:
        task_phrase = top_tasks[0]
    elif len(top_tasks) == 2:
        task_phrase = f"{top_tasks[0]} and {top_tasks[1]}"
    else:
        task_phrase = f"{top_tasks[0]}, {top_tasks[1]}, and {top_tasks[2]}"

    return (
        f"Based on the uploaded usage data, Gradient estimates annualized AI spend of {spend_str} "
        f"and identifies approximately {savings_str} in potential annual savings. "
        f"The largest opportunities are {task_phrase} workloads currently running on "
        f"higher-tier models. "
        "These findings are heuristic estimates based on spend data and model tier analysis — "
        "not measured from actual output quality. "
        "Use the Replay Engine to validate savings and quality impact before making production changes."
    )


def _assess_risks(records: list[UsageRecord], spend: SpendSummary) -> list[RiskNote]:
    notes = []

    # Provider concentration
    if spend.provider_breakdown and spend.total_cost > 0:
        top_provider = max(spend.provider_breakdown, key=lambda p: spend.provider_breakdown[p])
        concentration = spend.provider_breakdown[top_provider] / spend.total_cost
        if concentration > 0.80:
            notes.append(RiskNote(
                category="provider_concentration",
                description=(
                    f"{round(concentration * 100)}% of spend is with {top_provider}. "
                    "Single-vendor dependency creates pricing and availability risk."
                ),
                severity="medium",
            ))

    # Model concentration
    if spend.model_breakdown and spend.total_cost > 0:
        top_model = max(spend.model_breakdown, key=lambda m: spend.model_breakdown[m])
        model_conc = spend.model_breakdown[top_model] / spend.total_cost
        if model_conc > 0.80:
            notes.append(RiskNote(
                category="model_concentration",
                description=(
                    f"{round(model_conc * 100)}% of spend is concentrated on {top_model}. "
                    "Dependency on a single model creates vendor lock-in and pricing risk."
                ),
                severity="medium",
            ))

    # Frontier-model dependency
    if records and spend.total_cost > 0:
        frontier_cost = sum(
            r.cost for r in records
            if get_quality_tier(r.model) in _FRONTIER_TIERS
        )
        frontier_pct = frontier_cost / spend.total_cost
        if frontier_pct > 0.60:
            notes.append(RiskNote(
                category="frontier_model_dependency",
                description=(
                    f"{round(frontier_pct * 100)}% of spend is on frontier-tier models. "
                    "Many workloads may not require this capability level."
                ),
                severity="medium",
            ))

    # High annualized spend
    if spend.annualized_cost > 50_000:
        notes.append(RiskNote(
            category="spend_level",
            description=(
                f"Annualized AI spend of ${spend.annualized_cost:,.0f} warrants formal "
                "procurement review and budget governance."
            ),
            severity="high" if spend.annualized_cost > 200_000 else "medium",
        ))

    # Low feedback coverage
    if records:
        feedback_rate = sum(1 for r in records if r.feedback) / len(records)
        if feedback_rate < 0.10:
            notes.append(RiskNote(
                category="low_feedback_coverage",
                description=(
                    f"Only {round(feedback_rate * 100)}% of records include feedback. "
                    "Without output quality signals, savings estimates cannot be validated."
                ),
                severity="low",
            ))

    # Short data history
    if spend.date_range_days < 14:
        notes.append(RiskNote(
            category="short_data_history",
            description="Usage data covers fewer than 14 days. Annualized projections may be unreliable.",
            severity="low",
        ))

    # Large unclassified spend
    unclassified_spend = sum(
        v for k, v in spend.task_breakdown.items() if k in _UNCLASSIFIED
    )
    if spend.total_cost > 0 and unclassified_spend / spend.total_cost > 0.20:
        notes.append(RiskNote(
            category="unclassified_spend",
            description=(
                f"${unclassified_spend:,.2f} ({round(unclassified_spend / spend.total_cost * 100)}%) "
                "of spend is in unclassified tasks. These cannot be automatically optimized."
            ),
            severity="low",
        ))

    # Replay validation (always present)
    notes.append(RiskNote(
        category="replay_validation_required",
        description=(
            "All model migration recommendations should be validated through a replay analysis "
            "before production changes. Output quality differences may not be visible in cost data alone."
        ),
        severity="medium",
    ))

    return notes


def _generate_next_actions(spend: SpendSummary, opportunities: list, risk_notes: list) -> list[str]:
    actions = []

    if opportunities:
        top = opportunities[0]
        actions.append(
            f"Pilot {top.recommended_model} for {top.task_type.replace('_', ' ')} tasks "
            f"— estimated ${top.estimated_annual_savings:,.2f}/yr in savings."
        )

    if any(r.category == "model_concentration" for r in risk_notes):
        actions.append(
            "Reduce dependency on the dominant model by validating equivalent workloads "
            "on at least one alternate model family."
        )

    if any(r.category == "provider_concentration" for r in risk_notes):
        actions.append("Diversify across model providers to reduce single-vendor concentration risk.")

    if any(r.category == "frontier_model_dependency" for r in risk_notes):
        actions.append(
            "Audit frontier-model usage by task type. "
            "Most summarization, classification, and extraction tasks do not require frontier models."
        )

    if any(r.category == "low_feedback_coverage" for r in risk_notes):
        actions.append(
            "Instrument output quality feedback before migrating models. "
            "Even a 20% sample significantly improves audit confidence."
        )

    if spend.annualized_cost > 50_000:
        actions.append("Establish per-team spend budgets and real-time cost alerts.")

    actions.append("Upload 90+ days of usage data to increase audit confidence and annualization accuracy.")

    return actions
