"""
Migration simulator: estimates what would happen if N% of a workload
migrated from one model to another, using historical cost data and the
model catalog for pricing ratios.

This is a Phase 1-style heuristic — no actual LLM calls.
Phase 2 replay results will replace these estimates with evidence.
"""
from app.audit.model_catalog import CATALOG, get_quality_tier
from app.replay.replay_models import MigrationScenario, MigrationSimulationResult
from app.schemas import UsageRecord

# Quality tier ordering: lower index = higher quality
_TIER_ORDER: dict[str, int] = {
    "frontier": 0,
    "balanced": 1,
    "cheap": 2,
    "open_source": 3,
    "unknown": 2,
}

# Estimated quality delta per tier step (negative = quality loss)
_QUALITY_DELTA_PER_TIER_STEP = -0.01


def simulate_migration(
    records: list[UsageRecord],
    scenario: MigrationScenario,
    date_range_days: int = 30,
) -> MigrationSimulationResult:
    """
    Simulate migrating `migration_percentage` of matching records from
    `source_model` to `target_model`.

    Matching criteria:
      - record.model == scenario.source_model
      - record.task_type in scenario.task_types_included (or list is empty → all)
    """
    included_tasks = set(scenario.task_types_included)

    matching = [
        r for r in records
        if r.model == scenario.source_model
        and (
            not included_tasks
            or (r.task_type is not None and r.task_type.value in included_tasks)
        )
    ]

    period_cost = sum(r.cost for r in matching)
    annualization_factor = 365 / max(date_range_days, 1)
    current_annualized = period_cost * annualization_factor

    # Estimate cost ratio from model catalog
    source_entry = CATALOG.get(scenario.source_model)
    target_entry = CATALOG.get(scenario.target_model)
    if source_entry and target_entry and source_entry.relative_cost > 0:
        cost_ratio = target_entry.relative_cost / source_entry.relative_cost
    else:
        cost_ratio = 0.5  # conservative default

    migrated_portion = current_annualized * scenario.migration_percentage
    remaining_portion = current_annualized * (1 - scenario.migration_percentage)
    simulated_annualized = remaining_portion + migrated_portion * cost_ratio

    estimated_annual_savings = current_annualized - simulated_annualized
    estimated_savings_pct = (
        estimated_annual_savings / current_annualized * 100
        if current_annualized > 0 else 0.0
    )

    # Confidence: increases with record volume and catalog coverage
    confidence = 0.60
    if len(matching) >= 100:
        confidence += 0.15
    elif len(matching) >= 10:
        confidence += 0.08
    if source_entry and target_entry:
        confidence += 0.10
    if date_range_days >= 30:
        confidence += 0.05
    confidence = round(min(confidence, 0.95), 2)

    # Quality delta: based on tier change
    source_tier = get_quality_tier(scenario.source_model)
    target_tier = get_quality_tier(scenario.target_model)
    tier_steps = _TIER_ORDER.get(target_tier, 2) - _TIER_ORDER.get(source_tier, 0)
    average_quality_delta = round(tier_steps * _QUALITY_DELTA_PER_TIER_STEP, 4)

    # Recommendation
    if estimated_annual_savings <= 0:
        recommendation = "hold"
    elif confidence >= 0.75:
        recommendation = "proceed"
    else:
        recommendation = "investigate"

    rationale = (
        f"Migrating {round(scenario.migration_percentage * 100)}% of "
        f"{scenario.source_model} workloads to {scenario.target_model} "
        f"is estimated to save ${estimated_annual_savings:,.0f}/yr "
        f"({estimated_savings_pct:.1f}% reduction). "
        f"Estimated quality delta: {average_quality_delta:+.2%}. "
        "Validate with replay testing before production migration."
    )

    return MigrationSimulationResult(
        scenario_name=scenario.scenario_name,
        source_model=scenario.source_model,
        target_model=scenario.target_model,
        current_annualized_cost=round(current_annualized, 2),
        simulated_annualized_cost=round(simulated_annualized, 2),
        estimated_annual_savings=round(estimated_annual_savings, 2),
        estimated_savings_pct=round(estimated_savings_pct, 2),
        average_quality_delta=average_quality_delta,
        confidence_score=confidence,
        recommendation=recommendation,
        rationale=rationale,
    )
