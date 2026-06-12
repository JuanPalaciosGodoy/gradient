"""
Migration simulator: two modes.

1. simulate_migration — Phase 1 heuristic using catalog pricing ratios.
2. simulate_from_replay_data — Phase 2 evidence-based using actual replay results.
"""
from collections import defaultdict
from datetime import datetime

from app.audit.model_catalog import CATALOG, get_quality_tier
from app.replay.replay_models import MigrationScenario, MigrationSimulationResult
from app.schemas import UsageRecord
from app.utils.date_range import calculate_date_range_days  # noqa: F401 — re-exported


# Quality tier ordering: lower index = higher quality
_TIER_ORDER: dict[str, int] = {
    "frontier": 0,
    "balanced": 1,
    "cheap": 2,
    "open_source": 3,
    "unknown": 2,
}

_QUALITY_DELTA_PER_TIER_STEP = -0.01

# Task types that require more conservative migration thresholds
_HIGH_RISK_TASKS = {"research", "coding"}
# Task types that tolerate more aggressive migration recommendations
_LOWER_RISK_TASKS = {"classification", "summarization"}


def simulate_migration(
    records: list[UsageRecord],
    scenario: MigrationScenario,
    date_range_days: int = 30,
) -> MigrationSimulationResult:
    """
    Phase 1: estimate migration outcomes using catalog pricing ratios.
    No actual replay data required.
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

    source_entry = CATALOG.get(scenario.source_model)
    target_entry = CATALOG.get(scenario.target_model)
    if source_entry and target_entry and source_entry.relative_cost > 0:
        cost_ratio = target_entry.relative_cost / source_entry.relative_cost
    else:
        cost_ratio = 0.5

    migrated_portion = current_annualized * scenario.migration_percentage
    remaining_portion = current_annualized * (1 - scenario.migration_percentage)
    simulated_annualized = remaining_portion + migrated_portion * cost_ratio

    estimated_annual_savings = current_annualized - simulated_annualized
    estimated_savings_pct = (
        estimated_annual_savings / current_annualized * 100
        if current_annualized > 0 else 0.0
    )

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

    source_tier = get_quality_tier(scenario.source_model)
    target_tier = get_quality_tier(scenario.target_model)
    tier_steps = _TIER_ORDER.get(target_tier, 2) - _TIER_ORDER.get(source_tier, 0)
    average_quality_delta = round(tier_steps * _QUALITY_DELTA_PER_TIER_STEP, 4)

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


# ── Evidence-based simulation ─────────────────────────────────────────────────

def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _compute_replay_confidence(
    records_analyzed: int,
    avg_evaluator_confidence: float,
    failed_replay_pct: float,
    quality_std: float,
    task_type: str | None,
) -> float:
    """Score 0–1 based on evidence strength from actual replay data."""
    score = 0.50

    # Volume
    if records_analyzed >= 200:
        score += 0.20
    elif records_analyzed >= 50:
        score += 0.15
    elif records_analyzed >= 20:
        score += 0.10
    elif records_analyzed >= 5:
        score += 0.05

    # Evaluator confidence
    score += avg_evaluator_confidence * 0.15

    # Failure penalty
    score -= failed_replay_pct * 0.25

    # Score consistency
    if quality_std < 0.05:
        score += 0.05
    elif quality_std > 0.20:
        score -= 0.05

    # Task risk
    if task_type in _HIGH_RISK_TASKS:
        score -= 0.10
    elif task_type in _LOWER_RISK_TASKS:
        score += 0.05

    return round(min(max(score, 0.0), 0.95), 2)


def _recommend_from_replay(
    quality_loss_pct: float,
    savings_pct: float,
    confidence: float,
    task_type: str | None,
    records_analyzed: int,
) -> tuple[str, str]:
    """Return (recommendation, rationale) based on evidence from replay results."""
    if savings_pct <= 0:
        return (
            "hold",
            f"Target model costs more than source model (savings: {savings_pct:.1f}%). "
            "No migration recommended.",
        )

    if records_analyzed < 3:
        return (
            "investigate",
            f"Insufficient data ({records_analyzed} record(s) analyzed). "
            "Run a larger replay before making a migration decision.",
        )

    # Threshold matrix by task risk level
    if task_type in _HIGH_RISK_TASKS:
        migrate_loss_max = 1.0
        pilot_loss_max = 3.0
        migrate_savings_min = 30.0
        pilot_savings_min = 40.0
    elif task_type in _LOWER_RISK_TASKS:
        migrate_loss_max = 4.0
        pilot_loss_max = 8.0
        migrate_savings_min = 15.0
        pilot_savings_min = 25.0
    else:
        migrate_loss_max = 2.0
        pilot_loss_max = 5.0
        migrate_savings_min = 20.0
        pilot_savings_min = 30.0

    task_label = task_type or "this"

    if quality_loss_pct <= migrate_loss_max and savings_pct >= migrate_savings_min:
        return (
            "migrate",
            f"Quality loss of {quality_loss_pct:.1f}% is within the "
            f"{migrate_loss_max:.0f}% threshold for {task_label} tasks. "
            f"Savings of {savings_pct:.1f}% exceed the {migrate_savings_min:.0f}% minimum. "
            f"Confidence: {round(confidence * 100)}%. Recommended for full migration.",
        )

    if quality_loss_pct <= pilot_loss_max and savings_pct >= pilot_savings_min:
        return (
            "controlled_pilot",
            f"Moderate quality loss ({quality_loss_pct:.1f}%) with strong savings "
            f"({savings_pct:.1f}%). Run a controlled pilot on a subset of {task_label} "
            f"workload before full migration. Confidence: {round(confidence * 100)}%.",
        )

    if quality_loss_pct > pilot_loss_max:
        return (
            "no_migration",
            f"Quality loss of {quality_loss_pct:.1f}% exceeds the "
            f"{pilot_loss_max:.0f}% threshold for {task_label} tasks. "
            "Do not migrate. Consider a higher-quality alternative model.",
        )

    return (
        "investigate",
        f"Mixed signals: quality loss {quality_loss_pct:.1f}%, savings {savings_pct:.1f}%. "
        f"Savings or quality data are below thresholds for a definitive recommendation. "
        f"Expand replay sample to improve confidence ({round(confidence * 100)}%).",
    )


def simulate_from_replay_data(
    original_records: list[dict],
    replay_results: list[dict],
    date_range_days: int = 30,
) -> list[MigrationSimulationResult]:
    """
    Phase 2: evidence-based migration simulation using actual replay outcomes.

    Groups replay results by (source_model, target_model, task_type) and computes
    cost savings, quality impact, latency, and a recommendation for each combination.
    """
    if not original_records or not replay_results:
        return []

    annualization_factor = 365 / max(date_range_days, 1)

    # Build lookup: str(record_id) → record dict
    orig_by_id: dict[str, dict] = {str(r["id"]): r for r in original_records}

    # Group replay results by (source_model, target_model, task_type)
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rr in replay_results:
        rec_id = str(rr["original_record_id"])
        orig = orig_by_id.get(rec_id)
        if orig is None:
            continue
        source_model = orig.get("model", "unknown")
        task_type = orig.get("task_type") or "other"
        key = (source_model, rr["candidate_model"], task_type)
        groups[key].append({"result": rr, "original": orig})

    simulations = []
    for (source_model, target_model, task_type), pairs in groups.items():
        successful = [p for p in pairs if not p["result"].get("error_message")]
        failed = [p for p in pairs if p["result"].get("error_message")]

        if not successful:
            # No successful results for this scenario — can't make a recommendation
            simulations.append(MigrationSimulationResult(
                scenario_name=f"{source_model}→{target_model}:{task_type}",
                source_model=source_model,
                target_model=target_model,
                current_annualized_cost=0.0,
                simulated_annualized_cost=0.0,
                estimated_annual_savings=0.0,
                estimated_savings_pct=0.0,
                average_quality_delta=0.0,
                confidence_score=0.0,
                recommendation="investigate",
                rationale=(
                    f"All {len(failed)} replay attempt(s) failed for this scenario. "
                    "No cost or quality data available."
                ),
                avg_current_quality=0.0,
                avg_simulated_quality=0.0,
                avg_latency_delta_ms=0.0,
                records_analyzed=0,
                failed_replays=len(failed),
            ))
            continue

        # Cost computation (only from successful replays)
        orig_period_cost = sum(p["original"]["cost"] for p in successful)
        cand_period_cost = sum(p["result"]["estimated_cost"] for p in successful)
        current_annualized = orig_period_cost * annualization_factor
        simulated_annualized = cand_period_cost * annualization_factor
        savings = current_annualized - simulated_annualized
        savings_pct = savings / current_annualized * 100 if current_annualized > 0 else 0.0

        # Quality: original is 1.0 by definition (it IS the baseline)
        avg_current_quality = 1.0
        quality_scores = [p["result"]["quality_score"] for p in successful]
        avg_simulated_quality = _safe_mean(quality_scores)
        avg_quality_delta = avg_simulated_quality - avg_current_quality
        quality_loss_pct = max(0.0, -avg_quality_delta * 100)

        # Latency (average candidate latency; we don't have original latency)
        avg_latency = _safe_mean([p["result"]["latency_ms"] for p in successful])

        # Confidence
        avg_eval_confidence = _safe_mean(
            [p["result"].get("quality_confidence", 1.0) for p in successful]
        )
        total = len(successful) + len(failed)
        failed_pct = len(failed) / total if total > 0 else 0.0
        quality_std = _safe_stdev(quality_scores)

        confidence = _compute_replay_confidence(
            records_analyzed=len(successful),
            avg_evaluator_confidence=avg_eval_confidence,
            failed_replay_pct=failed_pct,
            quality_std=quality_std,
            task_type=task_type,
        )

        recommendation, rationale = _recommend_from_replay(
            quality_loss_pct=quality_loss_pct,
            savings_pct=savings_pct,
            confidence=confidence,
            task_type=task_type,
            records_analyzed=len(successful),
        )

        simulations.append(MigrationSimulationResult(
            scenario_name=f"{source_model}→{target_model}:{task_type}",
            source_model=source_model,
            target_model=target_model,
            current_annualized_cost=round(current_annualized, 2),
            simulated_annualized_cost=round(simulated_annualized, 2),
            estimated_annual_savings=round(savings, 2),
            estimated_savings_pct=round(savings_pct, 2),
            average_quality_delta=round(avg_quality_delta, 4),
            confidence_score=confidence,
            recommendation=recommendation,
            rationale=rationale,
            avg_current_quality=round(avg_current_quality, 4),
            avg_simulated_quality=round(avg_simulated_quality, 4),
            avg_latency_delta_ms=round(avg_latency, 2),
            records_analyzed=len(successful),
            failed_replays=len(failed),
        ))

    return sorted(simulations, key=lambda s: -s.estimated_annual_savings)
