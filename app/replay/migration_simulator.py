"""
Migration simulator: two modes.

1. simulate_migration — Phase 1 heuristic using catalog pricing ratios.
2. simulate_from_replay_data — Phase 2 evidence-based using actual replay results.
"""
from collections import Counter, defaultdict
from datetime import datetime

from app.audit.model_catalog import CATALOG, get_quality_tier
from app.replay.replay_models import MigrationScenario, MigrationSimulationResult
from app.schemas import EvidenceLevel, UsageRecord, ValidationStatus
from app.utils.date_range import calculate_date_range_days  # noqa: F401 — re-exported
from app.utils.evidence import adjust_confidence_for_evidence

# Rank ordering for evidence level and validation status — higher = stronger
_EVIDENCE_RANK: dict[str, int] = {
    "heuristic": 0,
    "estimated": 1,
    "observed_replay": 2,
    "llm_judge": 3,
    "human_reviewed": 4,
    "production_validated": 5,
}
_VALIDATION_RANK: dict[str, int] = {
    "not_validated": 0,
    "replayed": 1,
    "evaluator_scored": 2,
    "human_reviewed": 3,
    "ready_for_pilot": 4,
    "production_validated": 5,
}


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
        base_confidence_score=confidence,
        confidence_score=adjust_confidence_for_evidence(
            confidence, EvidenceLevel.ESTIMATED, ValidationStatus.NOT_VALIDATED
        ),
        recommendation=recommendation,
        rationale=rationale,
        evidence_level=EvidenceLevel.ESTIMATED,
        evidence_summary=(
            f"Catalog-based estimate using pricing ratios. {len(matching)} historical "
            f"request(s) on {scenario.source_model} analyzed. Cost ratio: {cost_ratio:.2f}x. "
            "No replay data used."
        ),
        limitations=[
            "Savings estimated from catalog pricing ratios, not live API pricing.",
            "Quality delta estimated from model tier difference, not measured from real outputs.",
            "Migration percentage is a user-defined assumption, not observed behavior.",
        ],
        validation_status=ValidationStatus.NOT_VALIDATED,
    )


# ── Evidence-based simulation ─────────────────────────────────────────────────

_PROMOTION_THRESHOLD = 0.80  # fraction of results that must be at/above a level to claim it

_EVIDENCE_LABEL: dict[str, str] = {
    "production_validated": "production-validated",
    "human_reviewed": "human-reviewed",
    "llm_judge": "LLM-judged",
    "observed_replay": "evaluator-scored",
    "estimated": "estimated",
    "heuristic": "heuristic",
}


def _derive_scenario_evidence(
    results: list[dict],
) -> tuple[EvidenceLevel, ValidationStatus, float, dict[str, int], float, str]:
    """Coverage-aware evidence promotion for a scenario result group.

    Returns:
        evidence_level      — claimed level (only promoted when ≥80% of results qualify)
        validation_status   — status matching the claimed level
        coverage_pct        — fraction of results at or above the claimed level
        evidence_counts     — raw count per evidence level key
        partial_higher_pct  — fraction above the claimed level (non-zero only when level is
                              OBSERVED_REPLAY and partial higher-evidence data exists)
        evidence_summary    — human-readable coverage breakdown string
    """
    total = len(results)
    counts: Counter[str] = Counter(r.get("evidence_level") or "observed_replay" for r in results)

    def _frac_at_or_above(min_rank: int) -> float:
        if total == 0:
            return 0.0
        return sum(
            cnt for lvl, cnt in counts.items()
            if _EVIDENCE_RANK.get(lvl, 0) >= min_rank
        ) / total

    pv_frac = _frac_at_or_above(_EVIDENCE_RANK["production_validated"])
    hr_frac = _frac_at_or_above(_EVIDENCE_RANK["human_reviewed"])
    lj_frac = _frac_at_or_above(_EVIDENCE_RANK["llm_judge"])
    obs_frac = _frac_at_or_above(_EVIDENCE_RANK["observed_replay"])

    if pv_frac >= _PROMOTION_THRESHOLD:
        level = EvidenceLevel.PRODUCTION_VALIDATED
        status = ValidationStatus.PRODUCTION_VALIDATED
        coverage_pct = pv_frac
        partial_higher_pct = 0.0
    elif hr_frac >= _PROMOTION_THRESHOLD:
        level = EvidenceLevel.HUMAN_REVIEWED
        status = ValidationStatus.HUMAN_REVIEWED
        coverage_pct = hr_frac
        partial_higher_pct = 0.0
    elif lj_frac >= _PROMOTION_THRESHOLD:
        level = EvidenceLevel.LLM_JUDGE
        status = ValidationStatus.EVALUATOR_SCORED
        coverage_pct = lj_frac
        partial_higher_pct = 0.0
    else:
        level = EvidenceLevel.OBSERVED_REPLAY
        status = ValidationStatus.EVALUATOR_SCORED
        coverage_pct = obs_frac
        # Fraction above observed_replay — used for a small confidence boost
        partial_higher_pct = lj_frac

    summary = _build_evidence_summary(counts, total, level)
    return level, status, coverage_pct, dict(counts), partial_higher_pct, summary


def _build_evidence_summary(
    counts: Counter,
    total: int,
    level: EvidenceLevel,
) -> str:
    parts = [
        f"{counts[lvl]} {_EVIDENCE_LABEL.get(lvl, lvl)}"
        for lvl in ("production_validated", "human_reviewed", "llm_judge", "observed_replay", "heuristic", "estimated")
        if counts.get(lvl)
    ]
    breakdown = ", ".join(parts) if parts else f"{total} replayed"
    base = f"{total} replayed: {breakdown}."

    if level == EvidenceLevel.OBSERVED_REPLAY:
        hr = counts.get("human_reviewed", 0) + counts.get("production_validated", 0)
        lj = counts.get("llm_judge", 0)
        if hr > 0:
            pct = round(hr / total * 100) if total else 0
            base += f" Scenario remains replay-validated; human review coverage is {pct}% (threshold: 80%)."
        elif lj > 0:
            pct = round(lj / total * 100) if total else 0
            base += f" Scenario remains replay-validated; LLM-judge coverage is {pct}% (threshold: 80%)."

    return base


def _compute_sim_confidence(
    base_score: float,
    evidence_level: EvidenceLevel,
    validation_status: ValidationStatus,
    partial_higher_pct: float,
) -> float:
    """Evidence-adjusted confidence with a small linear boost for partial higher coverage.

    When the scenario stays at OBSERVED_REPLAY because the 80% threshold was not met,
    but some fraction of results have higher-quality evidence, we add up to +3pp to
    acknowledge that signal without overclaiming full promotion.
    """
    adj = adjust_confidence_for_evidence(base_score, evidence_level, validation_status)
    if partial_higher_pct > 0 and evidence_level == EvidenceLevel.OBSERVED_REPLAY:
        boost = 0.03 * partial_higher_pct
        adj = min(round(adj + boost, 4), 0.95)
    return adj


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
                base_confidence_score=0.0,
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
                evidence_level=EvidenceLevel.HEURISTIC,
                evidence_summary=(
                    f"All {len(failed)} replay attempt(s) failed. "
                    "No usable quality or cost data was produced."
                ),
                limitations=["All replays failed; cannot assess cost savings or quality impact."],
                validation_status=ValidationStatus.NOT_VALIDATED,
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

        # Derive coverage-aware scenario evidence from the actual replay results in this group
        successful_rr = [p["result"] for p in successful]
        evidence_level, validation_status, coverage_pct, ev_counts, partial_higher_pct, ev_summary = (
            _derive_scenario_evidence(successful_rr)
        )

        failed_note = (
            [f"{len(failed)} of {len(successful) + len(failed)} replay(s) failed and were excluded."]
            if failed else []
        )

        if evidence_level == EvidenceLevel.HUMAN_REVIEWED:
            quality_note = "Quality validated by human reviewers."
        elif evidence_level == EvidenceLevel.LLM_JUDGE:
            quality_note = "Quality validated by LLM judge; human review may differ."
        else:
            quality_note = "Quality scored by heuristic evaluator; human review may yield different results."

        simulations.append(MigrationSimulationResult(
            scenario_name=f"{source_model}→{target_model}:{task_type}",
            source_model=source_model,
            target_model=target_model,
            current_annualized_cost=round(current_annualized, 2),
            simulated_annualized_cost=round(simulated_annualized, 2),
            estimated_annual_savings=round(savings, 2),
            estimated_savings_pct=round(savings_pct, 2),
            average_quality_delta=round(avg_quality_delta, 4),
            base_confidence_score=confidence,
            confidence_score=_compute_sim_confidence(
                confidence, evidence_level, validation_status, partial_higher_pct
            ),
            recommendation=recommendation,
            rationale=rationale,
            avg_current_quality=round(avg_current_quality, 4),
            avg_simulated_quality=round(avg_simulated_quality, 4),
            avg_latency_delta_ms=round(avg_latency, 2),
            records_analyzed=len(successful),
            failed_replays=len(failed),
            evidence_level=evidence_level,
            evidence_summary=(
                ev_summary + f" "
                f"Avg quality: {avg_simulated_quality:.2f} vs. baseline 1.00 "
                f"({quality_loss_pct:.1f}% loss). "
                f"Estimated savings: ${savings:,.2f}/yr."
            ),
            limitations=[
                quality_note,
                f"Based on {len(successful)} request sample; larger samples improve confidence.",
                "Latency measured in test conditions; production latency may vary.",
            ] + failed_note,
            validation_status=validation_status,
            evidence_coverage_pct=round(coverage_pct, 4),
            evidence_counts=ev_counts,
        ))

    return sorted(simulations, key=lambda s: -s.estimated_annual_savings)
