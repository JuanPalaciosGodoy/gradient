"""
Replay report models and builders.

ModelReplaySummary / ReplayReport — per-model summaries from raw replay results.
ScenarioOutcome / ExecutiveReplayReport — executive-level migration analysis.
"""
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel

from app.replay.migration_simulator import simulate_from_replay_data
from app.replay.replay_models import MigrationSimulationResult, ReplayRequest, ReplayResult


# ── Per-model replay summary ──────────────────────────────────────────────────

class ModelReplaySummary(BaseModel):
    model: str
    provider: str
    result_count: int
    success_count: int
    error_count: int
    coverage_rate: float               # fraction of requests with a successful result
    avg_quality_score: float
    avg_latency_ms: float
    avg_cost_per_request: float
    estimated_annual_savings: float    # vs matched original subset, annualized


class ReplayReport(BaseModel):
    replay_run_id: str
    generated_at: datetime
    total_requests: int
    total_results: int
    error_count: int
    model_summaries: list[ModelReplaySummary]  # sorted by savings descending


def build_replay_report(
    replay_run_id: str,
    requests: list[ReplayRequest],
    results: list[ReplayResult],
    date_range_days: int = 30,
) -> ReplayReport:
    annualization_factor = 365 / max(date_range_days, 1)
    orig_cost_by_id = {req.original_record_id: req.original_cost for req in requests}

    by_model: dict[str, list[ReplayResult]] = defaultdict(list)
    for r in results:
        by_model[r.candidate_model].append(r)

    summaries = []
    for model, model_results in by_model.items():
        successful = [r for r in model_results if not r.error_message]
        success_count = len(successful)
        error_count = len(model_results) - success_count
        coverage_rate = success_count / len(requests) if requests else 0.0

        avg_quality = (
            sum(r.quality_score for r in successful) / success_count
            if successful else 0.0
        )
        avg_latency = (
            sum(r.latency_ms for r in successful) / success_count
            if successful else 0.0
        )
        avg_cost = (
            sum(r.estimated_cost for r in successful) / success_count
            if successful else 0.0
        )

        # Compare candidate cost only against original costs for requests that
        # actually succeeded — failed calls are not "free" and must not inflate savings.
        matched_original_annualized = (
            sum(orig_cost_by_id.get(r.original_record_id, 0.0) for r in successful)
            * annualization_factor
        )
        candidate_annualized = sum(r.estimated_cost for r in successful) * annualization_factor
        savings = matched_original_annualized - candidate_annualized

        provider = model_results[0].candidate_provider if model_results else "unknown"

        summaries.append(ModelReplaySummary(
            model=model,
            provider=provider,
            result_count=len(model_results),
            success_count=success_count,
            error_count=error_count,
            coverage_rate=round(coverage_rate, 4),
            avg_quality_score=round(avg_quality, 4),
            avg_latency_ms=round(avg_latency, 2),
            avg_cost_per_request=round(avg_cost, 8),
            estimated_annual_savings=round(savings, 2),
        ))

    summaries.sort(key=lambda s: -s.estimated_annual_savings)

    return ReplayReport(
        replay_run_id=replay_run_id,
        generated_at=datetime.now(timezone.utc),
        total_requests=len(requests),
        total_results=len(results),
        error_count=sum(1 for r in results if r.error_message),
        model_summaries=summaries,
    )


# ── Executive replay report ───────────────────────────────────────────────────

class ScenarioOutcome(BaseModel):
    """Display-oriented view of one migration scenario."""
    source_model: str
    target_model: str
    task_type: str
    estimated_annual_savings: float
    quality_loss_pct: float     # positive = quality deteriorated (0 = no loss)
    avg_quality_delta: float    # signed; negative = candidate worse than original
    avg_latency_ms: float
    confidence_pct: int         # 0–100
    recommendation: str
    rationale: str
    records_analyzed: int


class ExecutiveReplayReport(BaseModel):
    replay_run_id: str
    generated_at: datetime
    total_requests: int
    total_results: int
    current_annualized_spend: float
    simulated_annualized_spend: float
    estimated_annual_savings: float
    avg_quality_impact: float
    overall_confidence: int          # 0–100
    executive_summary: str
    recommended_migrations: list[ScenarioOutcome]
    do_not_migrate: list[ScenarioOutcome]
    risk_notes: list[str]
    next_actions: list[str]


def _portfolio_savings(simulations: list[MigrationSimulationResult]) -> float:
    """Non-overlapping annual savings.

    For each (source_model, task_type) slot, only the best recommended scenario
    counts — you can only migrate a workload to one target model. Different slots
    are additive because they cover non-overlapping traffic.
    """
    best_per_slot: dict[tuple[str, str], float] = {}
    for s in simulations:
        if s.recommendation not in ("migrate", "controlled_pilot"):
            continue
        task_type = s.scenario_name.split(":")[-1] if ":" in s.scenario_name else "other"
        key = (s.source_model, task_type)
        best_per_slot[key] = max(best_per_slot.get(key, 0.0), s.estimated_annual_savings)
    return max(sum(v for v in best_per_slot.values() if v > 0), 0.0)


def _sim_to_outcome(sim: MigrationSimulationResult) -> ScenarioOutcome:
    task_type = sim.scenario_name.split(":")[-1] if ":" in sim.scenario_name else "other"
    return ScenarioOutcome(
        source_model=sim.source_model,
        target_model=sim.target_model,
        task_type=task_type,
        estimated_annual_savings=sim.estimated_annual_savings,
        quality_loss_pct=round(max(0.0, -sim.average_quality_delta * 100), 1),
        avg_quality_delta=sim.average_quality_delta,
        avg_latency_ms=sim.avg_latency_delta_ms,
        confidence_pct=round(sim.confidence_score * 100),
        recommendation=sim.recommendation,
        rationale=sim.rationale,
        records_analyzed=sim.records_analyzed,
    )


def _executive_summary(
    simulations: list[MigrationSimulationResult],
    current_annualized: float,
    best_savings: float,
    overall_confidence: int,
    total_requests: int,
) -> str:
    migrate_count = sum(1 for s in simulations if s.recommendation == "migrate")
    pilot_count = sum(1 for s in simulations if s.recommendation == "controlled_pilot")
    no_migrate_count = sum(1 for s in simulations if s.recommendation == "no_migration")
    savings_rate = best_savings / current_annualized if current_annualized > 0 else 0.0

    parts = [
        f"Gradient analyzed {total_requests} historical prompt(s) across "
        f"{len(simulations)} migration scenario(s).",
        f"At current run rate, annualized spend is ${current_annualized:,.2f}.",
    ]

    if best_savings > 0:
        parts.append(
            f"Evidence-based replay simulation identifies ${best_savings:,.2f}/yr in "
            f"potential savings ({savings_rate:.0%} of current spend) "
            f"with {overall_confidence}% confidence."
        )
    else:
        parts.append("No cost savings were identified in the tested migration scenarios.")

    if migrate_count:
        parts.append(
            f"{migrate_count} scenario{'s' if migrate_count > 1 else ''} "
            "recommended for immediate migration."
        )
    if pilot_count:
        parts.append(
            f"{pilot_count} scenario{'s' if pilot_count > 1 else ''} "
            "should proceed to controlled pilot."
        )
    if no_migrate_count:
        parts.append(
            f"{no_migrate_count} scenario{'s' if no_migrate_count > 1 else ''} "
            "show unacceptable quality loss — do not migrate."
        )

    return " ".join(parts)


def _risk_notes(simulations: list[MigrationSimulationResult]) -> list[str]:
    notes = []

    low_conf = [s for s in simulations if s.confidence_score < 0.50]
    if low_conf:
        notes.append(
            f"{len(low_conf)} scenario(s) have low confidence (<50%) due to limited "
            "replay data. Expand replay sample size before acting on these recommendations."
        )

    high_loss = [s for s in simulations if s.average_quality_delta < -0.10]
    if high_loss:
        bad_models = sorted({s.target_model for s in high_loss})
        notes.append(
            f"Model(s) {', '.join(bad_models)} show >10% quality degradation on some "
            "task types. Not suitable replacements for quality-sensitive workloads."
        )

    if simulations and all(s.estimated_annual_savings <= 0 for s in simulations):
        notes.append(
            "No cost savings identified across all tested migrations. "
            "Current model selection may already be cost-optimized, or the replay "
            "sample was too small to detect savings."
        )

    return notes


def _next_actions(simulations: list[MigrationSimulationResult]) -> list[str]:
    actions = []

    migrate_sims = sorted(
        [s for s in simulations if s.recommendation == "migrate"],
        key=lambda s: -s.estimated_annual_savings,
    )
    if migrate_sims:
        top = migrate_sims[0]
        actions.append(
            f"Initiate migration of {top.source_model} → {top.target_model} workloads "
            f"(estimated ${top.estimated_annual_savings:,.2f}/yr savings)."
        )

    pilot_sims = sorted(
        [s for s in simulations if s.recommendation == "controlled_pilot"],
        key=lambda s: -s.estimated_annual_savings,
    )
    if pilot_sims:
        top = pilot_sims[0]
        actions.append(
            f"Start controlled pilot: {top.source_model} → {top.target_model} on "
            "10–20% of traffic. Monitor quality metrics for 2 weeks before full rollout."
        )

    investigate_sims = [s for s in simulations if s.recommendation == "investigate"]
    if investigate_sims:
        actions.append(
            f"Expand replay sample for {len(investigate_sims)} inconclusive scenario(s) "
            "to improve recommendation confidence."
        )

    actions.append(
        "Run Phase 3 LLM judge evaluation to validate heuristic quality scores "
        "with human-level judgement."
    )

    return actions


def build_executive_replay_report_from_simulations(
    replay_run_id: str,
    simulations: list[MigrationSimulationResult],
    current_annualized_spend: float,
    total_requests: int,
    total_results: int,
) -> ExecutiveReplayReport:
    """
    Build an executive report from pre-computed (persisted) simulation results.

    Use this in the report endpoint after /simulate has been called and
    simulation rows have been written to the database.
    """
    portfolio_savings = _portfolio_savings(simulations)

    successful_sims = [s for s in simulations if s.records_analyzed > 0]
    avg_quality_impact = (
        sum(s.average_quality_delta for s in successful_sims) / len(successful_sims)
        if successful_sims else 0.0
    )
    overall_confidence = (
        round(sum(s.confidence_score for s in successful_sims) / len(successful_sims) * 100)
        if successful_sims else 0
    )

    recommended = [s for s in simulations if s.recommendation in ("migrate", "controlled_pilot")]
    do_not_migrate = [s for s in simulations if s.recommendation == "no_migration"]

    return ExecutiveReplayReport(
        replay_run_id=replay_run_id,
        generated_at=datetime.now(timezone.utc),
        total_requests=total_requests,
        total_results=total_results,
        current_annualized_spend=round(current_annualized_spend, 2),
        simulated_annualized_spend=round(max(current_annualized_spend - portfolio_savings, 0.0), 2),
        estimated_annual_savings=round(portfolio_savings, 2),
        avg_quality_impact=round(avg_quality_impact, 4),
        overall_confidence=overall_confidence,
        executive_summary=_executive_summary(
            simulations=simulations,
            current_annualized=current_annualized_spend,
            best_savings=portfolio_savings,
            overall_confidence=overall_confidence,
            total_requests=total_requests,
        ),
        recommended_migrations=[_sim_to_outcome(s) for s in recommended],
        do_not_migrate=[_sim_to_outcome(s) for s in do_not_migrate],
        risk_notes=_risk_notes(simulations),
        next_actions=_next_actions(simulations),
    )


def build_executive_replay_report(
    replay_run_id: str,
    original_records: list[dict],
    replay_results_raw: list[dict],
    date_range_days: int = 30,
) -> ExecutiveReplayReport:
    """
    Build an executive report on-demand from raw replay data.

    original_records — rows from usage_records (each has id, cost, model, task_type, …)
    replay_results_raw — rows from replay_results (decoded quality_flags)
    """
    simulations = simulate_from_replay_data(original_records, replay_results_raw, date_range_days)

    annualization_factor = 365 / max(date_range_days, 1)
    replayed_orig_ids = {str(rr["original_record_id"]) for rr in replay_results_raw}
    current_annualized = (
        sum(r["cost"] for r in original_records if str(r["id"]) in replayed_orig_ids)
        * annualization_factor
    )

    return build_executive_replay_report_from_simulations(
        replay_run_id=replay_run_id,
        simulations=simulations,
        current_annualized_spend=current_annualized,
        total_requests=len(original_records),
        total_results=len(replay_results_raw),
    )
