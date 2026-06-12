"""
Builds a structured summary from raw replay results.

Answers: "For each candidate model, what were the actual cost, quality,
and latency outcomes relative to the original?"
"""
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel

from app.replay.replay_models import ReplayRequest, ReplayResult


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
