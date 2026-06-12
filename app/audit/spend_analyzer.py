from collections import defaultdict

from app.audit.model_catalog import get_provider
from app.schemas import SpendSummary, TaskCostItem, UsageRecord


def analyze_spend(records: list[UsageRecord]) -> SpendSummary:
    if not records:
        return SpendSummary(
            total_cost=0.0,
            annualized_cost=0.0,
            daily_avg_spend=0.0,
            avg_cost_per_request=0.0,
            date_range_days=0,
            model_breakdown={},
            provider_breakdown={},
            task_breakdown={},
            top_cost_driving_task_types=[],
        )

    timestamps = [r.timestamp for r in records]
    min_date = min(timestamps)
    max_date = max(timestamps)
    date_range_days = max((max_date - min_date).days, 1)

    total_cost = sum(r.cost for r in records)
    annualized_cost = total_cost * (365 / date_range_days)
    daily_avg_spend = total_cost / date_range_days

    model_breakdown: dict[str, float] = defaultdict(float)
    provider_breakdown: dict[str, float] = defaultdict(float)
    task_breakdown: dict[str, float] = defaultdict(float)

    for r in records:
        model_breakdown[r.model] += r.cost
        provider_breakdown[get_provider(r.model)] += r.cost
        task_key = r.task_type.value if r.task_type else "other"
        task_breakdown[task_key] += r.cost

    sorted_tasks = sorted(task_breakdown.items(), key=lambda x: x[1], reverse=True)
    top_tasks = [TaskCostItem(task_type=k, cost=round(v, 4)) for k, v in sorted_tasks]

    return SpendSummary(
        total_cost=round(total_cost, 4),
        annualized_cost=round(annualized_cost, 2),
        daily_avg_spend=round(daily_avg_spend, 6),
        avg_cost_per_request=round(total_cost / len(records), 6),
        date_range_days=date_range_days,
        model_breakdown={k: round(v, 4) for k, v in model_breakdown.items()},
        provider_breakdown={k: round(v, 4) for k, v in provider_breakdown.items()},
        task_breakdown={k: round(v, 4) for k, v in task_breakdown.items()},
        top_cost_driving_task_types=top_tasks,
    )
