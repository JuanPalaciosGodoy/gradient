from collections import defaultdict
from dataclasses import dataclass

from app.audit.model_catalog import get_quality_tier
from app.schemas import EvidenceLevel, Opportunity, TaskType, UsageRecord, ValidationStatus
from app.utils.evidence import adjust_confidence_for_evidence

# Lower index = more expensive tier
TIER_ORDER: dict[str, int] = {
    "frontier": 0,
    "balanced": 1,
    "cheap": 2,
    "open_source": 3,
}


@dataclass
class _TaskRule:
    target_tier: str
    savings_rate: float   # conservative estimate of cost reduction
    confidence: float
    rationale: str


TASK_RULES: dict[str, _TaskRule] = {
    "summarization": _TaskRule(
        target_tier="balanced",
        savings_rate=0.45,
        confidence=0.88,
        rationale=(
            "Summarization tasks tolerate small quality differences and rarely require frontier "
            "model capabilities. Balanced models achieve comparable results at significantly lower cost."
        ),
    ),
    "classification": _TaskRule(
        target_tier="cheap",
        savings_rate=0.70,
        confidence=0.85,
        rationale=(
            "Classification is a well-bounded task. Smaller, cheaper models match frontier "
            "accuracy on most classification benchmarks."
        ),
    ),
    "extraction": _TaskRule(
        target_tier="cheap",
        savings_rate=0.60,
        confidence=0.82,
        rationale=(
            "Structured extraction generalizes well to smaller models. "
            "Schema-constrained outputs reduce quality variance."
        ),
    ),
    "customer_support": _TaskRule(
        target_tier="balanced",
        savings_rate=0.40,
        confidence=0.75,
        rationale=(
            "Customer support benefits from human-in-the-loop escalation. "
            "Balanced models handle routine queries; frontier models can be reserved for escalations."
        ),
    ),
    # research, coding, other → no automatic downgrade recommendation
}

# Best specific model to recommend for each (current_model, target_tier) pair.
_BALANCED_ALT: dict[str, str] = {
    "gpt-4o": "claude-3-5-sonnet-20241022",
    "gpt-4-turbo": "claude-3-5-sonnet-20241022",
    "gpt-4": "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229": "claude-3-5-sonnet-20241022",
    "gemini-1.5-pro": "gemini-1.5-flash",  # no dedicated balanced Gemini yet
}

_CHEAP_ALT: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4o-mini",
    "gpt-4": "gpt-4o-mini",
    "claude-3-opus-20240229": "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022": "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229": "claude-3-haiku-20240307",
    "gemini-1.5-pro": "gemini-1.5-flash",
}

_ALT_BY_TIER: dict[str, dict[str, str]] = {
    "balanced": _BALANCED_ALT,
    "cheap": _CHEAP_ALT,
}


def find_opportunities(records: list[UsageRecord], date_range_days: int) -> list[Opportunity]:
    by_task_model: dict[tuple[str, str], list[UsageRecord]] = defaultdict(list)
    for r in records:
        task = r.task_type.value if r.task_type else "other"
        by_task_model[(task, r.model)].append(r)

    denominator = max(date_range_days, 1)
    opportunities = []

    for (task, model), group in by_task_model.items():
        rule = TASK_RULES.get(task)
        if rule is None:
            continue  # research, coding, other → no automatic recommendation

        current_tier = get_quality_tier(model)
        target_tier = rule.target_tier

        # Only suggest if the current model is in a more expensive tier than the target
        if TIER_ORDER.get(current_tier, 99) >= TIER_ORDER.get(target_tier, 99):
            continue

        alt = _ALT_BY_TIER.get(target_tier, {}).get(model)
        if not alt:
            continue

        current_spend = round(sum(r.cost for r in group), 4)
        current_annualized_spend = round(current_spend * (365 / denominator), 2)
        estimated_annual_savings = round(current_annualized_spend * rule.savings_rate, 2)
        task_label = task.replace("_", " ")

        opportunities.append(
            Opportunity(
                task_type=task,
                current_model=model,
                recommended_model=alt,
                recommended_model_group=target_tier,
                current_spend=current_spend,
                current_annualized_spend=current_annualized_spend,
                estimated_annual_savings=estimated_annual_savings,
                estimated_savings_pct=round(rule.savings_rate * 100, 1),
                confidence=rule.confidence,
                rationale=rule.rationale,
                recommended_action=(
                    f"Move {task_label} workloads from {model} to a {target_tier} model such as {alt}."
                ),
                evidence_level=EvidenceLevel.HEURISTIC,
                evidence_summary=(
                    f"Heuristic estimate based on model tier analysis. {len(group)} historical "
                    f"{task_label} request(s) on {model}. No replay data available."
                ),
                limitations=[
                    "Savings rate is a conservative heuristic estimate, not measured from real outputs.",
                    "Quality impact estimated from tier difference; individual results may vary.",
                    "Validate with replay analysis before making production changes.",
                ],
                validation_status=ValidationStatus.NOT_VALIDATED,
                confidence_score=adjust_confidence_for_evidence(
                    rule.confidence, EvidenceLevel.HEURISTIC, ValidationStatus.NOT_VALIDATED
                ),
            )
        )

    return sorted(opportunities, key=lambda o: o.estimated_annual_savings, reverse=True)
