from app.schemas import TaskType, UsageRecord

_UNCLASSIFIED = {TaskType.OTHER.value, TaskType.UNKNOWN.value}


def compute_confidence(records: list[UsageRecord]) -> float:
    """
    Point-based confidence score (0–100 internally, returned as 0.0–1.0).
    Baseline 50. Floor 40. Cap 95.
    """
    if not records:
        return 0.0

    score = 50
    n = len(records)

    # Volume: more records → better statistical signal
    if n >= 1000:
        score += 15
    elif n >= 100:
        score += 10
    elif n < 10:
        score -= 10

    # Date range: longer history → better annualization accuracy
    timestamps = [r.timestamp for r in records]
    days = (max(timestamps) - min(timestamps)).days
    if days >= 90:
        score += 15
    elif days >= 30:
        score += 10
    elif days >= 14:
        score += 5
    else:
        score -= 5

    # Feedback coverage: signals whether output quality has been validated
    feedback_count = sum(1 for r in records if r.feedback)
    feedback_rate = feedback_count / n
    if feedback_rate >= 0.20:
        score += 10
    elif feedback_rate >= 0.10:
        score += 5

    # Cost validity: all records should have non-negative costs (enforced at ingest)
    valid_cost_count = sum(1 for r in records if r.cost >= 0)
    if valid_cost_count / n >= 0.95:
        score += 10

    # Task diversity: broader coverage enables better opportunity detection
    classified_types = {
        r.task_type for r in records
        if r.task_type and r.task_type.value not in _UNCLASSIFIED
    }
    if len(classified_types) >= 4:
        score += 10
    elif len(classified_types) >= 2:
        score += 5

    return round(min(max(score, 40), 95) / 100, 2)
