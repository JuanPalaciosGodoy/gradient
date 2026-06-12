from pydantic import BaseModel, Field, field_validator


class QualityEvaluation(BaseModel):
    score: float                              # 0.0–1.0; higher = closer match to original
    method: str                               # heuristic | prometheus | llm_judge
    explanation: str                          # human-readable rationale for the score
    confidence: float                         # 0.0–1.0; evaluator's own confidence in its score
    flags: list[str] = Field(default_factory=list)  # machine-readable signal tags

    @field_validator("score", "confidence")
    @classmethod
    def must_be_in_unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return v


# ── Utilities ─────────────────────────────────────────────────────────────────

def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def average_quality(evals: list[QualityEvaluation]) -> float:
    if not evals:
        return 0.0
    return round(sum(e.score for e in evals) / len(evals), 4)


def quality_delta(baseline: float, candidate: float) -> float:
    """Positive = candidate improved over baseline; negative = degradation."""
    return round(candidate - baseline, 4)


def quality_loss_percentage(baseline: float, candidate: float) -> float:
    """Percentage of quality lost relative to baseline. Returns 0.0 if no loss."""
    if baseline == 0.0:
        return 0.0
    loss = baseline - candidate
    return round(max(loss / baseline * 100, 0.0), 2)


def summarize_quality_distribution(
    evals: list[QualityEvaluation],
    threshold: float = 0.7,
) -> dict:
    if not evals:
        return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "below_threshold": 0}
    scores = [e.score for e in evals]
    return {
        "count": len(scores),
        "avg": round(sum(scores) / len(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "below_threshold": sum(1 for s in scores if s < threshold),
    }
