"""
Centralized confidence adjustment for evidence quality.

Higher evidence level + stronger validation status → higher adjusted confidence.
Result is always bounded [0.0, 0.95].
"""
from app.schemas import EvidenceLevel, ValidationStatus

_LEVEL_FACTOR: dict[EvidenceLevel, float] = {
    EvidenceLevel.HEURISTIC: 0.65,
    EvidenceLevel.ESTIMATED: 0.78,
    EvidenceLevel.OBSERVED_REPLAY: 0.90,
    EvidenceLevel.LLM_JUDGE: 0.95,
    EvidenceLevel.HUMAN_REVIEWED: 0.98,
    EvidenceLevel.PRODUCTION_VALIDATED: 1.00,
}

_STATUS_FACTOR: dict[ValidationStatus, float] = {
    ValidationStatus.NOT_VALIDATED: 0.80,
    ValidationStatus.REPLAYED: 0.90,
    ValidationStatus.EVALUATOR_SCORED: 1.00,
    ValidationStatus.HUMAN_REVIEWED: 1.05,
    ValidationStatus.READY_FOR_PILOT: 1.08,
    ValidationStatus.PRODUCTION_VALIDATED: 1.10,
}


def adjust_confidence_for_evidence(
    base_score: float,
    evidence_level: EvidenceLevel,
    validation_status: ValidationStatus,
) -> float:
    """Return `base_score` adjusted downward for weaker evidence, upward for stronger.

    The same base score will always satisfy:
      HEURISTIC/not_validated < ESTIMATED/not_validated
        < OBSERVED_REPLAY/evaluator_scored
        < HUMAN_REVIEWED/human_reviewed
        < PRODUCTION_VALIDATED/production_validated
    """
    factor = (
        _LEVEL_FACTOR.get(evidence_level, 0.80)
        * _STATUS_FACTOR.get(validation_status, 0.90)
    )
    return round(min(max(base_score * factor, 0.0), 0.95), 4)
