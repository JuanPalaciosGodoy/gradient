from abc import ABC, abstractmethod
from typing import Optional

from app.evaluation.quality_score import QualityEvaluation


class BaseEvaluator(ABC):
    """
    Evaluates whether a candidate model response is a quality equivalent
    of the original. Takes the original prompt as context.

    Returns a QualityEvaluation with score, explanation, confidence, and flags.
    """

    @abstractmethod
    def evaluate(
        self,
        prompt: str,
        original_response: str,
        candidate_response: str,
        task_type: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> QualityEvaluation: ...
