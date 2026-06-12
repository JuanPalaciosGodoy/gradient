from abc import ABC, abstractmethod
from typing import Optional

from app.evaluation.quality_score import QualityScore


class BaseEvaluator(ABC):
    """
    Compares an original model response with a candidate response and
    returns a quality score representing how well the candidate matches.
    """

    @abstractmethod
    def evaluate(
        self,
        original: str,
        candidate: str,
        task_type: Optional[str] = None,
    ) -> QualityScore: ...
