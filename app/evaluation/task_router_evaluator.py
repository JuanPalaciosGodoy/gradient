"""
Task-router evaluator: selects the best evaluator for each task type.

Routes:
  classification  → ExactMatchEvaluator
  all others      → HeuristicEvaluator
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.exact_match_evaluator import ExactMatchEvaluator
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.evaluation.quality_score import QualityEvaluation

_CLASSIFICATION_TYPES = {"classification"}


class TaskRouterEvaluator(BaseEvaluator):
    """Routes each evaluation to the most appropriate evaluator for the task type."""

    def __init__(self) -> None:
        self._heuristic = HeuristicEvaluator()
        self._exact_match = ExactMatchEvaluator()

    def evaluate(
        self,
        prompt: str,
        original_response: str,
        candidate_response: str,
        task_type: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> QualityEvaluation:
        if task_type and task_type.lower() in _CLASSIFICATION_TYPES:
            return self._exact_match.evaluate(
                prompt, original_response, candidate_response, task_type, feedback
            )
        return self._heuristic.evaluate(
            prompt, original_response, candidate_response, task_type, feedback
        )
