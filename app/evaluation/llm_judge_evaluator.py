"""
LLM-as-judge evaluator stub.

Will use a judge model (e.g. GPT-4o) to score candidate responses
against originals with task-specific rubrics.
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityScore


class LLMJudgeEvaluator(BaseEvaluator):

    def evaluate(
        self,
        original: str,
        candidate: str,
        task_type: Optional[str] = None,
    ) -> QualityScore:
        raise NotImplementedError(
            "LLMJudgeEvaluator requires a live judge model. "
            "Use HeuristicEvaluator for local testing."
        )
