"""
Prometheus evaluator stub.

Will use the Prometheus evaluation framework (reference-guided scoring)
when real provider credentials are available.
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityScore


class PrometheusEvaluator(BaseEvaluator):

    def evaluate(
        self,
        original: str,
        candidate: str,
        task_type: Optional[str] = None,
    ) -> QualityScore:
        raise NotImplementedError(
            "PrometheusEvaluator requires a live model endpoint. "
            "Use HeuristicEvaluator for local testing."
        )
