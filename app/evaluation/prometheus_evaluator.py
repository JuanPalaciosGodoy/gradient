"""
Prometheus evaluator stub.

Prometheus is a reference-guided LLM evaluator that scores outputs on a
rubric defined alongside the reference (original) response. It requires a
running Prometheus model endpoint.

Expected input when implemented:
    - instruction (prompt)
    - reference_answer (original_response)
    - response (candidate_response)
    - score_rubric: task-specific rubric string
    Expected output parsing: extract integer score 1–5 from model output,
    normalize to 0.0–1.0 range.

TODO (Phase 3):
    - Wire up Prometheus API endpoint via settings.prometheus_endpoint
    - Build task-specific rubrics per TaskType
    - Parse score from structured Prometheus output format
    - Fall back to HeuristicEvaluator when endpoint is unreachable
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityEvaluation


class PrometheusEvaluator(BaseEvaluator):

    def evaluate(
        self,
        prompt: str,
        original_response: str,
        candidate_response: str,
        task_type: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> QualityEvaluation:
        raise NotImplementedError(
            "PrometheusEvaluator requires a live model endpoint. "
            "Set EVALUATOR_MODE=heuristic for local testing."
        )
