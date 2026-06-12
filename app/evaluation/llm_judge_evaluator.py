"""
LLM-as-judge evaluator stub.

Uses a judge model (e.g. GPT-4o, Claude Sonnet) to score candidate
responses against the original using a structured prompt rubric.

Expected judge prompt format when implemented:
    System: "You are an expert evaluator. Score the candidate response
             on a scale from 0.0 to 1.0 relative to the reference response."
    User:   "Task type: {task_type}
             Original prompt: {prompt}
             Reference response: {original_response}
             Candidate response: {candidate_response}
             Score (JSON): {"score": float, "explanation": str, "flags": [str]}"

    Expected output parsing: parse JSON from judge model output;
    extract score, explanation, flags.

TODO (Phase 3):
    - Wire up judge model via GenerationProvider interface
    - Build per-task-type rubric templates
    - Add retry + timeout handling
    - Cache identical (prompt, original, candidate) evaluations
    - Fall back to HeuristicEvaluator on parse error or timeout
"""
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.evaluation.quality_score import QualityEvaluation


class LLMJudgeEvaluator(BaseEvaluator):

    def evaluate(
        self,
        prompt: str,
        original_response: str,
        candidate_response: str,
        task_type: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> QualityEvaluation:
        raise NotImplementedError(
            "LLMJudgeEvaluator requires a live judge model. "
            "Set EVALUATOR_MODE=heuristic for local testing."
        )
