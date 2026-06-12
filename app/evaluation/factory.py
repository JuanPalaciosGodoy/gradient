"""
Evaluator factory: returns the right BaseEvaluator for a given mode string.

EVALUATOR_MODE=heuristic   — no external calls; safe for local dev and CI
EVALUATOR_MODE=prometheus  — requires Prometheus model endpoint (Phase 3)
EVALUATOR_MODE=llm_judge   — requires judge model API credentials (Phase 3)
"""
from app.evaluation.base import BaseEvaluator
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.evaluation.llm_judge_evaluator import LLMJudgeEvaluator
from app.evaluation.prometheus_evaluator import PrometheusEvaluator


def get_evaluator(mode: str = "heuristic") -> BaseEvaluator:
    if mode == "heuristic":
        return HeuristicEvaluator()
    if mode == "prometheus":
        return PrometheusEvaluator()
    if mode == "llm_judge":
        return LLMJudgeEvaluator()
    raise ValueError(
        f"Unknown evaluator mode '{mode}'. "
        "Valid options: heuristic, prometheus, llm_judge"
    )
