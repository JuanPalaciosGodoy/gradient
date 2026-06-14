"""
Evaluator factory: returns the right BaseEvaluator for a given mode string.

Modes:
  heuristic    — rule-based; no external calls; default for local dev and CI
  exact_match  — label-exact comparison; best for classification tasks
  task_router  — auto-selects exact_match for classification, heuristic otherwise
  llm_judge    — LLM-as-judge; requires a real provider to be configured
  prometheus   — Prometheus model endpoint (legacy Phase 3 placeholder)
"""
from app.evaluation.base import BaseEvaluator
from app.evaluation.exact_match_evaluator import ExactMatchEvaluator
from app.evaluation.heuristic_evaluator import HeuristicEvaluator
from app.evaluation.llm_judge_evaluator import LLMJudgeEvaluator
from app.evaluation.prometheus_evaluator import PrometheusEvaluator
from app.evaluation.task_router_evaluator import TaskRouterEvaluator


def get_evaluator(mode: str = "heuristic") -> BaseEvaluator:
    if mode == "heuristic":
        return HeuristicEvaluator()
    if mode == "exact_match":
        return ExactMatchEvaluator()
    if mode == "task_router":
        return TaskRouterEvaluator()
    if mode == "prometheus":
        return PrometheusEvaluator()
    if mode == "llm_judge":
        from app.config import settings
        from app.providers.router import get_generation_provider
        judge_provider = get_generation_provider(settings.llm_judge_model)
        return LLMJudgeEvaluator(provider=judge_provider, model=settings.llm_judge_model)
    raise ValueError(
        f"Unknown evaluator mode '{mode}'. "
        "Valid options: heuristic, exact_match, task_router, llm_judge, prometheus"
    )
