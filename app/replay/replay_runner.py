"""
Replay runner: runs historical prompts against candidate models.

Design principles:
- A single failing candidate never aborts the full replay.
- The runner is stateless; callers decide what to persist.
- Works with any GenerationProvider (fake or real).
"""
import uuid
from datetime import datetime
from typing import Optional

from app.evaluation.base import BaseEvaluator
from app.providers.base import GenerationProvider
from app.replay.replay_models import ReplayCandidate, ReplayRequest, ReplayResult
from app.schemas import UsageRecord


def build_replay_requests(records: list[UsageRecord]) -> list[ReplayRequest]:
    """Convert in-memory UsageRecord objects into ReplayRequest objects.

    Assigns synthetic UUIDs as original_record_id — suitable for in-memory
    analysis but the IDs cannot be traced back to database rows. Use
    build_replay_requests_from_rows when DB traceability is required.
    """
    requests = []
    for record in records:
        requests.append(ReplayRequest(
            original_record_id=str(uuid.uuid4()),
            prompt=record.prompt,
            original_response=record.response,
            original_model=record.model,
            original_cost=record.cost,
            task_type=record.task_type.value if record.task_type else None,
            feedback=record.feedback,
            timestamp=record.timestamp,
        ))
    return requests


def build_replay_requests_from_rows(rows: list[dict]) -> list[ReplayRequest]:
    """Convert usage_records DB rows into ReplayRequest objects.

    Uses the DB row `id` as original_record_id so replay results can be
    traced back to specific rows in the database.
    """
    requests = []
    for row in rows:
        requests.append(ReplayRequest(
            original_record_id=str(row["id"]),
            prompt=row["prompt"],
            original_response=row["response"],
            original_model=row["model"],
            original_cost=row["cost"],
            task_type=row.get("task_type"),
            feedback=row.get("feedback"),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        ))
    return requests


class ReplayRunner:

    def __init__(self, provider: GenerationProvider, evaluator: BaseEvaluator):
        self.provider = provider
        self.evaluator = evaluator

    def run(
        self,
        requests: list[ReplayRequest],
        candidates: list[ReplayCandidate],
    ) -> list[ReplayResult]:
        """
        Run every request against every enabled candidate.
        Returns one ReplayResult per (request, candidate) pair.
        Errors are captured in result.error_message; they do not raise.
        """
        results = []
        for req in requests:
            for candidate in candidates:
                if not candidate.enabled:
                    continue
                if candidate.model == req.original_model:
                    continue  # never compare a model against itself
                results.append(self._run_single(req, candidate))
        return results

    def _run_single(self, req: ReplayRequest, candidate: ReplayCandidate) -> ReplayResult:
        replay_id = str(uuid.uuid4())
        try:
            response = self.provider.generate(req.prompt, candidate.model)
            quality = self.evaluator.evaluate(
                prompt=req.prompt,
                original_response=req.original_response,
                candidate_response=response.text,
                task_type=req.task_type,
                feedback=req.feedback,
            )
            return ReplayResult(
                replay_id=replay_id,
                original_record_id=req.original_record_id,
                candidate_provider=candidate.provider,
                candidate_model=candidate.model,
                candidate_response=response.text,
                estimated_cost=response.estimated_cost,
                latency_ms=response.latency_ms,
                quality_score=quality.score,
                quality_method=quality.method,
                quality_explanation=quality.explanation,
                quality_confidence=quality.confidence,
                quality_flags=quality.flags,
                error_message=None,
            )
        except Exception as exc:
            return ReplayResult(
                replay_id=replay_id,
                original_record_id=req.original_record_id,
                candidate_provider=candidate.provider,
                candidate_model=candidate.model,
                candidate_response="",
                estimated_cost=0.0,
                latency_ms=0.0,
                quality_score=0.0,
                quality_method="error",
                quality_explanation=f"Evaluation failed: {exc}",
                quality_confidence=0.0,
                quality_flags=["evaluation_error"],
                error_message=str(exc),
            )
