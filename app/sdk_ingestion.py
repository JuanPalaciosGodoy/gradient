"""Backend ingestion for Gradient SDK events."""
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from app.database import get_connection


class SDKAuditEvent(BaseModel):
    """Incoming AuditEvent from the Gradient SDK."""
    event_id: str
    timestamp: Optional[str] = None
    provider: str = "unknown"
    model: str = "unknown"
    workflow: str
    task_type: str
    status: Literal["success", "error", "timeout"] = "success"
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    error_message: Optional[str] = None
    user_id_hash: Optional[str] = None
    team: Optional[str] = None
    business_unit: Optional[str] = None
    process_name: Optional[str] = None
    tool_name: Optional[str] = None
    environment: Optional[str] = None
    risk_level: Optional[str] = None
    value_metric_name: Optional[str] = None
    value_metric_value: Optional[float] = None
    feedback_score: Optional[float] = None
    feedback_label: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None

    @field_validator("workflow", "task_type")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v


class SDKEventsRequest(BaseModel):
    events: list[SDKAuditEvent]


class SDKEventsResponse(BaseModel):
    accepted: int
    duplicate: int = 0
    rejected: int
    errors: list[str]


class FeedbackRequest(BaseModel):
    feedback_score: Optional[float] = None
    feedback_label: Optional[str] = None
    notes: Optional[str] = None


def save_sdk_event(event: SDKAuditEvent) -> bool:
    """Insert an SDK event. Returns True if inserted, False if duplicate event_id."""
    received_at = datetime.now(timezone.utc).isoformat()
    ts = event.timestamp or received_at
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO sdk_events (
                event_id, received_at, timestamp, provider, model, workflow, task_type, status,
                latency_ms, input_tokens, output_tokens, estimated_cost, error_message,
                user_id_hash, team, business_unit, process_name, tool_name, environment,
                risk_level, value_metric_name, value_metric_value,
                feedback_score, feedback_label, prompt, response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                received_at,
                ts,
                event.provider,
                event.model,
                event.workflow,
                event.task_type,
                event.status,
                event.latency_ms,
                event.input_tokens,
                event.output_tokens,
                event.estimated_cost,
                event.error_message,
                event.user_id_hash,
                event.team,
                event.business_unit,
                event.process_name,
                event.tool_name,
                event.environment,
                event.risk_level,
                event.value_metric_name,
                event.value_metric_value,
                event.feedback_score,
                event.feedback_label,
                event.prompt,
                event.response,
            ),
        )
        return cursor.rowcount > 0


def update_sdk_event_feedback(
    event_id: str,
    feedback_score: Optional[float],
    feedback_label: Optional[str],
    notes: Optional[str],
) -> bool:
    """Update feedback fields on a stored event. Returns True if the event was found."""
    with get_connection() as conn:
        cursor = conn.execute(
            """UPDATE sdk_events
               SET feedback_score = ?, feedback_label = ?, notes = ?
               WHERE event_id = ?""",
            (feedback_score, feedback_label, notes, event_id),
        )
        return cursor.rowcount > 0


def get_sdk_events(
    workflow: Optional[str] = None,
    team: Optional[str] = None,
    limit: int = 1000,
) -> list[dict]:
    with get_connection() as conn:
        clauses: list[str] = []
        params: list = []
        if workflow:
            clauses.append("workflow = ?")
            params.append(workflow)
        if team:
            clauses.append("team = ?")
            params.append(team)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM sdk_events {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
