"""AuditEvent: the core data structure for Gradient SDK audit logging."""
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AuditEvent(BaseModel):
    # ── Required identity ─────────────────────────────────────────────────────
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider: str
    model: str
    workflow: str
    task_type: str
    status: Literal["success", "error", "timeout"] = "success"

    @field_validator("provider", "model", "workflow", "task_type")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v

    # ── Optional performance ──────────────────────────────────────────────────
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    error_message: Optional[str] = None

    # ── Optional business context ─────────────────────────────────────────────
    user_id_hash: Optional[str] = None
    team: Optional[str] = None
    business_unit: Optional[str] = None
    process_name: Optional[str] = None
    tool_name: Optional[str] = None
    environment: Optional[str] = None
    risk_level: Optional[Literal["low", "medium", "high", "critical"]] = None
    value_metric_name: Optional[str] = None
    value_metric_value: Optional[float] = None
    feedback_score: Optional[float] = None
    feedback_label: Optional[str] = None

    # ── Optional sensitive (opt-in only) ──────────────────────────────────────
    prompt: Optional[str] = None
    response: Optional[str] = None


class FeedbackUpdate(BaseModel):
    event_id: str
    feedback_score: Optional[float] = None
    feedback_label: Optional[str] = None
    notes: Optional[str] = None
