from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator


class TaskType(str, Enum):
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    RESEARCH = "research"
    CODING = "coding"
    CUSTOMER_SUPPORT = "customer_support"
    OTHER = "other"      # unmatched; flag for manual review
    UNKNOWN = "unknown"  # kept for backward compat with stored records


class UsageRecord(BaseModel):
    prompt: str
    response: str
    timestamp: datetime
    model: str
    cost: float
    feedback: Optional[str] = None
    task_type: Optional[TaskType] = None

    @field_validator("cost")
    @classmethod
    def cost_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cost must be non-negative")
        return v

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must not be empty")
        return v.strip()


class AuditRunStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class AuditRun(BaseModel):
    id: str
    created_at: datetime
    record_count: int
    status: AuditRunStatus
    file_name: str


class TaskCostItem(BaseModel):
    task_type: str
    cost: float


class SpendSummary(BaseModel):
    total_cost: float
    annualized_cost: float
    daily_avg_spend: float
    avg_cost_per_request: float
    date_range_days: int
    model_breakdown: dict[str, float]
    provider_breakdown: dict[str, float]
    task_breakdown: dict[str, float]
    top_cost_driving_task_types: list[TaskCostItem]


class Opportunity(BaseModel):
    task_type: str
    current_model: str
    recommended_model: str
    recommended_model_group: str     # frontier | balanced | cheap | open_source
    current_spend: float             # observed period spend for this task/model group
    current_annualized_spend: float  # extrapolated to a full year
    estimated_annual_savings: float  # annual savings if recommendation is adopted
    estimated_savings_pct: float     # savings as % of current_annualized_spend
    confidence: float
    rationale: str
    recommended_action: str


class RiskNote(BaseModel):
    category: str
    description: str
    severity: str  # low | medium | high


class AuditReport(BaseModel):
    report_id: str
    audit_run_id: str
    generated_at: datetime
    executive_summary: str
    current_annual_spend: float      # convenience alias for spend_summary.annualized_cost
    spend_summary: SpendSummary
    top_opportunities: list[Opportunity]
    risk_notes: list[RiskNote]
    confidence_score: float
    potential_annual_savings: float  # sum of opportunity savings, capped at annualized spend
    savings_rate: float              # potential_annual_savings / annualized_spend; fraction 0.0–1.0
    recommended_next_actions: list[str]


class ValidationSummary(BaseModel):
    total_rows: int
    valid_rows: int
    invalid_rows: int
    error_samples: list[str]


class UploadResponse(BaseModel):
    audit_run_id: str
    record_count: int
    status: AuditRunStatus
    validation_summary: ValidationSummary


class GenerateResponse(BaseModel):
    audit_run_id: str
    report_id: str
    status: str
