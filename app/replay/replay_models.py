"""
Core data models for the Phase 2 Replay Engine.

These models represent the counterfactual question:
"What would have happened if this workload had run on a different model?"
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas import EvidenceLevel, ValidationStatus


class ReplayCandidate(BaseModel):
    """A model that is a candidate for replacing the original model in a replay."""

    provider: str
    model: str
    model_group: str                         # frontier | balanced | cheap | open_source
    enabled: bool = True
    estimated_input_cost_per_1k_tokens: float
    estimated_output_cost_per_1k_tokens: float
    notes: str = ""

    @field_validator("provider", "model", "model_group")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("estimated_input_cost_per_1k_tokens", "estimated_output_cost_per_1k_tokens")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cost must be non-negative")
        return v


class ReplayRequest(BaseModel):
    """A single historical prompt prepared for replay against candidate models."""

    original_record_id: str
    prompt: str
    original_response: str
    original_model: str
    original_cost: float
    task_type: Optional[str] = None
    feedback: Optional[str] = None  # original user feedback signal, carried for evaluator context
    timestamp: datetime


class ReplayResult(BaseModel):
    """The outcome of running one ReplayRequest against one candidate model."""

    replay_id: str
    original_record_id: str
    candidate_provider: str
    candidate_model: str
    candidate_response: str
    estimated_cost: float
    latency_ms: float
    quality_score: float          # 0.0–1.0; populated from evaluator
    quality_method: str           # heuristic | prometheus | llm_judge | error
    quality_explanation: str = ""
    quality_confidence: float = 1.0
    quality_flags: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    # Cost and latency provenance
    input_tokens: int = 0
    output_tokens: int = 0
    cost_source: str = "estimated_catalog"   # observed | estimated_catalog | fake | missing
    latency_source: str = "fake"             # observed | fake | missing
    # Phase 2.5 evidence fields — conservative by default; success path sets stronger evidence
    evidence_level: EvidenceLevel = EvidenceLevel.HEURISTIC
    evidence_summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("quality_score", "quality_confidence")
    @classmethod
    def must_be_in_unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return v

    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def tokens_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("cost_source")
    @classmethod
    def cost_source_must_be_valid(cls, v: str) -> str:
        valid = {"observed", "estimated_catalog", "fake", "missing"}
        if v not in valid:
            raise ValueError(f"must be one of {sorted(valid)}")
        return v

    @field_validator("latency_source")
    @classmethod
    def latency_source_must_be_valid(cls, v: str) -> str:
        valid = {"observed", "fake", "missing"}
        if v not in valid:
            raise ValueError(f"must be one of {sorted(valid)}")
        return v


class MigrationScenario(BaseModel):
    """
    Describes a proposed model migration: move N% of tasks that match
    `task_types_included` from `source_model` to `target_model`.
    """

    scenario_name: str
    source_model: str
    target_model: str
    task_types_included: list[str]   # empty list = all task types
    migration_percentage: float      # fraction 0.0–1.0
    notes: str = ""

    @field_validator("migration_percentage")
    @classmethod
    def migration_percentage_must_be_valid(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("migration_percentage must be between 0.0 and 1.0")
        return v


class MigrationSimulationResult(BaseModel):
    """Outcome of a MigrationScenario simulation run against historical records."""

    scenario_name: str
    source_model: str
    target_model: str
    current_annualized_cost: float
    simulated_annualized_cost: float
    estimated_annual_savings: float
    estimated_savings_pct: float      # 0–100
    average_quality_delta: float      # negative = quality loss; positive = quality gain
    base_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)  # raw algorithm score
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)       # evidence-adjusted
    recommendation: str               # migrate | controlled_pilot | no_migration | hold | investigate | proceed
    rationale: str
    # Evidence-based fields (populated by simulate_from_replay_data; default 0 for catalog-based)
    avg_current_quality: float = 0.0
    avg_simulated_quality: float = 0.0
    avg_latency_delta_ms: float = 0.0   # average candidate latency (original latency not tracked)
    records_analyzed: int = 0
    failed_replays: int = 0
    # Phase 2.5 evidence fields
    evidence_level: EvidenceLevel = EvidenceLevel.HEURISTIC
    evidence_summary: str = ""
    limitations: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    evidence_coverage_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_counts: dict[str, int] = Field(default_factory=dict)


# ── Candidate catalog ─────────────────────────────────────────────────────────
# Hardcoded for Phase 2 development. Real pricing is approximate.

REPLAY_CANDIDATES: list[ReplayCandidate] = [
    ReplayCandidate(
        provider="openai", model="gpt-4o",
        model_group="frontier",
        estimated_input_cost_per_1k_tokens=0.005,
        estimated_output_cost_per_1k_tokens=0.015,
        notes="OpenAI flagship frontier model",
    ),
    ReplayCandidate(
        provider="openai", model="gpt-4o-mini",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00015,
        estimated_output_cost_per_1k_tokens=0.0006,
        notes="OpenAI low-cost model; strong for classification and extraction",
    ),
    ReplayCandidate(
        provider="anthropic", model="claude-3-5-sonnet-20241022",
        model_group="balanced",
        estimated_input_cost_per_1k_tokens=0.003,
        estimated_output_cost_per_1k_tokens=0.015,
        notes="Anthropic balanced model; strong at coding and reasoning",
    ),
    ReplayCandidate(
        provider="anthropic", model="claude-3-haiku-20240307",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.00025,
        estimated_output_cost_per_1k_tokens=0.00125,
        notes="Anthropic low-cost model; fast, good for summarization and support",
    ),
    ReplayCandidate(
        provider="google", model="gemini-1.5-pro",
        model_group="balanced",
        estimated_input_cost_per_1k_tokens=0.00125,
        estimated_output_cost_per_1k_tokens=0.005,
        notes="Google balanced model; long context, strong multimodal",
    ),
    ReplayCandidate(
        provider="google", model="gemini-1.5-flash",
        model_group="cheap",
        estimated_input_cost_per_1k_tokens=0.000075,
        estimated_output_cost_per_1k_tokens=0.0003,
        notes="Google low-cost model; very fast, suitable for high-volume tasks",
    ),
    ReplayCandidate(
        provider="meta", model="llama-3-70b",
        model_group="open_source",
        estimated_input_cost_per_1k_tokens=0.00059,
        estimated_output_cost_per_1k_tokens=0.00079,
        notes="Open-source; self-hosted or via inference API",
    ),
]


def get_candidate(model: str) -> ReplayCandidate | None:
    return next((c for c in REPLAY_CANDIDATES if c.model == model), None)
