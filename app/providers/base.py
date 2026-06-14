from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ── Usage ingestion (Phase 1) ────────────────────────────────────────────────

@dataclass
class UsageStats:
    total_tokens: int
    total_cost: float
    request_count: int


class BaseProvider(ABC):
    """Fetches historical usage data from a provider's API."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fetch_usage(self, start_date: str, end_date: str) -> list[dict]: ...

    @abstractmethod
    def validate_credentials(self) -> bool: ...


# ── Completion generation (Phase 2 / Replay Engine) ──────────────────────────

@dataclass
class ProviderResponse:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    provider: str = ""
    model: str = ""
    # "observed" = measured from real API; "estimated_catalog" = computed from token counts + pricing;
    # "fake" = deterministic fake; "missing" = error path
    cost_source: str = "estimated_catalog"
    latency_source: str = "fake"
    raw_response: Optional[dict] = field(default=None)


class GenerationProvider(ABC):
    """Runs a single prompt against a model and returns a structured response."""

    @abstractmethod
    def generate(self, prompt: str, model: str) -> ProviderResponse: ...
