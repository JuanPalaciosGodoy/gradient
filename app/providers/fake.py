"""
Fake (deterministic) provider for local testing and development.

Returns predictable responses without making real API calls.
Uses model_catalog relative costs to estimate realistic pricing.
"""
import hashlib

from app.audit.model_catalog import CATALOG, get_quality_tier
from app.providers.base import GenerationProvider, ProviderResponse

_FAKE_RESPONSES = [
    "Based on the provided context, here is a concise and relevant response.",
    "The analysis indicates several key findings worth considering.",
    "After reviewing the input, the following summary captures the main points.",
    "This response addresses the core question with appropriate detail.",
    "The requested information has been processed and structured accordingly.",
    "Here is a structured response based on the available information.",
    "The following output reflects careful consideration of the prompt.",
    "This generated text provides a relevant answer to the query.",
]

# Base input/output cost per 1k tokens (USD), anchored to gpt-4o = 1.0
_BASE_INPUT_COST_PER_1K = 0.005
_BASE_OUTPUT_COST_PER_1K = 0.015

# Base latency (ms) by quality tier before jitter
_BASE_LATENCY_MS: dict[str, int] = {
    "frontier": 1500,
    "balanced": 900,
    "cheap": 400,
    "open_source": 650,
}


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


class FakeProvider(GenerationProvider):
    """
    Deterministic fake provider.

    - Responses are stable across runs for the same (prompt, model) pair.
    - Costs are estimated from the model catalog's relative cost factor.
    - Latency is tier-based with a deterministic jitter.
    """

    def generate(self, prompt: str, model: str) -> ProviderResponse:
        h = _stable_hash(f"{prompt}:{model}")

        # Deterministic response text
        text = f"[fake:{model}] " + _FAKE_RESPONSES[h % len(_FAKE_RESPONSES)]

        # Rough token estimates (~4 chars per token)
        input_tokens = max(len(prompt) // 4, 1)
        output_tokens = max(len(text) // 4, 1)

        # Cost from model catalog relative pricing
        entry = CATALOG.get(model)
        relative_cost = entry.relative_cost if entry else 0.5
        estimated_cost = (
            (input_tokens / 1000) * relative_cost * _BASE_INPUT_COST_PER_1K
            + (output_tokens / 1000) * relative_cost * _BASE_OUTPUT_COST_PER_1K
        )

        # Latency: tier-based base + deterministic jitter (0–499 ms)
        tier = get_quality_tier(model)
        base_latency = _BASE_LATENCY_MS.get(tier, 800)
        latency_ms = float(base_latency + (h % 500))

        return ProviderResponse(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(estimated_cost, 8),
        )
