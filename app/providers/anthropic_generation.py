"""
Anthropic generation provider for replay engine real-call mode.

Requires: pip install anthropic>=0.20.0
Falls back safely when not installed — raises ImportError only on first use.
"""
import time

from app.providers.base import GenerationProvider, ProviderResponse
from app.replay.replay_models import get_candidate


class AnthropicGenerationProvider(GenerationProvider):
    """Calls the Anthropic messages API to generate replay responses."""

    def __init__(self, api_key: str):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for real provider mode. "
                "Install it with: pip install anthropic>=0.20.0"
            ) from exc
        self._client = _anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, model: str) -> ProviderResponse:
        start = time.monotonic()
        response = self._client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = round((time.monotonic() - start) * 1000, 2)

        text = response.content[0].text if response.content else ""
        usage = response.usage
        input_tokens = usage.input_tokens if usage else max(len(prompt) // 4, 1)
        output_tokens = usage.output_tokens if usage else 50

        cost = _estimate_cost(model, input_tokens, output_tokens)
        cost_source = "observed" if usage else "estimated_catalog"

        return ProviderResponse(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(cost, 8),
            provider="anthropic",
            model=model,
            cost_source=cost_source,
            latency_source="observed",
        )


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    candidate = get_candidate(model)
    if candidate:
        return (
            (input_tokens / 1000) * candidate.estimated_input_cost_per_1k_tokens
            + (output_tokens / 1000) * candidate.estimated_output_cost_per_1k_tokens
        )
    return 0.0
