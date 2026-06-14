"""
OpenAI generation provider for replay engine real-call mode.

Requires: pip install openai>=1.0.0
Falls back safely when not installed — raises ImportError only on first use.
"""
import time
from typing import Optional

from app.providers.base import GenerationProvider, ProviderResponse
from app.replay.replay_models import get_candidate


class OpenAIGenerationProvider(GenerationProvider):
    """Calls the OpenAI chat completions API to generate replay responses."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for real provider mode. "
                "Install it with: pip install openai>=1.0.0"
            ) from exc
        self._client = OpenAI(api_key=api_key)  # type: ignore[arg-type]

    def generate(self, prompt: str, model: str) -> ProviderResponse:
        start = time.monotonic()
        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        latency_ms = round((time.monotonic() - start) * 1000, 2)

        choice = response.choices[0]
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else max(len(prompt) // 4, 1)
        output_tokens = usage.completion_tokens if usage else 50
        text = choice.message.content or ""

        cost = _estimate_cost(model, input_tokens, output_tokens)
        cost_source = "observed" if usage else "estimated_catalog"

        return ProviderResponse(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(cost, 8),
            provider="openai",
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
