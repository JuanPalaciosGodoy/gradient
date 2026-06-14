"""
Google Gemini generation provider for replay engine real-call mode.

Uses the Gemini REST API via httpx (already a project dependency).
No extra package install needed.
"""
import time

import httpx

from app.providers.base import GenerationProvider, ProviderResponse
from app.replay.replay_models import get_candidate

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiGenerationProvider(GenerationProvider):
    """Calls the Google Gemini REST API to generate replay responses."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    def generate(self, prompt: str, model: str) -> ProviderResponse:
        url = f"{_GEMINI_API_BASE}/{model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 512},
        }
        start = time.monotonic()
        resp = httpx.post(
            url,
            params={"key": self._api_key},
            json=payload,
            timeout=30.0,
        )
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        resp.raise_for_status()
        data = resp.json()

        candidate_data = data.get("candidates", [{}])[0]
        parts = candidate_data.get("content", {}).get("parts", [])
        text = parts[0].get("text", "") if parts else ""

        usage_meta = data.get("usageMetadata", {})
        input_tokens = usage_meta.get("promptTokenCount") or max(len(prompt) // 4, 1)
        output_tokens = usage_meta.get("candidatesTokenCount") or 50
        cost_source = "observed" if usage_meta else "estimated_catalog"

        cost = _estimate_cost(model, input_tokens, output_tokens)

        return ProviderResponse(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(cost, 8),
            provider="google",
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
