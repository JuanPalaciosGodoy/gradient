"""Lightweight cost estimation for common models (offline catalog)."""
from typing import Optional

# (input_cost_per_1k, output_cost_per_1k) in USD
_CATALOG: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3-opus-20240229": (0.015, 0.075),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "llama-3-70b": (0.00059, 0.00079),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Return estimated USD cost or None if model is not in the catalog."""
    entry = _CATALOG.get(model)
    if entry is None:
        return None
    in_cost, out_cost = entry
    return round(
        (input_tokens / 1000) * in_cost + (output_tokens / 1000) * out_cost,
        8,
    )


def known_models() -> list[str]:
    return list(_CATALOG.keys())
