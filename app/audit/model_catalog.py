from dataclasses import dataclass, field


@dataclass
class ModelEntry:
    provider: str
    model_name: str
    quality_tier: str    # frontier | balanced | cheap | open_source
    relative_cost: float  # normalized; gpt-4o = 1.0
    suggested_tasks: list[str] = field(default_factory=list)


CATALOG: dict[str, ModelEntry] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4o": ModelEntry("openai", "gpt-4o", "frontier", 1.00,
                         ["research", "coding", "complex_analysis"]),
    "gpt-4-turbo": ModelEntry("openai", "gpt-4-turbo", "frontier", 1.10,
                              ["research", "coding"]),
    "gpt-4": ModelEntry("openai", "gpt-4", "frontier", 1.20,
                        ["research", "coding"]),
    "gpt-4o-mini": ModelEntry("openai", "gpt-4o-mini", "cheap", 0.07,
                              ["summarization", "classification", "extraction", "customer_support"]),
    "gpt-3.5-turbo": ModelEntry("openai", "gpt-3.5-turbo", "cheap", 0.05,
                                ["summarization", "classification", "extraction"]),

    # ── Anthropic ───────────────────────────────────────────────────────────
    "claude-3-opus-20240229": ModelEntry("anthropic", "claude-3-opus-20240229", "frontier", 1.00,
                                         ["research", "coding", "complex_analysis"]),
    "claude-3-5-sonnet-20241022": ModelEntry("anthropic", "claude-3-5-sonnet-20241022", "balanced", 0.20,
                                              ["summarization", "research", "coding", "customer_support"]),
    "claude-3-sonnet-20240229": ModelEntry("anthropic", "claude-3-sonnet-20240229", "balanced", 0.18,
                                            ["summarization", "research", "customer_support"]),
    "claude-3-haiku-20240307": ModelEntry("anthropic", "claude-3-haiku-20240307", "cheap", 0.025,
                                           ["summarization", "classification", "extraction", "customer_support"]),

    # ── Google ──────────────────────────────────────────────────────────────
    "gemini-1.5-pro": ModelEntry("google", "gemini-1.5-pro", "balanced", 0.25,
                                  ["research", "coding", "summarization"]),
    "gemini-1.5-flash": ModelEntry("google", "gemini-1.5-flash", "cheap", 0.025,
                                    ["summarization", "classification", "extraction"]),
    "gemini-flash-1.5": ModelEntry("google", "gemini-flash-1.5", "cheap", 0.025,
                                    ["summarization", "classification", "extraction"]),

    # ── Meta (self-hosted / open-source) ────────────────────────────────────
    "llama-3-70b": ModelEntry("meta", "llama-3-70b", "open_source", 0.02,
                               ["classification", "extraction", "summarization"]),
    "llama-3-8b": ModelEntry("meta", "llama-3-8b", "open_source", 0.005,
                              ["classification", "extraction"]),

    # ── Mistral ─────────────────────────────────────────────────────────────
    "mistral-large": ModelEntry("mistral", "mistral-large", "balanced", 0.15,
                                 ["research", "coding", "summarization"]),
    "mistral-7b": ModelEntry("mistral", "mistral-7b", "open_source", 0.01,
                              ["classification", "extraction", "summarization"]),
}


def get_model_info(model: str) -> ModelEntry | None:
    return CATALOG.get(model)


def get_provider(model: str) -> str:
    entry = CATALOG.get(model)
    if entry:
        return entry.provider
    m = model.lower()
    if "gpt" in m or "o1" in m:
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "google"
    if "llama" in m:
        return "meta"
    if "mistral" in m:
        return "mistral"
    return "unknown"


def get_quality_tier(model: str) -> str:
    entry = CATALOG.get(model)
    if entry:
        return entry.quality_tier
    m = model.lower()
    if any(x in m for x in ["opus", "gpt-4-turbo", "gpt-4o", "gemini-1.5-pro"]):
        return "frontier"
    if any(x in m for x in ["sonnet", "pro", "large"]):
        return "balanced"
    if any(x in m for x in ["haiku", "mini", "flash", "3.5-turbo"]):
        return "cheap"
    return "unknown"
