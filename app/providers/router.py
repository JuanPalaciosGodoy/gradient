"""
Provider router: maps a model name to the right GenerationProvider.

When real_provider_mode=False (default), always returns FakeProvider.
When real_provider_mode=True, selects the real provider for each model
family (openai/anthropic/gemini) if the corresponding API key is set.
Falls back to FakeProvider for unknown model families or missing keys.
"""
from app.providers.base import GenerationProvider
from app.providers.fake import FakeProvider


def get_generation_provider(model: str) -> GenerationProvider:
    """Return the appropriate GenerationProvider for the given model.

    Reads settings at call time so tests can monkeypatch settings without
    re-importing the module.
    """
    from app.config import settings

    if not settings.real_provider_mode:
        return FakeProvider()

    if _is_openai_model(model) and settings.openai_api_key:
        from app.providers.openai_generation import OpenAIGenerationProvider
        return OpenAIGenerationProvider(settings.openai_api_key)

    if _is_anthropic_model(model) and settings.anthropic_api_key:
        from app.providers.anthropic_generation import AnthropicGenerationProvider
        return AnthropicGenerationProvider(settings.anthropic_api_key)

    if _is_gemini_model(model) and settings.gemini_api_key:
        from app.providers.gemini_generation import GeminiGenerationProvider
        return GeminiGenerationProvider(settings.gemini_api_key)

    return FakeProvider()


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "text-"))


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini-")
