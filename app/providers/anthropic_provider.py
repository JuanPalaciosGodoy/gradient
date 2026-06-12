from app.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def name(self) -> str:
        return "anthropic"

    def fetch_usage(self, start_date: str, end_date: str) -> list[dict]:
        # TODO: Anthropic usage API
        raise NotImplementedError

    def validate_credentials(self) -> bool:
        # TODO: verify api_key
        raise NotImplementedError
