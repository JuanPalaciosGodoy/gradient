from app.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def name(self) -> str:
        return "gemini"

    def fetch_usage(self, start_date: str, end_date: str) -> list[dict]:
        # TODO: Google AI usage API
        raise NotImplementedError

    def validate_credentials(self) -> bool:
        # TODO: verify api_key
        raise NotImplementedError
