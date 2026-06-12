from app.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def name(self) -> str:
        return "openai"

    def fetch_usage(self, start_date: str, end_date: str) -> list[dict]:
        # TODO: GET https://api.openai.com/v1/usage
        raise NotImplementedError

    def validate_credentials(self) -> bool:
        # TODO: lightweight call to verify api_key
        raise NotImplementedError
