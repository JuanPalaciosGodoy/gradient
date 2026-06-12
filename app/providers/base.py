from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class UsageStats:
    total_tokens: int
    total_cost: float
    request_count: int


class BaseProvider(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fetch_usage(self, start_date: str, end_date: str) -> list[dict]: ...

    @abstractmethod
    def validate_credentials(self) -> bool: ...
