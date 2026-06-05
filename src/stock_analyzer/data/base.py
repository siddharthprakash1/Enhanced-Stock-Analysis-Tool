from typing import Protocol, runtime_checkable
from .models import PriceHistory, Fundamentals, NewsItem

class DataUnavailableError(Exception):
    def __init__(self, symbol: str, field: str):
        super().__init__(f"Data unavailable for {symbol!r}: {field}")
        self.symbol, self.field = symbol, field

@runtime_checkable
class DataProvider(Protocol):
    def get_price_history(self, symbol: str, period: str) -> PriceHistory: ...
    def get_fundamentals(self, symbol: str) -> Fundamentals: ...
    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]: ...
