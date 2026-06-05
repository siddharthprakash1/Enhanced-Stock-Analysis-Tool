from datetime import datetime
import pandas as pd
from pydantic import BaseModel, ConfigDict

class PriceHistory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    symbol: str
    period: str
    currency: str = "USD"
    bars: pd.DataFrame  # columns: Open High Low Close Volume; DatetimeIndex

    @property
    def latest_close(self) -> float:
        return float(self.bars["Close"].iloc[-1])

class Fundamentals(BaseModel):
    symbol: str
    pe: float | None = None
    pb: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    eps_growth: float | None = None
    market_cap: float | None = None
    dividend_yield: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    sector: str | None = None
    industry: str | None = None
    beta_reported: float | None = None
    name: str | None = None
    free_cash_flow: float | None = None
    shares_outstanding: float | None = None

class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    published_at: datetime | None = None
    url: str | None = None
