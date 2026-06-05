import pandas as pd
import yfinance as yf
from .models import PriceHistory, Fundamentals, NewsItem
from .base import DataUnavailableError

class YFinanceProvider:
    def get_price_history(self, symbol: str, period: str) -> PriceHistory:
        df = yf.download(symbol, period=period, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise DataUnavailableError(symbol, "price_history")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return PriceHistory(symbol=symbol, period=period, bars=df[["Open","High","Low","Close","Volume"]])

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        info = yf.Ticker(symbol).info or {}
        return Fundamentals(
            symbol=symbol, pe=info.get("trailingPE"), pb=info.get("priceToBook"),
            debt_to_equity=info.get("debtToEquity"), roe=info.get("returnOnEquity"),
            eps_growth=info.get("earningsQuarterlyGrowth"), market_cap=info.get("marketCap"),
            dividend_yield=info.get("dividendYield"), profit_margin=info.get("profitMargins"),
            revenue_growth=info.get("revenueGrowth"), sector=info.get("sector"),
            industry=info.get("industry"), beta_reported=info.get("beta"),
        )

    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        raw = getattr(yf.Ticker(symbol), "news", []) or []
        items = []
        for a in raw[:limit]:
            t = a.get("title") or (a.get("content") or {}).get("title")
            if t:
                items.append(NewsItem(title=t, publisher=a.get("publisher")))
        return items
