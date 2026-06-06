import pandas as pd
import yfinance as yf
from .models import PriceHistory, Fundamentals, NewsItem
from .base import DataUnavailableError


def _normalize_tnx(raw: float) -> float | None:
    """Raw ^TNX close -> decimal yield (e.g. 0.0455), or None if implausible/mis-scaled.

    ^TNX is usually quoted in percent (4.55); some feeds scale it x10 (45.5). We de-scale
    only a clearly x10 value (>30) and reject anything outside a credible 0.5%-10% band so
    a glitchy feed makes the caller fall back to the configured default rather than poison
    the WACC with a bogus rate.
    """
    if raw > 30:
        raw /= 10.0
    rate = raw / 100.0
    return rate if 0.005 <= rate <= 0.10 else None

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
            name=info.get("longName") or info.get("shortName"),
            free_cash_flow=info.get("freeCashflow"),
            shares_outstanding=info.get("sharesOutstanding"),
            enterprise_to_ebitda=info.get("enterpriseToEbitda"),
            ebitda=info.get("ebitda"), enterprise_value=info.get("enterpriseValue"),
            total_debt=info.get("totalDebt"), quote_type=info.get("quoteType"),
            currency=info.get("currency") or info.get("financialCurrency"),
        )

    def get_risk_free_rate(self) -> float | None:
        """Live 10-year US Treasury yield from ^TNX, as a decimal (e.g. 0.0455).

        Returns None on any failure so the caller can fall back to a configured default.
        """
        try:
            df = yf.download("^TNX", period="5d", auto_adjust=False, progress=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return _normalize_tnx(float(df["Close"].dropna().iloc[-1]))
        except Exception:
            return None

    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        raw = getattr(yf.Ticker(symbol), "news", []) or []
        items = []
        for a in raw[:limit]:
            t = a.get("title") or (a.get("content") or {}).get("title")
            if t:
                items.append(NewsItem(title=t, publisher=a.get("publisher")))
        return items
