from datetime import date
from ..data.models import Fundamentals
from .value import MetricValue


def compute_fundamental(f: Fundamentals, as_of: date) -> dict[str, MetricValue]:
    def mv(key, label, value, unit):
        return MetricValue(key=key, label=label, value=value, unit=unit, category="fundamental", as_of=as_of)

    return {
        "pe_ratio": mv("pe_ratio", "P/E Ratio", f.pe, "x"),
        "pb_ratio": mv("pb_ratio", "P/B Ratio", f.pb, "x"),
        "debt_to_equity": mv("debt_to_equity", "Debt/Equity", f.debt_to_equity, ""),
        "roe": mv("roe", "Return on Equity", f.roe, "%"),
        "eps_growth": mv("eps_growth", "EPS Growth (Q)", f.eps_growth, "%"),
        "profit_margin": mv("profit_margin", "Profit Margin", f.profit_margin, "%"),
        "revenue_growth": mv("revenue_growth", "Revenue Growth", f.revenue_growth, "%"),
        "dividend_yield": mv("dividend_yield", "Dividend Yield", f.dividend_yield, "%"),
        "market_cap": mv("market_cap", "Market Cap", f.market_cap, "$"),
    }
