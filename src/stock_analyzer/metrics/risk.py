import numpy as np
import pandas as pd
from .value import MetricValue


def _pct(x):
    return x * 100 if (x is not None and x == x) else x  # x==x is False for NaN


def compute_beta(close: pd.Series, benchmark_close: pd.Series) -> float | None:
    """Beta of the security vs. the benchmark over the aligned return window."""
    ret = close.pct_change().dropna()
    bret = benchmark_close.pct_change().reindex(ret.index).dropna()
    aligned = ret.reindex(bret.index)
    if not bret.var():
        return None
    b = aligned.cov(bret) / bret.var()
    return None if b is None or pd.isna(b) else float(b)


def compute_risk(bars: pd.DataFrame, benchmark_close: pd.Series, rf: float = 0.0) -> dict[str, MetricValue]:
    as_of = bars.index[-1].date()
    c = bars["Close"]
    ret = c.pct_change().dropna()

    def mv(key, label, value, unit):
        return MetricValue(
            key=key,
            label=label,
            value=None if value is None or pd.isna(value) else float(value),
            unit=unit,
            category="risk",
            as_of=as_of,
        )

    beta = compute_beta(c, benchmark_close)

    hv = ret.std() * np.sqrt(252)

    tr = pd.concat(
        [bars["High"] - bars["Low"], (bars["High"] - c.shift()).abs(), (bars["Low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    cum = (1 + ret).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    sharpe = ((ret.mean() - rf / 252) / ret.std() * np.sqrt(252)) if ret.std() else None

    var95 = -np.percentile(ret, 5)

    downside = ret[ret < 0].std() * np.sqrt(252)

    return {
        "beta": mv("beta", "Beta (vs SPY)", beta, ""),
        "hist_vol": mv("hist_vol", "Annualized Volatility", _pct(hv), "%"),
        "atr": mv("atr", "Average True Range", atr, "$"),
        "max_drawdown": mv("max_drawdown", "Max Drawdown", _pct(max_dd), "%"),
        "sharpe": mv("sharpe", "Sharpe Ratio", sharpe, ""),
        "var_95": mv("var_95", "Value at Risk (95%)", _pct(var95), "%"),
        "downside_dev": mv("downside_dev", "Downside Deviation", _pct(downside), "%"),
    }
