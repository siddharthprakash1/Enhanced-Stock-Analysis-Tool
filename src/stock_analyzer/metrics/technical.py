import pandas as pd
from .value import MetricValue


def _mv(key, label, value, unit, as_of):
    return MetricValue(
        key=key,
        label=label,
        value=None if value is None or pd.isna(value) else float(value),
        unit=unit,
        category="technical",
        as_of=as_of,
    )


def compute_technical(bars: pd.DataFrame) -> dict[str, MetricValue]:
    c = bars["Close"]
    as_of = bars.index[-1].date()

    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()

    return {
        "last_close": _mv("last_close", "Last Close", c.iloc[-1], "$", as_of),
        "sma_50": _mv("sma_50", "SMA 50", sma50.iloc[-1], "$", as_of),
        "sma_200": _mv("sma_200", "SMA 200", sma200.iloc[-1], "$", as_of),
        "rsi_14": _mv("rsi_14", "RSI (14)", rsi.iloc[-1], "", as_of),
        "macd": _mv("macd", "MACD", macd.iloc[-1], "", as_of),
        "signal_line": _mv("signal_line", "Signal Line", signal.iloc[-1], "", as_of),
        "bb_upper": _mv("bb_upper", "Bollinger Upper", (mid + 2 * sd).iloc[-1], "$", as_of),
        "bb_lower": _mv("bb_lower", "Bollinger Lower", (mid - 2 * sd).iloc[-1], "$", as_of),
        "wk52_high": _mv("wk52_high", "52-Week High", c.tail(252).max(), "$", as_of),
        "wk52_low": _mv("wk52_low", "52-Week Low", c.tail(252).min(), "$", as_of),
    }
