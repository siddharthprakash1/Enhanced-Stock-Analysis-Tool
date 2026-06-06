import pandas as pd
import pytest
from stock_analyzer.data.yfinance_provider import YFinanceProvider, _normalize_tnx


@pytest.mark.parametrize("raw, expected", [
    (4.55, 0.0455),    # percent-quoted (Yahoo)
    (45.5, 0.0455),    # x10-scaled feed
    (0.55, 0.0055),    # low but credible
    (8.0, 0.08),       # high but credible
    (19.0, None),      # glitchy x10-of-~1.9% -> rejected, not a 19% rate
    (20.0, None),      # ambiguous band -> rejected
    (12.0, None),      # implausible 10Y yield -> rejected
    (0.1, None),       # too low -> rejected
])
def test_normalize_tnx(raw, expected):
    out = _normalize_tnx(raw)
    if expected is None:
        assert out is None
    else:
        assert out == pytest.approx(expected)

def test_get_price_history(mocker):
    df = pd.DataFrame(
        {"Open":[1.0], "High":[2.0], "Low":[0.5], "Close":[1.5], "Volume":[100]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    mocker.patch("stock_analyzer.data.yfinance_provider.yf.download", return_value=df)
    ph = YFinanceProvider().get_price_history("AAPL", "1y")
    assert ph.symbol == "AAPL" and ph.latest_close == 1.5

def test_get_fundamentals_maps_info(mocker):
    fake = mocker.Mock()
    fake.info = {"trailingPE": 28.4, "priceToBook": 5.0, "returnOnEquity": 0.3, "marketCap": 3e12}
    mocker.patch("stock_analyzer.data.yfinance_provider.yf.Ticker", return_value=fake)
    f = YFinanceProvider().get_fundamentals("AAPL")
    assert f.pe == 28.4 and f.market_cap == 3e12
