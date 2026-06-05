import pandas as pd
from stock_analyzer.data.yfinance_provider import YFinanceProvider

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
