import pandas as pd
from stock_analyzer.data.models import PriceHistory, Fundamentals, NewsItem

def test_price_history_latest_close():
    df = pd.DataFrame(
        {"Open":[1,2], "High":[2,3], "Low":[1,1], "Close":[1.5, 2.5], "Volume":[100,200]},
        index=pd.to_datetime(["2026-01-01","2026-01-02"]),
    )
    ph = PriceHistory(symbol="AAPL", period="1y", currency="USD", bars=df)
    assert ph.latest_close == 2.5
    assert len(ph.bars) == 2

def test_fundamentals_optional_fields_default_none():
    f = Fundamentals(symbol="AAPL")
    assert f.pe is None and f.market_cap is None
