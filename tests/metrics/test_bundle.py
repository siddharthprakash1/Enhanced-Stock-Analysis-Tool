from tests.fixtures.prices import linear_prices
from stock_analyzer.data.models import Fundamentals, PriceHistory
from stock_analyzer.metrics.bundle import MetricsBundle


def test_bundle_flat_and_prompt_block():
    ph = PriceHistory(symbol="AAPL", period="1y", bars=linear_prices())
    b = MetricsBundle.from_data(
        ph,
        Fundamentals(symbol="AAPL", pe=28.4),
        linear_prices()["Close"],
        news=[],
        fcf0=100,
        growth=0.05,
        wacc=0.10,
    )
    flat = b.as_flat()
    assert "rsi_14" in flat and "pe_ratio" in flat and "sharpe" in flat
    pb = b.prompt_block()
    assert "RSI (14)" in pb and "P/E Ratio" in pb
    assert b.prompt_block() == pb
