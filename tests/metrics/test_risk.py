from tests.fixtures.prices import linear_prices
from stock_analyzer.metrics.risk import compute_risk


def test_risk_keys_and_drawdown_nonpositive():
    bench = linear_prices()["Close"]
    out = compute_risk(linear_prices(), bench)
    for k in ("beta", "hist_vol", "atr", "max_drawdown", "sharpe", "var_95", "downside_dev"):
        assert k in out
    assert out["max_drawdown"].value <= 0
