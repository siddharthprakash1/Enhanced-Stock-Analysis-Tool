from tests.fixtures.prices import linear_prices
from stock_analyzer.metrics.technical import compute_technical


def test_rsi_all_gains_is_100():
    out = compute_technical(linear_prices())
    assert round(out["rsi_14"].value, 1) == 100.0


def test_sma_keys_present():
    out = compute_technical(linear_prices())
    for k in ("sma_50", "sma_200", "macd", "signal_line", "bb_upper", "bb_lower"):
        assert k in out and out[k].value is not None
