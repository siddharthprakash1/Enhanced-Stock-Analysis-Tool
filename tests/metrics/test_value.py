from datetime import date
from stock_analyzer.metrics.value import MetricValue


def test_metric_value_render():
    mv = MetricValue(key="rsi_14", label="RSI (14)", value=66.8, unit="", category="technical", as_of=date(2026, 6, 5))
    assert mv.display() == "66.80"
    assert MetricValue(key="x", label="x", value=None, unit="$", category="technical", as_of=date(2026, 6, 5)).display() == "N/A"


def test_nan_and_inf_become_none():
    base = dict(key="x", label="x", unit="", category="technical", as_of=date(2026, 6, 5))
    assert MetricValue(**base, value=float("nan")).value is None
    assert MetricValue(**base, value=float("inf")).value is None
    assert MetricValue(**base, value=float("-inf")).value is None
