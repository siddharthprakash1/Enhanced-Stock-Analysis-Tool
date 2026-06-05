from datetime import date
from stock_analyzer.metrics.valuation import simple_dcf, compute_valuation


def test_dcf_growth_increases_value():
    low = simple_dcf(fcf0=100, growth=0.03, wacc=0.10, years=5, terminal_growth=0.02)
    high = simple_dcf(fcf0=100, growth=0.08, wacc=0.10, years=5, terminal_growth=0.02)
    assert high > low > 0


def test_compute_valuation_returns_grid():
    out = compute_valuation(fcf0=100, growth=0.05, wacc=0.10, as_of=date(2026, 6, 5))
    assert out["dcf_value"].value > 0
    assert isinstance(out["dcf_value"].value, float)
