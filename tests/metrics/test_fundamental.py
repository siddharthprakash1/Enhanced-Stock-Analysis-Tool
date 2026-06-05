from datetime import date
from stock_analyzer.data.models import Fundamentals
from stock_analyzer.metrics.fundamental import compute_fundamental


def test_maps_fundamentals():
    out = compute_fundamental(Fundamentals(symbol="AAPL", pe=28.4, roe=0.3), as_of=date(2026, 6, 5))
    assert out["pe_ratio"].value == 28.4
    assert out["roe"].value == 30.0 and out["roe"].unit == "%"
    assert out["pe_ratio"].value is not None and out["debt_to_equity"].value is None
