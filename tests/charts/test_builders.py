from pathlib import Path
from tests.fixtures.prices import linear_prices
from stock_analyzer.charts.builders import build_charts


def test_build_charts_writes_images_and_facts(tmp_path):
    refs = build_charts(linear_prices(), out_dir=tmp_path)
    names = {r.name for r in refs}
    assert {"price", "rsi", "macd", "returns_dist", "drawdown"} <= names
    for r in refs:
        assert Path(r.image_path).exists()
        assert isinstance(r.fact, dict)
