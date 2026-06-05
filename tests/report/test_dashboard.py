from tests.fixtures.prices import linear_prices
from stock_analyzer.report.html_dashboard import render_dashboard


def test_dashboard_self_contained(tmp_path):
    out = tmp_path / "dash.html"
    render_dashboard(
        symbol="AAPL",
        bars=linear_prices(),
        kpis=[{"label": "RSI", "value": "66.80", "unit": ""}],
        verdict_rows=[{"text": "RSI is 66.8", "status": "supported"}],
        out_path=out,
    )
    html = out.read_text()
    assert out.exists() and "plotly" in html.lower()
    assert "AAPL" in html and "sidebar" in html.lower()
