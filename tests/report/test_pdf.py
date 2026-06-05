from stock_analyzer.report.pdf import render_pdf


def test_pdf_bytes_produced(tmp_path):
    ctx = {
        "symbol": "AAPL", "company": "Apple Inc.", "as_of": "05 Jun 2026",
        "price": "$307.34", "recommendation": "buy", "confidence": "high",
        "targets": {"base": "$120.00", "bull": "$160.00", "bear": "$90.00",
                    "base_upside": "-61.0%", "bull_upside": "-48.0%", "bear_upside": "-70.7%"},
        "kpis": [{"label": "RSI", "display": "66.80"}, {"label": "P/E", "display": "37.30x"}],
        "sections": [{"title": "Technical Analysis",
                      "paras": ["RSI is 66.8.", "Momentum remains constructive."], "charts": []}],
        "fundamentals_table": [{"label": "P/E Ratio", "value": "37.30x"}],
        "risk_table": [{"label": "Sharpe Ratio", "value": "2.03"}],
        "valuation_table": [{"label": "DCF Enterprise Value", "value": "$1.45T"}],
        "metrics_appendix": [{"category": "Technical", "rows": [{"label": "RSI (14)", "value": "66.80"}]}],
        "news": [{"title": "Apple reaches a new 52-week high"}],
        "sensitivity": {"growths": [0.03, 0.05, 0.07], "waccs": [0.08, 0.10, 0.12],
                        "grid": [[1.1e12, 9e11, 8e11], [1.3e12, 1.0e12, 9e11], [1.6e12, 1.2e12, 1.0e12]]},
        "audit": {"total_claims": 1, "supported": 1, "contradicted": 0, "unsupported": 0,
                  "corrections_applied": [], "residual_unverified": []},
        "residual": [],
    }
    out = tmp_path / "r.pdf"
    render_pdf(ctx, out)
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes()[:4] == b"%PDF"
