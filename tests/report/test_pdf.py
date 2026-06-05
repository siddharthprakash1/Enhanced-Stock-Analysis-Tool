from stock_analyzer.report.pdf import render_pdf


def test_pdf_bytes_produced(tmp_path):
    ctx = {
        "symbol": "AAPL",
        "recommendation": "buy",
        "confidence": "high",
        "sections": [{"id": "technical", "prose": "RSI is 66.8", "charts": []}],
        "kpis": [{"label": "RSI", "value": "66.80", "unit": ""}],
        "audit": {
            "total_claims": 1,
            "supported": 1,
            "contradicted": 0,
            "unsupported": 0,
            "corrections_applied": [],
            "residual_unverified": [],
        },
    }
    out = tmp_path / "r.pdf"
    render_pdf(ctx, out)
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes()[:4] == b"%PDF"
