from datetime import date

from tests.fixtures.prices import linear_prices
from stock_analyzer.data.models import Fundamentals, PriceHistory
from stock_analyzer.metrics.bundle import MetricsBundle
from stock_analyzer.metrics.valuation import ValuationInputs
from stock_analyzer.metrics.wacc import compute_wacc


def test_bundle_flat_and_prompt_block_without_valuation():
    ph = PriceHistory(symbol="AAPL", period="1y", bars=linear_prices())
    b = MetricsBundle.from_data(
        ph,
        Fundamentals(symbol="AAPL", pe=28.4),
        linear_prices()["Close"],
        news=[],
    )
    flat = b.as_flat()
    assert "rsi_14" in flat and "pe_ratio" in flat and "sharpe" in flat
    assert "dcf_value" not in flat  # no valuation inputs -> no fabricated DCF
    pb = b.prompt_block()
    assert "RSI (14)" in pb and "P/E Ratio" in pb
    assert b.prompt_block() == pb


def test_bundle_includes_valuation_when_inputs_given():
    ph = PriceHistory(symbol="AAPL", period="1y", bars=linear_prices())
    wc = compute_wacc(risk_free=0.0455, beta=1.0, erp=0.0423, market_cap=1e12,
                      total_debt=1e11, cost_of_debt_spread=0.015, tax_rate=0.21)
    vi = ValuationInputs(fcf0=1e10, growth=0.05, growth_source="company revenue growth",
                         wacc=wc, terminal_growth=0.02, years=5, shares_outstanding=1e9)
    b = MetricsBundle.from_data(
        ph, Fundamentals(symbol="AAPL", pe=28.4), linear_prices()["Close"], news=[], valuation=vi,
    )
    flat = b.as_flat()
    assert flat["dcf_value"].value > 0
    assert "wacc" in flat and flat["wacc"].unit == "%"
    assert "cost_of_equity" in flat
