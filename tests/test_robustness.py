"""Robustness: the pipeline degrades gracefully for ETFs/funds, thin-data
securities, and missing financials — and never fabricates a DCF."""

from tests.fixtures.prices import linear_prices
from stock_analyzer.data.models import PriceHistory, Fundamentals
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection
from stock_analyzer.verification.models import ClaimList, Verdict
from stock_analyzer.pipeline import run_analysis, _build_valuation_inputs, _resolve_risk_free
from stock_analyzer.metrics.risk import compute_beta

VAL_KW = dict(erp=0.0423, tax_rate=0.21, cod_spread=0.015, terminal_growth=0.02, years=5,
              growth_default=0.05, growth_min=0.0, growth_max=0.15)


def fake_factory(schema):
    class F:
        def invoke(self, msgs):
            if schema is ReportDraft:
                return ReportDraft(sections=[ReportSection(id="technical", prose="ok",
                                   referenced_metrics=[], charts=[])],
                                   recommendation="hold", confidence="medium")
            if schema is ClaimList:
                return ClaimList(claims=[])
            if schema is Verdict:
                return Verdict(claim_id="x", status="supported", rationale="ok")
            return AnalystFinding(summary="s", key_points=[], outlook="neutral", rationale="r", cited_metrics=[])
    return F()


class EquityProvider:
    """Operating company with full data -> DCF + comps should run."""
    def get_price_history(self, s, p):
        return PriceHistory(symbol=s, period=p, bars=linear_prices())

    def get_fundamentals(self, s):
        return Fundamentals(symbol=s, pe=28.0, pb=10.0, enterprise_to_ebitda=20.0,
                            profit_margin=0.25, revenue_growth=0.08, market_cap=3e12,
                            total_debt=1e11, free_cash_flow=2e10, shares_outstanding=1.6e10,
                            sector="Technology", quote_type="EQUITY", name=f"{s} Inc.")

    def get_news(self, s, limit=20):
        return []

    def get_risk_free_rate(self):
        return 0.045


class ETFProvider(EquityProvider):
    def get_fundamentals(self, s):
        return Fundamentals(symbol=s, quote_type="ETF", name=f"{s} ETF")


def _bench():
    return linear_prices()["Close"]


# --- risk-free resolution ---------------------------------------------------

def test_resolve_risk_free_prefers_provider():
    assert _resolve_risk_free(EquityProvider(), 0.04) == 0.045


def test_resolve_risk_free_falls_back_without_method():
    class NoRF:
        pass
    assert _resolve_risk_free(NoRF(), 0.04) == 0.04


def test_resolve_risk_free_falls_back_on_error():
    class RaisingRF:
        def get_risk_free_rate(self):
            raise RuntimeError("boom")
    assert _resolve_risk_free(RaisingRF(), 0.04) == 0.04


# --- valuation inputs gating ------------------------------------------------

def test_valuation_inputs_built_for_operating_company():
    f = EquityProvider().get_fundamentals("AAPL")
    beta = compute_beta(linear_prices()["Close"], _bench())
    vi = _build_valuation_inputs(f, beta, 0.045, **VAL_KW)
    assert vi is not None and 0 < vi.wacc.wacc < 1
    assert vi.growth_source.startswith("company")


def test_valuation_skipped_for_etf():
    f = ETFProvider().get_fundamentals("SPY")
    assert _build_valuation_inputs(f, 1.0, 0.045, **VAL_KW) is None


def test_valuation_skipped_without_fcf():
    f = Fundamentals(symbol="X", quote_type="EQUITY", shares_outstanding=1e9)  # no FCF
    assert _build_valuation_inputs(f, 1.0, 0.045, **VAL_KW) is None


# --- end-to-end graceful degradation ----------------------------------------

def test_pipeline_full_equity(tmp_path):
    res = run_analysis("AAPL", "1y", tmp_path, EquityProvider(), fake_factory, _bench())
    assert res["pdf"].exists() and res["html"].exists()


def test_pipeline_etf_no_crash(tmp_path):
    res = run_analysis("SPY", "1y", tmp_path, ETFProvider(), fake_factory, _bench())
    assert res["pdf"].exists() and res["html"].exists()


def test_pipeline_thin_data_no_crash(tmp_path):
    class ThinProvider(EquityProvider):
        def get_price_history(self, s, p):
            return PriceHistory(symbol=s, period=p, bars=linear_prices(n=20))
    res = run_analysis("NEW", "1mo", tmp_path, ThinProvider(), fake_factory, linear_prices(n=20)["Close"])
    assert res["pdf"].exists() and res["html"].exists()
