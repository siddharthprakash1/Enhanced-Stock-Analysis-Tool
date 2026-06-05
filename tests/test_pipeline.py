"""
Task 8.1 – end-to-end pipeline test (fully offline / fake-injected).

All external I/O (network, LLM) is replaced with in-process fakes so the test
runs deterministically without any credentials.
"""

import pytest
from tests.fixtures.prices import linear_prices
from stock_analyzer.data.models import PriceHistory, Fundamentals
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection
from stock_analyzer.verification.models import ClaimList, Verdict
from stock_analyzer.pipeline import run_analysis


class FakeProvider:
    def get_price_history(self, s, p):
        return PriceHistory(symbol=s, period=p, bars=linear_prices())

    def get_fundamentals(self, s):
        return Fundamentals(symbol=s, pe=28.4)

    def get_news(self, s, limit=20):
        return []


def fake_factory(schema):
    class F:
        def invoke(self, msgs):
            if schema is ReportDraft:
                return ReportDraft(
                    sections=[
                        ReportSection(
                            id="technical",
                            prose="RSI ~66.8",
                            referenced_metrics=[],
                            charts=["rsi"],
                        )
                    ],
                    recommendation="buy",
                    confidence="high",
                )
            if schema is ClaimList:
                return ClaimList(claims=[])
            if schema is Verdict:
                return Verdict(claim_id="x", status="supported", rationale="ok")
            return AnalystFinding(
                summary="s",
                key_points=[],
                outlook="neutral",
                rationale="r",
                cited_metrics=[],
            )

    return F()


def test_run_analysis_writes_pdf_and_html(tmp_path):
    res = run_analysis(
        "AAPL",
        period="1y",
        out_dir=tmp_path,
        provider=FakeProvider(),
        structured_factory=fake_factory,
        benchmark=linear_prices()["Close"],
    )
    assert (tmp_path / "AAPL_report.pdf").exists()
    assert (tmp_path / "AAPL_dashboard.html").exists()


def test_run_analysis_returns_paths_and_audit(tmp_path):
    """Return dict has expected keys and paths point to real files."""
    res = run_analysis(
        "TSLA",
        period="6mo",
        out_dir=tmp_path,
        provider=FakeProvider(),
        structured_factory=fake_factory,
        benchmark=linear_prices()["Close"],
    )
    assert res["pdf"].exists()
    assert res["html"].exists()
    assert "audit" in res
