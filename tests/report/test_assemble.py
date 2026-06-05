from datetime import date

from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.agents.state import ReportDraft, ReportSection
from stock_analyzer.report.assemble import build_context


def test_context_has_sections_kpis_audit():
    draft = ReportDraft(
        sections=[
            ReportSection(
                id="technical",
                prose="p",
                referenced_metrics=["rsi_14"],
                charts=["rsi"],
            )
        ],
        recommendation="buy",
        confidence="high",
    )
    gt = {
        "rsi_14": MetricValue(
            key="rsi_14",
            label="RSI",
            value=66.8,
            unit="",
            category="technical",
            as_of=date(2026, 6, 5),
        )
    }
    ctx = build_context(
        symbol="AAPL",
        draft=draft,
        ground_truth=gt,
        audit={
            "total_claims": 1,
            "supported": 1,
            "contradicted": 0,
            "unsupported": 0,
            "corrections_applied": [],
            "residual_unverified": [],
        },
        charts={"rsi": "out/rsi.png"},
    )
    assert ctx["symbol"] == "AAPL" and ctx["recommendation"] == "buy"
    assert ctx["kpis"][0]["label"] == "RSI" and ctx["audit"]["supported"] == 1
