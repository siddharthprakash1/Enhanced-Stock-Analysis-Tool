from stock_analyzer.agents.writer import make_writer_node
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection


class FakeStructured:
    def invoke(self, msgs):
        return ReportDraft(
            sections=[ReportSection(id="technical", prose="p", referenced_metrics=["rsi_14"], charts=["rsi"])],
            recommendation="buy",
            confidence="high",
        )


def test_writer_emits_report():
    node = make_writer_node(lambda schema: FakeStructured())
    f = AnalystFinding(summary="s", key_points=[], outlook="bullish", rationale="r", cited_metrics=[])
    out = node({"metrics_block": "...", "symbol": "AAPL", "fundamental": f, "technical": f, "risk": f, "valuation": f})
    assert isinstance(out["report"], ReportDraft) and out["report"].recommendation == "buy"
