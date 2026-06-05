from stock_analyzer.agents.graph import build_graph
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection


def fake_factory(schema):
    class F:
        def invoke(self, msgs):
            if schema is ReportDraft:
                return ReportDraft(
                    sections=[ReportSection(id="technical", prose="p", referenced_metrics=[], charts=[])],
                    recommendation="hold",
                    confidence="medium",
                )
            return AnalystFinding(summary="s", key_points=[], outlook="neutral", rationale="r", cited_metrics=[])

    return F()


def test_graph_runs_to_report():
    app = build_graph(fake_factory, verify=False)
    out = app.invoke({"symbol": "AAPL", "metrics_block": "RSI (14): 66.8", "revisions": 0})
    assert out["report"].recommendation == "hold"
