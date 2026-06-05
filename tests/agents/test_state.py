from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection


def test_models_construct():
    f = AnalystFinding(summary="s", key_points=["a"], outlook="bullish", rationale="r", cited_metrics=["rsi_14"])
    assert f.outlook == "bullish"
    d = ReportDraft(sections=[ReportSection(id="technical", prose="p", referenced_metrics=["rsi_14"], charts=["rsi"])],
                    recommendation="buy", confidence="high")
    assert d.recommendation == "buy"
