from stock_analyzer.agents.analysts import make_analyst_node
from stock_analyzer.agents.state import AnalystFinding


class FakeStructured:
    def invoke(self, msgs):
        return AnalystFinding(summary="ok", key_points=["RSI high"],
                              outlook="bullish", rationale="r", cited_metrics=["rsi_14"])


def test_analyst_node_writes_its_key():
    node = make_analyst_node("technical", lambda schema: FakeStructured())
    out = node({"metrics_block": "RSI (14): 66.8", "symbol": "AAPL"})
    assert "technical" in out and out["technical"].outlook == "bullish"
