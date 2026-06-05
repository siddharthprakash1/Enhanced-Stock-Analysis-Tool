from datetime import date
from stock_analyzer.agents.llm import grounding_block
from stock_analyzer.metrics.bundle import MetricsBundle
from stock_analyzer.metrics.value import MetricValue


def test_grounding_block_includes_metrics():
    b = MetricsBundle(symbol="AAPL", period="1y",
        metrics={"rsi_14": MetricValue(key="rsi_14", label="RSI (14)", value=66.8, unit="", category="technical", as_of=date(2026, 6, 5))})
    block = grounding_block(b)
    assert "RSI (14)" in block and "only use" in block.lower()
