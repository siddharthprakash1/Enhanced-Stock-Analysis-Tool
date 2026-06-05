from stock_analyzer.verification.extract import make_extract_node
from stock_analyzer.verification.models import ClaimList, Claim
from stock_analyzer.agents.state import ReportDraft, ReportSection


class Fake:
    def invoke(self, msgs):
        return ClaimList(claims=[
            Claim(id="c1", text="RSI is 66.8", metric_key="rsi_14",
                  claimed_value=66.8, claim_type="numeric", source_section="technical")
        ])


def test_extract_populates_claims():
    node = make_extract_node(lambda schema: Fake())
    draft = ReportDraft(
        sections=[ReportSection(id="technical", prose="RSI is 66.8",
                                referenced_metrics=["rsi_14"], charts=[])],
        recommendation="buy",
        confidence="high",
    )
    out = node({"report": draft})
    assert out["claims"][0].metric_key == "rsi_14"
