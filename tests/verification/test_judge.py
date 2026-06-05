from datetime import date
from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.verification.models import Claim, Verdict
from stock_analyzer.verification.judge import make_verify_node

GT = {
    "rsi_14": MetricValue(key="rsi_14", label="RSI", value=66.8, unit="",
                          category="technical", as_of=date(2026, 6, 5))
}


class FakeJudge:
    def invoke(self, msgs):
        return Verdict(claim_id="d1", status="supported", rationale="ok")


def test_numeric_goes_programmatic_directional_goes_judge():
    node = make_verify_node(lambda schema: FakeJudge(), ground_truth=GT,
                            tol_rel=0.01, tol_abs=0.05)
    claims = [
        Claim(id="n1", text="RSI 80", metric_key="rsi_14", claimed_value=80.0,
              claim_type="numeric", source_section="technical"),
        Claim(id="d1", text="RSI signals overbought", metric_key=None,
              claim_type="directional", source_section="technical"),
    ]
    out = node({"claims": claims, "revisions": 0})
    by = {v.claim_id: v for v in out["verdicts"]}
    assert by["n1"].status == "contradicted"
    assert by["d1"].status == "supported"
    assert out["audit"]["contradicted"] == 1 and out["revisions"] == 1
    assert isinstance(out["verdict_feedback"], str)
