from datetime import date
from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.verification.models import Claim
from stock_analyzer.verification.reconcile import reconcile_numeric, gate

GT = {
    "rsi_14": MetricValue(key="rsi_14", label="RSI", value=66.8, unit="",
                          category="technical", as_of=date(2026, 6, 5))
}


def test_supported_within_tolerance():
    c = Claim(id="c1", text="RSI ~ 66.8", metric_key="rsi_14", claimed_value=66.9,
              claim_type="numeric", source_section="technical")
    v = reconcile_numeric(c, GT, tol_rel=0.01, tol_abs=0.05)
    assert v.status == "supported"


def test_contradicted_outside_tolerance():
    c = Claim(id="c2", text="RSI is 80", metric_key="rsi_14", claimed_value=80.0,
              claim_type="numeric", source_section="technical")
    v = reconcile_numeric(c, GT, tol_rel=0.01, tol_abs=0.05)
    assert v.status == "contradicted" and "66.8" in (v.correction or "")


def test_gate_routes_revise_then_finalize():
    assert gate({"verdicts": [type("V", (), {"status": "contradicted"})()],
                 "revisions": 0, "max_revisions": 2}) == "revise"
    assert gate({"verdicts": [], "revisions": 0, "max_revisions": 2}) == "finalize"
    assert gate({"verdicts": [type("V", (), {"status": "contradicted"})()],
                 "revisions": 2, "max_revisions": 2}) == "finalize"
