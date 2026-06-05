from datetime import date
from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.agents.graph import build_graph
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection
from stock_analyzer.verification.models import ClaimList, Claim, Verdict

GT = {
    "rsi_14": MetricValue(key="rsi_14", label="RSI", value=66.8, unit="",
                          category="technical", as_of=date(2026, 6, 5))
}


def make_fake_factory():
    state = {"pass": 0}

    def factory(schema):
        class F:
            def invoke(self, msgs):
                if schema is ReportDraft:
                    return ReportDraft(
                        sections=[ReportSection(id="technical", prose="x",
                                                referenced_metrics=[], charts=[])],
                        recommendation="hold",
                        confidence="medium",
                    )
                if schema is ClaimList:
                    state["pass"] += 1
                    val = 80.0 if state["pass"] == 1 else 66.8
                    return ClaimList(claims=[
                        Claim(id="c1", text=f"RSI {val}", metric_key="rsi_14",
                              claimed_value=val, claim_type="numeric", source_section="technical")
                    ])
                if schema is Verdict:
                    return Verdict(claim_id="x", status="supported", rationale="ok")
                return AnalystFinding(summary="s", key_points=[], outlook="neutral",
                                      rationale="r", cited_metrics=[])
        return F()

    return factory


def test_revision_loop_resolves():
    app = build_graph(make_fake_factory(), verify=True, ground_truth=GT, max_revisions=2)
    out = app.invoke({"symbol": "AAPL", "metrics_block": "RSI (14): 66.8", "revisions": 0})
    assert out["audit"]["contradicted"] == 0
    assert out["revisions"] == 2
