from stock_analyzer.verification.models import Claim, Verdict, VerificationAudit


def test_construct():
    c = Claim(id="c1", text="RSI is 66.8", metric_key="rsi_14", claimed_value=66.8,
              claim_type="numeric", source_section="technical")
    v = Verdict(claim_id="c1", status="supported", expected_value=66.8, claimed_value=66.8,
                delta=0.0, rationale="match", correction=None)
    a = VerificationAudit(total_claims=1, supported=1, contradicted=0, unsupported=0,
                          corrections_applied=[], residual_unverified=[])
    assert c.claim_type == "numeric" and v.status == "supported" and a.total_claims == 1
