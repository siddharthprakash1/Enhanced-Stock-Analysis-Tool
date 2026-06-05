from ..metrics.value import MetricValue
from .models import Claim, Verdict


def reconcile_numeric(claim: Claim, ground_truth: dict[str, MetricValue],
                      tol_rel: float, tol_abs: float) -> Verdict:
    mv = ground_truth.get(claim.metric_key) if claim.metric_key else None
    if mv is None or mv.value is None or claim.claimed_value is None:
        return Verdict(claim_id=claim.id, status="unsupported",
                       rationale="No matching ground-truth metric")
    expected = mv.value
    claimed = claim.claimed_value
    # Models often state large $ figures in K/M/B/T rather than raw dollars; accept any common scale.
    candidates = [expected]
    if mv.unit == "$" and abs(expected) >= 1e6:
        candidates += [expected / s for s in (1e3, 1e6, 1e9, 1e12)]
    best = min(candidates, key=lambda e: abs(claimed - e))
    delta = claimed - best
    tol = max(tol_abs, abs(best) * tol_rel)
    if abs(delta) <= tol:
        return Verdict(claim_id=claim.id, status="supported", expected_value=expected,
                       claimed_value=claimed, delta=delta, rationale="within tolerance")
    return Verdict(claim_id=claim.id, status="contradicted", expected_value=expected,
                   claimed_value=claimed, delta=delta, rationale="outside tolerance",
                   correction=f"{mv.label} is {mv.display()}, not {claimed:g}")


def gate(state) -> str:
    verdicts = state.get("verdicts", []) or []
    contradictions = sum(1 for v in verdicts if getattr(v, "status", None) == "contradicted")
    if contradictions and state.get("revisions", 0) < state.get("max_revisions", 2):
        return "revise"
    return "finalize"
