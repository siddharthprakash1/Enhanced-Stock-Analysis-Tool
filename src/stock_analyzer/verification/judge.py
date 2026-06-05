from .models import Verdict, VerificationAudit
from .reconcile import reconcile_numeric


def make_verify_node(structured_factory, ground_truth, tol_rel, tol_abs):
    judge = structured_factory(Verdict)

    def node(state):
        gt = ground_truth or {}
        verdicts = []
        for c in state["claims"]:
            if c.claim_type == "numeric" and c.metric_key:
                verdicts.append(reconcile_numeric(c, gt, tol_rel, tol_abs))
            else:
                msgs = [
                    {"role": "system", "content": (
                        "Judge the claim using ONLY the ground-truth metrics. "
                        "Return supported/unsupported/contradicted + a correction if contradicted."
                    )},
                    {"role": "user", "content": (
                        f"Claim: {c.text}\n\nGround truth keys/values:\n"
                        + "\n".join(f"{k}={v.display()}" for k, v in gt.items())
                    )},
                ]
                v = judge.invoke(msgs)
                v.claim_id = c.id
                verdicts.append(v)

        contradicted = [v for v in verdicts if v.status == "contradicted"]
        audit = VerificationAudit(
            total_claims=len(verdicts),
            supported=sum(v.status == "supported" for v in verdicts),
            contradicted=len(contradicted),
            unsupported=sum(v.status == "unsupported" for v in verdicts),
            corrections_applied=contradicted,
            residual_unverified=[],
        ).model_dump()

        feedback = "\n".join(f"- {v.correction or v.rationale}" for v in contradicted) or None
        return {
            "verdicts": verdicts,
            "audit": audit,
            "revisions": state.get("revisions", 0) + 1,
            "verdict_feedback": feedback,
        }

    return node
