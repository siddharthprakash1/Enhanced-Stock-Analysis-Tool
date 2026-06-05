from .models import Verdict, JudgeVerdict, VerificationAudit
from .reconcile import reconcile_numeric


def make_verify_node(structured_factory, ground_truth, tol_rel, tol_abs):
    judge = structured_factory(JudgeVerdict)

    def node(state):
        gt = ground_truth or {}
        verdicts = []
        for c in state["claims"]:
            if c.claim_type == "numeric" and c.metric_key:
                verdicts.append(reconcile_numeric(c, gt, tol_rel, tol_abs))
                continue
            msgs = [
                {"role": "system", "content": (
                    "You verify a research report against computed ground-truth metrics. Judge ONLY "
                    "factual/quantitative claims. Mark 'supported' when a claim agrees with the metrics, "
                    "INCLUDING reasonable rounding or approximations (e.g. '~60%' for 59.6%, 'about 209%' "
                    "for 208.65% are SUPPORTED). Mark 'contradicted' ONLY when a claim is materially wrong "
                    "or clearly conflicts with the metrics. Do NOT contradict the investment recommendation, "
                    "subjective judgments, opinions, or framing — mark those 'supported'. Give a one-sentence "
                    "rationale; add a correction only when contradicted."
                )},
                {"role": "user", "content": (
                    f"Claim: {c.text}\n\nGround-truth metrics:\n"
                    + "\n".join(f"{v.label}: {v.display()}" for v in gt.values())
                )},
            ]
            try:
                jv = judge.invoke(msgs)
                verdicts.append(Verdict(
                    claim_id=c.id, status=jv.status, claimed_value=c.claimed_value,
                    rationale=jv.rationale, correction=jv.correction,
                ))
            except Exception as exc:  # a single malformed LLM response must not crash the pipeline
                verdicts.append(Verdict(
                    claim_id=c.id, status="unsupported", claimed_value=c.claimed_value,
                    rationale=f"judge unavailable ({type(exc).__name__})",
                ))

        contradicted = [v for v in verdicts if v.status == "contradicted"]
        audit = VerificationAudit(
            total_claims=len(verdicts),
            supported=sum(v.status == "supported" for v in verdicts),
            contradicted=len(contradicted),
            unsupported=sum(v.status == "unsupported" for v in verdicts),
            corrections_applied=contradicted,
            residual_unverified=[v for v in verdicts if v.status != "supported"],
        ).model_dump()

        feedback = "\n".join(f"- {v.correction or v.rationale}" for v in contradicted) or None
        return {
            "verdicts": verdicts,
            "audit": audit,
            "revisions": state.get("revisions", 0) + 1,
            "verdict_feedback": feedback,
        }

    return node
