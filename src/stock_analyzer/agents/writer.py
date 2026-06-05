from .state import ReportDraft


def make_writer_node(structured_factory):
    chain = structured_factory(ReportDraft)

    def node(state):
        findings = "\n\n".join(
            f"## {r}\n{getattr(state[r], 'summary', '')}\n{getattr(state[r], 'rationale', '')}"
            for r in ("fundamental", "technical", "risk", "valuation") if r in state
        )
        feedback = state.get("verdict_feedback")
        extra = (f"\n\nFIX THESE VERIFICATION ISSUES (correct only the affected sentences):\n{feedback}"
                 if feedback else "")
        msgs = [{"role": "system", "content": "You are an equity report writer. Use only grounded metrics; "
                 "produce sections: exec_summary, overview, technical, fundamental, risk, valuation, recommendation."},
                {"role": "user", "content": state["metrics_block"] + "\n\nAnalyst findings:\n" + findings + extra}]
        return {"report": chain.invoke(msgs)}

    return node
