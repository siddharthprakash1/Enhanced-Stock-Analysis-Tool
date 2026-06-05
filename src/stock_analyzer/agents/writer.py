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
        msgs = [{"role": "system", "content":
                 "You are a senior equity-research writer. Using ONLY the provided ground-truth metrics and "
                 "analyst findings, write a polished, investor-ready report. Produce these section ids exactly: "
                 "exec_summary, overview, technical, fundamental, risk, valuation, recommendation. Write clear, "
                 "professional prose (2-4 sentences per section; exec_summary may be longer). Refer to metrics by "
                 "their human names and values (e.g. 'an RSI of 63.5', 'a P/E of 37.7x'). NEVER include internal "
                 "identifiers or bracketed keys (like [rsi_14] or [roe]), code, or markdown symbols (#, *, `). "
                 "Do not invent numbers — use only the provided values."},
                {"role": "user", "content": state["metrics_block"] + "\n\nAnalyst findings:\n" + findings + extra}]
        return {"report": chain.invoke(msgs)}

    return node
