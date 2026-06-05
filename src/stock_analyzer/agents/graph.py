from langgraph.graph import StateGraph, START, END
from .state import AnalysisState
from .analysts import make_analyst_node
from .writer import make_writer_node


def build_graph(structured_factory, verify: bool = True, ground_truth=None, max_revisions: int = 2):
    g = StateGraph(AnalysisState)
    for role in ("fundamental", "technical", "risk", "valuation"):
        g.add_node(role, make_analyst_node(role, structured_factory))
        g.add_edge(START, role)          # parallel fan-out from START
        g.add_edge(role, "writer")       # join at writer (waits for all 4)
    g.add_node("writer", make_writer_node(structured_factory))
    if not verify:
        g.add_edge("writer", END)
        return g.compile()
    from ..verification.extract import make_extract_node
    from ..verification.judge import make_verify_node
    from ..verification.reconcile import gate
    g.add_node("extract_claims", make_extract_node(structured_factory, ground_truth))
    g.add_node("verify_claims", make_verify_node(structured_factory, ground_truth, 0.01, 0.05))

    def gate_with_cap(state):
        return gate({**state, "max_revisions": max_revisions})

    g.add_edge("writer", "extract_claims")
    g.add_edge("extract_claims", "verify_claims")
    g.add_conditional_edges("verify_claims", gate_with_cap, {"revise": "writer", "finalize": END})
    return g.compile()
