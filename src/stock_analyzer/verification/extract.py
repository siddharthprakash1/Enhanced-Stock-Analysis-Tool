from .models import ClaimList


def make_extract_node(structured_factory):
    chain = structured_factory(ClaimList)

    def node(state):
        text = "\n\n".join(f"[{s.id}] {s.prose}" for s in state["report"].sections)
        msgs = [
            {"role": "system", "content": (
                "Extract every atomic factual/numeric/directional claim. "
                "For numeric claims set metric_key (the referenced ground-truth key) and claimed_value."
            )},
            {"role": "user", "content": text},
        ]
        return {"claims": chain.invoke(msgs).claims}

    return node
