from .models import ClaimList


def make_extract_node(structured_factory, ground_truth=None):
    chain = structured_factory(ClaimList)

    def node(state):
        gt = ground_truth or {}
        ref = "\n".join(f"- {k}: {mv.label} = {mv.display()}" for k, mv in gt.items())
        text = "\n\n".join(f"[{s.id}] {s.prose}" for s in state["report"].sections)
        msgs = [{"role": "system", "content":
                 "Extract every atomic factual/numeric/directional claim from the report. For each NUMERIC claim, "
                 "set metric_key to the matching key from this reference (match by metric name and the value stated) "
                 "and claimed_value to the number in the prose:\n" + ref},
                {"role": "user", "content": text}]
        return {"claims": chain.invoke(msgs).claims}

    return node
