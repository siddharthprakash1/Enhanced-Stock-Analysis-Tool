HERO_KEYS = ["last_close", "pe_ratio", "rsi_14", "beta", "sharpe", "max_drawdown", "var_95"]


def build_context(symbol, draft, ground_truth, audit, charts):
    kpis = [
        {"label": ground_truth[k].label, "value": ground_truth[k].display(), "unit": ground_truth[k].unit}
        for k in HERO_KEYS
        if k in ground_truth
    ]
    return {
        "symbol": symbol,
        "recommendation": draft.recommendation,
        "confidence": draft.confidence,
        "sections": [
            {
                "id": s.id,
                "prose": s.prose,
                "charts": [charts.get(c) for c in s.charts if charts.get(c)],
            }
            for s in draft.sections
        ],
        "kpis": kpis,
        "audit": audit,
    }
