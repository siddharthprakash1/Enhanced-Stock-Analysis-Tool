from pathlib import Path
import re

HERO_KEYS = ["last_close", "pe_ratio", "rsi_14", "beta", "sharpe", "max_drawdown", "var_95"]
SECTION_TITLES = {
    "exec_summary": "Executive Summary", "overview": "Company Overview",
    "technical": "Technical Analysis", "fundamental": "Fundamental Analysis",
    "risk": "Risk Assessment", "valuation": "Valuation & Forecast", "recommendation": "Recommendation",
}
SECTION_CHARTS = {"technical": ["price", "rsi", "macd"], "risk": ["returns_dist", "drawdown"]}
CHART_CAPTIONS = {
    "price": "Price with 50- and 200-day moving averages", "rsi": "Relative Strength Index (14-day)",
    "macd": "MACD vs. signal line", "returns_dist": "Distribution of daily returns",
    "drawdown": "Drawdown from the running peak",
}


def _paras(prose):
    parts = [p.strip() for p in re.split(r"\n\s*\n", (prose or "").strip()) if p.strip()]
    return parts or [(prose or "").strip()]


def _charts_for(section_id, charts):
    out = []
    for name in SECTION_CHARTS.get(section_id, []):
        p = charts.get(name)
        if p:
            out.append({"src": Path(p).resolve().as_uri(), "caption": CHART_CAPTIONS.get(name, name)})
    return out


def build_context(symbol, draft, ground_truth, audit, charts):
    as_of = ""
    if ground_truth:
        mv0 = next(iter(ground_truth.values()))
        as_of = mv0.as_of.strftime("%d %b %Y") if hasattr(mv0.as_of, "strftime") else str(mv0.as_of)
    kpis = [{"label": ground_truth[k].label, "display": ground_truth[k].display()}
            for k in HERO_KEYS if k in ground_truth]
    sections = [{
        "title": SECTION_TITLES.get(s.id, s.id.replace("_", " ").title()),
        "paras": _paras(s.prose),
        "charts": _charts_for(s.id, charts),
    } for s in draft.sections]
    residual = [{"status": v.get("status"), "note": v.get("correction") or v.get("rationale") or ""}
                for v in (audit.get("residual_unverified") or [])]
    return {
        "symbol": symbol, "recommendation": draft.recommendation, "confidence": draft.confidence,
        "as_of": as_of, "kpis": kpis, "sections": sections, "audit": audit, "residual": residual,
    }
