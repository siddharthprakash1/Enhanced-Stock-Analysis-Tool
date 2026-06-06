from pathlib import Path
import re

HERO_KEYS = ["last_close", "pe_ratio", "rsi_14", "beta", "sharpe", "max_drawdown", "var_95"]
SHORT_LABELS = {"last_close": "Last", "pe_ratio": "P/E", "rsi_14": "RSI", "beta": "Beta",
                "sharpe": "Sharpe", "max_drawdown": "Max DD", "var_95": "VaR 95%"}
SECTION_TITLES = {
    "exec_summary": "Executive Summary", "overview": "Company Overview",
    "technical": "Technical Analysis", "fundamental": "Fundamental Analysis",
    "risk": "Risk Assessment", "valuation": "Valuation & Forecast", "recommendation": "Recommendation",
}
SECTION_CHARTS = {"technical": ["candlestick", "rsi", "macd"], "risk": ["returns_dist", "drawdown"]}
CHART_CAPTIONS = {
    "price": "Price with 50- and 200-day moving averages", "rsi": "Relative Strength Index (14-day)",
    "macd": "MACD vs. signal line", "returns_dist": "Distribution of daily returns",
    "drawdown": "Drawdown from the running peak",
    "candlestick": "Candlestick with SMA 50/200 and volume",
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


def build_context(symbol, draft, ground_truth, audit, charts, company=None, news=None,
                  sensitivity=None, assumptions=None, comps=None, currency=None, logo=None):
    as_of = ""
    if ground_truth:
        mv0 = next(iter(ground_truth.values()))
        as_of = mv0.as_of.strftime("%d %b %Y") if hasattr(mv0.as_of, "strftime") else str(mv0.as_of)
    kpis = [{"label": SHORT_LABELS.get(k, ground_truth[k].label), "display": ground_truth[k].display()}
            for k in HERO_KEYS if k in ground_truth]
    sections = [{
        "title": SECTION_TITLES.get(s.id, s.id.replace("_", " ").title()),
        "paras": _paras(s.prose),
        "charts": _charts_for(s.id, charts),
    } for s in draft.sections]
    residual = [{"status": v.get("status"), "note": v.get("correction") or v.get("rationale") or ""}
                for v in (audit.get("residual_unverified") or [])]

    by_cat = {}
    for mv in (ground_truth or {}).values():
        by_cat.setdefault(mv.category, []).append({"key": mv.key, "label": mv.label, "value": mv.display()})

    # Rate/comps-summary metrics live in the assumptions & comps blocks (and the
    # appendix), so keep them out of the headline Valuation table to avoid dupes.
    _ASSUMPTION_KEYS = {"wacc", "cost_of_equity", "cost_of_debt", "peer_median_pe", "pe_premium_to_peers"}
    valuation_table = [r for r in by_cat.get("valuation", []) if r["key"] not in _ASSUMPTION_KEYS]

    def tgt(key):
        return ground_truth[key].display() if key in ground_truth else None

    last = ground_truth.get("last_close")
    targets = None
    if "target_base" in ground_truth:
        def upside(key):
            if last and last.value and ground_truth[key].value is not None:
                return f"{(ground_truth[key].value / last.value - 1) * 100:+.1f}%"
            return ""
        targets = {
            "base": tgt("target_base"), "bull": tgt("target_bull"), "bear": tgt("target_bear"),
            "base_upside": upside("target_base"), "bull_upside": upside("target_bull"),
            "bear_upside": upside("target_bear"),
        }

    return {
        "symbol": symbol, "recommendation": draft.recommendation, "confidence": draft.confidence,
        "as_of": as_of, "kpis": kpis, "sections": sections, "audit": audit, "residual": residual,
        "company": company or symbol,
        "price": ground_truth["last_close"].display() if "last_close" in ground_truth else "",
        "targets": targets,
        "fundamentals_table": by_cat.get("fundamental", []),
        "risk_table": by_cat.get("risk", []),
        "valuation_table": valuation_table,
        "valuation_available": bool(valuation_table),
        "assumptions": assumptions,
        "comps": comps,
        "currency": currency,
        "logo": (Path(logo).resolve().as_uri() if logo else None),
        "metrics_appendix": [{"category": cat.title(), "rows": rows} for cat, rows in by_cat.items()],
        "news": [{"title": n.title} for n in (news or [])][:10],
        "sensitivity": sensitivity,
    }
