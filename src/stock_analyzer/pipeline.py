from pathlib import Path

from .data.models import Fundamentals
from .metrics.bundle import MetricsBundle
from .metrics.risk import compute_beta
from .metrics.wacc import compute_wacc
from .metrics.valuation import ValuationInputs, derive_growth, sensitivity_grid
from .metrics.comps import peers_for, build_comps, comps_metrics
from .charts.builders import build_charts
from .agents.graph import build_graph
from .agents.llm import grounding_block
from .report.assemble import build_context
from .report.pdf import render_pdf
from .report.html_dashboard import render_dashboard


def _resolve_risk_free(provider, fallback: float) -> float:
    """Live risk-free rate from the provider if it offers one, else the fallback."""
    getter = getattr(provider, "get_risk_free_rate", None)
    if callable(getter):
        try:
            rate = getter()
            if rate is not None:
                return float(rate)
        except Exception:
            pass
    return fallback


def _build_valuation_inputs(fund: Fundamentals, beta, rf, *, erp, tax_rate, cod_spread,
                            terminal_growth, years, growth_default, growth_min, growth_max):
    """ValuationInputs for an operating company with real FCF + shares, else None."""
    if not fund.is_operating_company:
        return None
    fcf0 = fund.free_cash_flow
    if not fcf0 or fcf0 <= 0 or not fund.shares_outstanding or fund.shares_outstanding <= 0:
        return None
    wc = compute_wacc(
        risk_free=rf, beta=beta, erp=erp, market_cap=fund.market_cap,
        total_debt=fund.total_debt, cost_of_debt_spread=cod_spread, tax_rate=tax_rate,
    )
    growth, gsrc = derive_growth(fund.revenue_growth, fund.eps_growth,
                                 growth_default, growth_min, growth_max)
    return ValuationInputs(
        fcf0=fcf0, growth=growth, growth_source=gsrc, wacc=wc,
        terminal_growth=terminal_growth, years=years,
        shares_outstanding=fund.shares_outstanding,
    )


def _fetch_peers(provider, symbol: str, sector: str | None) -> list[Fundamentals]:
    peers = []
    for tkr in peers_for(symbol, sector):
        try:
            peers.append(provider.get_fundamentals(tkr))
        except Exception:
            continue  # a flaky peer fetch must not break the subject's report
    return peers


def run_analysis(
    symbol, period, out_dir, provider, structured_factory, benchmark, max_revisions: int = 2,
    *, risk_free_rate: float = 0.0455, equity_risk_premium: float = 0.0423, tax_rate: float = 0.21,
    cost_of_debt_spread: float = 0.015, terminal_growth: float = 0.02, dcf_years: int = 5,
    growth_default: float = 0.05, growth_min: float = 0.0, growth_max: float = 0.15,
):
    """Run the full analysis pipeline and write PDF + HTML dashboard to out_dir.

    DCF uses a CAPM-derived WACC (live 10Y Treasury for the risk-free rate, falling
    back to ``risk_free_rate``) with growth taken from the company's own fundamentals.
    Valuation and comps are skipped gracefully for ETFs/funds or thin-data securities.

    Returns a dict with keys: pdf, html, audit.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data
    ph = provider.get_price_history(symbol, period)
    fund = provider.get_fundamentals(symbol)
    news = provider.get_news(symbol)
    as_of = ph.bars.index[-1].date()

    # 2. Discount rate (WACC) + growth + valuation inputs (None => skip valuation)
    rf = _resolve_risk_free(provider, risk_free_rate)
    beta = compute_beta(ph.bars["Close"], benchmark)
    vi = _build_valuation_inputs(
        fund, beta, rf, erp=equity_risk_premium, tax_rate=tax_rate,
        cod_spread=cost_of_debt_spread, terminal_growth=terminal_growth, years=dcf_years,
        growth_default=growth_default, growth_min=growth_min, growth_max=growth_max,
    )

    # 3. Deterministic ground-truth metrics
    bundle = MetricsBundle.from_data(ph, fund, benchmark, news, valuation=vi)

    # 4. Relative valuation (comps) against the sector peer set; inject verifiable summary metrics
    comps = None
    if fund.is_operating_company:
        peers = _fetch_peers(provider, symbol, fund.sector)
        comps = build_comps(fund, peers)
        if comps:
            bundle.metrics.update(comps_metrics(comps, as_of))

    gt = bundle.as_flat()

    # 5. Charts (PNG files written to out/charts/)
    charts = {r.name: r.image_path for r in build_charts(ph.bars, out / "charts")}

    # 6. LangGraph multi-agent pipeline (writer grounded only on `bundle`)
    app = build_graph(structured_factory, verify=True, ground_truth=gt, max_revisions=max_revisions)
    state = app.invoke({"symbol": symbol, "metrics_block": grounding_block(bundle), "revisions": 0})

    # 7. Assemble report context — DCF sensitivity + assumptions only when a DCF was run
    sens = assumptions = None
    if vi is not None and "dcf_value" in gt:
        growths = [round(max(vi.growth - 0.02, 0.0), 4), round(vi.growth, 4), round(vi.growth + 0.02, 4)]
        w = vi.wacc.wacc
        waccs = [round(max(w - 0.02, vi.terminal_growth + 0.005), 4), round(w, 4), round(w + 0.02, 4)]
        sens = {"growths": growths, "waccs": waccs,
                "grid": sensitivity_grid(vi.fcf0, growths, waccs, vi.years, vi.terminal_growth)}
        assumptions = _assumptions_rows(vi)

    ctx = build_context(symbol, state["report"], gt, state.get("audit", {}), charts,
                        company=fund.name, news=news, sensitivity=sens,
                        assumptions=assumptions, comps=_comps_view(comps),
                        currency=(fund.currency or ph.currency))

    # 8. Render PDF
    render_pdf(ctx, out / f"{symbol}_report.pdf")

    # 9. Render interactive HTML dashboard
    dash_kpis = [{"label": k["label"], "value": k["display"], "unit": ""} for k in ctx["kpis"]]
    verdict_rows = [{"text": v.rationale, "status": v.status} for v in state.get("verdicts", [])]
    render_dashboard(symbol, ph.bars, dash_kpis, verdict_rows, out / f"{symbol}_dashboard.html")

    return {
        "pdf": out / f"{symbol}_report.pdf",
        "html": out / f"{symbol}_dashboard.html",
        "audit": state.get("audit"),
    }


def _pct(x):
    return f"{x * 100:.2f}%"


def _assumptions_rows(vi: ValuationInputs):
    """Render-ready (label, value) rows for the report's DCF assumptions block."""
    wc = vi.wacc
    return [
        {"label": "Risk-free rate (10Y UST)", "value": _pct(wc.risk_free)},
        {"label": "Equity risk premium", "value": _pct(wc.erp)},
        {"label": "Beta", "value": f"{wc.beta:.2f}"},
        {"label": "Cost of equity (CAPM)", "value": _pct(wc.cost_of_equity)},
        {"label": "Cost of debt (pre-tax)", "value": _pct(wc.cost_of_debt)},
        {"label": "Tax rate", "value": _pct(wc.tax_rate)},
        {"label": "Equity / debt weight", "value": f"{wc.weight_equity * 100:.0f}% / {wc.weight_debt * 100:.0f}%"},
        {"label": "WACC (discount rate)", "value": _pct(wc.wacc)},
        {"label": "FCF growth", "value": f"{vi.growth * 100:.1f}%  ({vi.growth_source})"},
        {"label": "Terminal growth", "value": _pct(vi.terminal_growth)},
        {"label": "Forecast horizon", "value": f"{vi.years} years"},
    ]


def _fmt_comp(v, unit):
    if v is None:
        return "—"
    return f"{v:.1f}%" if unit == "%" else f"{v:.1f}x"


def _comps_view(comps):
    """Convert raw comps into display-ready rows for the template, or None."""
    if not comps:
        return None
    rows = []
    for r in comps["rows"]:
        rows.append({
            "label": r["label"],
            "subject": _fmt_comp(r["subject"], r["unit"]),
            "median": _fmt_comp(r["median"], r["unit"]),
            "premium": (None if r["premium"] is None else f"{r['premium']:+.0f}%"),
        })
    return {"subject": comps["subject"], "tickers": comps["tickers"], "rows": rows}
