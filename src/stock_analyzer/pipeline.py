from pathlib import Path

from .metrics.bundle import MetricsBundle
from .charts.builders import build_charts
from .agents.graph import build_graph
from .agents.llm import grounding_block
from .report.assemble import build_context
from .report.pdf import render_pdf
from .report.html_dashboard import render_dashboard


def run_analysis(symbol, period, out_dir, provider, structured_factory, benchmark, max_revisions: int = 2):
    """Run the full analysis pipeline and write PDF + HTML dashboard to out_dir.

    Parameters
    ----------
    symbol:            ticker symbol, e.g. "AAPL"
    period:            look-back window, e.g. "1y"
    out_dir:           destination directory (created if absent)
    provider:          DataProvider instance
    structured_factory: callable(schema) -> LLM-like object with .invoke(msgs)
    benchmark:         pd.Series of benchmark close prices (e.g. SPY)

    Returns
    -------
    dict with keys: pdf, html, audit
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data
    ph = provider.get_price_history(symbol, period)
    fund = provider.get_fundamentals(symbol)
    news = provider.get_news(symbol)

    # 2. Compute deterministic ground-truth metrics
    bundle = MetricsBundle.from_data(ph, fund, benchmark, news, fcf0=100.0, growth=0.05, wacc=0.10)
    gt = bundle.as_flat()

    # 3. Build charts (PNG files written to out/charts/)
    charts = {r.name: r.image_path for r in build_charts(ph.bars, out / "charts")}

    # 4. Run the LangGraph multi-agent pipeline
    app = build_graph(structured_factory, verify=True, ground_truth=gt, max_revisions=max_revisions)
    state = app.invoke({"symbol": symbol, "metrics_block": grounding_block(bundle), "revisions": 0})

    # 5. Assemble report context
    from .metrics.valuation import sensitivity_grid
    fcf0 = fund.free_cash_flow or 100.0
    sens = {
        "growths": [0.03, 0.05, 0.07],
        "waccs": [0.08, 0.10, 0.12],
        "grid": sensitivity_grid(fcf0, [0.03, 0.05, 0.07], [0.08, 0.10, 0.12]),
    }
    ctx = build_context(symbol, state["report"], gt, state.get("audit", {}), charts,
                        company=fund.name, news=news, sensitivity=sens)

    # 6. Render PDF (base_url = out dir so file:// image embeds work)
    render_pdf(ctx, out / f"{symbol}_report.pdf")

    # 7. Render interactive HTML dashboard
    # Dashboard expects kpis with "value"/"unit"; adapt from the new "display" key.
    dash_kpis = [{"label": k["label"], "value": k["display"], "unit": ""} for k in ctx["kpis"]]
    verdict_rows = [{"text": v.rationale, "status": v.status} for v in state.get("verdicts", [])]
    render_dashboard(symbol, ph.bars, dash_kpis, verdict_rows, out / f"{symbol}_dashboard.html")

    return {
        "pdf": out / f"{symbol}_report.pdf",
        "html": out / f"{symbol}_dashboard.html",
        "audit": state.get("audit"),
    }
