"""
Command-line interface for the Enhanced Stock Analysis Tool.

Usage:
    stock-analyzer analyze AAPL --period 1y --out out --benchmark SPY
"""

import typer

app = typer.Typer(help="AI-powered equity analysis with self-verifying reports.")


@app.callback()
def _main() -> None:
    """Keep 'analyze' as an explicit subcommand (e.g. `stock-analyzer analyze AAPL`)."""


@app.command()
def analyze(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    period: str = typer.Option("1y", help="Look-back window, e.g. 1y, 6mo"),
    out: str = typer.Option("out", help="Output directory for PDF + HTML"),
    benchmark: str = typer.Option("SPY", help="Benchmark ticker for beta/Sharpe"),
):
    """Run a full analysis pipeline and write a verified PDF report + HTML dashboard."""
    from .config import Settings
    from .data import get_provider
    from .agents.llm import make_llm, structured
    from .pipeline import run_analysis

    s = Settings()
    provider = get_provider(s)
    llm = make_llm(s)
    factory = lambda schema: structured(llm, schema)  # noqa: E731
    bench = provider.get_price_history(benchmark, period).bars["Close"]
    res = run_analysis(symbol, period, out, provider, factory, bench, max_revisions=s.max_revisions)
    typer.echo(f"PDF:   {res['pdf']}")
    typer.echo(f"HTML:  {res['html']}")
    typer.echo(f"Audit: {res['audit']}")


if __name__ == "__main__":
    app()
