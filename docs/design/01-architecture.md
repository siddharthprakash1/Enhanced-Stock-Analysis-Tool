# 01 — Architecture Skeleton (APPROVED)

## The LangGraph graph (analysis brain)
```
START
 → compute_metrics      deterministic → builds MetricsBundle (ground truth)
 → build_charts         deterministic → chart images + "chart facts"
 → ┌─ fundamental ─┐
   ├─ technical  ──┤    parallel fan-out; each a structured Claude call,
   ├─ risk       ──┤    grounded ONLY in the MetricsBundle
   └─ valuation  ──┘
 → writer               synthesizes analyst findings → structured ReportDraft
 → extract_claims       pulls atomic numeric/factual claims from the draft
 → verify_claims        reconciles each claim vs MetricsBundle → verdicts
 → GATE (conditional edge):
       contradictions found & revisions < MAX_REVISIONS → back to writer w/ verdict feedback
       otherwise                                        → finalize
 → finalize             attaches verdicts + audit log to state
END
 → render PDF  +  render interactive HTML
```
The conditional edge + a bounded `revisions` counter implement the grounding gate / self-correction loop. LangGraph chosen for: explicit typed state, parallel fan-out, the loop edge, and built-in checkpointing/observability (show the graph + trace in the portfolio).

## Project structure (fresh `src/` layout)
```
stock-analyzer/
  pyproject.toml          .env.example          README.md
  src/stock_analyzer/
    config.py             # pydantic-settings: model, effort, thresholds, paths, MAX_REVISIONS
    cli.py                # Typer: `analyze AAPL --period 1y --out report.pdf`
    pipeline.py           # data → metrics → charts → graph → render
    data/                 # DataProvider protocol + yfinance (default) + fmp (optional) + pydantic models
    metrics/              # technical, fundamental, risk, valuation, sentiment → bundle.py (MetricsBundle)
    charts/               # theme + mplfinance/matplotlib builders → {name: image}
    agents/               # state.py, llm.py (ChatAnthropic factory), analysts, writer, verifier, graph.py
    verification/         # Claim/Verdict models, atomic extraction, numeric reconciliation
    report/               # templates/ (Jinja2+CSS), pdf.py (WeasyPrint), html_dashboard.py (Plotly), assemble.py
  tests/                  # deterministic metric math, reconciliation, provider (mocked), graph smoke test
```

## Claude integration specifics
- All LLM calls go through `langchain-anthropic` `ChatAnthropic`, pinned to **`claude-opus-4-8`**.
- Adaptive thinking on; `effort` likely `high` for analysts + verifier, tunable in `config.py`.
- Structured nodes use `.with_structured_output(PydanticModel)` (Anthropic tool-calling under the hood) → analyst findings, claims, and verdicts return as validated objects.
- The `MetricsBundle` context block is **prompt-cached** so the 4 analysts + verifier reuse it cheaply.
