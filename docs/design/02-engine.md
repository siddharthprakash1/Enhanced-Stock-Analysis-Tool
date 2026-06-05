# 02 — The Engine (detailed)

Covers the five engine layers: **A. Data provider**, **B. Metrics → MetricsBundle (ground truth)**, **C. Charts**, **D. Agents + state**, **E. Verification gate**. (Report rendering is doc 03.)

---

## A. Data provider layer (`data/`)
Provider-agnostic; the rest of the system never touches yfinance/FMP directly.

**`base.py` — `DataProvider` Protocol** (structural typing):
- `get_price_history(symbol: str, period: str) -> PriceHistory`
- `get_fundamentals(symbol: str) -> Fundamentals`
- `get_news(symbol: str, limit: int = 20) -> list[NewsItem]`

**`models.py` — pydantic data models** (normalized, validated):
- `PriceHistory`: OHLCV bars (DataFrame + typed accessors), index of dates, `symbol`, `period`, `currency`.
- `Fundamentals`: `pe, pb, debt_to_equity, roe, eps_growth, market_cap, dividend_yield, profit_margin, revenue_growth, sector, industry, beta_reported` (each `Optional` — missing data is explicit, never silently 0).
- `NewsItem`: `title, publisher, published_at, url`.

**Providers:**
- `yfinance_provider.py` — default, zero-config. Maps yfinance raw → our models.
- `fmp_provider.py` — optional, keyed (env `FMP_API_KEY`). Deeper fundamentals + SEC data. Same interface.
- Factory in `data/__init__.py`: if a keyed provider is configured+enabled use it, else yfinance.

**Errors:** `DataUnavailableError(symbol, field)` when required data is missing. Pipeline degrades gracefully — a missing field becomes `N/A` in the bundle and downstream agents are told it is unavailable (never fabricate).

---

## B. Metrics → `MetricsBundle` (ground truth) (`metrics/`)
Pure functions. No LLM, no network beyond the provider. **This is the single source of truth.**

**The atomic unit — `MetricValue`:**
```python
class MetricValue(BaseModel):
    key: str            # stable id, e.g. "rsi_14", "pe_ratio", "sharpe_ratio"
    label: str          # human label, e.g. "RSI (14-day)"
    value: float | None # None == N/A (data unavailable)
    unit: str           # "", "$", "%", "x", "days", "ratio"
    category: str       # technical | fundamental | risk | valuation | sentiment
    as_of: date
```

**Per-module computations:**
- `technical.py` — SMA(50/200), EMA(12/26), RSI(14), MACD + signal line, Bollinger(20, 2σ), volume MA(20/50), support/resistance (rolling pivots), 52-week high/low. *(carried over from old `test.py`, vectorized + cleaned.)*
- `fundamental.py` — P/E, P/B, D/E, ROE, EPS growth **+ added:** profit margin, revenue growth, dividend yield, market cap.
- `risk.py` — beta vs SPY, annualized historical volatility, ATR **+ added for rigor:** max drawdown, Sharpe ratio, 95% Value-at-Risk (historical), downside deviation.
- `valuation.py` — simple **DCF** (explicit assumptions + a small sensitivity grid), comparables inputs, price-target scaffolding (bull/base/bear).
- `sentiment.py` — headline sentiment. **Default:** VADER (zero-config, fast). **Optional upgrade:** a Claude-based finance-aware sentiment node (more nuanced). Config flag chooses.

**`bundle.py` — `MetricsBundle`:**
- Aggregates all sub-categories + metadata (`symbol`, `period`, `as_of`, `provider`, `currency`).
- Exposes `as_flat() -> dict[str, MetricValue]` (key → value) — the lookup table the verifier reconciles against and the report references by key.
- Exposes `prompt_block() -> str` — a deterministic, cache-friendly rendering of all metrics for grounding the agents (stable ordering → prompt cache stays warm).

---

## C. Charts (`charts/`) — deterministic
From `PriceHistory` + `MetricsBundle`. mplfinance for candlestick + overlays; matplotlib for the rest. Full visual/theme treatment is **doc 03**; engine contract here:
- Each chart returns a `ChartRef`: `{name, image_path, caption, fact: dict}` where `fact` holds the exact values drawn (e.g. `{"latest_rsi": 66.8, "ma_cross": "2026-04-02"}`).
- `fact` lets the writer reference charts accurately AND lets the verifier check chart captions against ground truth too.
- Charts: candlestick + SMA/EMA + Bollinger; RSI panel; MACD panel; volume; returns distribution; drawdown curve.

---

## D. Agents + state (`agents/`)

**`state.py` — `AnalysisState` (LangGraph TypedDict):**
```
inputs:      symbol, period, config
ground_truth: metrics: MetricsBundle, charts: list[ChartRef]
analysts:    fundamental, technical, risk, valuation : AnalystFinding
draft:       report: ReportDraft
verify:      claims: list[Claim], verdicts: list[Verdict]
control:     revisions: int, verdict_feedback: str | None
finalize:    audit: VerificationAudit
```

**`llm.py`:**
- `make_llm(effort="high") -> ChatAnthropic(model="claude-opus-4-8", thinking=adaptive, ...)`.
- `structured(llm, Model)` → `llm.with_structured_output(Model)`.
- Injects the cached `MetricsBundle.prompt_block()` as grounding context.

**Analyst nodes (`analysts.py`)** — fundamental / technical / risk / valuation, run in parallel. Each returns:
```python
class AnalystFinding(BaseModel):
    summary: str
    key_points: list[str]
    outlook: Literal["bullish", "bearish", "neutral"]
    rationale: str
    cited_metrics: list[str]   # MetricsBundle keys this analyst used
```
System prompt rule: **"Use only values present in the metrics block; cite each metric key you reference in `cited_metrics`; if a value is N/A, say so — never estimate or invent."** `cited_metrics` makes verification tight.

**Writer node (`writer.py`)** — synthesizes the 4 findings into:
```python
class ReportSection(BaseModel):
    id: str            # exec_summary | overview | technical | fundamental | risk | valuation | recommendation
    prose: str
    referenced_metrics: list[str]
    charts: list[str]  # ChartRef names to embed
class ReportDraft(BaseModel):
    sections: list[ReportSection]
    recommendation: Literal["buy", "hold", "sell"]
    confidence: Literal["low", "medium", "high"]
```
On a **revision pass**, the writer also receives `verdict_feedback` (the failed claims + corrections) and must fix only the offending sections.

---

## E. Verification gate (`verification/`) — the heart

**`claims.py`:**
```python
class Claim(BaseModel):
    id: str
    text: str                         # the asserted sentence
    metric_key: str | None            # MetricsBundle key it refers to, if identifiable
    claimed_value: float | None
    claim_type: Literal["numeric", "directional", "categorical", "qualitative"]
    source_section: str
```
`extract_claims` node — structured Claude call over `ReportDraft`. Every sentence asserting a number, comparison, or directional statement becomes an atomic `Claim`. Numeric claims tagged with their `metric_key` + `claimed_value` when identifiable.

**`reconcile.py`:**
```python
class Verdict(BaseModel):
    claim_id: str
    status: Literal["supported", "contradicted", "unsupported"]
    expected_value: float | None      # from MetricsBundle
    claimed_value: float | None
    delta: float | None
    rationale: str
    correction: str | None            # corrected sentence if contradicted
```
- **Programmatic check:** for numeric claims with a `metric_key`, compare `claimed_value` to `MetricsBundle[key].value` within unit-aware tolerance (relative + absolute, from config) → `supported`/`contradicted` + `delta`.
- **Grounded LLM judge:** for directional/derived/unkeyed claims (e.g. "RSI signals overbought"), a structured Claude call evaluates the claim **against the MetricsBundle ONLY** (not the open web) → status + rationale + correction. This is LLM-as-judge, but grounded on exact numbers.

**GATE (conditional edge in `graph.py`):**
- Count `contradictions` (and track `unsupported`).
- If `contradictions > 0 and revisions < MAX_REVISIONS` → set `verdict_feedback` (failing claims + corrections), `revisions += 1`, route → `writer`.
- Else → `finalize`.

**`finalize` → `VerificationAudit`:**
```python
class VerificationAudit(BaseModel):
    total_claims: int
    supported: int
    contradicted: int
    unsupported: int
    corrections_applied: list[Verdict]
    residual_unverified: list[Verdict]   # if MAX_REVISIONS hit, surfaced honestly
```
If the loop hits `MAX_REVISIONS` with residual issues, the PDF **flags** the remaining unverified claims rather than silently shipping them. The audit renders as a **"Verification" appendix** in the report — the portfolio differentiator: the report shows it fact-checked itself against N metrics and applied M corrections.

**Config (`config.py`):** `MODEL="claude-opus-4-8"`, `EFFORT="high"`, `MAX_REVISIONS=2`, `NUMERIC_TOL_REL=0.01`, `NUMERIC_TOL_ABS=...`, provider selection, output paths.
