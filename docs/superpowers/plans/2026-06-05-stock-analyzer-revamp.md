# Stock Analyzer Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `Enhanced-Stock-Analysis-Tool` into a portfolio-grade, Claude-powered equity analyzer that computes a deterministic metrics ground-truth, has LangGraph agents write a report, verifies every numeric claim against the ground-truth, and renders a polished PDF + interactive HTML dashboard.

**Architecture:** Deterministic `MetricsBundle` (single source of truth) → LangGraph graph (parallel analyst fan-out → writer → claim-extraction → reconciliation → conditional revision gate → finalize) → WeasyPrint PDF + Plotly HTML. All LLM calls use Claude Opus 4.8 via `langchain-anthropic` with structured outputs.

**Tech Stack:** Python 3.11+, `uv`/pip, pydantic v2, pandas/numpy, yfinance, mplfinance/matplotlib, plotly, langgraph, langchain-anthropic, jinja2, weasyprint, typer, pytest.

**Design reference:** `docs/design/00-overview.md` … `03-report-design.md`. Re-read these before starting — they are the source of truth and the locked decisions.

---

## File structure (responsibilities)

```
src/stock_analyzer/
  config.py            # pydantic-settings: model id, effort, tolerances, MAX_REVISIONS, provider, paths
  tokens.py            # design tokens (colors, fonts) shared by matplotlib + plotly + CSS
  cli.py               # Typer entrypoint: `analyze AAPL --period 1y --out out/`
  pipeline.py          # orchestration: data → metrics → charts → graph → render
  data/
    models.py          # PriceHistory, Fundamentals, NewsItem (pydantic)
    base.py            # DataProvider Protocol + DataUnavailableError
    yfinance_provider.py
    fmp_provider.py    # optional keyed (stub interface ok for v1)
    __init__.py        # get_provider() factory
  metrics/
    value.py           # MetricValue
    technical.py  fundamental.py  risk.py  valuation.py  sentiment.py
    bundle.py          # MetricsBundle (+ from_data, as_flat, prompt_block)
  charts/
    theme.py           # matplotlib/mplfinance style from tokens
    builders.py        # ChartRef + build_charts() -> list[ChartRef]
  agents/
    state.py           # AnalysisState (TypedDict) + AnalystFinding, ReportDraft, ReportSection
    llm.py             # make_llm(), structured(), metrics grounding block
    analysts.py        # analyst node factory (fundamental/technical/risk/valuation)
    writer.py          # writer node
    graph.py           # build_graph()
  verification/
    models.py          # Claim, Verdict, VerificationAudit
    extract.py         # extract_claims node
    reconcile.py       # programmatic numeric reconciliation
    judge.py           # grounded LLM judge node
  report/
    templates/report.html.j2  dashboard.html.j2  styles.css
    assemble.py        # build render context from state
    pdf.py             # WeasyPrint render
    html_dashboard.py  # Plotly self-contained dashboard
tests/                 # mirrors src; fixtures/ holds canned data
```

**Testing approach for LLM code:** nodes never construct their own LLM — `build_graph(llm_factory)` injects it, so tests pass a **fake** that returns preset pydantic objects. No network in unit tests. Data providers are tested against recorded/mocked responses.

---

## Phase 0 — Scaffold & tooling

### Task 0.1: Project skeleton

**Files:**
- Create: `pyproject.toml`, `src/stock_analyzer/__init__.py`, `.env.example`, `tests/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports():
    import stock_analyzer
    assert stock_analyzer.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL (ModuleNotFoundError / no `__version__`)

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "stock-analyzer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7", "pydantic-settings>=2.3", "pandas>=2.2", "numpy>=1.26",
  "yfinance>=0.2.40", "mplfinance>=0.12.10b0", "matplotlib>=3.8", "plotly>=5.22",
  "langgraph>=0.2.0", "langchain-anthropic>=0.3.0", "langchain-core>=0.3.0",
  "jinja2>=3.1", "weasyprint>=62", "typer>=0.12", "vaderSentiment>=3.3.2",
]
[project.optional-dependencies]
dev = ["pytest>=8", "pytest-mock>=3.14", "ruff>=0.5"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/stock_analyzer"]
[tool.pytest.ini_options]
pythonpath = ["src"]
```

```python
# src/stock_analyzer/__init__.py
__version__ = "0.1.0"
```

```bash
# .env.example
ANTHROPIC_API_KEY=sk-ant-...
# Optional reliable data provider (else yfinance is used):
FMP_API_KEY=
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_smoke.py -v`
Expected: PASS
(macOS note: WeasyPrint needs native libs — `brew install pango cairo gdk-pixbuf libffi` before install.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/stock_analyzer/__init__.py .env.example tests/
git commit -m "chore: scaffold src layout, deps, and smoke test"
```

### Task 0.2: Config

**Files:**
- Create: `src/stock_analyzer/config.py`, `tests/test_config.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_config.py
from stock_analyzer.config import Settings

def test_defaults():
    s = Settings(_env_file=None)
    assert s.model == "claude-opus-4-8"
    assert s.effort == "high"
    assert s.max_revisions == 2
    assert s.numeric_tol_rel == 0.01
    assert s.provider == "yfinance"
```

- [ ] **Step 2: Run / fail** — `pytest tests/test_config.py -v` → FAIL (no module)

- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str | None = None
    fmp_api_key: str | None = None
    model: str = "claude-opus-4-8"
    effort: str = "high"
    max_revisions: int = 2
    numeric_tol_rel: float = 0.01
    numeric_tol_abs: float = 0.05
    provider: str = "yfinance"
    out_dir: str = "out"
```

- [ ] **Step 4: Run / pass** — `pytest tests/test_config.py -v` → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: settings via pydantic-settings"`

---

## Phase 1 — Data layer

### Task 1.1: Data models

**Files:** Create `src/stock_analyzer/data/models.py`, `tests/data/test_models.py`

- [ ] **Step 1: Failing test**

```python
# tests/data/test_models.py
import pandas as pd
from stock_analyzer.data.models import PriceHistory, Fundamentals, NewsItem

def test_price_history_latest_close():
    df = pd.DataFrame(
        {"Open":[1,2], "High":[2,3], "Low":[1,1], "Close":[1.5, 2.5], "Volume":[100,200]},
        index=pd.to_datetime(["2026-01-01","2026-01-02"]),
    )
    ph = PriceHistory(symbol="AAPL", period="1y", currency="USD", bars=df)
    assert ph.latest_close == 2.5
    assert len(ph.bars) == 2

def test_fundamentals_optional_fields_default_none():
    f = Fundamentals(symbol="AAPL")
    assert f.pe is None and f.market_cap is None
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/data/models.py
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, ConfigDict

class PriceHistory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    symbol: str
    period: str
    currency: str = "USD"
    bars: pd.DataFrame  # columns: Open High Low Close Volume; DatetimeIndex

    @property
    def latest_close(self) -> float:
        return float(self.bars["Close"].iloc[-1])

class Fundamentals(BaseModel):
    symbol: str
    pe: float | None = None
    pb: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    eps_growth: float | None = None
    market_cap: float | None = None
    dividend_yield: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    sector: str | None = None
    industry: str | None = None
    beta_reported: float | None = None

class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    published_at: datetime | None = None
    url: str | None = None
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(data): pydantic data models"`

### Task 1.2: Provider protocol + error

**Files:** Create `src/stock_analyzer/data/base.py`, `tests/data/test_base.py`

- [ ] **Step 1: Failing test**

```python
# tests/data/test_base.py
from stock_analyzer.data.base import DataProvider, DataUnavailableError

def test_error_message():
    e = DataUnavailableError("AAPL", "pe")
    assert "AAPL" in str(e) and "pe" in str(e)

def test_protocol_is_runtime_checkable():
    class Dummy:
        def get_price_history(self, s, p): ...
        def get_fundamentals(self, s): ...
        def get_news(self, s, limit=20): ...
    assert isinstance(Dummy(), DataProvider)
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/data/base.py
from typing import Protocol, runtime_checkable
from .models import PriceHistory, Fundamentals, NewsItem

class DataUnavailableError(Exception):
    def __init__(self, symbol: str, field: str):
        super().__init__(f"Data unavailable for {symbol!r}: {field}")
        self.symbol, self.field = symbol, field

@runtime_checkable
class DataProvider(Protocol):
    def get_price_history(self, symbol: str, period: str) -> PriceHistory: ...
    def get_fundamentals(self, symbol: str) -> Fundamentals: ...
    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]: ...
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(data): provider protocol + DataUnavailableError"`

### Task 1.3: yfinance provider

**Files:** Create `src/stock_analyzer/data/yfinance_provider.py`, `tests/data/test_yfinance_provider.py`

- [ ] **Step 1: Failing test** (mock yfinance — no network)

```python
# tests/data/test_yfinance_provider.py
import pandas as pd
from stock_analyzer.data.yfinance_provider import YFinanceProvider

def test_get_price_history(mocker):
    df = pd.DataFrame(
        {"Open":[1.0], "High":[2.0], "Low":[0.5], "Close":[1.5], "Volume":[100]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    mocker.patch("stock_analyzer.data.yfinance_provider.yf.download", return_value=df)
    ph = YFinanceProvider().get_price_history("AAPL", "1y")
    assert ph.symbol == "AAPL" and ph.latest_close == 1.5

def test_get_fundamentals_maps_info(mocker):
    fake = mocker.Mock()
    fake.info = {"trailingPE": 28.4, "priceToBook": 5.0, "returnOnEquity": 0.3, "marketCap": 3e12}
    mocker.patch("stock_analyzer.data.yfinance_provider.yf.Ticker", return_value=fake)
    f = YFinanceProvider().get_fundamentals("AAPL")
    assert f.pe == 28.4 and f.market_cap == 3e12
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/data/yfinance_provider.py
import yfinance as yf
from .models import PriceHistory, Fundamentals, NewsItem
from .base import DataUnavailableError

class YFinanceProvider:
    def get_price_history(self, symbol: str, period: str) -> PriceHistory:
        df = yf.download(symbol, period=period, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise DataUnavailableError(symbol, "price_history")
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        return PriceHistory(symbol=symbol, period=period, bars=df[["Open","High","Low","Close","Volume"]])

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        info = yf.Ticker(symbol).info or {}
        return Fundamentals(
            symbol=symbol, pe=info.get("trailingPE"), pb=info.get("priceToBook"),
            debt_to_equity=info.get("debtToEquity"), roe=info.get("returnOnEquity"),
            eps_growth=info.get("earningsQuarterlyGrowth"), market_cap=info.get("marketCap"),
            dividend_yield=info.get("dividendYield"), profit_margin=info.get("profitMargins"),
            revenue_growth=info.get("revenueGrowth"), sector=info.get("sector"),
            industry=info.get("industry"), beta_reported=info.get("beta"),
        )

    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        raw = getattr(yf.Ticker(symbol), "news", []) or []
        items = []
        for a in raw[:limit]:
            t = a.get("title") or (a.get("content") or {}).get("title")
            if t:
                items.append(NewsItem(title=t, publisher=a.get("publisher")))
        return items
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(data): yfinance provider with mocked tests"`

### Task 1.4: Provider factory

**Files:** Create `src/stock_analyzer/data/__init__.py`, `tests/data/test_factory.py`

- [ ] **Step 1: Failing test**

```python
# tests/data/test_factory.py
from stock_analyzer.config import Settings
from stock_analyzer.data import get_provider
from stock_analyzer.data.yfinance_provider import YFinanceProvider

def test_default_is_yfinance():
    assert isinstance(get_provider(Settings(_env_file=None)), YFinanceProvider)
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/data/__init__.py
from ..config import Settings
from .base import DataProvider, DataUnavailableError
from .yfinance_provider import YFinanceProvider

def get_provider(settings: Settings) -> DataProvider:
    if settings.provider == "fmp" and settings.fmp_api_key:
        from .fmp_provider import FMPProvider
        return FMPProvider(settings.fmp_api_key)
    return YFinanceProvider()
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(data): provider factory (yfinance default, fmp optional)"`

---

## Phase 2 — Metrics (ground truth)

### Task 2.1: MetricValue

**Files:** Create `src/stock_analyzer/metrics/value.py`, `tests/metrics/test_value.py`

- [ ] **Step 1: Failing test**

```python
# tests/metrics/test_value.py
from datetime import date
from stock_analyzer.metrics.value import MetricValue

def test_metric_value_render():
    mv = MetricValue(key="rsi_14", label="RSI (14)", value=66.8, unit="", category="technical", as_of=date(2026,6,5))
    assert mv.display() == "66.80"
    assert MetricValue(key="x", label="x", value=None, unit="$", category="technical", as_of=date(2026,6,5)).display() == "N/A"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/metrics/value.py
from datetime import date
from pydantic import BaseModel

class MetricValue(BaseModel):
    key: str
    label: str
    value: float | None
    unit: str
    category: str
    as_of: date

    def display(self) -> str:
        if self.value is None:
            return "N/A"
        return f"{self.value:.2f}"
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(metrics): MetricValue"`

### Task 2.2: Technical metrics

**Files:** Create `src/stock_analyzer/metrics/technical.py`, `tests/metrics/test_technical.py`, `tests/fixtures/prices.py`

- [ ] **Step 1: Failing test** (deterministic golden values)

```python
# tests/fixtures/prices.py
import numpy as np, pandas as pd
def linear_prices(n=260, start=100.0, step=0.5):
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(start + step*np.arange(n), index=idx)
    return pd.DataFrame({"Open":close, "High":close+1, "Low":close-1, "Close":close, "Volume":1_000_000}, index=idx)
```

```python
# tests/metrics/test_technical.py
from tests.fixtures.prices import linear_prices
from stock_analyzer.metrics.technical import compute_technical

def test_rsi_all_gains_is_100():
    out = compute_technical(linear_prices())
    assert round(out["rsi_14"].value, 1) == 100.0  # strictly rising series

def test_sma_keys_present():
    out = compute_technical(linear_prices())
    for k in ("sma_50","sma_200","macd","signal_line","bb_upper","bb_lower"):
        assert k in out and out[k].value is not None
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement** (port + clean from old `test.py`)

```python
# src/stock_analyzer/metrics/technical.py
from datetime import date
import pandas as pd
from .value import MetricValue

def _mv(key, label, value, unit, as_of):
    return MetricValue(key=key, label=label, value=None if value is None or pd.isna(value) else float(value),
                       unit=unit, category="technical", as_of=as_of)

def compute_technical(bars: pd.DataFrame) -> dict[str, MetricValue]:
    c = bars["Close"]; as_of = bars.index[-1].date()
    sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss; rsi = 100 - 100/(1+rs)
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; signal = macd.ewm(span=9, adjust=False).mean()
    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    out = {
        "last_close": _mv("last_close","Last Close", c.iloc[-1], "$", as_of),
        "sma_50": _mv("sma_50","SMA 50", sma50.iloc[-1], "$", as_of),
        "sma_200": _mv("sma_200","SMA 200", sma200.iloc[-1], "$", as_of),
        "rsi_14": _mv("rsi_14","RSI (14)", rsi.iloc[-1], "", as_of),
        "macd": _mv("macd","MACD", macd.iloc[-1], "", as_of),
        "signal_line": _mv("signal_line","Signal Line", signal.iloc[-1], "", as_of),
        "bb_upper": _mv("bb_upper","Bollinger Upper", (mid+2*sd).iloc[-1], "$", as_of),
        "bb_lower": _mv("bb_lower","Bollinger Lower", (mid-2*sd).iloc[-1], "$", as_of),
        "wk52_high": _mv("wk52_high","52-Week High", c.tail(252).max(), "$", as_of),
        "wk52_low": _mv("wk52_low","52-Week Low", c.tail(252).min(), "$", as_of),
    }
    return out
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(metrics): technical indicators with golden tests"`

### Task 2.3: Fundamental metrics

**Files:** Create `src/stock_analyzer/metrics/fundamental.py`, `tests/metrics/test_fundamental.py`

- [ ] **Step 1: Failing test**

```python
# tests/metrics/test_fundamental.py
from datetime import date
from stock_analyzer.data.models import Fundamentals
from stock_analyzer.metrics.fundamental import compute_fundamental

def test_maps_fundamentals():
    out = compute_fundamental(Fundamentals(symbol="AAPL", pe=28.4, roe=0.3), as_of=date(2026,6,5))
    assert out["pe_ratio"].value == 28.4
    assert out["roe"].value == 0.3 and out["roe"].unit == "%"
    assert out["pe_ratio"].value is not None and out["debt_to_equity"].value is None
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/metrics/fundamental.py
from datetime import date
from ..data.models import Fundamentals
from .value import MetricValue

def compute_fundamental(f: Fundamentals, as_of: date) -> dict[str, MetricValue]:
    def mv(key, label, value, unit):
        return MetricValue(key=key, label=label, value=value, unit=unit, category="fundamental", as_of=as_of)
    return {
        "pe_ratio": mv("pe_ratio","P/E Ratio", f.pe, "x"),
        "pb_ratio": mv("pb_ratio","P/B Ratio", f.pb, "x"),
        "debt_to_equity": mv("debt_to_equity","Debt/Equity", f.debt_to_equity, ""),
        "roe": mv("roe","Return on Equity", f.roe, "%"),
        "eps_growth": mv("eps_growth","EPS Growth (Q)", f.eps_growth, "%"),
        "profit_margin": mv("profit_margin","Profit Margin", f.profit_margin, "%"),
        "revenue_growth": mv("revenue_growth","Revenue Growth", f.revenue_growth, "%"),
        "dividend_yield": mv("dividend_yield","Dividend Yield", f.dividend_yield, "%"),
        "market_cap": mv("market_cap","Market Cap", f.market_cap, "$"),
    }
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(metrics): fundamental ratios"`

### Task 2.4: Risk metrics (incl. modern additions)

**Files:** Create `src/stock_analyzer/metrics/risk.py`, `tests/metrics/test_risk.py`

- [ ] **Step 1: Failing test**

```python
# tests/metrics/test_risk.py
import numpy as np, pandas as pd
from tests.fixtures.prices import linear_prices
from stock_analyzer.metrics.risk import compute_risk

def test_risk_keys_and_drawdown_nonpositive():
    bench = linear_prices()["Close"]
    out = compute_risk(linear_prices(), bench)
    for k in ("beta","hist_vol","atr","max_drawdown","sharpe","var_95","downside_dev"):
        assert k in out
    assert out["max_drawdown"].value <= 0  # drawdown is <= 0
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/metrics/risk.py
import numpy as np, pandas as pd
from .value import MetricValue

def compute_risk(bars: pd.DataFrame, benchmark_close: pd.Series, rf: float = 0.0) -> dict[str, MetricValue]:
    as_of = bars.index[-1].date(); c = bars["Close"]; ret = c.pct_change().dropna()
    def mv(key, label, value, unit):
        return MetricValue(key=key, label=label, value=None if value is None or pd.isna(value) else float(value),
                           unit=unit, category="risk", as_of=as_of)
    bret = benchmark_close.pct_change().reindex(ret.index).dropna(); aligned = ret.reindex(bret.index)
    beta = aligned.cov(bret) / bret.var() if bret.var() else None
    hv = ret.std() * np.sqrt(252)
    tr = pd.concat([bars["High"]-bars["Low"], (bars["High"]-c.shift()).abs(), (bars["Low"]-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    cum = (1+ret).cumprod(); max_dd = (cum/cum.cummax() - 1).min()
    sharpe = ((ret.mean()-rf/252)/ret.std()*np.sqrt(252)) if ret.std() else None
    var95 = -np.percentile(ret, 5)
    downside = ret[ret < 0].std() * np.sqrt(252)
    return {
        "beta": mv("beta","Beta (vs SPY)", beta, ""),
        "hist_vol": mv("hist_vol","Annualized Volatility", hv, "%"),
        "atr": mv("atr","Average True Range", atr, "$"),
        "max_drawdown": mv("max_drawdown","Max Drawdown", max_dd, "%"),
        "sharpe": mv("sharpe","Sharpe Ratio", sharpe, ""),
        "var_95": mv("var_95","Value at Risk (95%)", var95, "%"),
        "downside_dev": mv("downside_dev","Downside Deviation", downside, "%"),
    }
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(metrics): risk incl. max drawdown, Sharpe, VaR, downside dev"`

### Task 2.5: Valuation (DCF + sensitivity)

**Files:** Create `src/stock_analyzer/metrics/valuation.py`, `tests/metrics/test_valuation.py`

- [ ] **Step 1: Failing test**

```python
# tests/metrics/test_valuation.py
from datetime import date
from stock_analyzer.metrics.valuation import simple_dcf, compute_valuation

def test_dcf_growth_increases_value():
    low = simple_dcf(fcf0=100, growth=0.03, wacc=0.10, years=5, terminal_growth=0.02)
    high = simple_dcf(fcf0=100, growth=0.08, wacc=0.10, years=5, terminal_growth=0.02)
    assert high > low > 0

def test_compute_valuation_returns_grid():
    out = compute_valuation(fcf0=100, growth=0.05, wacc=0.10, as_of=date(2026,6,5))
    assert out["dcf_value"].value > 0
    assert isinstance(out["dcf_value"].value, float)
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/metrics/valuation.py
from datetime import date
from .value import MetricValue

def simple_dcf(fcf0: float, growth: float, wacc: float, years: int = 5, terminal_growth: float = 0.02) -> float:
    pv = 0.0; fcf = fcf0
    for t in range(1, years+1):
        fcf *= (1+growth); pv += fcf / (1+wacc)**t
    terminal = fcf*(1+terminal_growth)/(wacc-terminal_growth)
    pv += terminal / (1+wacc)**years
    return pv

def compute_valuation(fcf0: float, growth: float, wacc: float, as_of: date) -> dict[str, MetricValue]:
    val = simple_dcf(fcf0, growth, wacc)
    return {"dcf_value": MetricValue(key="dcf_value", label="DCF Enterprise Value",
            value=val, unit="$", category="valuation", as_of=as_of)}

def sensitivity_grid(fcf0: float, growths: list[float], waccs: list[float]) -> list[list[float]]:
    return [[round(simple_dcf(fcf0, g, w), 2) for w in waccs] for g in growths]
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(metrics): DCF valuation + sensitivity grid"`

### Task 2.6: Sentiment (VADER default)

**Files:** Create `src/stock_analyzer/metrics/sentiment.py`, `tests/metrics/test_sentiment.py`

- [ ] **Step 1: Failing test**

```python
# tests/metrics/test_sentiment.py
from datetime import date
from stock_analyzer.data.models import NewsItem
from stock_analyzer.metrics.sentiment import compute_sentiment

def test_positive_headlines_positive_score():
    news = [NewsItem(title="Company crushes earnings, soars to record high"),
            NewsItem(title="Analysts upgrade with strong buy and optimism")]
    out = compute_sentiment(news, as_of=date(2026,6,5))
    assert out["sentiment_score"].value > 0

def test_no_news_is_none():
    out = compute_sentiment([], as_of=date(2026,6,5))
    assert out["sentiment_score"].value is None
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/metrics/sentiment.py
from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..data.models import NewsItem
from .value import MetricValue

def compute_sentiment(news: list[NewsItem], as_of: date) -> dict[str, MetricValue]:
    if not news:
        score = None
    else:
        sia = SentimentIntensityAnalyzer()
        score = sum(sia.polarity_scores(n.title)["compound"] for n in news) / len(news)
    return {"sentiment_score": MetricValue(key="sentiment_score", label="News Sentiment",
            value=score, unit="", category="sentiment", as_of=as_of)}
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(metrics): VADER headline sentiment"`

### Task 2.7: MetricsBundle assembly

**Files:** Create `src/stock_analyzer/metrics/bundle.py`, `tests/metrics/test_bundle.py`

- [ ] **Step 1: Failing test**

```python
# tests/metrics/test_bundle.py
from tests.fixtures.prices import linear_prices
from stock_analyzer.data.models import Fundamentals, PriceHistory
from stock_analyzer.metrics.bundle import MetricsBundle

def test_bundle_flat_and_prompt_block():
    ph = PriceHistory(symbol="AAPL", period="1y", bars=linear_prices())
    b = MetricsBundle.from_data(ph, Fundamentals(symbol="AAPL", pe=28.4), linear_prices()["Close"],
                                news=[], fcf0=100, growth=0.05, wacc=0.10)
    flat = b.as_flat()
    assert "rsi_14" in flat and "pe_ratio" in flat and "sharpe" in flat
    pb = b.prompt_block()
    assert "RSI (14)" in pb and "P/E Ratio" in pb
    # deterministic ordering (cache-friendly): same output across calls
    assert b.prompt_block() == pb
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/metrics/bundle.py
import pandas as pd
from pydantic import BaseModel
from ..data.models import PriceHistory, Fundamentals, NewsItem
from .value import MetricValue
from .technical import compute_technical
from .fundamental import compute_fundamental
from .risk import compute_risk
from .valuation import compute_valuation
from .sentiment import compute_sentiment

class MetricsBundle(BaseModel):
    symbol: str
    period: str
    metrics: dict[str, MetricValue]

    @classmethod
    def from_data(cls, ph: PriceHistory, fundamentals: Fundamentals, benchmark_close: pd.Series,
                  news: list[NewsItem], fcf0: float, growth: float, wacc: float) -> "MetricsBundle":
        as_of = ph.bars.index[-1].date()
        m: dict[str, MetricValue] = {}
        m.update(compute_technical(ph.bars))
        m.update(compute_fundamental(fundamentals, as_of))
        m.update(compute_risk(ph.bars, benchmark_close))
        m.update(compute_valuation(fcf0, growth, wacc, as_of))
        m.update(compute_sentiment(news, as_of))
        return cls(symbol=ph.symbol, period=ph.period, metrics=m)

    def as_flat(self) -> dict[str, MetricValue]:
        return self.metrics

    def prompt_block(self) -> str:
        lines = [f"# Ground-truth metrics for {self.symbol} ({self.period})"]
        for key in sorted(self.metrics):  # sorted = deterministic = cache-friendly
            mv = self.metrics[key]
            lines.append(f"- {mv.label} [{mv.key}]: {mv.display()} {mv.unit}".rstrip())
        return "\n".join(lines)
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(metrics): MetricsBundle ground-truth (as_flat, prompt_block)"`

---

## Phase 3 — Charts

### Task 3.1: Design tokens + matplotlib theme

**Files:** Create `src/stock_analyzer/tokens.py`, `src/stock_analyzer/charts/theme.py`, `tests/charts/test_theme.py`

- [ ] **Step 1: Failing test**

```python
# tests/charts/test_theme.py
from stock_analyzer.tokens import COLORS, FONTS
from stock_analyzer.charts.theme import apply_theme

def test_tokens_present():
    assert COLORS["accent"] == "#2347D9" and COLORS["up"] == "#16a34a"
    assert "mono" in FONTS

def test_apply_theme_runs():
    apply_theme()  # should not raise
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/tokens.py
COLORS = {
    "bg": "#FFFFFF", "panel": "#0D1117", "ink": "#0B0F19", "muted": "#9ca3af",
    "hairline": "#E5E7EB", "accent": "#2347D9", "up": "#16a34a", "down": "#ef4444",
    "caution": "#D29922", "data_text": "#E6EDF3", "verified": "#047857",
}
FONTS = {"sans": "Space Grotesk, system-ui, sans-serif", "mono": "JetBrains Mono, ui-monospace, monospace"}
```

```python
# src/stock_analyzer/charts/theme.py
import matplotlib as mpl
from ..tokens import COLORS

def apply_theme() -> None:
    mpl.rcParams.update({
        "figure.facecolor": COLORS["bg"], "axes.facecolor": COLORS["bg"],
        "axes.edgecolor": COLORS["hairline"], "axes.grid": True,
        "grid.color": COLORS["hairline"], "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"], "font.size": 9,
    })
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(charts): design tokens + matplotlib theme"`

### Task 3.2: Chart builders → ChartRef

**Files:** Create `src/stock_analyzer/charts/builders.py`, `tests/charts/test_builders.py`

- [ ] **Step 1: Failing test**

```python
# tests/charts/test_builders.py
from pathlib import Path
from tests.fixtures.prices import linear_prices
from stock_analyzer.charts.builders import build_charts

def test_build_charts_writes_images_and_facts(tmp_path):
    refs = build_charts(linear_prices(), out_dir=tmp_path)
    names = {r.name for r in refs}
    assert {"price","rsi","macd","returns_dist","drawdown"} <= names
    for r in refs:
        assert Path(r.image_path).exists()
        assert isinstance(r.fact, dict)
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement** (each builder is small; all enumerated — no "similar to")

```python
# src/stock_analyzer/charts/builders.py
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pydantic import BaseModel
from ..tokens import COLORS
from .theme import apply_theme

class ChartRef(BaseModel):
    name: str
    image_path: str
    caption: str
    fact: dict

def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return str(path)

def build_charts(bars: pd.DataFrame, out_dir) -> list[ChartRef]:
    apply_theme(); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    c = bars["Close"]; refs: list[ChartRef] = []

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(c.index, c, color=COLORS["accent"], lw=1.5, label="Close")
    ax.plot(c.index, c.rolling(50).mean(), color=COLORS["caution"], lw=1, label="SMA50")
    ax.plot(c.index, c.rolling(200).mean(), color=COLORS["down"], lw=1, label="SMA200")
    ax.legend(); ax.set_title("Price & Moving Averages")
    refs.append(ChartRef(name="price", image_path=_save(fig, out/"price.png"),
                         caption="Price with SMA50/SMA200", fact={"last_close": round(float(c.iloc[-1]),2)}))

    delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100-100/(1+gain/loss)
    fig, ax = plt.subplots(figsize=(9,2.5)); ax.plot(rsi.index, rsi, color=COLORS["caution"])
    ax.axhline(70, ls="--", color=COLORS["down"]); ax.axhline(30, ls="--", color=COLORS["up"]); ax.set_title("RSI (14)")
    refs.append(ChartRef(name="rsi", image_path=_save(fig, out/"rsi.png"),
                         caption="RSI (14)", fact={"latest_rsi": round(float(rsi.iloc[-1]),1)}))

    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26; sig=macd.ewm(span=9,adjust=False).mean()
    fig, ax = plt.subplots(figsize=(9,2.5)); ax.bar(macd.index,(macd-sig),color=COLORS["accent"]); ax.plot(macd.index,macd,color=COLORS["ink"])
    ax.set_title("MACD")
    refs.append(ChartRef(name="macd", image_path=_save(fig, out/"macd.png"),
                         caption="MACD vs signal", fact={"macd": round(float(macd.iloc[-1]),3)}))

    ret = c.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(6,3)); ax.hist(ret, bins=40, color=COLORS["accent"], alpha=0.8); ax.set_title("Returns Distribution")
    refs.append(ChartRef(name="returns_dist", image_path=_save(fig, out/"returns_dist.png"),
                         caption="Daily returns distribution", fact={"mean_daily": round(float(ret.mean()),5)}))

    cum=(1+ret).cumprod(); dd=cum/cum.cummax()-1
    fig, ax = plt.subplots(figsize=(9,2.5)); ax.fill_between(dd.index, dd, color=COLORS["down"], alpha=0.4); ax.set_title("Drawdown")
    refs.append(ChartRef(name="drawdown", image_path=_save(fig, out/"drawdown.png"),
                         caption="Drawdown curve", fact={"max_drawdown": round(float(dd.min()),4)}))
    return refs
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(charts): builders for price/RSI/MACD/returns/drawdown"`

---

## Phase 4 — Agents & graph

### Task 4.0: Integration spike (validate Claude + structured output) — NO network in CI

**Files:** Create `src/stock_analyzer/agents/llm.py`, `tests/agents/test_llm.py`
**Purpose:** lock the `langchain-anthropic` + Opus 4.8 + structured-output call shape behind an injectable factory before building nodes. Real-API call is a manual/integration check (marked, skipped by default); unit test uses a fake.

- [ ] **Step 1: Failing test**

```python
# tests/agents/test_llm.py
from stock_analyzer.agents.llm import grounding_block
from stock_analyzer.metrics.bundle import MetricsBundle
from stock_analyzer.metrics.value import MetricValue
from datetime import date

def test_grounding_block_includes_metrics():
    b = MetricsBundle(symbol="AAPL", period="1y",
        metrics={"rsi_14": MetricValue(key="rsi_14",label="RSI (14)",value=66.8,unit="",category="technical",as_of=date(2026,6,5))})
    block = grounding_block(b)
    assert "RSI (14)" in block and "only use" in block.lower()
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/agents/llm.py
from langchain_anthropic import ChatAnthropic
from ..config import Settings
from ..metrics.bundle import MetricsBundle

def make_llm(settings: Settings):
    # Opus 4.8: adaptive thinking + effort. Verify kwargs against installed langchain-anthropic.
    return ChatAnthropic(model=settings.model, max_tokens=8000, anthropic_api_key=settings.anthropic_api_key,
                         model_kwargs={"thinking": {"type": "adaptive"},
                                       "output_config": {"effort": settings.effort}})

def structured(llm, schema):
    return llm.with_structured_output(schema)

def grounding_block(bundle: MetricsBundle) -> str:
    return ("You are grounded ONLY in the metrics below. Use only these values; "
            "if a value is N/A, say so — never estimate or invent numbers.\n\n" + bundle.prompt_block())
```

- [ ] **Step 4: Run / pass** → PASS. (Manual: a `@pytest.mark.integration` test that does one real `make_llm` call is added but excluded from default `pytest` via marker; run it once by hand to confirm the kwargs are accepted by the installed SDK, and adjust `model_kwargs` if the version expects top-level `thinking`/`output_config`.)
- [ ] **Step 5: Commit** — `git commit -am "feat(agents): llm factory + grounding block (+integration spike)"`

### Task 4.1: State + finding/draft models

**Files:** Create `src/stock_analyzer/agents/state.py`, `tests/agents/test_state.py`

- [ ] **Step 1: Failing test**

```python
# tests/agents/test_state.py
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection

def test_models_construct():
    f = AnalystFinding(summary="s", key_points=["a"], outlook="bullish", rationale="r", cited_metrics=["rsi_14"])
    assert f.outlook == "bullish"
    d = ReportDraft(sections=[ReportSection(id="technical", prose="p", referenced_metrics=["rsi_14"], charts=["rsi"])],
                    recommendation="buy", confidence="high")
    assert d.recommendation == "buy"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/agents/state.py
from typing import TypedDict, Literal
from pydantic import BaseModel

class AnalystFinding(BaseModel):
    summary: str
    key_points: list[str]
    outlook: Literal["bullish","bearish","neutral"]
    rationale: str
    cited_metrics: list[str]

class ReportSection(BaseModel):
    id: str
    prose: str
    referenced_metrics: list[str]
    charts: list[str]

class ReportDraft(BaseModel):
    sections: list[ReportSection]
    recommendation: Literal["buy","hold","sell"]
    confidence: Literal["low","medium","high"]

class AnalysisState(TypedDict, total=False):
    symbol: str
    metrics_block: str           # cached grounding text
    fundamental: AnalystFinding
    technical: AnalystFinding
    risk: AnalystFinding
    valuation: AnalystFinding
    report: ReportDraft
    claims: list
    verdicts: list
    revisions: int
    verdict_feedback: str | None
    audit: dict
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(agents): graph state + structured models"`

### Task 4.2: Analyst node factory

**Files:** Create `src/stock_analyzer/agents/analysts.py`, `tests/agents/test_analysts.py`

- [ ] **Step 1: Failing test** (fake structured LLM)

```python
# tests/agents/test_analysts.py
from stock_analyzer.agents.analysts import make_analyst_node
from stock_analyzer.agents.state import AnalystFinding

class FakeStructured:
    def invoke(self, msgs): return AnalystFinding(summary="ok", key_points=["RSI high"],
        outlook="bullish", rationale="r", cited_metrics=["rsi_14"])

def test_analyst_node_writes_its_key():
    node = make_analyst_node("technical", lambda schema: FakeStructured())
    out = node({"metrics_block": "RSI (14): 66.8", "symbol": "AAPL"})
    assert "technical" in out and out["technical"].outlook == "bullish"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/agents/analysts.py
from .state import AnalystFinding

PROMPTS = {
    "fundamental": "You are a fundamental analyst. Assess valuation, profitability, growth, leverage.",
    "technical": "You are a technical analyst. Assess trend, momentum (RSI/MACD), Bollinger, support/resistance.",
    "risk": "You are a risk analyst. Assess beta, volatility, drawdown, Sharpe, VaR.",
    "valuation": "You are a valuation analyst. Assess DCF value, multiples, and a price target.",
}

def make_analyst_node(role: str, structured_factory):
    chain = structured_factory(AnalystFinding)
    def node(state):
        msgs = [{"role":"system","content": PROMPTS[role]},
                {"role":"user","content": state["metrics_block"] +
                 f"\n\nAnalyze {state['symbol']}. Cite each metric key you use."}]
        return {role: chain.invoke(msgs)}
    return node
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(agents): analyst node factory"`

### Task 4.3: Writer node

**Files:** Create `src/stock_analyzer/agents/writer.py`, `tests/agents/test_writer.py`

- [ ] **Step 1: Failing test**

```python
# tests/agents/test_writer.py
from stock_analyzer.agents.writer import make_writer_node
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection

class FakeStructured:
    def invoke(self, msgs):
        return ReportDraft(sections=[ReportSection(id="technical", prose="p", referenced_metrics=["rsi_14"], charts=["rsi"])],
                           recommendation="buy", confidence="high")

def test_writer_emits_report():
    node = make_writer_node(lambda schema: FakeStructured())
    f = AnalystFinding(summary="s", key_points=[], outlook="bullish", rationale="r", cited_metrics=[])
    out = node({"metrics_block":"...","symbol":"AAPL","fundamental":f,"technical":f,"risk":f,"valuation":f})
    assert isinstance(out["report"], ReportDraft) and out["report"].recommendation == "buy"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/agents/writer.py
from .state import ReportDraft

def make_writer_node(structured_factory):
    chain = structured_factory(ReportDraft)
    def node(state):
        findings = "\n\n".join(f"## {r}\n{getattr(state[r], 'summary', '')}\n{getattr(state[r],'rationale','')}"
                               for r in ("fundamental","technical","risk","valuation") if r in state)
        feedback = state.get("verdict_feedback")
        extra = f"\n\nFIX THESE VERIFICATION ISSUES (correct only the affected sentences):\n{feedback}" if feedback else ""
        msgs = [{"role":"system","content":"You are an equity report writer. Use only grounded metrics; "
                 "produce sections: exec_summary, overview, technical, fundamental, risk, valuation, recommendation."},
                {"role":"user","content": state["metrics_block"] + "\n\nAnalyst findings:\n" + findings + extra}]
        return {"report": chain.invoke(msgs)}
    return node
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(agents): writer node (handles revision feedback)"`

### Task 4.4: Graph wiring (no verification yet)

**Files:** Create `src/stock_analyzer/agents/graph.py`, `tests/agents/test_graph_smoke.py`

- [ ] **Step 1: Failing test** (fake LLM; assert parallel analysts + writer run)

```python
# tests/agents/test_graph_smoke.py
from stock_analyzer.agents.graph import build_graph
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection

def fake_factory(schema):
    class F:
        def invoke(self, msgs):
            if schema is ReportDraft:
                return ReportDraft(sections=[ReportSection(id="technical",prose="p",referenced_metrics=[],charts=[])],
                                   recommendation="hold", confidence="medium")
            return AnalystFinding(summary="s",key_points=[],outlook="neutral",rationale="r",cited_metrics=[])
    return F()

def test_graph_runs_to_report():
    app = build_graph(fake_factory, verify=False)
    out = app.invoke({"symbol":"AAPL","metrics_block":"RSI (14): 66.8","revisions":0})
    assert out["report"].recommendation == "hold"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/agents/graph.py
from langgraph.graph import StateGraph, START, END
from .state import AnalysisState
from .analysts import make_analyst_node
from .writer import make_writer_node

def build_graph(structured_factory, verify: bool = True):
    g = StateGraph(AnalysisState)
    for role in ("fundamental","technical","risk","valuation"):
        g.add_node(role, make_analyst_node(role, structured_factory))
        g.add_edge(START, role)            # parallel fan-out from START
        g.add_edge(role, "writer")         # join at writer (waits for all 4)
    g.add_node("writer", make_writer_node(structured_factory))
    if not verify:
        g.add_edge("writer", END)
        return g.compile()
    # verification nodes + gate are wired in Phase 5 (Task 5.6)
    from ..verification.extract import make_extract_node
    from ..verification.judge import make_verify_node
    from ..verification.reconcile import gate
    g.add_node("extract_claims", make_extract_node(structured_factory))
    g.add_node("verify_claims", make_verify_node(structured_factory))
    g.add_edge("writer", "extract_claims")
    g.add_edge("extract_claims", "verify_claims")
    g.add_conditional_edges("verify_claims", gate, {"revise": "writer", "finalize": END})
    return g.compile()
```

- [ ] **Step 4: Run / pass** → PASS (run with `verify=False` until Phase 5 lands; the import of verification modules is inside the `verify` branch so the smoke test passes before Phase 5)
- [ ] **Step 5: Commit** — `git commit -am "feat(agents): LangGraph fan-out + writer (verify branch stubbed)"`

---

## Phase 5 — Verification gate

### Task 5.1: Claim/Verdict/Audit models

**Files:** Create `src/stock_analyzer/verification/models.py`, `tests/verification/test_models.py`

- [ ] **Step 1: Failing test**

```python
# tests/verification/test_models.py
from stock_analyzer.verification.models import Claim, Verdict, VerificationAudit

def test_construct():
    c = Claim(id="c1", text="RSI is 66.8", metric_key="rsi_14", claimed_value=66.8, claim_type="numeric", source_section="technical")
    v = Verdict(claim_id="c1", status="supported", expected_value=66.8, claimed_value=66.8, delta=0.0, rationale="match", correction=None)
    a = VerificationAudit(total_claims=1, supported=1, contradicted=0, unsupported=0, corrections_applied=[], residual_unverified=[])
    assert c.claim_type=="numeric" and v.status=="supported" and a.total_claims==1
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/verification/models.py
from typing import Literal
from pydantic import BaseModel

class Claim(BaseModel):
    id: str
    text: str
    metric_key: str | None = None
    claimed_value: float | None = None
    claim_type: Literal["numeric","directional","categorical","qualitative"]
    source_section: str

class Verdict(BaseModel):
    claim_id: str
    status: Literal["supported","contradicted","unsupported"]
    expected_value: float | None = None
    claimed_value: float | None = None
    delta: float | None = None
    rationale: str
    correction: str | None = None

class ClaimList(BaseModel):
    claims: list[Claim]

class VerificationAudit(BaseModel):
    total_claims: int
    supported: int
    contradicted: int
    unsupported: int
    corrections_applied: list[Verdict]
    residual_unverified: list[Verdict]
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(verify): Claim/Verdict/Audit models"`

### Task 5.2: Programmatic reconciliation + gate

**Files:** Create `src/stock_analyzer/verification/reconcile.py`, `tests/verification/test_reconcile.py`

- [ ] **Step 1: Failing test**

```python
# tests/verification/test_reconcile.py
from datetime import date
from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.verification.models import Claim
from stock_analyzer.verification.reconcile import reconcile_numeric, gate

GT = {"rsi_14": MetricValue(key="rsi_14",label="RSI",value=66.8,unit="",category="technical",as_of=date(2026,6,5))}

def test_supported_within_tolerance():
    c = Claim(id="c1", text="RSI ~ 66.8", metric_key="rsi_14", claimed_value=66.9, claim_type="numeric", source_section="technical")
    v = reconcile_numeric(c, GT, tol_rel=0.01, tol_abs=0.05)
    assert v.status == "supported"

def test_contradicted_outside_tolerance():
    c = Claim(id="c2", text="RSI is 80", metric_key="rsi_14", claimed_value=80.0, claim_type="numeric", source_section="technical")
    v = reconcile_numeric(c, GT, tol_rel=0.01, tol_abs=0.05)
    assert v.status == "contradicted" and "66.8" in (v.correction or "")

def test_gate_routes_revise_then_finalize():
    assert gate({"verdicts":[type("V",(),{"status":"contradicted"})()], "revisions":0, "max_revisions":2}) == "revise"
    assert gate({"verdicts":[], "revisions":0, "max_revisions":2}) == "finalize"
    assert gate({"verdicts":[type("V",(),{"status":"contradicted"})()], "revisions":2, "max_revisions":2}) == "finalize"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/verification/reconcile.py
from ..metrics.value import MetricValue
from .models import Claim, Verdict

def reconcile_numeric(claim: Claim, ground_truth: dict[str, MetricValue], tol_rel: float, tol_abs: float) -> Verdict:
    mv = ground_truth.get(claim.metric_key) if claim.metric_key else None
    if mv is None or mv.value is None or claim.claimed_value is None:
        return Verdict(claim_id=claim.id, status="unsupported", rationale="No matching ground-truth metric")
    expected = mv.value; delta = claim.claimed_value - expected
    tol = max(tol_abs, abs(expected)*tol_rel)
    if abs(delta) <= tol:
        return Verdict(claim_id=claim.id, status="supported", expected_value=expected,
                       claimed_value=claim.claimed_value, delta=delta, rationale="within tolerance")
    return Verdict(claim_id=claim.id, status="contradicted", expected_value=expected, claimed_value=claim.claimed_value,
                   delta=delta, rationale="outside tolerance",
                   correction=f"{mv.label} is {expected:.2f}, not {claim.claimed_value:.2f}")

def gate(state) -> str:
    verdicts = state.get("verdicts", []) or []
    contradictions = sum(1 for v in verdicts if getattr(v, "status", None) == "contradicted")
    if contradictions and state.get("revisions", 0) < state.get("max_revisions", 2):
        return "revise"
    return "finalize"
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(verify): numeric reconciliation + gate logic"`

### Task 5.3: extract_claims node

**Files:** Create `src/stock_analyzer/verification/extract.py`, `tests/verification/test_extract.py`

- [ ] **Step 1: Failing test** (fake structured LLM returns ClaimList)

```python
# tests/verification/test_extract.py
from stock_analyzer.verification.extract import make_extract_node
from stock_analyzer.verification.models import ClaimList, Claim
from stock_analyzer.agents.state import ReportDraft, ReportSection

class Fake:
    def invoke(self, msgs):
        return ClaimList(claims=[Claim(id="c1", text="RSI is 66.8", metric_key="rsi_14",
                                       claimed_value=66.8, claim_type="numeric", source_section="technical")])

def test_extract_populates_claims():
    node = make_extract_node(lambda schema: Fake())
    draft = ReportDraft(sections=[ReportSection(id="technical", prose="RSI is 66.8", referenced_metrics=["rsi_14"], charts=[])],
                        recommendation="buy", confidence="high")
    out = node({"report": draft})
    assert out["claims"][0].metric_key == "rsi_14"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/verification/extract.py
from .models import ClaimList

def make_extract_node(structured_factory):
    chain = structured_factory(ClaimList)
    def node(state):
        text = "\n\n".join(f"[{s.id}] {s.prose}" for s in state["report"].sections)
        msgs = [{"role":"system","content":"Extract every atomic factual/numeric/directional claim. "
                 "For numeric claims set metric_key (the referenced ground-truth key) and claimed_value."},
                {"role":"user","content": text}]
        return {"claims": chain.invoke(msgs).claims}
    return node
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(verify): claim-extraction node"`

### Task 5.4: verify node (programmatic + grounded judge) + audit

**Files:** Create `src/stock_analyzer/verification/judge.py`, `tests/verification/test_judge.py`

- [ ] **Step 1: Failing test**

```python
# tests/verification/test_judge.py
from datetime import date
from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.verification.models import Claim, Verdict
from stock_analyzer.verification.judge import make_verify_node

GT = {"rsi_14": MetricValue(key="rsi_14",label="RSI",value=66.8,unit="",category="technical",as_of=date(2026,6,5))}

class FakeJudge:
    def invoke(self, msgs): return Verdict(claim_id="d1", status="supported", rationale="ok")

def test_numeric_goes_programmatic_directional_goes_judge():
    node = make_verify_node(lambda schema: FakeJudge(), ground_truth=GT, tol_rel=0.01, tol_abs=0.05)
    claims = [
        Claim(id="n1", text="RSI 80", metric_key="rsi_14", claimed_value=80.0, claim_type="numeric", source_section="technical"),
        Claim(id="d1", text="RSI signals overbought", metric_key=None, claim_type="directional", source_section="technical"),
    ]
    out = node({"claims": claims, "revisions": 0})
    by = {v.claim_id: v for v in out["verdicts"]}
    assert by["n1"].status == "contradicted"   # programmatic
    assert by["d1"].status == "supported"       # judged
    assert out["audit"]["contradicted"] == 1 and out["revisions"] == 1
    assert isinstance(out["verdict_feedback"], str)
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/verification/judge.py
from .models import Verdict, VerificationAudit
from .reconcile import reconcile_numeric

def make_verify_node(structured_factory, ground_truth, tol_rel, tol_abs):
    judge = structured_factory(Verdict)
    def node(state):
        verdicts = []
        for c in state["claims"]:
            if c.claim_type == "numeric" and c.metric_key:
                verdicts.append(reconcile_numeric(c, ground_truth, tol_rel, tol_abs))
            else:
                msgs = [{"role":"system","content":"Judge the claim using ONLY the ground-truth metrics. "
                         "Return supported/unsupported/contradicted + a correction if contradicted."},
                        {"role":"user","content": f"Claim: {c.text}\n\nGround truth keys/values:\n" +
                         "\n".join(f"{k}={v.display()}" for k,v in ground_truth.items())}]
                v = judge.invoke(msgs); v.claim_id = c.id; verdicts.append(v)
        contradicted = [v for v in verdicts if v.status=="contradicted"]
        audit = VerificationAudit(total_claims=len(verdicts),
            supported=sum(v.status=="supported" for v in verdicts),
            contradicted=len(contradicted), unsupported=sum(v.status=="unsupported" for v in verdicts),
            corrections_applied=contradicted, residual_unverified=[]).model_dump()
        feedback = "\n".join(f"- {v.correction or v.rationale}" for v in contradicted) or None
        return {"verdicts": verdicts, "audit": audit, "revisions": state.get("revisions",0)+1,
                "verdict_feedback": feedback}
    return node
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(verify): verify node (programmatic + grounded judge) + audit"`

### Task 5.5: Wire verification into the graph (gate + revision loop)

**Files:** Modify `src/stock_analyzer/agents/graph.py`, Create `tests/agents/test_graph_verify.py`

- [ ] **Step 1: Failing test** (contradiction on pass 1 → revise → clean on pass 2)

```python
# tests/agents/test_graph_verify.py
from datetime import date
from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.agents.graph import build_graph
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection
from stock_analyzer.verification.models import ClaimList, Claim, Verdict

GT = {"rsi_14": MetricValue(key="rsi_14",label="RSI",value=66.8,unit="",category="technical",as_of=date(2026,6,5))}

def make_fake_factory():
    state = {"pass": 0}
    def factory(schema):
        class F:
            def invoke(self, msgs):
                if schema is ReportDraft:
                    return ReportDraft(sections=[ReportSection(id="technical",prose="x",referenced_metrics=[],charts=[])],
                                       recommendation="hold", confidence="medium")
                if schema is ClaimList:
                    state["pass"] += 1
                    val = 80.0 if state["pass"] == 1 else 66.8   # wrong first, correct second
                    return ClaimList(claims=[Claim(id="c1", text=f"RSI {val}", metric_key="rsi_14",
                                     claimed_value=val, claim_type="numeric", source_section="technical")])
                if schema is Verdict:
                    return Verdict(claim_id="x", status="supported", rationale="ok")
                return AnalystFinding(summary="s",key_points=[],outlook="neutral",rationale="r",cited_metrics=[])
        return F()
    return factory

def test_revision_loop_resolves():
    app = build_graph(make_fake_factory(), verify=True, ground_truth=GT, max_revisions=2)
    out = app.invoke({"symbol":"AAPL","metrics_block":"RSI (14): 66.8","revisions":0})
    assert out["audit"]["contradicted"] == 0  # resolved after one revision
    assert out["revisions"] == 2
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Modify `build_graph`** to accept `ground_truth`, `max_revisions`, inject them into the verify node and `gate`, and seed `max_revisions` into state.

```python
# graph.py (verify branch, replacing the Phase-4 stub)
def build_graph(structured_factory, verify=True, ground_truth=None, max_revisions=2):
    ...
    from ..verification.extract import make_extract_node
    from ..verification.judge import make_verify_node
    from ..verification.reconcile import gate
    g.add_node("extract_claims", make_extract_node(structured_factory))
    g.add_node("verify_claims", make_verify_node(structured_factory, ground_truth, 0.01, 0.05))
    def gate_with_cap(state):  # inject cap
        return gate({**state, "max_revisions": max_revisions})
    g.add_edge("writer", "extract_claims")
    g.add_edge("extract_claims", "verify_claims")
    g.add_conditional_edges("verify_claims", gate_with_cap, {"revise": "writer", "finalize": END})
    return g.compile()
```

(Seed `revisions: 0` from the caller; `verdict_feedback` flows back to the writer automatically via state.)

- [ ] **Step 4: Run / pass** → PASS (also re-run `test_graph_smoke.py`)
- [ ] **Step 5: Commit** — `git commit -am "feat(agents): wire verification gate + revision loop"`

---

## Phase 6 — PDF rendering

### Task 6.1: Render context assembler

**Files:** Create `src/stock_analyzer/report/assemble.py`, `tests/report/test_assemble.py`

- [ ] **Step 1: Failing test**

```python
# tests/report/test_assemble.py
from datetime import date
from stock_analyzer.metrics.value import MetricValue
from stock_analyzer.agents.state import ReportDraft, ReportSection
from stock_analyzer.report.assemble import build_context

def test_context_has_sections_kpis_audit():
    draft = ReportDraft(sections=[ReportSection(id="technical",prose="p",referenced_metrics=["rsi_14"],charts=["rsi"])],
                        recommendation="buy", confidence="high")
    gt = {"rsi_14": MetricValue(key="rsi_14",label="RSI",value=66.8,unit="",category="technical",as_of=date(2026,6,5))}
    ctx = build_context(symbol="AAPL", draft=draft, ground_truth=gt,
                        audit={"total_claims":1,"supported":1,"contradicted":0,"unsupported":0,
                               "corrections_applied":[],"residual_unverified":[]},
                        charts={"rsi":"out/rsi.png"})
    assert ctx["symbol"]=="AAPL" and ctx["recommendation"]=="buy"
    assert ctx["kpis"][0]["label"]=="RSI" and ctx["audit"]["supported"]==1
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/report/assemble.py
HERO_KEYS = ["last_close","pe_ratio","rsi_14","beta","sharpe","max_drawdown","var_95"]

def build_context(symbol, draft, ground_truth, audit, charts):
    kpis = [{"label": ground_truth[k].label, "value": ground_truth[k].display(), "unit": ground_truth[k].unit}
            for k in HERO_KEYS if k in ground_truth]
    return {
        "symbol": symbol,
        "recommendation": draft.recommendation,
        "confidence": draft.confidence,
        "sections": [{"id":s.id, "prose":s.prose, "charts":[charts.get(c) for c in s.charts if charts.get(c)]}
                     for s in draft.sections],
        "kpis": kpis,
        "audit": audit,
    }
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(report): render-context assembler"`

### Task 6.2: Jinja2 templates + WeasyPrint PDF

**Files:** Create `src/stock_analyzer/report/templates/report.html.j2`, `templates/styles.css`, `src/stock_analyzer/report/pdf.py`, `tests/report/test_pdf.py`

- [ ] **Step 1: Failing test**

```python
# tests/report/test_pdf.py
from stock_analyzer.report.pdf import render_pdf

def test_pdf_bytes_produced(tmp_path):
    ctx = {"symbol":"AAPL","recommendation":"buy","confidence":"high",
           "sections":[{"id":"technical","prose":"RSI is 66.8","charts":[]}],
           "kpis":[{"label":"RSI","value":"66.80","unit":""}],
           "audit":{"total_claims":1,"supported":1,"contradicted":0,"unsupported":0,
                    "corrections_applied":[],"residual_unverified":[]}}
    out = tmp_path/"r.pdf"; render_pdf(ctx, out)
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes()[:4] == b"%PDF"
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement** (template references tokens via CSS; minimal but real)

```jinja
{# templates/report.html.j2 #}
<!DOCTYPE html><html><head><meta charset="utf-8"><style>{{ css }}</style></head><body>
<header><div class="label">EQUITY ANALYSIS</div><h1>{{ symbol }}</h1>
<span class="pill">{{ recommendation|upper }} · {{ confidence }}</span></header>
<div class="strip">{% for k in kpis %}<div class="kpi"><div class="k-label">{{ k.label }}</div>
<div class="k-val">{{ k.value }}{{ k.unit }}</div></div>{% endfor %}</div>
{% for s in sections %}<section><h2>{{ s.id|replace("_"," ")|title }}</h2><p>{{ s.prose }}</p>
{% for img in s.charts %}<img src="file://{{ img }}"/>{% endfor %}</section>{% endfor %}
<section class="verify"><h2>Verification</h2><p>{{ audit.supported }} supported · {{ audit.contradicted }} contradicted ·
{{ audit.unsupported }} unsupported (of {{ audit.total_claims }}).</p></section>
</body></html>
```

```css
/* templates/styles.css */
body{font-family:'Space Grotesk',system-ui,sans-serif;color:#0B0F19;margin:32px}
.label{letter-spacing:2px;color:#2347D9;font-size:10px;font-weight:700}
h1{font-size:28px;margin:4px 0} h2{font-size:16px;border-left:4px solid #2347D9;padding-left:8px}
.pill{background:#2347D9;color:#fff;border-radius:999px;padding:3px 10px;font-size:11px}
.strip{background:#0D1117;color:#E6EDF3;border-radius:10px;padding:12px;display:flex;gap:18px;font-family:'JetBrains Mono',monospace;margin:16px 0}
.k-label{color:#8B949E;font-size:9px} .k-val{font-size:15px}
img{width:100%;margin:8px 0} .verify{color:#047857}
```

```python
# src/stock_analyzer/report/pdf.py
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

_TPL = Path(__file__).parent / "templates"

def render_pdf(context: dict, out_path) -> None:
    env = Environment(loader=FileSystemLoader(str(_TPL)))
    css = (_TPL / "styles.css").read_text()
    html = env.get_template("report.html.j2").render(css=css, **context)
    HTML(string=html, base_url=str(_TPL)).write_pdf(str(out_path))
```

- [ ] **Step 4: Run / pass** → PASS (requires native libs from Task 0.1 macOS note)
- [ ] **Step 5: Commit** — `git commit -am "feat(report): Jinja2 template + WeasyPrint PDF"`

---

## Phase 7 — Interactive HTML dashboard

### Task 7.1: Plotly dashboard (self-contained)

**Files:** Create `src/stock_analyzer/report/html_dashboard.py`, `tests/report/test_dashboard.py`

- [ ] **Step 1: Failing test**

```python
# tests/report/test_dashboard.py
from tests.fixtures.prices import linear_prices
from stock_analyzer.report.html_dashboard import render_dashboard

def test_dashboard_self_contained(tmp_path):
    out = tmp_path/"dash.html"
    render_dashboard(symbol="AAPL", bars=linear_prices(),
                     kpis=[{"label":"RSI","value":"66.80","unit":""}],
                     verdict_rows=[{"text":"RSI is 66.8","status":"supported"}], out_path=out)
    html = out.read_text()
    assert out.exists() and "plotly" in html.lower()
    assert "AAPL" in html and "sidebar" in html.lower()
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement** (candlestick + RSI subplot; sidebar + data strip + verdict table; inline Plotly)

```python
# src/stock_analyzer/report/html_dashboard.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..tokens import COLORS

def render_dashboard(symbol, bars, kpis, verdict_rows, out_path) -> None:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7,0.3], shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=bars.index, open=bars.Open, high=bars.High, low=bars.Low, close=bars.Close,
                  increasing_line_color=COLORS["up"], decreasing_line_color=COLORS["down"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bars.index, y=bars.Close.rolling(50).mean(), line=dict(color=COLORS["accent"]), name="SMA50"), row=1, col=1)
    ret = bars.Close.pct_change()
    fig.add_trace(go.Bar(x=bars.index, y=ret, marker_color=COLORS["accent"], name="Daily return"), row=2, col=1)
    fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=True, height=560,
                      font=dict(family="JetBrains Mono"), margin=dict(l=20,r=20,t=20,b=20))
    chart_html = fig.to_html(full_html=False, include_plotlyjs="inline")

    strip = "".join(f'<div class="kpi"><div class="kl">{k["label"]}</div><div class="kv">{k["value"]}{k["unit"]}</div></div>' for k in kpis)
    rows = "".join(f'<tr><td>{r["text"]}</td><td class="st-{r["status"]}">{r["status"]}</td></tr>' for r in verdict_rows)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{symbol}</title><style>
    body{{margin:0;font-family:'Space Grotesk',system-ui,sans-serif;color:{COLORS['ink']}}}
    .layout{{display:flex}} .sidebar{{width:150px;background:#FAFBFC;border-right:1px solid {COLORS['hairline']};padding:14px;min-height:100vh}}
    .sidebar a{{display:block;padding:8px 0;color:#6b7280;text-decoration:none;font-size:13px}}
    .main{{flex:1;padding:18px}} .strip{{background:{COLORS['panel']};color:{COLORS['data_text']};border-radius:10px;padding:12px;display:flex;gap:18px;font-family:'JetBrains Mono',monospace}}
    .kl{{color:#8B949E;font-size:9px}} .kv{{font-size:15px}}
    table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}} td{{padding:6px;border-bottom:1px solid {COLORS['hairline']}}}
    .st-supported{{color:{COLORS['verified']}}} .st-contradicted{{color:{COLORS['down']}}} .st-corrected{{color:{COLORS['caution']}}}
    </style></head><body><div class="layout">
    <nav class="sidebar"><b>{symbol}</b><a>Overview</a><a>Technical</a><a>Fundamental</a><a>Risk</a><a>Valuation</a><a>Verification</a></nav>
    <div class="main"><div class="strip">{strip}</div>{chart_html}
    <h3>Verification</h3><table>{rows}</table></div></div></body></html>"""
    from pathlib import Path; Path(out_path).write_text(html)
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(report): self-contained Plotly dashboard"`

---

## Phase 8 — Pipeline, CLI, end-to-end

### Task 8.1: Pipeline orchestration

**Files:** Create `src/stock_analyzer/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Failing test** (inject fake provider + fake llm factory; assert both outputs written)

```python
# tests/test_pipeline.py
from datetime import date
from tests.fixtures.prices import linear_prices
from stock_analyzer.data.models import PriceHistory, Fundamentals
from stock_analyzer.agents.state import AnalystFinding, ReportDraft, ReportSection
from stock_analyzer.verification.models import ClaimList, Verdict
from stock_analyzer.pipeline import run_analysis

class FakeProvider:
    def get_price_history(self,s,p): return PriceHistory(symbol=s,period=p,bars=linear_prices())
    def get_fundamentals(self,s): return Fundamentals(symbol=s, pe=28.4)
    def get_news(self,s,limit=20): return []

def fake_factory(schema):
    class F:
        def invoke(self,msgs):
            if schema is ReportDraft:
                return ReportDraft(sections=[ReportSection(id="technical",prose="RSI ~66.8",referenced_metrics=[],charts=["rsi"])],
                                   recommendation="buy", confidence="high")
            if schema is ClaimList: return ClaimList(claims=[])
            if schema is Verdict: return Verdict(claim_id="x",status="supported",rationale="ok")
            return AnalystFinding(summary="s",key_points=[],outlook="neutral",rationale="r",cited_metrics=[])
    return F()

def test_run_analysis_writes_pdf_and_html(tmp_path):
    res = run_analysis("AAPL", period="1y", out_dir=tmp_path, provider=FakeProvider(),
                       structured_factory=fake_factory, benchmark=linear_prices()["Close"])
    assert (tmp_path/"AAPL_report.pdf").exists()
    assert (tmp_path/"AAPL_dashboard.html").exists()
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement**

```python
# src/stock_analyzer/pipeline.py
from pathlib import Path
from .metrics.bundle import MetricsBundle
from .charts.builders import build_charts
from .agents.graph import build_graph
from .agents.llm import grounding_block
from .report.assemble import build_context
from .report.pdf import render_pdf
from .report.html_dashboard import render_dashboard

def run_analysis(symbol, period, out_dir, provider, structured_factory, benchmark):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ph = provider.get_price_history(symbol, period)
    fund = provider.get_fundamentals(symbol); news = provider.get_news(symbol)
    bundle = MetricsBundle.from_data(ph, fund, benchmark, news, fcf0=100.0, growth=0.05, wacc=0.10)
    gt = bundle.as_flat()
    charts = {r.name: r.image_path for r in build_charts(ph.bars, out/"charts")}
    app = build_graph(structured_factory, verify=True, ground_truth=gt, max_revisions=2)
    state = app.invoke({"symbol":symbol, "metrics_block": grounding_block(bundle), "revisions":0})
    ctx = build_context(symbol, state["report"], gt, state.get("audit",{}), charts)
    render_pdf(ctx, out/f"{symbol}_report.pdf")
    verdict_rows = [{"text": v.rationale, "status": v.status} for v in state.get("verdicts", [])]
    render_dashboard(symbol, ph.bars, ctx["kpis"], verdict_rows, out/f"{symbol}_dashboard.html")
    return {"pdf": out/f"{symbol}_report.pdf", "html": out/f"{symbol}_dashboard.html", "audit": state.get("audit")}
```

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: end-to-end pipeline (fake-injected test)"`

### Task 8.2: CLI

**Files:** Create `src/stock_analyzer/cli.py`, `tests/test_cli.py`; Modify `pyproject.toml` (add script entrypoint)

- [ ] **Step 1: Failing test**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from stock_analyzer.cli import app

def test_cli_help():
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0 and "analyze" in res.output
```

- [ ] **Step 2: Run / fail** → FAIL
- [ ] **Step 3: Implement** (CLI wires real provider + real llm factory)

```python
# src/stock_analyzer/cli.py
import typer
from .config import Settings
from .data import get_provider
from .agents.llm import make_llm, structured
from .pipeline import run_analysis

app = typer.Typer()

@app.command()
def analyze(symbol: str, period: str = "1y", out: str = "out", benchmark: str = "SPY"):
    s = Settings()
    provider = get_provider(s); llm = make_llm(s)
    factory = lambda schema: structured(llm, schema)
    bench = provider.get_price_history(benchmark, period).bars["Close"]
    res = run_analysis(symbol, period, out, provider, factory, bench)
    typer.echo(f"PDF: {res['pdf']}\nHTML: {res['html']}\nAudit: {res['audit']}")

if __name__ == "__main__":
    app()
```

Add to `pyproject.toml`: `[project.scripts]` → `stock-analyzer = "stock_analyzer.cli:app"`.

- [ ] **Step 4: Run / pass** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: Typer CLI entrypoint"`

### Task 8.3: README + archive old code

**Files:** Modify `README.md`; move legacy files to `legacy/`

- [ ] **Step 1:** Move old root files into `legacy/` (`git mv test.py tools.py chat_groq_manager.py test.ipynb visualizeReport.ipynb legacy/`), keep the sample report PDFs for reference.
- [ ] **Step 2:** Rewrite `README.md`: overview, the verification/ground-truth design, setup (incl. macOS WeasyPrint native libs + `ANTHROPIC_API_KEY`), `stock-analyzer analyze AAPL` usage, sample outputs, architecture diagram (link `docs/design/`).
- [ ] **Step 3:** Run full suite: `pytest -q` → all pass.
- [ ] **Step 4:** Manual smoke (real API + network), once: `stock-analyzer analyze AAPL` → open the PDF + HTML, sanity-check the verification appendix.
- [ ] **Step 5: Commit** — `git commit -am "docs: README + archive legacy implementation"`

---

## Self-Review

**Spec coverage:** data provider (P1) ✓ · all 2-yr metrics + modern risk additions (P2) ✓ · charts (P3) ✓ · LangGraph parallel analysts + writer (P4) ✓ · verification gate: atomic claims + programmatic + grounded judge + loop + audit (P5) ✓ · WeasyPrint PDF w/ verification appendix (P6) ✓ · Plotly sidebar dashboard + verdict table (P7) ✓ · yfinance default + pluggable provider (P1) ✓ · Claude Opus 4.8 via langchain-anthropic + structured output + cached grounding (P4) ✓ · CLI/config/tokens ✓.

**Open risk flagged in-plan:** Task 4.0 validates the exact `langchain-anthropic` kwargs for Opus 4.8 adaptive thinking + `effort` + structured output before nodes are built; adjust `model_kwargs` placement if the installed SDK version differs.

**Placeholders:** none — every step has runnable code/commands.

**Type consistency:** `MetricsBundle.as_flat()/prompt_block()`, `AnalystFinding`, `ReportDraft`, `Claim/Verdict/ClaimList/VerificationAudit`, `ChartRef`, `build_graph(structured_factory, verify, ground_truth, max_revisions)`, and the `structured_factory(schema)` injection convention are used consistently across tasks.
