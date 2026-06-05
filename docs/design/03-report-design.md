# 03 — Report Visual Design (LOCKED)

Approved via interactive mockups. Visual language: **Quant Minimal base + Terminal data accents** — clean minimal everywhere, but *any displayed number* gets a terminal/data-dense (monospace, dark strip, directional color) treatment. Same language for **both** the PDF and the HTML dashboard.

## Design tokens (single source of truth — reuse in PDF CSS, Plotly, matplotlib)
- Base bg `#FFFFFF`; subtle panel `#FAFBFC`/`#F6F8FA`; hairline `#E5E7EB`
- Ink `#0B0F19`; muted `#6b7280` / `#9ca3af`
- Accent (cobalt) `#2347D9`; accent-soft `#EEF2FF`
- Terminal/data panel bg `#0D1117`; data text `#E6EDF3`; data muted `#8B949E`
- Directional: positive `#16a34a` (green `#3FB950` on dark) · caution `#D29922` (amber) · negative `#ef4444`/`#f85149`
- Verified chip: bg `#ecfdf5`, border `#a7f3d0`, text `#047857`; corrected = amber `⟳`; unresolved = red `⚠`
- Fonts: headings/body = geometric humanist sans — **bundle an open font** (e.g. Space Grotesk / Archivo) with `system-ui` fallback; **explicitly avoid Inter/Roboto/default-Arial** (anti generic-AI look). Numbers/data = monospace — **bundle** JetBrains Mono / IBM Plex Mono, `ui-monospace` fallback.
- Charts: cobalt line + soft cobalt gradient area fill; candles green/red; minimal axes + hairlines.

## Treatment rules
- Layout / headings / prose / chrome → **minimal**: white, whitespace, cobalt accent-bar on headings, hairlines.
- **Any number** (hero KPIs, inline figures, table values) → **monospace**; hero KPI cluster lives in a dark terminal **data strip**; inline figures render as small dark monospace **data chips**; apply directional colors.
- Verification status → green `✓ Verified …` chip / amber `⟳ corrected` / red `⚠ unresolved`.

## PDF report (WeasyPrint + Jinja2)
One Jinja2 HTML template + CSS (tokens above) → WeasyPrint → PDF. *macOS: needs Pango/Cairo via Homebrew; bundle the webfonts so rendering is deterministic.*
**Page flow:**
1. **Header band** — cobalt label ("EQUITY ANALYSIS"), large company name, ticker · exchange, date, recommendation pill (BUY/HOLD/SELL) + confidence; hero **terminal data strip** (LAST price + Δ, P/E, RSI, Beta, Sharpe, Max DD, VaR).
2. **Executive Summary** · 3. **Company Overview** · 4. **Technical Analysis** (candlestick+SMA/Bollinger, RSI, MACD) · 5. **Fundamental Analysis** (ratios strip/table, margins, growth) · 6. **Risk Assessment** (beta, vol, ATR, max DD, Sharpe, VaR; returns dist + drawdown charts) · 7. **Valuation & Forecasts** (DCF assumptions + sensitivity grid, comparables, bull/base/bear) · 8. **Recommendation**.
9. **Verification Appendix** — full claim→verdict table (status chips), totals (supported/contradicted/unsupported), corrections applied, **residual unresolved flagged honestly**.
10. **Footer** — disclaimer, data provenance (provider + as-of), model id `claude-opus-4-8`.
Per-section pattern: cobalt accent-bar heading → prose (inline mono chips) → chart(s) → optional mini data strip.

## Interactive HTML dashboard (Plotly, single self-contained .html)
- **Left sidebar nav**: Overview / Technical / Fundamental / Risk / Valuation / Verification.
- Sticky app header (name, ticker, BUY pill, live price) + sticky **terminal data strip**.
- Interactive charts (hover + zoom + range slider): candlestick + SMA50/200 + Bollinger (range slider, OHLC hover); RSI (overbought/oversold lines); MACD (histogram + signal); volume; returns distribution; drawdown.
- **Verification section**: filterable claim→verdict table (✓ supported / ⟳ corrected / ⚠ unresolved) with expected vs claimed + delta.
- Self-contained single file (inline Plotly + CSS) for easy sharing. Same tokens as the PDF.

## Consistency mandate
Define tokens **once** (`report/tokens.py` or shared CSS variables + matching matplotlib & Plotly themes) so the static PDF charts and the interactive HTML look like one product.
