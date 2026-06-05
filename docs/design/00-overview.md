# Stock Analyzer Revamp — Design Overview

**Project:** Revamp of `Enhanced-Stock-Analysis-Tool` (cloned 2026-06-05 into `~/Desktop/Personal/Goldmansachs/`).
**Goal:** Portfolio / showcase-grade rebuild aimed at finance / quant / SWE roles (the parent folder is "Goldmansachs"). Optimize for **polish, analytical rigor, and modern best practices** — not just "it runs."
**Status:** Design phase (brainstorming workflow). No code written yet.
- ✅ Part 1 — Architecture skeleton: approved (see `01-architecture.md`).
- ✅ Part 2 — Engine: approved (see `02-engine.md`).
- ✅ Part 3 — Report visual design: approved/locked (see `03-report-design.md`).
- ⏳ Implementation plan (writing-plans) → **user reviews plan before any code** → build.

## North-star principle — ONE SOURCE OF TRUTH
A deterministic **`MetricsBundle`** (the exact computed numbers) is the ground truth. Analyst agents may **only** reason over it. The verification agent reconciles **every** claim in the generated narrative back against it. **No number reaches the PDF without reconciling to a computed value.** The verification audit is shown in the report itself.

## The headline feature (what the user explicitly asked for)
"A good amount of agent work like the agent checks the content properly over all the metrics" = a **verification gate**: atomic claim extraction → reconcile each claim vs `MetricsBundle` (programmatic numeric check + grounded LLM judge) → conditional loop back to revise the offending sections → an audit appendix that proves the report fact-checked itself.

## Locked decisions (do NOT re-litigate without the user)
| Decision | Choice |
|---|---|
| Scope | Rebuild fresh; reuse the solid financial math from old `test.py`/`tools.py`, restructure everything else |
| LLM | **Claude Opus 4.8** (`claude-opus-4-8`) via `langchain-anthropic` `ChatAnthropic`; adaptive thinking + `effort`; structured output via `.with_structured_output()`; prompt-cache the `MetricsBundle` |
| Orchestration | **LangGraph** state graph — parallel analyst fan-out + conditional verification-loop edge |
| Data | **yfinance** default (zero-config) behind a `DataProvider` interface; optional keyed FMP / Alpha Vantage provider |
| Output | **Polished PDF** (Jinja2 + WeasyPrint) **AND** interactive **HTML dashboard** (Plotly) |
| Build location | In-place in the cloned repo, new `src/stock_analyzer/` layout; old files archived during impl |

## Guardrails — don't drift
- Do NOT reintroduce the old stack: CrewAI, Groq, `llama3-groq-*`, Ollama, or NLTK-as-core-engine.
- Do NOT let any LLM invent numbers — everything grounded in `MetricsBundle`.
- Keep modules single-responsibility, isolated, and unit-testable.
- Model id is exactly `claude-opus-4-8` (no date suffix). Opus 4.8 = adaptive thinking only; no `temperature`/`top_p`/`budget_tokens`.
- WeasyPrint needs Pango/Cairo native libs (Homebrew on macOS) — call this out in setup.

## Doc index
- `01-architecture.md` — graph shape, module layout, Claude integration (APPROVED)
- `02-engine.md` — data provider, metrics, agents, verification gate (detailed)
- `03-report-design.md` — PDF layout + HTML dashboard + theme (LOCKED)
