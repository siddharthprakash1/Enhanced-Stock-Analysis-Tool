from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    fmp_api_key: str | None = None
    llm_provider: str = "gemini"            # "gemini" (dev) | "anthropic" (final)
    model: str = "claude-haiku-4-5"  # cheapest Claude; used when llm_provider == "anthropic"
    gemini_model: str = "gemini-2.5-flash-lite"  # free-tier-friendly default; used when llm_provider == "gemini"
    effort: str = "high"
    max_revisions: int = 2
    numeric_tol_rel: float = 0.01
    numeric_tol_abs: float = 0.05
    provider: str = "yfinance"
    out_dir: str = "out"

    # --- DCF / WACC assumptions (macro inputs; fetched live where possible, these are fallbacks) ---
    risk_free_rate: float = 0.0455       # 10Y UST, ~Jun 2026 (fallback if ^TNX fetch fails)
    equity_risk_premium: float = 0.0423  # Damodaran 2026 implied US ERP
    tax_rate: float = 0.21               # US federal statutory corporate rate
    cost_of_debt_spread: float = 0.015   # over risk-free; investment-grade approximation
    terminal_growth: float = 0.02        # perpetuity growth
    dcf_years: int = 5
    growth_default: float = 0.05         # used when company growth signals are unavailable
    growth_min: float = 0.0
    growth_max: float = 0.15             # clamp company-derived growth to a sane band
