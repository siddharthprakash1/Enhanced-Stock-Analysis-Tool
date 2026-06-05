from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    fmp_api_key: str | None = None
    llm_provider: str = "gemini"            # "gemini" (dev) | "anthropic" (final)
    model: str = "claude-opus-4-8"          # used when llm_provider == "anthropic"
    gemini_model: str = "gemini-2.5-flash"  # used when llm_provider == "gemini"
    effort: str = "high"
    max_revisions: int = 2
    numeric_tol_rel: float = 0.01
    numeric_tol_abs: float = 0.05
    provider: str = "yfinance"
    out_dir: str = "out"
