from stock_analyzer.config import Settings

def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini"
    assert s.gemini_model == "gemini-2.5-flash-lite"
    assert s.model == "claude-opus-4-8"
    assert s.effort == "high"
    assert s.max_revisions == 2
    assert s.numeric_tol_rel == 0.01
    assert s.provider == "yfinance"
