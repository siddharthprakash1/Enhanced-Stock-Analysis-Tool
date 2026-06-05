from stock_analyzer.tokens import COLORS, FONTS
from stock_analyzer.charts.theme import apply_theme


def test_tokens_present():
    assert COLORS["accent"] == "#2347D9" and COLORS["up"] == "#16a34a"
    assert "mono" in FONTS


def test_apply_theme_runs():
    apply_theme()  # should not raise
