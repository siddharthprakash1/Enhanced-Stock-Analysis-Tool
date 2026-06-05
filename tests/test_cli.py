"""CLI smoke tests (no network / LLM calls)."""

from typer.testing import CliRunner
from stock_analyzer.cli import app


def test_cli_help_lists_analyze():
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "analyze" in res.output


def test_analyze_help_shows_options():
    res = CliRunner().invoke(app, ["analyze", "--help"])
    assert res.exit_code == 0
    assert "SYMBOL" in res.output
    assert "--period" in res.output
    assert "--out" in res.output
    assert "--benchmark" in res.output
