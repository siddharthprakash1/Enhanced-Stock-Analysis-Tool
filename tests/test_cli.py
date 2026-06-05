"""
Task 8.2 – CLI smoke tests (no network / LLM calls).

The Typer app uses a single command (`analyze`) so invoking `--help` at the
top level directly shows that command's usage.  We assert the relevant
options are present rather than looking for "analyze" as a sub-command label
(that's a Typer single-command-app detail, not a regression risk).
"""

from typer.testing import CliRunner
from stock_analyzer.cli import app


def test_cli_help_exits_cleanly():
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0


def test_cli_help_shows_analyze_options():
    """The analyze command's signature must be visible in --help output."""
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0
    # Key options that prove the analyze command is wired correctly
    assert "SYMBOL" in res.output          # required positional argument
    assert "--period" in res.output        # look-back window option
    assert "--out" in res.output           # output directory option
    assert "--benchmark" in res.output     # benchmark ticker option
