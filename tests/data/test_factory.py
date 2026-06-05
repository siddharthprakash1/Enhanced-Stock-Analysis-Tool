from stock_analyzer.config import Settings
from stock_analyzer.data import get_provider
from stock_analyzer.data.yfinance_provider import YFinanceProvider

def test_default_is_yfinance():
    assert isinstance(get_provider(Settings(_env_file=None)), YFinanceProvider)
