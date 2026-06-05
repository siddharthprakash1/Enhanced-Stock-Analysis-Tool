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
