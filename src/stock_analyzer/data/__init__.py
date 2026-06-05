from ..config import Settings
from .base import DataProvider, DataUnavailableError
from .yfinance_provider import YFinanceProvider

def get_provider(settings: Settings) -> DataProvider:
    if settings.provider == "fmp" and settings.fmp_api_key:
        from .fmp_provider import FMPProvider
        return FMPProvider(settings.fmp_api_key)
    return YFinanceProvider()
