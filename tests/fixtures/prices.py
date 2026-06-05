import numpy as np
import pandas as pd


def linear_prices(n=260, start=100.0, step=0.5):
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(start + step * np.arange(n), index=idx)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000_000},
        index=idx,
    )
