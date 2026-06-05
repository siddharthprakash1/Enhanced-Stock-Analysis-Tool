from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt
from pydantic import BaseModel
from ..tokens import COLORS
from .theme import apply_theme


class ChartRef(BaseModel):
    name: str
    image_path: str
    caption: str
    fact: dict


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); return str(path)


def build_charts(bars: pd.DataFrame, out_dir) -> list[ChartRef]:
    apply_theme(); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    c = bars["Close"]; refs: list[ChartRef] = []

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(c.index, c, color=COLORS["accent"], lw=1.5, label="Close")
    ax.plot(c.index, c.rolling(50).mean(), color=COLORS["caution"], lw=1, label="SMA50")
    ax.plot(c.index, c.rolling(200).mean(), color=COLORS["down"], lw=1, label="SMA200")
    ax.legend(); ax.set_title("Price & Moving Averages")
    refs.append(ChartRef(name="price", image_path=_save(fig, out / "price.png"),
                         caption="Price with SMA50/SMA200", fact={"last_close": round(float(c.iloc[-1]), 2)}))

    delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    fig, ax = plt.subplots(figsize=(9, 2.5)); ax.plot(rsi.index, rsi, color=COLORS["caution"])
    ax.axhline(70, ls="--", color=COLORS["down"]); ax.axhline(30, ls="--", color=COLORS["up"]); ax.set_title("RSI (14)")
    refs.append(ChartRef(name="rsi", image_path=_save(fig, out / "rsi.png"),
                         caption="RSI (14)", fact={"latest_rsi": round(float(rsi.iloc[-1]), 1)}))

    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; sig = macd.ewm(span=9, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(9, 2.5)); ax.bar(macd.index, (macd - sig), color=COLORS["accent"]); ax.plot(macd.index, macd, color=COLORS["ink"])
    ax.set_title("MACD")
    refs.append(ChartRef(name="macd", image_path=_save(fig, out / "macd.png"),
                         caption="MACD vs signal", fact={"macd": round(float(macd.iloc[-1]), 3)}))

    ret = c.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(ret, bins=40, color=COLORS["accent"], alpha=0.8); ax.set_title("Returns Distribution")
    refs.append(ChartRef(name="returns_dist", image_path=_save(fig, out / "returns_dist.png"),
                         caption="Daily returns distribution", fact={"mean_daily": round(float(ret.mean()), 5)}))

    cum = (1 + ret).cumprod(); dd = cum / cum.cummax() - 1
    fig, ax = plt.subplots(figsize=(9, 2.5)); ax.fill_between(dd.index, dd, color=COLORS["down"], alpha=0.4); ax.set_title("Drawdown")
    refs.append(ChartRef(name="drawdown", image_path=_save(fig, out / "drawdown.png"),
                         caption="Drawdown curve", fact={"max_drawdown": round(float(dd.min()), 4)}))
    return refs
