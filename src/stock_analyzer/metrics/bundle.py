import pandas as pd
from pydantic import BaseModel, ConfigDict
from ..data.models import PriceHistory, Fundamentals, NewsItem
from .value import MetricValue
from .technical import compute_technical
from .fundamental import compute_fundamental
from .risk import compute_risk
from .valuation import compute_valuation
from .sentiment import compute_sentiment


class MetricsBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    period: str
    metrics: dict[str, MetricValue]

    @classmethod
    def from_data(
        cls,
        ph: PriceHistory,
        fundamentals: Fundamentals,
        benchmark_close: pd.Series,
        news: list[NewsItem],
        fcf0: float,
        growth: float,
        wacc: float,
    ) -> "MetricsBundle":
        as_of = ph.bars.index[-1].date()
        m: dict[str, MetricValue] = {}
        m.update(compute_technical(ph.bars))
        m.update(compute_fundamental(fundamentals, as_of))
        m.update(compute_risk(ph.bars, benchmark_close))
        m.update(compute_valuation(fcf0, growth, wacc, as_of))
        m.update(compute_sentiment(news, as_of))
        return cls(symbol=ph.symbol, period=ph.period, metrics=m)

    def as_flat(self) -> dict[str, MetricValue]:
        return self.metrics

    def prompt_block(self) -> str:
        lines = [f"# Ground-truth metrics for {self.symbol} ({self.period})"]
        for key in sorted(self.metrics):
            mv = self.metrics[key]
            lines.append(f"- {mv.label}: {mv.display()}")
        return "\n".join(lines)
