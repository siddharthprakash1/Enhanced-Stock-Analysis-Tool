import pandas as pd
from pydantic import BaseModel, ConfigDict
from ..data.models import PriceHistory, Fundamentals, NewsItem
from .value import MetricValue
from .technical import compute_technical
from .fundamental import compute_fundamental
from .risk import compute_risk
from .valuation import ValuationInputs, compute_valuation_metrics
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
        valuation: ValuationInputs | None = None,
    ) -> "MetricsBundle":
        """Build the deterministic ground-truth.

        ``valuation`` carries the DCF inputs (FCF, WACC, growth). Pass ``None`` to
        skip valuation entirely — e.g. for ETFs/funds or securities with no usable
        free-cash-flow data — rather than fabricating a meaningless DCF.
        """
        as_of = ph.bars.index[-1].date()
        m: dict[str, MetricValue] = {}
        m.update(compute_technical(ph.bars))
        m.update(compute_fundamental(fundamentals, as_of))
        m.update(compute_risk(ph.bars, benchmark_close))
        if valuation is not None:
            m.update(compute_valuation_metrics(valuation, as_of))
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
