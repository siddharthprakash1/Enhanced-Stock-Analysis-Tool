from datetime import date
from .value import MetricValue


def simple_dcf(fcf0: float, growth: float, wacc: float, years: int = 5, terminal_growth: float = 0.02) -> float:
    if wacc <= terminal_growth:
        raise ValueError(f"wacc ({wacc}) must exceed terminal_growth ({terminal_growth})")
    pv = 0.0
    fcf = fcf0
    for t in range(1, years + 1):
        fcf *= (1 + growth)
        pv += fcf / (1 + wacc) ** t
    terminal = fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv += terminal / (1 + wacc) ** years
    return pv


def compute_valuation(fcf0: float, growth: float, wacc: float, as_of: date) -> dict[str, MetricValue]:
    val = simple_dcf(fcf0, growth, wacc)
    return {
        "dcf_value": MetricValue(
            key="dcf_value",
            label="DCF Enterprise Value",
            value=val,
            unit="$",
            category="valuation",
            as_of=as_of,
        )
    }


