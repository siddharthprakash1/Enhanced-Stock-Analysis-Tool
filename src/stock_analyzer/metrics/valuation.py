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


def sensitivity_grid(fcf0, growths, waccs, years=5, terminal_growth=0.02):
    return [[round(simple_dcf(fcf0, g, w, years, terminal_growth), 2) for w in waccs] for g in growths]


def price_targets(fcf0, growth, wacc, shares_outstanding):
    """Per-share DCF fair-value targets (bear = lower growth/higher WACC ... bull = higher growth/lower WACC)."""
    if not shares_outstanding or shares_outstanding <= 0 or fcf0 is None:
        return None
    base = simple_dcf(fcf0, growth, wacc) / shares_outstanding
    bull = simple_dcf(fcf0, growth + 0.02, max(wacc - 0.01, 0.04)) / shares_outstanding
    bear = simple_dcf(fcf0, max(growth - 0.02, 0.0), wacc + 0.01) / shares_outstanding
    return {"base": base, "bull": bull, "bear": bear}


def compute_valuation(fcf0: float, growth: float, wacc: float, as_of: date, shares_outstanding=None) -> dict[str, MetricValue]:
    val = simple_dcf(fcf0, growth, wacc)
    def mv(key, label, value):
        return MetricValue(key=key, label=label, value=value, unit="$", category="valuation", as_of=as_of)
    out = {"dcf_value": mv("dcf_value", "DCF Enterprise Value", val)}
    tg = price_targets(fcf0, growth, wacc, shares_outstanding)
    if tg:
        out["dcf_per_share"] = mv("dcf_per_share", "DCF Fair Value / Share", tg["base"])
        out["target_base"] = mv("target_base", "Price Target (Base)", tg["base"])
        out["target_bull"] = mv("target_bull", "Price Target (Bull)", tg["bull"])
        out["target_bear"] = mv("target_bear", "Price Target (Bear)", tg["bear"])
    return out


