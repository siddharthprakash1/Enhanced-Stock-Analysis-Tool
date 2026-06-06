from dataclasses import dataclass
from datetime import date

from .value import MetricValue
from .wacc import WaccComponents


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
    def cell(g, w):
        try:
            return round(simple_dcf(fcf0, g, w, years, terminal_growth), 2)
        except ValueError:
            return None
    return [[cell(g, w) for w in waccs] for g in growths]


def derive_growth(revenue_growth, eps_growth, default=0.05, lo=0.0, hi=0.15):
    """Near-term FCF growth from the company's own fundamentals, clamped to a sane band.

    Returns (growth, human-readable source label).
    """
    def ok(x):
        return x is not None and x == x  # not None, not NaN

    if ok(revenue_growth):
        cand, src = revenue_growth, "company revenue growth"
    elif ok(eps_growth):
        cand, src = eps_growth, "company EPS growth"
    else:
        return default, "default assumption"
    clamped = max(lo, min(hi, cand))
    if clamped != cand:
        src += " (clamped)"
    return clamped, src


def price_targets(fcf0, growth, wacc, shares_outstanding, terminal_growth=0.02, years=5):
    """Per-share DCF fair-value targets (bear = lower growth/higher WACC ... bull = higher growth/lower WACC)."""
    if not shares_outstanding or shares_outstanding <= 0 or fcf0 is None:
        return None
    floor = terminal_growth + 0.005  # keep bull-case WACC safely above terminal growth
    try:
        base = simple_dcf(fcf0, growth, wacc, years, terminal_growth) / shares_outstanding
        bull = simple_dcf(fcf0, growth + 0.02, max(wacc - 0.01, floor), years, terminal_growth) / shares_outstanding
        bear = simple_dcf(fcf0, max(growth - 0.02, 0.0), wacc + 0.01, years, terminal_growth) / shares_outstanding
    except ValueError:
        return None
    return {"base": base, "bull": bull, "bear": bear}


def compute_valuation(
    fcf0: float, growth: float, wacc: float, as_of: date,
    shares_outstanding=None, terminal_growth: float = 0.02, years: int = 5,
) -> dict[str, MetricValue]:
    val = simple_dcf(fcf0, growth, wacc, years, terminal_growth)

    def mv(key, label, value, unit="$"):
        return MetricValue(key=key, label=label, value=value, unit=unit, category="valuation", as_of=as_of)

    out = {"dcf_value": mv("dcf_value", "DCF Enterprise Value", val)}
    tg = price_targets(fcf0, growth, wacc, shares_outstanding, terminal_growth, years)
    if tg:
        out["dcf_per_share"] = mv("dcf_per_share", "DCF Fair Value / Share", tg["base"])
        out["target_base"] = mv("target_base", "DCF Value — Base Case", tg["base"])
        out["target_bull"] = mv("target_bull", "DCF Value — Bull Case", tg["bull"])
        out["target_bear"] = mv("target_bear", "DCF Value — Bear Case", tg["bear"])
    return out


@dataclass
class ValuationInputs:
    """Everything needed to run a defensible DCF, with the discount rate and growth
    derived from real inputs (WACC from CAPM, growth from company fundamentals)."""
    fcf0: float
    growth: float
    growth_source: str
    wacc: WaccComponents
    terminal_growth: float
    years: int
    shares_outstanding: float | None


def compute_valuation_metrics(vi: ValuationInputs, as_of: date) -> dict[str, MetricValue]:
    """DCF outputs + the discount-rate assumptions, as ground-truth metrics.

    Returns {} (graceful skip) when a sensible perpetuity DCF can't be run.
    """
    if vi.wacc.wacc <= vi.terminal_growth:
        return {}
    try:
        out = compute_valuation(
            vi.fcf0, vi.growth, vi.wacc.wacc, as_of,
            shares_outstanding=vi.shares_outstanding,
            terminal_growth=vi.terminal_growth, years=vi.years,
        )
    except ValueError:
        return {}

    def pct(key, label, value):
        return MetricValue(key=key, label=label, value=value * 100, unit="%",
                           category="valuation", as_of=as_of)

    wc = vi.wacc
    out["wacc"] = pct("wacc", "WACC (Discount Rate)", wc.wacc)
    out["cost_of_equity"] = pct("cost_of_equity", "Cost of Equity (CAPM)", wc.cost_of_equity)
    out["cost_of_debt"] = pct("cost_of_debt", "Cost of Debt (pre-tax)", wc.cost_of_debt)
    return out
