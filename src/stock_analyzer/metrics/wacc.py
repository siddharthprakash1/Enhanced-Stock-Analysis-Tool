"""Weighted Average Cost of Capital (WACC) from CAPM + after-tax cost of debt.

    cost of equity (CAPM) = rf + beta * ERP
    cost of debt          = rf + credit spread
    WACC = w_e * r_e + w_d * r_d * (1 - tax)

Capital weights use equity = market cap and debt = total debt. Macro inputs
(risk-free rate, equity risk premium, tax rate, credit spread) come from the
config / a live Treasury fetch; everything is surfaced in the report's
assumptions block so the discount rate is fully auditable.
"""
from dataclasses import dataclass


@dataclass
class WaccComponents:
    risk_free: float
    erp: float
    beta: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    weight_equity: float
    weight_debt: float
    wacc: float


def cost_of_equity(rf: float, beta: float, erp: float) -> float:
    return rf + beta * erp


def cost_of_debt(rf: float, spread: float) -> float:
    return rf + spread


def compute_wacc(
    *,
    risk_free: float,
    beta: float | None,
    erp: float,
    market_cap: float | None,
    total_debt: float | None,
    cost_of_debt_spread: float,
    tax_rate: float,
    beta_default: float = 1.0,
) -> WaccComponents:
    b = beta if (beta is not None and beta == beta) else beta_default  # b==b guards NaN
    re = cost_of_equity(risk_free, b, erp)
    rd = cost_of_debt(risk_free, cost_of_debt_spread)

    e = float(market_cap) if market_cap and market_cap > 0 else 0.0
    d = float(total_debt) if total_debt and total_debt > 0 else 0.0
    total = e + d
    if total <= 0:
        we, wd = 1.0, 0.0            # no capital structure data → treat as all-equity
    else:
        we, wd = e / total, d / total

    wacc = we * re + wd * rd * (1 - tax_rate)
    return WaccComponents(
        risk_free=risk_free, erp=erp, beta=b, cost_of_equity=re, cost_of_debt=rd,
        tax_rate=tax_rate, weight_equity=we, weight_debt=wd, wacc=wacc,
    )
