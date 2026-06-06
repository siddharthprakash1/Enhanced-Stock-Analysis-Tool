from stock_analyzer.metrics.wacc import compute_wacc, cost_of_equity, cost_of_debt


def test_capm_cost_of_equity():
    # rf 4.55% + beta 1.2 * ERP 4.23% = 9.626%
    assert abs(cost_of_equity(0.0455, 1.2, 0.0423) - (0.0455 + 1.2 * 0.0423)) < 1e-12


def test_cost_of_debt_is_rf_plus_spread():
    assert abs(cost_of_debt(0.0455, 0.015) - 0.0605) < 1e-12


def test_wacc_blends_equity_and_after_tax_debt():
    wc = compute_wacc(
        risk_free=0.0455, beta=1.0, erp=0.0423, market_cap=300.0, total_debt=100.0,
        cost_of_debt_spread=0.015, tax_rate=0.21,
    )
    assert abs(wc.weight_equity - 0.75) < 1e-9 and abs(wc.weight_debt - 0.25) < 1e-9
    expected = 0.75 * wc.cost_of_equity + 0.25 * wc.cost_of_debt * (1 - 0.21)
    assert abs(wc.wacc - expected) < 1e-12
    # blended WACC sits below cost of equity because after-tax debt is cheaper
    assert wc.wacc < wc.cost_of_equity


def test_no_debt_means_wacc_equals_cost_of_equity():
    wc = compute_wacc(
        risk_free=0.0455, beta=1.1, erp=0.0423, market_cap=1000.0, total_debt=0.0,
        cost_of_debt_spread=0.015, tax_rate=0.21,
    )
    assert wc.weight_debt == 0.0
    assert abs(wc.wacc - wc.cost_of_equity) < 1e-12


def test_missing_beta_and_capital_structure_falls_back_safely():
    wc = compute_wacc(
        risk_free=0.0455, beta=None, erp=0.0423, market_cap=None, total_debt=None,
        cost_of_debt_spread=0.015, tax_rate=0.21,
    )
    assert wc.beta == 1.0                  # default beta
    assert wc.weight_equity == 1.0         # all-equity fallback
    assert wc.wacc > 0
