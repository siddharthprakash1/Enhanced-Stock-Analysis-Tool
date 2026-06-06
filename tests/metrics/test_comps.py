from datetime import date
from stock_analyzer.data.models import Fundamentals
from stock_analyzer.metrics.comps import peers_for, build_comps, comps_metrics, SECTOR_PEERS


def test_peers_for_excludes_self_and_limits():
    peers = peers_for("AAPL", "Technology", limit=3)
    assert "AAPL" not in peers
    assert len(peers) == 3
    assert all(p in SECTOR_PEERS["Technology"] for p in peers)


def test_peers_for_unknown_sector_is_empty():
    assert peers_for("XYZ", None) == []
    assert peers_for("XYZ", "Nonexistent Sector") == []


def _f(sym, pe=None, pb=None, ev=None, margin=None, rev=None):
    return Fundamentals(symbol=sym, pe=pe, pb=pb, enterprise_to_ebitda=ev,
                        profit_margin=margin, revenue_growth=rev)


def test_build_comps_computes_median_and_premium():
    subject = _f("AAPL", pe=30.0, margin=0.25)
    peers = [_f("MSFT", pe=20.0, margin=0.30), _f("ORCL", pe=10.0, margin=0.20)]
    comps = build_comps(subject, peers)
    pe_row = next(r for r in comps["rows"] if r["label"] == "P/E")
    assert pe_row["median"] == 15.0                 # median(20, 10)
    assert abs(pe_row["premium"] - 100.0) < 1e-9    # 30 vs median 15 = +100%
    margin_row = next(r for r in comps["rows"] if r["label"] == "Profit Margin")
    assert margin_row["subject"] == 25.0            # percent-scaled
    assert comps["tickers"] == ["MSFT", "ORCL"]


def test_build_comps_returns_none_without_peers():
    assert build_comps(_f("AAPL", pe=30.0), []) is None


def test_comps_metrics_emit_verifiable_summary():
    subject = _f("AAPL", pe=30.0)
    peers = [_f("MSFT", pe=20.0), _f("ORCL", pe=10.0)]
    comps = build_comps(subject, peers)
    m = comps_metrics(comps, date(2026, 6, 5))
    assert m["peer_median_pe"].value == 15.0
    assert abs(m["pe_premium_to_peers"].value - 100.0) < 1e-9
