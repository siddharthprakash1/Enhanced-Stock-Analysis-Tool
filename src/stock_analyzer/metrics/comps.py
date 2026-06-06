"""Relative valuation (comps) against a built-in sector peer set.

yFinance does not expose a reliable peer list, so we ship a curated
sector -> large-cap peers map (keyed by yFinance's `sector` strings). For a
subject in a known sector we pull each peer's fundamentals and compare the
subject to the peer-group median on the standard multiples.
"""
import statistics
from datetime import date

from ..data.models import Fundamentals
from .value import MetricValue

# yFinance `sector` string -> representative large-cap peers
SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "ADBE", "CRM", "AVGO"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA"],
    "Healthcare": ["JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Industrials": ["CAT", "HON", "UPS", "BA", "GE", "LMT"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
    "Basic Materials": ["LIN", "SHW", "APD", "FCX", "NEM"],
}

# (Fundamentals attr, label, unit, is_percent)
COMP_METRICS = [
    ("pe", "P/E", "x", False),
    ("pb", "P/B", "x", False),
    ("enterprise_to_ebitda", "EV/EBITDA", "x", False),
    ("profit_margin", "Profit Margin", "%", True),
    ("revenue_growth", "Revenue Growth", "%", True),
]


def peers_for(symbol: str, sector: str | None, limit: int = 5) -> list[str]:
    """Peer tickers for a sector, excluding the subject itself."""
    if not sector:
        return []
    pool = SECTOR_PEERS.get(sector, [])
    out = [p for p in pool if p.upper() != (symbol or "").upper()]
    return out[:limit]


def _val(f: Fundamentals, attr: str, is_pct: bool):
    v = getattr(f, attr, None)
    if v is None or v != v:  # None or NaN
        return None
    return v * 100 if is_pct else v


def build_comps(subject: Fundamentals, peers: list[Fundamentals]) -> dict | None:
    """Build a comps table: subject vs each peer + peer-group median + premium/discount.

    Returns None when no usable peer data is available (graceful skip).
    """
    peers = [p for p in peers if p is not None and p.symbol != subject.symbol]
    if not peers:
        return None

    rows = []
    for attr, label, unit, is_pct in COMP_METRICS:
        sub = _val(subject, attr, is_pct)
        peer_vals = [_val(p, attr, is_pct) for p in peers]
        present = [x for x in peer_vals if x is not None]
        median = statistics.median(present) if present else None
        premium = None
        if sub is not None and median not in (None, 0):
            premium = (sub / median - 1.0) * 100.0
        rows.append({
            "label": label, "unit": unit, "is_pct": is_pct,
            "subject": sub, "peers": peer_vals, "median": median, "premium": premium,
        })

    return {"subject": subject.symbol, "tickers": [p.symbol for p in peers], "rows": rows}


def comps_metrics(comps: dict, as_of: date) -> dict[str, MetricValue]:
    """A couple of verifiable summary metrics (peer median P/E, premium to peers)
    so the narrative can discuss relative valuation and the gate can check it."""
    out: dict[str, MetricValue] = {}
    pe_row = next((r for r in comps["rows"] if r["label"] == "P/E"), None)
    if pe_row and pe_row["median"] is not None:
        out["peer_median_pe"] = MetricValue(
            key="peer_median_pe", label="Peer Median P/E", value=pe_row["median"],
            unit="x", category="valuation", as_of=as_of,
        )
        if pe_row["premium"] is not None:
            out["pe_premium_to_peers"] = MetricValue(
                key="pe_premium_to_peers", label="P/E Premium vs Peers",
                value=pe_row["premium"], unit="%", category="valuation", as_of=as_of,
            )
    return out
