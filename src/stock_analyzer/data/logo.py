"""Best-effort company logo fetching.

Resolves a logo from the company's website domain (Clearbit) with a ticker-based
fallback, downloads it to a local file, and returns the path. Never raises — a
missing logo simply means the report renders without one.
"""
import urllib.request
from pathlib import Path

# magic bytes for png, jpeg, gif, webp(RIFF)
_IMG_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF")


def domain_from_website(website: str | None) -> str | None:
    """Bare domain from a website URL, e.g. 'https://www.nvidia.com/en-us/' -> 'nvidia.com'."""
    if not website:
        return None
    d = website.strip().split("//")[-1].split("/")[0].split("?")[0].strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d or None


def _looks_like_image(data: bytes) -> bool:
    return bool(data) and len(data) > 200 and any(data.startswith(m) for m in _IMG_MAGIC)


def _download(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (stock-analyzer)"})
        with urllib.request.urlopen(req, timeout=8) as r:
            if getattr(r, "status", 200) != 200:
                return None
            data = r.read(2_000_000)  # cap at ~2 MB
        return data if _looks_like_image(data) else None
    except Exception:
        return None


def fetch_logo(website: str | None, ticker: str, dest) -> str | None:
    """Download a company logo to ``dest`` and return its path, or None.

    Tries the website domain via Clearbit first, then a ticker-based fallback.
    Returns None (no network attempt) when neither a domain nor a ticker is given.
    """
    domain = domain_from_website(website)
    urls = []
    if domain:
        urls.append(f"https://logo.clearbit.com/{domain}?size=256")
    if ticker:
        urls.append(f"https://financialmodelingprep.com/image-stock/{ticker.upper()}.png")

    for url in urls:
        data = _download(url)
        if data:
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return str(dest)
    return None
