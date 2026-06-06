import pytest
from stock_analyzer.data.logo import domain_from_website, fetch_logo, _looks_like_image


@pytest.mark.parametrize("website, expected", [
    ("https://www.nvidia.com/en-us/", "nvidia.com"),
    ("http://abc.xyz", "abc.xyz"),
    ("https://www.apple.com", "apple.com"),
    ("google.com/finance", "google.com"),
    ("https://www.example.co.uk/path?x=1", "example.co.uk"),
    (None, None),
    ("", None),
])
def test_domain_from_website(website, expected):
    assert domain_from_website(website) == expected


def test_fetch_logo_no_inputs_returns_none_without_network(tmp_path):
    # No website and no ticker -> no candidate URLs -> returns None, never touches the network.
    assert fetch_logo(None, "", tmp_path / "logo.png") is None
    assert not (tmp_path / "logo.png").exists()


def test_looks_like_image_rejects_html_error_page():
    assert _looks_like_image(b"<!DOCTYPE html><html>not found</html>" * 10) is False
    assert _looks_like_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 300) is True
