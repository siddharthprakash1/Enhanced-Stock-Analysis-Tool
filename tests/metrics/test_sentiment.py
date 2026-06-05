from datetime import date
from stock_analyzer.data.models import NewsItem
from stock_analyzer.metrics.sentiment import compute_sentiment


def test_positive_headlines_positive_score():
    news = [
        NewsItem(title="Company crushes earnings, soars to record high"),
        NewsItem(title="Analysts upgrade with strong buy and optimism"),
    ]
    out = compute_sentiment(news, as_of=date(2026, 6, 5))
    assert out["sentiment_score"].value > 0


def test_no_news_is_none():
    out = compute_sentiment([], as_of=date(2026, 6, 5))
    assert out["sentiment_score"].value is None
