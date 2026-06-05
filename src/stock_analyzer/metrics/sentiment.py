from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..data.models import NewsItem
from .value import MetricValue


def compute_sentiment(news: list[NewsItem], as_of: date) -> dict[str, MetricValue]:
    if not news:
        score = None
    else:
        sia = SentimentIntensityAnalyzer()
        score = sum(sia.polarity_scores(n.title)["compound"] for n in news) / len(news)
    return {
        "sentiment_score": MetricValue(
            key="sentiment_score",
            label="News Sentiment",
            value=score,
            unit="",
            category="sentiment",
            as_of=as_of,
        )
    }
