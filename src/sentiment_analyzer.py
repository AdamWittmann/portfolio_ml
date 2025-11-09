# src/sentiment_analyzer.py

import yfinance as yf
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sentiment_score(ticker: str, max_articles: int = 10) -> float:
    """
    Fetch recent news for a ticker and calculate average sentiment.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        max_articles: Number of recent articles to analyze
        
    Returns:
        Sentiment score from -1 (very negative) to +1 (very positive)
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news or len(news) == 0:
            logger.warning(f"No news found for {ticker}, returning neutral sentiment")
            return 0.0
        
        sentiments = []
        for article in news[:max_articles]:
            # NEW: Extract from nested 'content' dict
            content = article.get('content', {})
            
            title = content.get('title', '')
            summary = content.get('summary', '')
            text = f"{title} {summary}".strip()
            
            # Skip if no meaningful text
            if len(text) < 20:
                logger.debug(f"{ticker} - Skipping article with insufficient text")
                continue
            
            # Calculate sentiment using TextBlob
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity  # -1 to +1
            sentiments.append(sentiment)
            
            logger.debug(f"{ticker} - '{title[:50]}...' | Sentiment: {sentiment:.3f}")
        
        if not sentiments:
            logger.warning(f"{ticker} - No valid articles found")
            return 0.0
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        logger.info(f"{ticker} average sentiment: {avg_sentiment:.3f} (from {len(sentiments)} articles)")
        
        return avg_sentiment
        
    except Exception as e:
        logger.error(f"Error fetching sentiment for {ticker}: {e}")
        return 0.0

# Test function
if __name__ == "__main__":
    test_tickers = ["AAPL", "TSLA", "META"]
    print("\nTesting sentiment analysis:\n")
    for ticker in test_tickers:
        sentiment = get_sentiment_score(ticker)
        print(f"{ticker}: {sentiment:.3f}")