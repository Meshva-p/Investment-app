import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def fetch_news(stock_name, count=5):
    """Fetch latest news from Google News RSS feed."""
    query = stock_name.replace(" ", "+") + "+stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "xml")
        items = soup.find_all("item")[:count]

        news_list = []
        for item in items:
            title = item.title.text if item.title else "No title"
            link = item.link.text if item.link else ""
            news_list.append(f"{title} - {link}")

        return news_list

    except Exception as e:
        return [f"Error fetching news: {e}"]

def analyze_sentiment(text):
    """Analyze sentiment using FinBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    sentiment = torch.argmax(probabilities).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment], probabilities[0][sentiment].item()

def summarize_news(news_list):
    """Summarize news with sentiment analysis."""
    if not news_list:
        return "⚠️ No news to summarize."

    summary = ""
    for i, article in enumerate(news_list, 1):
        title = article.split(" - ")[0]
        sentiment, confidence = analyze_sentiment(title)
        summary += f"{i}. {title} [{sentiment} - {confidence*100:.2f}%]\n"

    return summary

if __name__ == "__main__":
    tickers = ["Nifty 50", "Reliance Industries", "Tata Consultancy Services", "Apple", "Tesla"]

    for ticker in tickers:
        print(f"\n📰 Latest news for {ticker}:")
        news_list = fetch_news(ticker, count=5)
        for n in news_list:
            print("-", n)

        summary = summarize_news(news_list)
        print("\n💡 Summary:")
        print(summary)
        print("-" * 60)
