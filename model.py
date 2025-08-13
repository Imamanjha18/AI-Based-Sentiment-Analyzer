from transformers import pipeline

# Load model (cache locally for faster reloads)
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return result['label'], result['score']