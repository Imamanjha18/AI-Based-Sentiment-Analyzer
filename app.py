from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from model import analyze_sentiment  # Hugging Face model
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Connect to MongoDB (replace with your URI)
mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://beingamanjha:NFl1aFk1n6THMM2S@aman18.4dtxfsy.mongodb.net/?retryWrites=true&w=majority&appName=Aman18")
client = MongoClient(mongo_uri)
db = client.sentiment_db
collection = db.results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    model_type = request.form.get('model', 'vader')  # Default to VADER

    # Sentiment analysis logic
    if model_type == 'vader':
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "Positive 😊"
            sentiment_label = "POSITIVE"  # Added for consistent grouping
        elif compound <= -0.05:
            sentiment = "Negative 😔"
            sentiment_label = "NEGATIVE"  # Added for consistent grouping
        else:
            sentiment = "Neutral 😐"
            sentiment_label = "NEUTRAL"   # Added for consistent grouping
        score = compound
    else:  # Hugging Face model
        label, score = analyze_sentiment(text)
        sentiment = f"{label} {'😊' if label == 'POSITIVE' else '😔' if label == 'NEGATIVE' else '😐'}"
        sentiment_label = label  # Use Hugging Face's label directly

    # Store in MongoDB (with consistent labels)
    collection.insert_one({
        "text": text,
        "sentiment": sentiment,
        "sentiment_label": sentiment_label,  # New field for consistent charting
        "score": score,
        "model": model_type,
        "timestamp": datetime.now()
    })

    return render_template('index.html', text=text, sentiment=sentiment, model=model_type)

# New route for sentiment statistics
@app.route('/get_sentiment_stats')
def get_sentiment_stats():
    # Overall counts (for pie chart)
    positive = collection.count_documents({"sentiment_label": "POSITIVE"})
    neutral = collection.count_documents({"sentiment_label": "NEUTRAL"})
    negative = collection.count_documents({"sentiment_label": "NEGATIVE"})
    
    # Time-based hourly aggregation (for line chart)
    hourly_data = list(collection.aggregate([
        {
            "$group": {
                "_id": {"$hour": "$timestamp"},
                "positive": {"$sum": {"$cond": [{"$eq": ["$sentiment_label", "POSITIVE"]}, 1, 0]}},
                "neutral": {"$sum": {"$cond": [{"$eq": ["$sentiment_label", "NEUTRAL"]}, 1, 0]}},
                "negative": {"$sum": {"$cond": [{"$eq": ["$sentiment_label", "NEGATIVE"]}, 1, 0]}},
                "total": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}  # Sort by hour
    ]))
    
    return jsonify({
        "overall": {
            "positive": positive,
            "neutral": neutral,
            "negative": negative
        },
        "hourly": hourly_data
    })

# New route for dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)