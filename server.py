import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load reviews from CSV
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

# List of valid locations
VALID_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ.get('QUERY_STRING', ''))
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            filtered_reviews = reviews
            if location:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]
            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]
            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]

            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
            post_params = parse_qs(request_body)
            location = post_params.get('Location', [None])[0]
            review_body = post_params.get('ReviewBody', [None])[0]

            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Location and ReviewBody are required"]

            if location not in VALID_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Invalid location"]

            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "Location": location,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ReviewBody": review_body,
                "sentiment": self.analyze_sentiment(review_body)
            }
            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()