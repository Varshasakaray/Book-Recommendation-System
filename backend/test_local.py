import requests

# The URL where your FastAPI server is listening
url = "http://127.0.0.1:8000/predict"

# Realistic sample data for a user and a book
payload = {
    "num_pages": 350,
    "average_rating": 4.2,
    "publication_year": 2018,
    "previous_interactions": 12,
    "previous_conversions_90d": 4,
    "user_conversion_rate_90d": 0.33,
    "user_avg_rating": 4.0,
    "user_avg_pages_read": 320.5,
    "num_canonical_genres": 2
}

# Send the request to your API
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Model Prediction Response:", response.json())