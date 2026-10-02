from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Book Recommendation API")

# Load ML artifacts from the models folder
model = joblib.load('../models/xgboost_conversion_model.pkl')
scaler = joblib.load('../models/feature_scaler.pkl')

# Load your books dataset on startup
DATASET_PATH = 'goodreads_books.csv'
if os.path.exists(DATASET_PATH):
    print("Loading Goodreads dataset into memory...")
    books_df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {len(books_df)} books successfully!")
else:
    books_df = None

class RecommendationRequest(BaseModel):
    user_avg_rating: float
    user_avg_pages_read: float
    user_conversion_rate_90d: float
    previous_interactions: float
    previous_conversions_90d: float
    top_n: int = 5

@app.post("/recommend-from-dataset")
def recommend_books(data: RecommendationRequest):
    if books_df is None:
        return {"error": "goodreads_books.csv not found on the backend server."}
    
    # Copy the dataset so we don't modify the original
    df = books_df.copy()
    
    # 1. Clean and fill missing numerical values to prevent crashes
    df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce').fillna(300.0)
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce').fillna(4.0)
    df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce').fillna(2015.0)
    
    # 2. Derive num_canonical_genres from the 'genres' column safely
    if 'genres' in df.columns:
        df['num_canonical_genres'] = df['genres'].fillna('').apply(lambda x: len(str(x).split(',')) if str(x).strip() else 1)
    else:
        df['num_canonical_genres'] = 1

    # 3. Inject the user's real-time behavioral features into every book row
    df['user_avg_rating'] = data.user_avg_rating
    df['user_avg_pages_read'] = data.user_avg_pages_read
    df['user_conversion_rate_90d'] = data.user_conversion_rate_90d
    df['previous_interactions'] = data.previous_interactions
    df['previous_conversions_90d'] = data.previous_conversions_90d
    
    # Ensure columns match the exact 9 features your XGBoost model expects
    feature_columns = [
        'num_pages', 'average_rating', 'publication_year',
        'previous_interactions', 'previous_conversions_90d',
        'user_conversion_rate_90d', 'user_avg_rating',
        'user_avg_pages_read', 'num_canonical_genres'
    ]
    
    # Extract features, scale, and predict probabilities for all books
    X = df[feature_columns]
    scaled_features = scaler.transform(X)
    probabilities = model.predict_proba(scaled_features)[:, 1]
    
    # Attach predicted probability
    df['reading_probability'] = [round(float(p) * 100, 2) for p in probabilities]
    
    # Sort by highest reading probability and pick the Top N
    top_books = df.sort_values(by='reading_probability', ascending=False).head(data.top_n)
    
    # Ensure book_id and title columns exist for the response
    if 'book_id' not in top_books.columns:
        top_books['book_id'] = range(len(top_books))
    if 'title' not in top_books.columns:
        top_books['title'] = "Unknown Title"

    return {
        "top_n": data.top_n,
        "recommendations": top_books[['book_id', 'title', 'reading_probability']].to_dict(orient='records')
    }