# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# # Initialize the server
# app = FastAPI(title="Book Recommendation API")

# # Load your ML artifacts 
# model = joblib.load('xgboost_conversion_model.pkl')
# scaler = joblib.load('feature_scaler.pkl')

# # Define the exact 9 features the frontend must send
# class BookRequest(BaseModel):
#     num_pages: float
#     average_rating: float
#     publication_year: float
#     previous_interactions: float
#     previous_conversions_90d: float
#     user_conversion_rate_90d: float
#     user_avg_rating: float
#     user_avg_pages_read: float
#     num_canonical_genres: float

# @app.post("/predict")
# def predict_reading_probability(data: BookRequest):
#     # Convert incoming data into a DataFrame format the scaler understands
#     input_df = pd.DataFrame([data.model_dump()])
    
#     # Scale and predict
#     scaled_features = scaler.transform(input_df)
#     probability = model.predict_proba(scaled_features)[0][1]
    
#     # Return the probability to the frontend as a JSON response
#     return {"reading_probability": round(probability * 100, 2)}




# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# # Initialize the server
# app = FastAPI(title="Book Recommendation API")

# # Load your ML artifacts 
# model = joblib.load('xgboost_conversion_model.pkl')
# scaler = joblib.load('feature_scaler.pkl')

# # Define the exact 9 features the frontend must send
# class BookRequest(BaseModel):
#     num_pages: float
#     average_rating: float
#     publication_year: float
#     previous_interactions: float
#     previous_conversions_90d: float
#     user_conversion_rate_90d: float
#     user_avg_rating: float
#     user_avg_pages_read: float
#     num_canonical_genres: float

# @app.post("/predict")
# def predict_reading_probability(data: BookRequest):
#     # Use model_dump() for modern Pydantic compatibility
#     input_df = pd.DataFrame([data.model_dump()])
    
#     # Scale and predict
#     scaled_features = scaler.transform(input_df)
    
#     # Explicitly wrap the prediction in float() to fix the numpy formatting error
#     probability = float(model.predict_proba(scaled_features)[0][1])
    
#     # Return the probability to the frontend as a JSON response
#     return {"reading_probability": round(probability * 100, 2)}



# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import os

# app = FastAPI(title="Book Recommendation API")

# # 1. Load ML artifacts and your books dataset on startup
# model = joblib.load('xgboost_conversion_model.pkl')
# scaler = joblib.load('feature_scaler.pkl')

# # Load your books dataset (make sure your CSV file is in the same folder)
# # Replace 'books.csv' with your actual dataset filename if it's different
# DATASET_PATH = 'goodreads_books.csv'
# if os.path.exists(DATASET_PATH):
#     books_df = pd.read_csv(DATASET_PATH)
# else:
#     books_df = None

# class RecommendationRequest(BaseModel):
#     user_avg_rating: float
#     user_avg_pages_read: float
#     user_conversion_rate_90d: float
#     previous_interactions: float
#     previous_conversions_90d: float
#     top_n: int = 10

# @app.post("/recommend-from-dataset")
# def recommend_books(data: RecommendationRequest):
#     if books_df is None:
#         return {"error": "Books dataset not found on the backend server."}
    
#     # Copy the dataset so we don't modify the original
#     df = books_df.copy()
    
#     # Inject the user's real-time behavioral features into every book row
#     df['user_avg_rating'] = data.user_avg_rating
#     df['user_avg_pages_read'] = data.user_avg_pages_read
#     df['user_conversion_rate_90d'] = data.user_conversion_rate_90d
#     df['previous_interactions'] = data.previous_interactions
#     df['previous_conversions_90d'] = data.previous_conversions_90d
    
#     # Ensure the columns match the 9 features your XGBoost model expects
#     feature_columns = [
#         'num_pages', 'average_rating', 'publication_year',
#         'previous_interactions', 'previous_conversions_90d',
#         'user_conversion_rate_90d', 'user_avg_rating',
#         'user_avg_pages_read', 'num_canonical_genres'
#     ]
    
#     # Extract features, scale them, and predict for all books in the dataset
#     X = df[feature_columns]
#     scaled_features = scaler.transform(X)
#     probabilities = model.predict_proba(scaled_features)[:, 1]
    
#     # Attach the predicted probability to each book
#     df['reading_probability'] = [round(float(p) * 100, 2) for p in probabilities]
    
#     # Sort by highest reading probability and pick the Top N
#     top_books = df.sort_values(by='reading_probability', ascending=False).head(data.top_n)
    
#     # Return the top recommendations
#     return {
#         "top_n": data.top_n,
#         "recommendations": top_books[['book_id', 'title', 'reading_probability']].to_dict(orient='records')
#     }


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Book Recommendation API")

# Load ML artifacts 
model = joblib.load('xgboost_conversion_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

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