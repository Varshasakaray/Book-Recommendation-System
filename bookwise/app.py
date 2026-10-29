
# from pathlib import Path

# import joblib
# import numpy as np
# import pandas as pd
# import scipy.sparse as sp
# import streamlit as st
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------
# # 1. Paths and app configuration
# # -----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / "data"
# MODEL_DIR = BASE_DIR / "models"

# st.set_page_config(
#     page_title="BookWise",
#     page_icon="📚",
#     layout="wide",
# )

# st.title("📚 BookWise")
# st.caption("Personalized recommendations and plot-based book discovery")

# # -----------------------------
# # 2. Load discovery assets
# # -----------------------------
# @st.cache_resource
# def load_discovery_assets():
#     matrix_path = MODEL_DIR / "hybrid_discovery_matrix.npz"
#     mapping_path = MODEL_DIR / "discovery_books_mapping.pkl"

#     matrix = sp.load_npz(matrix_path)
#     books = pd.read_pickle(mapping_path).reset_index(drop=True)

#     if matrix.shape[0] != len(books):
#         raise ValueError(
#             "Discovery matrix rows do not match the book mapping."
#         )

#     books["book_id"] = books["book_id"].astype(str)
#     return matrix, books


# # -----------------------------
# # 3. Display book results
# # -----------------------------
# def show_books(results):
#     if results.empty:
#         st.info("No matching books found.")
#         return

#     for _, book in results.iterrows():
#         title = str(book.get("title", "Untitled"))
#         rating = book.get("average_rating", np.nan)
#         pages = book.get("num_pages", np.nan)

#         with st.container(border=True):
#             st.subheader(title)

#             details = []
#             if pd.notna(rating):
#                 details.append(f"⭐ {float(rating):.2f}")
#             if pd.notna(pages):
#                 details.append(f"📄 {float(pages):.0f} pages")

#             if details:
#                 st.caption(" · ".join(details))

#             if "similarity" in book.index:
#                 st.caption(
#                     f"Similarity score: {float(book['similarity']):.3f}"
#                 )


# # -----------------------------
# # 4. Navigation
# # -----------------------------
# # mode = st.sidebar.radio(
# #     "Choose a mode",
# #     [
# #         "Explore similar books",
# #         "Personalized recommendations",
# #     ],
# # )

# # if mode == "Explore similar books":
# #     st.header("Discover books with similar plots")
# #     st.write(
# #         "Choose a book to find other books using plot-description "
# #         "TF-IDF and scaled book metadata."
# #     )

# #     try:
# #         matrix, books = load_discovery_assets()

# #         book_options = books.drop_duplicates(
# #             subset=["book_id"]
# #         ).copy()

# #         selected_id = st.selectbox(
# #             "Choose a book",
# #             options=book_options["book_id"].tolist(),
# #             format_func=lambda book_id: str(
# #                 book_options.loc[
# #                     book_options["book_id"] == book_id, "title"
# #                 ].iloc[0]
# #             ),
# #         )

# #         number_of_results = st.slider(
# #             "Number of similar books",
# #             min_value=5,
# #             max_value=20,
# #             value=10,
# #         )

# #         if st.button("Find similar books", type="primary"):
# #             matching_rows = np.flatnonzero(
# #                 books["book_id"].to_numpy() == selected_id
# #             )

# #             if len(matching_rows) == 0:
# #                 st.error("The selected book was not found.")
# #             else:
# #                 idx = int(matching_rows[0])

# #                 scores = cosine_similarity(
# #                     matrix[idx], matrix
# #                 ).ravel()

# #                 # Exclude the selected book itself.
# #                 scores[idx] = -1

# #                 # Avoid returning duplicate book IDs.
# #                 ranked_indices = np.argsort(scores)[::-1]
# #                 selected_indices = []
# #                 seen_ids = {selected_id}

# #                 for candidate_idx in ranked_indices:
# #                     candidate_id = books.iloc[candidate_idx]["book_id"]

# #                     if candidate_id in seen_ids:
# #                         continue

# #                     seen_ids.add(candidate_id)
# #                     selected_indices.append(candidate_idx)

# #                     if len(selected_indices) >= number_of_results:
# #                         break

# #                 results = books.iloc[selected_indices].copy()
# #                 results["similarity"] = scores[selected_indices]

# #                 st.subheader("Books you might enjoy")
# #                 show_books(results)

# #     except FileNotFoundError:
# #         st.error(
# #             "Discovery files are missing. Put "
# #             "hybrid_discovery_matrix.npz and "
# #             "discovery_books_mapping.pkl inside the models folder."
# #         )
# #     except Exception as exc:
# #         st.error(f"Could not load the discovery engine: {exc}")


# # else:
# #     st.header("Personalized recommendations")
# #     st.info(
# #         "The Ranker and scaler will be connected here next. "
# #         "We need to prepare candidate books with the same features "
# #         "and preprocessing used during training before making predictions."
# #     )



# # -----------------------------
# # 4. Navigation
# # -----------------------------
# mode = st.sidebar.radio(
#     "Choose a mode",
#     [
#         "Explore similar books",
#         "Personalized recommendations",
#     ],
# )

# if mode == "Explore similar books":
#     st.header("Discover books with similar plots")
#     st.write(
#         "Choose a book to find other books using plot-description "
#         "TF-IDF and scaled book metadata."
#     )

#     try:
#         # Show a loading spinner while heavy model assets load into RAM
#         with st.spinner("Loading discovery engine assets into memory..."):
#             matrix, books = load_discovery_assets()

#         book_options = books.drop_duplicates(
#             subset=["book_id"]
#         ).copy()

#         selected_id = st.selectbox(
#             "Choose a book",
#             options=book_options["book_id"].tolist(),
#             format_func=lambda book_id: str(
#                 book_options.loc[
#                     book_options["book_id"] == book_id, "title"
#                 ].iloc[0]
#             ),
#         )

#         number_of_results = st.slider(
#             "Number of similar books",
#             min_value=5,
#             max_value=20,
#             value=10,
#         )

#         if st.button("Find similar books", type="primary"):
#             with st.spinner("Computing cosine similarities..."):
#                 matching_rows = np.flatnonzero(
#                     books["book_id"].to_numpy() == selected_id
#                 )

#                 if len(matching_rows) == 0:
#                     st.error("The selected book was not found.")
#                 else:
#                     idx = int(matching_rows[0])

#                     scores = cosine_similarity(
#                         matrix[idx], matrix
#                     ).ravel()

#                     # Exclude the selected book itself.
#                     scores[idx] = -1

#                     # Avoid returning duplicate book IDs.
#                     ranked_indices = np.argsort(scores)[::-1]
#                     selected_indices = []
#                     seen_ids = {selected_id}

#                     for candidate_idx in ranked_indices:
#                         candidate_id = books.iloc[candidate_idx]["book_id"]

#                         if candidate_id in seen_ids:
#                             continue

#                         seen_ids.add(candidate_id)
#                         selected_indices.append(candidate_idx)

#                         if len(selected_indices) >= number_of_results:
#                             break

#                     results = books.iloc[selected_indices].copy()
#                     results["similarity"] = scores[selected_indices]

#                     st.subheader("Books you might enjoy")
#                     show_books(results)

#     except FileNotFoundError:
#         st.error(
#             "Discovery files are missing. Put "
#             "hybrid_discovery_matrix.npz and "
#             "discovery_books_mapping.pkl inside the models folder."
#         )
#     except Exception as exc:
#         st.error(f"Could not load the discovery engine: {exc}")


# else:
#     st.header("Personalized recommendations")
#     st.info(
#         "The Ranker and scaler will be connected here next. "
#         "We need to prepare candidate books with the same features "
#         "and preprocessing used during training before making predictions."
#     )





from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Paths and app configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="BookWise",
    page_icon="📚",
    layout="wide",
)

st.title("📚 BookWise")
st.caption("Personalized recommendations and plot-based book discovery")

# -----------------------------
# 2. Load discovery assets
# -----------------------------
@st.cache_resource
def load_discovery_assets():
    matrix_path = MODEL_DIR / "hybrid_discovery_matrix.npz"
    mapping_path = MODEL_DIR / "discovery_books_mapping.pkl"

    matrix = sp.load_npz(matrix_path)
    books = pd.read_pickle(mapping_path).reset_index(drop=True)

    if matrix.shape[0] != len(books):
        raise ValueError("Discovery matrix rows do not match the book mapping.")

    books["book_id"] = books["book_id"].astype(str)
    return matrix, books

# -----------------------------
# 3. Load behavioral assets (SAFE MEMORY LIMIT)
# -----------------------------
@st.cache_resource
def load_behavioral_assets():
    ranker_path = MODEL_DIR / "xgboost_ranker_model.pkl"
    scaler_path = MODEL_DIR / "feature_scaler_2.pkl"
    
    ranker = joblib.load(ranker_path)
    scaler = joblib.load(scaler_path)
    return ranker, scaler

@st.cache_resource
def load_candidate_books():
    # HARD LIMIT: nrows=5000 strictly prevents the 2-million row freezing issue
    df = pd.read_csv(DATA_DIR / "goodreads_books.csv", nrows=5000)
    
    df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce').fillna(300)
    df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce').fillna(3.8)
    df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce').fillna(2005)
    
    # Mock extracted pipeline features for frontend testing
    df['num_canonical_genres'] = 2
    df['genre_conv_rate_extracted'] = 0.25
    df['author_conv_rate_extracted'] = 0.30
    
    return df

# -----------------------------
# 4. Display book results
# -----------------------------
def show_books(results):
    if results.empty:
        st.info("No matching books found.")
        return

    for _, book in results.iterrows():
        title = str(book.get("title", "Untitled"))
        rating = book.get("average_rating", np.nan)
        pages = book.get("num_pages", np.nan)

        with st.container(border=True):
            st.subheader(title)

            details = []
            if pd.notna(rating):
                details.append(f"⭐ {float(rating):.2f}")
            if pd.notna(pages):
                details.append(f"📄 {float(pages):.0f} pages")

            if details:
                st.caption(" · ".join(details))

            if "similarity" in book.index:
                st.caption(f"Score/Similarity: {float(book['similarity']):.3f}")

# -----------------------------
# 5. Navigation & Execution
# -----------------------------
mode = st.sidebar.radio(
    "Choose a mode",
    [
        "Explore similar books",
        "Personalized recommendations",
    ],
)

if mode == "Explore similar books":
    st.header("Discover books with similar plots")
    st.write("Choose a book to find other books using plot-description TF-IDF and scaled book metadata.")

    try:
        with st.spinner("Loading discovery engine assets into memory..."):
            matrix, books = load_discovery_assets()

        book_options = books.drop_duplicates(subset=["book_id"]).copy()

        selected_id = st.selectbox(
            "Choose a book",
            options=book_options["book_id"].tolist(),
            format_func=lambda book_id: str(
                book_options.loc[book_options["book_id"] == book_id, "title"].iloc[0]
            ),
        )

        number_of_results = st.slider("Number of similar books", min_value=5, max_value=20, value=10)

        if st.button("Find similar books", type="primary"):
            with st.spinner("Computing cosine similarities..."):
                matching_rows = np.flatnonzero(books["book_id"].to_numpy() == selected_id)

                if len(matching_rows) == 0:
                    st.error("The selected book was not found.")
                else:
                    idx = int(matching_rows[0])
                    scores = cosine_similarity(matrix[idx], matrix).ravel()
                    scores[idx] = -1  # Exclude the selected book itself.

                    ranked_indices = np.argsort(scores)[::-1]
                    selected_indices = []
                    seen_ids = {selected_id}

                    for candidate_idx in ranked_indices:
                        candidate_id = books.iloc[candidate_idx]["book_id"]
                        if candidate_id in seen_ids:
                            continue
                        seen_ids.add(candidate_id)
                        selected_indices.append(candidate_idx)
                        if len(selected_indices) >= number_of_results:
                            break

                    results = books.iloc[selected_indices].copy()
                    results["similarity"] = scores[selected_indices]

                    st.subheader("Books you might enjoy")
                    show_books(results)

    except FileNotFoundError:
        st.error("Discovery files missing. Put hybrid_discovery_matrix.npz and discovery_books_mapping.pkl inside the models folder.")
    except Exception as exc:
        st.error(f"Could not load the discovery engine: {exc}")

else:
    st.header("Personalized recommendations")
    st.write("Score candidate books in real-time based on your historical reading behavior.")
    
    try:
        with st.spinner("Loading ranker assets..."):
            ranker, scaler = load_behavioral_assets()
            candidates_df = load_candidate_books()
        
        with st.form("user_profile"):
            st.subheader("Simulate Your User Profile")
            col1, col2, col3 = st.columns(3)
            user_avg_rating = col1.slider("Your Avg Rating Given", 1.0, 5.0, 4.0, step=0.1)
            user_avg_pages = col2.slider("Preferred Page Count", 50, 1000, 320, step=10)
            interactions = col3.number_input("Previous Interactions", 0, 500, 15)
            
            submitted = st.form_submit_button("Generate Rankings", type="primary")
            
        if submitted:
            with st.spinner("Scoring candidate books using XGBRanker..."):
                score_df = candidates_df.copy()
                
                # Apply simulated user state to all candidates
                score_df['previous_interactions'] = interactions
                score_df['previous_conversions_90d'] = 5
                score_df['user_conversion_rate_90d'] = 0.33
                score_df['user_avg_rating'] = user_avg_rating
                score_df['user_avg_pages_read'] = user_avg_pages
                
                # Define exact feature order used during training
                features = [
                    'num_pages', 'average_rating', 'publication_year',
                    'previous_interactions', 'previous_conversions_90d',
                    'user_conversion_rate_90d', 'user_avg_rating',
                    'user_avg_pages_read', 'num_canonical_genres',
                    'genre_conv_rate_extracted', 'author_conv_rate_extracted'
                ]
                
                # Scale and Predict
                X_cand = scaler.transform(score_df[features])
                score_df['rank_score'] = ranker.predict(X_cand)
                
                # Sort by XGBoost output score
                top_recs = score_df.sort_values(by='rank_score', ascending=False).head(10)
                
                # Map rank score to similarity variable so show_books() renders it cleanly
                top_recs['similarity'] = top_recs['rank_score']  
                st.subheader("Top Ranked for You")
                show_books(top_recs)
                
    except FileNotFoundError:
        st.error("Ranker models missing. Ensure 'xgboost_ranker_model.pkl' and 'feature_scaler_2.pkl' are in the models folder.")
    except Exception as exc:
        st.error(f"Could not load the ranking engine: {exc}")