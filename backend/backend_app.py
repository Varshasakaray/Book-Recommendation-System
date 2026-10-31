
from pathlib import Path
from contextlib import closing
import ast
import json
import sqlite3
import random
import shap
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "bookwise" / "data"
MODEL_DIR = ROOT_DIR / "bookwise" / "models"

BOOKS_PATH = DATA_DIR / "goodreads_books.csv"
TEMPORAL_DATA_PATH = DATA_DIR / "training_data_temporal_corrected.csv"

# Optional mapping file containing user_id and name columns.
USER_NAMES_PATH = DATA_DIR / "user_names.csv"

RANKER_PATH = MODEL_DIR / "xgboost_ranker_model3.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler_3.pkl"

INDEX_PATH = Path(__file__).resolve().parent / "user_history_index.sqlite3"

MAX_CANDIDATE_BOOKS = 5000
MAX_USERS_IN_DROPDOWN = 100

FEATURES = [
    "num_pages",
    "average_rating",
    "publication_year",
    "previous_interactions",
    "previous_conversions_90d",
    "user_conversion_rate_90d",
    "user_avg_rating",
    "user_avg_pages_read",
    "num_canonical_genres",
    "genre_conv_rate_extracted",
    "author_conv_rate_extracted",
    "page_difference",
]

PROFILE_COLUMNS = [
    "previous_interactions",
    "previous_conversions_90d",
    "user_conversion_rate_90d",
    "user_avg_rating",
    "user_avg_pages_read",
]

TEMPORAL_COLUMNS = [
    "user_id",
    "book_id",
    "date_added",
    "previous_interactions",
    "previous_conversions_90d",
    "user_conversion_rate_90d",
    "user_avg_rating",
    "user_avg_pages_read",
    "genre_conversion_rates",
    "author_conversion_rates",
    "canonical_genres_list",
]

BOOK_COLUMNS = [
    "book_id",
    "title",
    "author_ids",
    "author_names",
    "average_rating",
    "publication_year",
    "num_pages",
]

# Human-readable names for the model features.
FEATURE_DISPLAY_NAMES = {
    "num_pages": "Book page count",
    "average_rating": "Book average rating",
    "publication_year": "Publication year",
    "previous_interactions": "Previous reading interactions",
    "previous_conversions_90d": "Recent completed interactions",
    "user_conversion_rate_90d": "Recent completion rate",
    "user_avg_rating": "Your average book rating",
    "user_avg_pages_read": "Your average pages read",
    "num_canonical_genres": "Number of book genres",
    "genre_conv_rate_extracted": "Genre preference",
    "author_conv_rate_extracted": "Author preference",
    "page_difference": "Closeness to your average book length",
}

app = FastAPI(title="BookWise Recommendation API")

ranker = None
scaler = None
candidate_books = None
shap_explainer = None
shap_error = None

# Maps user_id -> actual display name, when supplied.
user_name_map = {}


# ============================================================
# 2. SAFE VALUE PARSING
# ============================================================

def parse_list(value):
    """Parse list-like values stored as JSON or Python literals."""

    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    return parsed if isinstance(parsed, list) else []


def safe_float(value, default=0.0):
    """Convert a value to a finite float."""

    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def extract_mean_rate(rate_value):
    """
    Match the training notebook's extract_mean_rate logic.
    """

    items = parse_list(rate_value)

    total_eligible = 0.0
    total_converted = 0.0
    valid_counts_found = False
    individual_rates = []

    for item in items:
        if not isinstance(item, dict):
            continue

        eligible = safe_float(item.get("eligible"), -1)
        converted = safe_float(item.get("converted"), -1)

        if eligible > 0 and converted >= 0:
            total_eligible += eligible
            total_converted += converted
            valid_counts_found = True

        rate = safe_float(item.get("conversion_rate"), -1)

        if 0 <= rate <= 1:
            individual_rates.append(rate)

    if valid_counts_found and total_eligible > 0:
        return total_converted / total_eligible

    if individual_rates:
        return float(np.mean(individual_rates))

    return 0.0


# ============================================================
# 3. USER DISPLAY NAMES
# ============================================================

def load_user_names():
    """
    Load optional user names.

    Expected CSV columns:
        user_id,name

    If no mapping file exists, the application will use
    a readable fallback such as 'User 12345'.
    """

    global user_name_map

    user_name_map = {}

    if not USER_NAMES_PATH.exists():
        print(
            "No user_names.csv found. "
            "Using user ID labels as fallback."
        )
        return

    try:
        names = pd.read_csv(
            USER_NAMES_PATH,
            dtype={"user_id": str, "name": str},
        )

        if not {"user_id", "name"}.issubset(names.columns):
            print(
                "user_names.csv must contain user_id and name columns."
            )
            return

        names = names.dropna(subset=["user_id", "name"])

        user_name_map = {
            str(row.user_id).strip(): str(row.name).strip()
            for row in names.itertuples(index=False)
            if str(row.name).strip()
        }

        print(f"Loaded {len(user_name_map)} user display names.")

    except Exception as error:
        print(f"Could not load user names: {error}")


def get_display_name(user_id):
    """Return a real mapped name or a safe fallback."""

    user_id = str(user_id)

    return user_name_map.get(
        user_id,
        f"User {user_id}"
    )


# ============================================================
# 4. LOAD CANDIDATE BOOKS
# ============================================================

def load_candidate_books():
    if not BOOKS_PATH.exists():
        raise FileNotFoundError(f"Books dataset not found: {BOOKS_PATH}")

    print("Loading a random sample of candidate books...")

    # The Goodreads dataset has roughly 2.3 million books. 
    # We sample ~10,000 rows during read to be safe, then trim to 5,000.
    sample_probability = 10000 / 2300000

    books = pd.read_csv(
        BOOKS_PATH,
        usecols=lambda column: column in BOOK_COLUMNS,
        # Keep the header (i == 0), then randomly skip rows
        skiprows=lambda i: i > 0 and random.random() > sample_probability,
        encoding_errors="replace"
    )

    required = {"book_id", "title"}
    if not required.issubset(books.columns):
        raise ValueError("Books dataset must contain book_id and title.")

    # Shuffle and trim to exactly MAX_CANDIDATE_BOOKS
    books = books.sample(frac=1, random_state=42).head(MAX_CANDIDATE_BOOKS)

    books = books.dropna(subset=["book_id"]).copy()
    books["book_id"] = books["book_id"].astype(str)
    books = books.drop_duplicates(subset=["book_id"]).reset_index(drop=True)

    for column in ["num_pages", "average_rating", "publication_year"]:
        if column not in books.columns:
            books[column] = np.nan
        books[column] = pd.to_numeric(books[column], errors="coerce")
        
        median_value = books[column].median()
        if pd.isna(median_value):
            median_value = 0.0
        books[column] = books[column].fillna(median_value)

    for column in ["author_ids", "author_names"]:
        if column not in books.columns:
            books[column] = None

    print(f"Loaded {len(books)} random candidate books.")
    return books


# ============================================================
# 5. SQLITE USER HISTORY INDEX
# ============================================================

def index_is_current():

    if not INDEX_PATH.exists():
        return False

    if not TEMPORAL_DATA_PATH.exists():
        return False

    try:
        with closing(sqlite3.connect(INDEX_PATH)) as connection:

            row = connection.execute(
                """
                SELECT value
                FROM index_metadata
                WHERE key = 'source_mtime'
                """
            ).fetchone()

            return (
                row is not None
                and float(row[0]) == TEMPORAL_DATA_PATH.stat().st_mtime
            )

    except (sqlite3.Error, ValueError):
        return False


def build_user_history_index():

    if not TEMPORAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Temporal dataset not found: {TEMPORAL_DATA_PATH}"
        )

    print("Building user history index from temporal dataset...")
    print("This is a one-time streaming operation.")

    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    with closing(sqlite3.connect(INDEX_PATH)) as connection:

        connection.execute("""
            CREATE TABLE user_profiles (
                user_id TEXT PRIMARY KEY,
                interaction_count INTEGER NOT NULL,
                latest_date TEXT,
                previous_interactions REAL,
                previous_conversions_90d REAL,
                user_conversion_rate_90d REAL,
                user_avg_rating REAL,
                user_avg_pages_read REAL,
                genre_conversion_rates TEXT,
                author_conversion_rates TEXT
            )
        """)

        connection.execute("""
            CREATE TABLE user_books (
                user_id TEXT NOT NULL,
                book_id TEXT NOT NULL,
                PRIMARY KEY (user_id, book_id)
            )
        """)

        connection.execute("""
            CREATE TABLE book_genres (
                book_id TEXT PRIMARY KEY,
                canonical_genres_list TEXT
            )
        """)

        connection.execute("""
            CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        candidate_ids = set(
            candidate_books["book_id"].astype(str)
        )

        chunk_number = 0

        for chunk in pd.read_csv(
            TEMPORAL_DATA_PATH,
            usecols=lambda column: column in TEMPORAL_COLUMNS,
            chunksize=50000,
            encoding_errors="replace",
        ):

            chunk_number += 1

            # Remove missing IDs before converting to strings.
            chunk = chunk.dropna(
                subset=["user_id", "book_id"]
            ).copy()

            chunk["user_id"] = chunk["user_id"].astype(str)
            chunk["book_id"] = chunk["book_id"].astype(str)

            counts = (
                chunk.groupby("user_id")
                .size()
                .to_dict()
            )

            connection.executemany(
                """
                INSERT INTO user_profiles (
                    user_id,
                    interaction_count
                )
                VALUES (?, ?)

                ON CONFLICT(user_id)
                DO UPDATE SET
                    interaction_count =
                    interaction_count + excluded.interaction_count
                """,
                [
                    (str(user_id), int(count))
                    for user_id, count in counts.items()
                ],
            )

            connection.executemany(
                """
                INSERT OR IGNORE INTO user_books (
                    user_id,
                    book_id
                )
                VALUES (?, ?)
                """,
                chunk[["user_id", "book_id"]].itertuples(
                    index=False,
                    name=None
                ),
            )

            if "date_added" in chunk.columns:
                chunk = chunk.sort_values(
                    "date_added",
                    na_position="first"
                )

            latest_rows = (
                chunk.groupby("user_id", sort=False)
                .tail(1)
            )

            profile_records = []

            for row in latest_rows.itertuples(index=False):

                values = row._asdict()
                user_id = str(values["user_id"])

                date_value = values.get("date_added", "")

                latest_date = (
                    ""
                    if pd.isna(date_value)
                    else str(date_value)
                )

                profile_records.append((
                    user_id,
                    latest_date,
                    safe_float(values.get("previous_interactions")),
                    safe_float(values.get("previous_conversions_90d")),
                    safe_float(values.get("user_conversion_rate_90d")),
                    safe_float(values.get("user_avg_rating")),
                    safe_float(values.get("user_avg_pages_read")),
                    str(values.get("genre_conversion_rates", "[]")),
                    str(values.get("author_conversion_rates", "[]")),
                    latest_date,
                ))

            # Protect the latest profile snapshot from older chunks.
            connection.executemany(
                """
                UPDATE user_profiles
                SET
                    latest_date = ?,
                    previous_interactions = ?,
                    previous_conversions_90d = ?,
                    user_conversion_rate_90d = ?,
                    user_avg_rating = ?,
                    user_avg_pages_read = ?,
                    genre_conversion_rates = ?,
                    author_conversion_rates = ?

                WHERE user_id = ?
                  AND (
                      latest_date IS NULL
                      OR latest_date <= ?
                  )
                """,
                profile_records,
            )

            if "canonical_genres_list" in chunk.columns:

                genre_rows = chunk[
                    chunk["book_id"].isin(candidate_ids)
                ][
                    ["book_id", "canonical_genres_list"]
                ]

                genre_records = []

                for book_id, genres in genre_rows.itertuples(
                    index=False,
                    name=None
                ):

                    parsed = parse_list(genres)

                    if parsed:
                        genre_records.append((
                            str(book_id),
                            str(genres),
                        ))

                connection.executemany(
                    """
                    INSERT OR REPLACE INTO book_genres (
                        book_id,
                        canonical_genres_list
                    )
                    VALUES (?, ?)
                    """,
                    genre_records,
                )

            if chunk_number % 10 == 0:
                print(f"Processed {chunk_number} chunks...")

            connection.commit()

        connection.execute(
            """
            INSERT OR REPLACE INTO index_metadata (
                key,
                value
            )
            VALUES ('source_mtime', ?)
            """,
            (str(TEMPORAL_DATA_PATH.stat().st_mtime),),
        )

        connection.commit()

    print("User history index created successfully.")


def get_connection():

    if not INDEX_PATH.exists():
        raise RuntimeError(
            "User history index has not been built."
        )

    return sqlite3.connect(INDEX_PATH)


# ============================================================
# 6. USER PROFILE RETRIEVAL
# ============================================================

def get_user_profile(user_id):

    with closing(get_connection()) as connection:

        row = connection.execute(
            """
            SELECT
                user_id,
                interaction_count,
                previous_interactions,
                previous_conversions_90d,
                user_conversion_rate_90d,
                user_avg_rating,
                user_avg_pages_read,
                genre_conversion_rates,
                author_conversion_rates

            FROM user_profiles
            WHERE user_id = ?
            """,
            (str(user_id),),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="User not found."
        )

    return {
        "user_id": row[0],
        "interaction_count": int(row[1] or 0),
        "previous_interactions": safe_float(row[2]),
        "previous_conversions_90d": safe_float(row[3]),
        "user_conversion_rate_90d": safe_float(row[4]),
        "user_avg_rating": safe_float(row[5]),
        "user_avg_pages_read": safe_float(row[6]),
        "genre_conversion_rates": row[7] or "[]",
        "author_conversion_rates": row[8] or "[]",
    }


def get_interacted_book_ids(user_id):

    with closing(get_connection()) as connection:

        rows = connection.execute(
            """
            SELECT book_id
            FROM user_books
            WHERE user_id = ?
            """,
            (str(user_id),),
        ).fetchall()

    return {str(row[0]) for row in rows}


# ============================================================
# 7. BATCHED CANONICAL GENRE COUNT RETRIEVAL
# ============================================================

def get_canonical_genre_counts(book_ids):

    if not book_ids:
        return {}

    book_ids = list(dict.fromkeys(
        str(book_id)
        for book_id in book_ids
    ))

    genre_counts = {}

    with closing(get_connection()) as connection:

        for i in range(0, len(book_ids), 900):

            batch = book_ids[i:i + 900]

            placeholders = ",".join(
                ["?"] * len(batch)
            )

            query = f"""
                SELECT
                    book_id,
                    canonical_genres_list
                FROM book_genres
                WHERE book_id IN ({placeholders})
            """

            rows = connection.execute(
                query,
                batch
            ).fetchall()

            for book_id, genres in rows:

                genre_counts[str(book_id)] = len(
                    parse_list(genres)
                )

    return genre_counts


# ============================================================
# 8. SHAP EXPLANATIONS
# ============================================================

def initialize_shap_explainer():
    """
    Initialize SHAP using the exact XGBoost Booster.

    If SHAP initialization fails, recommendations remain
    available; the API reports that explanations are unavailable.
    """

    global shap_explainer, shap_error

    try:
        booster = ranker.get_booster()

        shap_explainer = shap.TreeExplainer(booster)
        shap_error = None

        print("SHAP TreeExplainer initialized successfully.")

    except Exception as error:

        shap_explainer = None
        shap_error = str(error)

        print(
            "WARNING: SHAP could not be initialized: "
            f"{shap_error}"
        )


def generate_shap_explanations(X_raw, X_scaled):
    """
    Explain the model predictions for the final recommended books.

    X_scaled:
        Exact scaled features passed to ranker.predict().

    X_raw:
        Corresponding original, unscaled feature values,
        used only to display understandable feature values.
    """

    if shap_explainer is None:

        return [
            {
                "status": "unavailable",
                "message": (
                    "SHAP explanations are unavailable. "
                    "Check backend startup logs."
                ),
                "positive_factors": [],
                "negative_factors": [],
            }
            for _ in range(len(X_raw))
        ]

    try:

        # Calculate SHAP values for the final recommendations only.
        shap_result = shap_explainer.shap_values(
            X_scaled,
            check_additivity=False
        )

        # Handle list output from some SHAP/model combinations.
        if isinstance(shap_result, list):
            shap_result = shap_result[0]

        shap_array = np.asarray(shap_result)

        # Handle a possible extra output dimension.
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 0]

        if shap_array.ndim == 1:
            shap_array = shap_array.reshape(1, -1)

        if shap_array.shape != (
            len(X_raw),
            len(FEATURES)
        ):
            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_array.shape}"
            )

        all_explanations = []

        for row_index in range(len(X_raw)):

            contributions = shap_array[row_index]
            raw_values = X_raw.iloc[row_index]

            factors = []

            for feature_index, feature in enumerate(FEATURES):

                contribution = safe_float(
                    contributions[feature_index]
                )

                factors.append({
                    "feature": FEATURE_DISPLAY_NAMES.get(
                        feature,
                        feature
                    ),
                    "feature_key": feature,
                    "feature_value": safe_float(
                        raw_values[feature]
                    ),
                    "contribution": contribution,
                })

            positive = sorted(
                [
                    factor for factor in factors
                    if factor["contribution"] > 0
                ],
                key=lambda item: item["contribution"],
                reverse=True,
            )[:3]

            negative = sorted(
                [
                    factor for factor in factors
                    if factor["contribution"] < 0
                ],
                key=lambda item: item["contribution"],
            )[:3]

            all_explanations.append({
                "status": "available",
                "positive_factors": positive,
                "negative_factors": negative,
            })

        return all_explanations

    except Exception as error:

        print(f"SHAP calculation failed: {error}")

        return [
            {
                "status": "unavailable",
                "message": str(error),
                "positive_factors": [],
                "negative_factors": [],
            }
            for _ in range(len(X_raw))
        ]


# ============================================================
# 9. API REQUEST MODEL
# ============================================================

class RecommendationRequest(BaseModel):

    user_id: str
    top_n: int = Field(
        default=10,
        ge=5,
        le=20
    )
    min_pages: int = Field(default=0)     # NEW
    max_pages: int = Field(default=5000)  # NEW


# ============================================================
# 10. STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    global ranker, scaler, candidate_books

    print("Starting BookWise backend...")

    for path in [RANKER_PATH, SCALER_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact missing: {path}"
            )

    print("Loading XGBRanker...")
    ranker = joblib.load(RANKER_PATH)

    print("Loading feature scaler...")
    scaler = joblib.load(SCALER_PATH)

    print("Loading user display names...")
    load_user_names()

    candidate_books = load_candidate_books()

    print("Initializing SHAP...")
    initialize_shap_explainer()

    if not index_is_current():
        build_user_history_index()
    else:
        print("Using existing user history index.")

    print("BookWise backend is ready.")


# ============================================================
# 11. GET AVAILABLE USERS
# ============================================================

@app.get("/users")
def get_users():

    if not INDEX_PATH.exists():

        raise HTTPException(
            status_code=503,
            detail="User history index is not ready."
        )

    with closing(get_connection()) as connection:

        rows = connection.execute(
            """
            SELECT
                user_id,
                interaction_count
            FROM user_profiles
            ORDER BY interaction_count DESC
            LIMIT ?
            """,
            (MAX_USERS_IN_DROPDOWN,),
        ).fetchall()

    return {
        "users": [
            {
                "user_id": str(user_id),
                "display_name": get_display_name(user_id),
                "interaction_count": int(count),
            }
            for user_id, count in rows
        ]
    }


# ============================================================
# 12. GET SELECTED USER PROFILE
# ============================================================

@app.get("/users/{user_id}/profile")
def user_profile_endpoint(user_id: str):

    profile = get_user_profile(user_id)

    return {
        "user_id": profile["user_id"],
        "display_name": get_display_name(user_id),

        "previous_interactions":
            profile["previous_interactions"],

        "user_avg_rating":
            profile["user_avg_rating"],

        "user_avg_pages_read":
            profile["user_avg_pages_read"],

        "previous_conversions_90d":
            profile["previous_conversions_90d"],

        "user_conversion_rate_90d":
            profile["user_conversion_rate_90d"],
    }


# ============================================================
# 13. PERSONALIZED RECOMMENDATIONS
# ============================================================

@app.post("/recommendations")
def recommend_books(request: RecommendationRequest):

    if (
        ranker is None
        or scaler is None
        or candidate_books is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Recommendation engine is not ready."
        )

    # A. Retrieve the user's historical profile.
    profile = get_user_profile(request.user_id)

    display_name = get_display_name(request.user_id)

    # B. Exclude books already present in the user's history.
    interacted_ids = get_interacted_book_ids(
        request.user_id
    )

    candidates = candidate_books[
        ~candidate_books["book_id"].isin(interacted_ids)
    ].copy()

    if candidates.empty:

        return {
            "user_id": request.user_id,
            "display_name": display_name,
            "top_n": request.top_n,
            "recommendations": [],
        }

    #added..........
    # Ensure num_pages is numeric so we can filter mathematically
    candidates["num_pages"] = pd.to_numeric(candidates["num_pages"], errors="coerce").fillna(0)

    # 1. ALWAYS filter out completely broken metadata (0-5 pages)
    # 2. ENFORCE the user's requested page range from the frontend slider
    candidates = candidates[
        (candidates["num_pages"] > 5) &
        (candidates["num_pages"] >= request.min_pages) & 
        (candidates["num_pages"] <= request.max_pages)
    ].copy()

    if candidates.empty:
        return {
            "user_id": request.user_id,
            "display_name": display_name,
            "top_n": request.top_n,
            "recommendations": [],
        }
    #added........

    # C. Add historical user features to every candidate.
    for column in PROFILE_COLUMNS:
        candidates[column] = profile[column]

    # D. Extract user-level genre and author conversion rates.
    candidates["genre_conv_rate_extracted"] = extract_mean_rate(
        profile["genre_conversion_rates"]
    )

    candidates["author_conv_rate_extracted"] = extract_mean_rate(
        profile["author_conversion_rates"]
    )

    # E. Retrieve canonical genre counts in batches.
    genre_counts = get_canonical_genre_counts(
        candidates["book_id"].astype(str).tolist()
    )

    candidates["num_canonical_genres"] = (
        candidates["book_id"]
        .astype(str)
        .map(genre_counts)
        .fillna(0)
    )

    # Calculate relative page difference for the model
    candidates["num_pages"] = pd.to_numeric(candidates["num_pages"], errors="coerce").fillna(0)
    candidates["page_difference"] = (candidates["num_pages"] - candidates["user_avg_pages_read"]).abs()


    # F. Ensure numerical feature values are valid.
    for column in FEATURES:

        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce"
        )

    candidates[FEATURES] = candidates[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    candidates[FEATURES] = candidates[FEATURES].fillna(0.0)

    # G. Use exact training feature order.
    X = candidates[FEATURES].copy()

    
    # TEMPORARY PAGE-PREFERENCE SENSITIVITY TEST

    print("\n========== PAGE PREFERENCE TEST ==========")

    for avg_pages in [100, 200, 288, 400, 600]:

        X_test = X.copy()

        X_test["user_avg_pages_read"] = avg_pages

        # YOU MUST ADD THIS LINE TO RECALCULATE THE DIFFERENCE FOR THE TEST:
        X_test["page_difference"] = (X_test["num_pages"] - avg_pages).abs()

        test_scores = ranker.predict(
            scaler.transform(X_test)
        )

        top_indices = np.argsort(test_scores)[::-1][:20]

        top_test = candidates.iloc[top_indices].copy()

        print(f"\nUser average pages: {avg_pages}")

        print("Top 20 average book pages:",
            round(top_test["num_pages"].mean(), 2))

        print("Top 20 median book pages:",
            top_test["num_pages"].median())

        print("Top 10 recommended books:")

        print(
            top_test[["title", "num_pages"]]
            .head(10)
            .to_string(index=False)
        )

    print("\n==========================================\n")




    # H. Scale and rank using the trained XGBoost Ranker.
    X_scaled = scaler.transform(X)

    ranking_scores = ranker.predict(X_scaled)

    # TEMPORARY DIAGNOSTICS
    candidates["rank_score"] = ranking_scores

    print("\n========== RANKING DIAGNOSTICS ==========")

    print("Total candidates:", len(candidates))

    print(
        "Unique ranking scores:",
        candidates["rank_score"].nunique()
    )

    print("\nTop 10 most frequent ranking scores:")
    print(
        candidates["rank_score"]
        .value_counts()
        .head(10)
    )

    print("\nTop 25 ranked candidates:")
    print(
        candidates.nlargest(25, "rank_score")[
            ["title", "num_pages", "user_avg_pages_read", "rank_score"]
        ].to_string(index=False)
    )

    print("\nPage-count distribution of top 25:")
    print(
        candidates.nlargest(25, "rank_score")["num_pages"]
        .describe()
    )

    print("=========================================\n")





    candidates["rank_score"] = ranking_scores

    # I. Select the requested top N books.
    top_books = (
        candidates
        .sort_values("rank_score", ascending=False)
        .head(request.top_n)
        .copy()
    )

    # J. Calculate SHAP explanations for only the final books.
    top_indices = top_books.index

    X_top_raw = X.loc[top_indices].reset_index(drop=True)

    X_top_scaled = scaler.transform(X_top_raw)

    explanations = generate_shap_explanations(
        X_top_raw,
        X_top_scaled
    )

    # K. Construct the API response.
    recommendations = []

    for position, (_, book) in enumerate(top_books.iterrows()):

        author_names = book.get("author_names")
        parsed_authors = parse_list(author_names)

        if parsed_authors:

            authors = ", ".join(
                str(author)
                for author in parsed_authors
            )

        elif (
            isinstance(author_names, str)
            and author_names.strip()
        ):

            authors = author_names

        else:

            authors = "Unknown Author"

        recommendations.append({

            "book_id": str(book["book_id"]),

            "title": str(
                book.get("title", "Unknown Title")
            ),

            "authors": authors,

            "average_rating": safe_float(
                book.get("average_rating")
            ),

            "num_pages": safe_float(
                book.get("num_pages")
            ),

            "rank_score": safe_float(
                book.get("rank_score")
            ),

            "explanation": explanations[position],
        })

    return {
        "user_id": request.user_id,
        "display_name": display_name,
        "top_n": request.top_n,
        "recommendations": recommendations,
    }


# ============================================================
# 14. SHAP STATUS
# ============================================================

@app.get("/explanation-status")
def explanation_status():

    return {
        "shap_available": shap_explainer is not None,
        "message": (
            "SHAP explanations are ready."
            if shap_explainer is not None
            else f"SHAP unavailable: {shap_error}"
        ),
    }




