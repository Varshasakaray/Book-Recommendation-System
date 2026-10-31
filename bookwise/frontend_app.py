
from pathlib import Path

import requests
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

API_URL = "http://127.0.0.1:8000"

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
        raise ValueError(
            "Discovery matrix rows do not match the book mapping."
        )

    books["book_id"] = books["book_id"].astype(str)

    return matrix, books


# -----------------------------
# 3. SHAP explanation display
# -----------------------------

def show_shap_explanation(explanation):
    """
    Display SHAP explanations returned by the backend.
    """

    if not isinstance(explanation, dict):
        st.info("SHAP explanation is not available for this book.")
        return

    if explanation.get("status") != "available":
        st.info("SHAP explanation is not available for this book.")
        return

    positive = explanation.get("positive_factors", [])
    negative = explanation.get("negative_factors", [])

    if not positive and not negative:
        st.info("No feature contributions are available.")
        return

    st.markdown("**Factors increasing the recommendation score**")

    for factor in positive:
        label = factor.get("feature", "Unknown feature")
        value = factor.get("feature_value", "N/A")
        contribution = float(factor.get("contribution", 0))

        st.markdown(
            f"- **{label}:** {value} "
            f"(SHAP contribution: `+{contribution:.4f}`)"
        )

    # st.markdown("**Factors decreasing the recommendation score**")

    # for factor in negative:
    #     label = factor.get("feature", "Unknown feature")
    #     value = factor.get("feature_value", "N/A")
    #     contribution = float(factor.get("contribution", 0))

    #     st.markdown(
    #         f"- **{label}:** {value} "
    #         f"(SHAP contribution: `{contribution:.4f}`)"
    #     )


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
                st.caption(
                    f"Similarity Score: {float(book['similarity']):.3f}"
                )

            if "rank_score" in book.index:
                st.caption(
                    f"Ranking Score: {float(book['rank_score']):.3f}"
                )


# -----------------------------
# 5. Navigation
# -----------------------------

mode = st.sidebar.radio(
    "Choose a mode",
    [
        "Explore similar books",
        "Personalized recommendations",
    ],
)


# ============================================================
# MODE 1: EXPLORE SIMILAR BOOKS
# ============================================================

if mode == "Explore similar books":

    st.header("Discover books with similar plots")

    st.write(
        "Choose a book to find other books using "
        "plot-description TF-IDF and scaled book metadata."
    )

    try:

        with st.spinner(
            "Loading discovery engine assets into memory..."
        ):
            matrix, books = load_discovery_assets()

        book_options = books.drop_duplicates(
            subset=["book_id"]
        ).copy()

        selected_id = st.selectbox(
            "Choose a book",
            options=book_options["book_id"].tolist(),
            format_func=lambda book_id: str(
                book_options.loc[
                    book_options["book_id"] == book_id,
                    "title"
                ].iloc[0]
            ),
        )

        number_of_results = st.slider(
            "Number of similar books",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
        )

        if st.button(
            "Find similar books",
            type="primary"
        ):

            with st.spinner("Computing cosine similarities..."):

                matching_rows = np.flatnonzero(
                    books["book_id"].to_numpy() == selected_id
                )

                if len(matching_rows) == 0:

                    st.error("The selected book was not found.")

                else:

                    idx = int(matching_rows[0])

                    scores = cosine_similarity(
                        matrix[idx],
                        matrix
                    ).ravel()

                    scores[idx] = -1

                    ranked_indices = np.argsort(scores)[::-1]

                    selected_indices = []
                    seen_ids = {selected_id}

                    for candidate_idx in ranked_indices:

                        candidate_id = books.iloc[
                            candidate_idx
                        ]["book_id"]

                        if candidate_id in seen_ids:
                            continue

                        seen_ids.add(candidate_id)
                        selected_indices.append(candidate_idx)

                        if len(selected_indices) >= number_of_results:
                            break

                    results = books.iloc[
                        selected_indices
                    ].copy()

                    results["similarity"] = scores[
                        selected_indices
                    ]

                    st.subheader("Books you might enjoy")

                    show_books(results)

    except FileNotFoundError:

        st.error(
            "Discovery files missing. Put "
            "hybrid_discovery_matrix.npz and "
            "discovery_books_mapping.pkl inside the models folder."
        )

    except Exception as exc:

        st.error(
            f"Could not load the discovery engine: {exc}"
        )


# ============================================================
# MODE 2: PERSONALIZED RECOMMENDATIONS
# ============================================================

else:

    st.header("Personalized recommendations")

    st.write(
        "Discover books ranked according to your historical "
        "reading preferences and interactions."
    )

    try:

        # ----------------------------------------------------
        # Step 1: Retrieve available Goodreads users
        # ----------------------------------------------------

        with st.spinner("Loading Goodreads users..."):

            response = requests.get(
                f"{API_URL}/users",
                timeout=30,
            )

            response.raise_for_status()

            users = response.json().get("users", [])

        if not users:

            st.warning(
                "No users are available in the dataset."
            )

        else:

            # ------------------------------------------------
            # Step 2: Create user options
            # ------------------------------------------------

            user_options = {
                str(user["user_id"]): user
                for user in users
            }

            selected_user_id = st.selectbox(
                "Select a Goodreads user",

                options=list(user_options.keys()),

                format_func=lambda uid: (
                    f"{user_options[uid].get('display_name') or f'User {uid}'} "
                    # f"({user_options[uid].get('interaction_count', 0)} interactions)"
                ),
            )

            selected_user = user_options[selected_user_id]

            selected_display_name = (
                selected_user.get("display_name")
                or f"User {selected_user_id}"
            )

            # ------------------------------------------------
            # Step 3: Fetch selected user's historical profile
            # ------------------------------------------------

            with st.spinner("Loading reading profile..."):

                profile_response = requests.get(
                    f"{API_URL}/users/{selected_user_id}/profile",
                    timeout=30,
                )

                profile_response.raise_for_status()

                profile = profile_response.json()

            display_name = (
                profile.get("display_name")
                or selected_display_name
            )

            st.subheader(f"Reading Profile: {display_name}")

            # Show only the three selected profile metrics.
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Previous Interactions",
                profile.get("previous_interactions", 0),
            )

            col2.metric(
                "Average Rating Given",
                round(
                    profile.get("user_avg_rating", 0),
                    2,
                ),
            )

            col3.metric(
                "Average Pages Read",
                round(
                    profile.get("user_avg_pages_read", 0),
                    0,
                ),
            )

            st.divider()

            # ------------------------------------------------
            # Step 4: Choose recommendation count
            # ------------------------------------------------

            st.subheader("Find Books for You")

            number_of_results = st.slider(
                "Number of recommendations",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                key="personalized_recommendation_count",
            )
            # NEW: Double-ended slider for page range
            page_range = st.slider(
                "Preferred book length (pages)",
                min_value=10,
                max_value=1500,
                value=(50, 800), # Default selected range
                step=10,
            )

            # ------------------------------------------------
            # Step 5: Generate recommendations
            # ------------------------------------------------

            if st.button(
                "Generate Recommendations",
                type="primary",
            ):

                with st.spinner(
                    "Finding books that match your reading preferences..."
                ):

                    recommendation_response = requests.post(
                        f"{API_URL}/recommendations",

                        json={
                            "user_id": selected_user_id,
                            "top_n": number_of_results,
                            "min_pages": page_range[0], # Sends the bottom slider value
                            "max_pages": page_range[1], # Sends the top slider value
                        },

                        timeout=120,
                    )

                    recommendation_response.raise_for_status()

                    recommendation_data = (
                        recommendation_response.json()
                    )

                    recommendations = recommendation_data.get(
                        "recommendations",
                        []
                    )

                # --------------------------------------------
                # Step 6: Display recommendations
                # --------------------------------------------

                if not recommendations:

                    st.info(
                        "No recommendations found for this user."
                    )

                else:

                    st.subheader(
                        f"Recommendations for {display_name}"
                    )

                    st.caption(
                        f"Showing {len(recommendations)} recommendations"
                    )

                    results = pd.DataFrame(recommendations)

                    # Display recommendation cards in two columns.
                    for start in range(0, len(results), 2):

                        columns = st.columns(2)

                        for offset, column in enumerate(columns):

                            index = start + offset

                            if index >= len(results):
                                break

                            book = results.iloc[index]

                            with column:

                                with st.container(border=True):

                                    # Book title
                                    st.markdown(
                                        f"### {book.get('title', 'Untitled')}"
                                    )

                                    # Author
                                    author = book.get("authors")

                                    if (
                                        author is None
                                        or (
                                            isinstance(author, float)
                                            and pd.isna(author)
                                        )
                                    ):
                                        author = "Unknown Author"

                                    st.caption(f"By {author}")

                                    # Rating and pages
                                    rating = book.get(
                                        "average_rating"
                                    )

                                    pages = book.get("num_pages")

                                    details = []

                                    if pd.notna(rating):
                                        details.append(
                                            f"⭐ {float(rating):.2f}"
                                        )

                                    if pd.notna(pages):
                                        details.append(
                                            f"📖 {int(float(pages))} pages"
                                        )

                                    if details:
                                        st.write(" · ".join(details))

                                    
                                    # --------------------------------
                                    # Step 7: SHAP explanation
                                    # --------------------------------

                                    explanation = book.get(
                                        "explanation"
                                    )

                                    with st.expander(
                                        "🔍 Why this book?"
                                    ):

                                        show_shap_explanation(
                                            explanation
                                        )

    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the recommendation backend. "
            "Please start the FastAPI server first."
        )

    except requests.exceptions.HTTPError as exc:

        st.error(
            f"Backend returned an error: {exc}"
        )

    except requests.exceptions.RequestException as exc:

        st.error(
            f"Could not complete the recommendation request: {exc}"
        )

    except Exception as exc:

        st.error(
            f"Something went wrong: {exc}"
        )
