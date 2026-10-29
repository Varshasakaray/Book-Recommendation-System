# Behavior-Aware Personalized Book Recommendation System

A behavior-aware personalized book recommendation system that predicts the probability that a user will read a book within **90 days** after adding it to their shelf.

Instead of recommending books only because they appear similar to a user's interests, the system focuses on a practical behavioral question:

> **After a user adds a book, how likely are they to actually read it within the next 90 days?**

The system uses historical user behavior, genre-specific behavior, author-specific behavior, ratings, book-length behavior, and book metadata to estimate this conversion probability. The resulting probability is used to rank books and generate transparent, rule-based explanations.

---

## Project Objective

The main objective is to build a personalized recommendation system around **shelf-to-read conversion**.

For a user \(u\), book \(b\), and fixed time horizon \(T = 90\) days:

$$
P(\text{Read within 90 days} \mid \text{User, Book, Historical Behavior})
$$

The system learns from historical interactions while preserving chronological order.

### Recommendation Pipeline

```text
Historical User Behavior
          +
     Book Information
          ↓
 Conversion Prediction
          ↓
P(Read within 90 days)
          ↓
     Book Ranking
          ↓
Explainable Recommendation
```

---

## Core Prediction Target

For an interaction with `date_added` and `read_at`:

### Target = 1

The book is recorded as read within 90 days after `date_added`.

### Target = 0

The complete 90-day observation window has passed and the book was not converted within that window.

### Immature Observations

Interactions whose complete 90-day observation window has not elapsed are **not used as supervised outcomes**.

This avoids incorrectly treating insufficiently observed interactions as negative examples.

---

## Why This Approach?

Many recommendation systems focus primarily on whether a user may like a book.

This project focuses on an additional behavioral objective:

> A book may be relevant to a user, but will the user actually convert it from a shelf addition into a read within a defined period?

The project therefore treats **shelf-to-read conversion within a fixed horizon** as the main prediction objective.

---

## Behavioral Signals

### Overall Conversion Behavior

The user's historical shelf-to-read conversion rate is:

$$
ConversionRate_{90d} = \frac {Previous books converted within 90 days} {Previous eligible books}
$$

### Genre-Specific Behavior

Users may behave differently across genres.

Example:

| Genre              | Historical Conversion Rate |
| ------------------ | -------------------------: |
| Fantasy            |                       0.82 |
| Mystery & Thriller |                       0.64 |
| Comedy             |                       0.31 |
| Science Fiction    |                       0.76 |

### Author-Specific Behavior

The system can also learn author-specific conversion behavior.

Example:

| Author   | Historical Conversion Rate |
| -------- | -------------------------: |
| Author A |                       0.90 |
| Author B |                       0.72 |
| Author C |                       0.41 |

### Historical Ratings

Ratings attached to previously interacted books are used as historical user behavior.


### Book-Length Behavior

`num_pages` can be used to represent the user's historical relationship with book length, such as the average size of previously read books.

---

## Temporal Feature Engineering

Temporal correctness is a central requirement of the project.

For a prediction event at time \(t\):

$$
X_{u,b,t} = History(u,<t)
$$

Only information available **before the current `date_added`** may be used to construct features.

The current interaction's future outcome must never be used to construct its own input features.

### Information That Must Not Be Used as Current-Row Features

* Current target
* Current `read_at`
* Current `is_read`
* Future rating for the current interaction
* Future reading information
* Future interactions of the same user

The current interaction is added to the historical state **only after its features have been calculated**.

---

## Recommendation Flow

```text
                    USER INTERACTION HISTORY
                               │
                               ▼
                       DATA PREPROCESSING
                               │
                               ▼
                  TEMPORAL BEHAVIOR
                   FEATURE ENGINEERING
                               │
                               ▼
                    USER BEHAVIOR SIGNALS
                               │
                               │
                               ▼
                       BOOK INFORMATION
                               │
                               ▼
                       CONVERSION MODEL
                               │
                               ▼
                   P(Read within 90 days)
                               │
                               ▼
                         BOOK RANKING
                               │
                               ▼
                      TOP RECOMMENDATIONS
                               │
                               ▼
                    RULE-BASED EXPLANATION
```

---

## Machine Learning

The conversion task is treated as a **binary classification** problem.

### Candidate Models

* Logistic Regression
* Random Forest
* XGBoost

Logistic Regression provides an interpretable baseline, while nonlinear models can capture more complex behavioral relationships.

The final model should be selected through experimental evaluation rather than assuming a particular algorithm will always perform best.

---

## Model Inputs

The final model combines historical user behavior with information about the book being evaluated.

### User Behavioral Information

Examples:

* Historical conversion rate
* Previous interaction counts
* Genre-specific conversion behavior
* Author-specific conversion behavior
* Historical ratings
* Historical average pages read

### Book Information

Examples:

* Number of pages
* Genre information
* Author information
* Other metadata available at recommendation time

The model outputs:

$$
P(\text{Read within 90 days})
$$

for a user-book pair.

---

## Recommendation Ranking

After the model produces conversion probabilities, books can be ranked by predicted probability.

Example:

| Book   | Predicted Conversion Probability |
| ------ | -------------------------------: |
| Book A |                             0.86 |
| Book B |                             0.72 |
| Book C |                             0.64 |
| Book D |                             0.31 |

The ranking is directly tied to the project's **90-day conversion objective**.

---

## Explainability

The system uses **rule-based explanations instead of an LLM**.

Example:

> Recommended because the user has historically converted 8 of 10 eligible Fantasy books within 90 days, has given high ratings to previous books, and the book matches their historical book-length behavior.

Explanation components are generated from calculated behavioral features, making them:

* Transparent
* Reproducible
* Auditable
* Directly connected to the recommendation logic

---

## Evaluation

### Classification Metrics

* ROC-AUC
* PR-AUC
* Log Loss
* Brier Score

## Data Processing

The data-processing pipeline includes:

1. Load interaction data
2. Clean and validate dates
3. Standardize genre information
4. Standardize author identifiers
5. Define the 90-day target
6. Exclude observations without a fully observable target
7. Build chronological behavioral features
8. Prepare data for model training

Particular attention is given to **temporal leakage** and **observation-window maturity**.

---

## Technology Stack

### Frontend

* React
* Vite
* Tailwind CSS

### Backend

* FastAPI
* Python

### Database

* PostgreSQL

### Machine Learning / Data Processing

* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Development

* Jupyter / Google Colab
* Git
* GitHub

---

## Project Structure

```text
book-recommendation-system/
│
├── README.md
├── .gitignore
├── .env.example
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── app/
│   ├── routes/
│   ├── services/
│   └── requirements.txt
│
├── ml/
│   ├── notebooks/
│   │   ├── 01_data_preparation_target_90d.ipynb
│   │   ├── 02_data_cleaning_and_genres.ipynb
│   │   └── 03_temporal_feature_engineering.ipynb
│   │
│   └── src/
│       ├── preprocessing/
│       ├── features/
│       ├── training/
│       ├── evaluation/
│       └── inference/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│
├── database/
│   ├── schema/
│   └── seed/
│
├── scripts/
├── tests/
└── docs/
```

---

## System Architecture

```text
                         ┌─────────────────────┐
                         │   User / Client     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   React Frontend    │
                         │   + Tailwind CSS    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    FastAPI API      │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼──────────────────┐
                   │                │                  │
                   ▼                ▼                  ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │ PostgreSQL   │ │ Feature      │ │ ML Model     │
           │              │ │ Engineering  │ │              │
           └──────────────┘ └──────┬───────┘ └──────┬───────┘
                                   │                  │
                                   └────────┬─────────┘
                                            ▼
                                  ┌─────────────────────┐
                                  │ Conversion          │
                                  │ Probability         │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ Book Ranking        │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ Rule-Based          │
                                  │ Explanations        │
                                  └──────────┬──────────┘
                                             │
                                             ▼
                                  ┌─────────────────────┐
                                  │ Recommendations     │
                                  └─────────────────────┘
```

---

## Development Roadmap

### Phase 1 — Data Preparation

* Clean interaction data
* Validate dates
* Standardize genres
* Standardize author IDs
* Define the 90-day target

### Phase 2 — Behavioral Modeling

* Build chronological user behavior features
* Calculate historical conversion statistics
* Calculate genre-specific behavior
* Calculate author-specific behavior
* Calculate historical rating and page signals

### Phase 3 — Model Development

* Prepare numerical ML features
* Train baseline models
* Train nonlinear models
* Compare performance
* Evaluate probability quality

### Phase 4 — Recommendation Engine

* Connect user behavior to book metadata
* Generate candidate books
* Calculate conversion probability
* Rank candidates
* Return top recommendations

### Phase 5 — Explainability

* Generate rule-based explanations
* Display relevant behavioral evidence
* Connect explanations to recommendation reasons

### Phase 6 — Application

* Build React interface
* Connect FastAPI backend
* Store application data in PostgreSQL
* Integrate the trained model
* Display personalized recommendations and explanations


---
