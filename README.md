# Behavior-Aware Personalized Book Recommendation System

## PROJECT PROPOSAL

### Behavior-Aware Personalized Book Recommendation System

**Using Shelf-to-Read Conversion Prediction**

---

## 1. Problem Statement

Traditional book recommendation systems mainly recommend books based on similarity, ratings, or popularity.

However, a book that a user may like is not always a book that the user will actually read. Our project studies historical user behavior and recommends books that both match the user's interests and have a higher probability of being converted from a saved/shelf book into a read book.

---

## 2. Completion Prediction of a Book

We use an observable outcome:

**Shelf-to-Read Conversion**

---

## 3. Project Objective

To build a personalized book recommendation system that uses each user's historical behavior to:

* Understand which genres, authors, and book lengths the user prefers.
* Measure the user's historical shelf-to-read conversion behavior.
* Predict the probability that a user will read a candidate book within a fixed time horizon **T**.
* Rank books using both preference similarity and predicted conversion probability.
* Provide simple rule-based explanations without using an LLM.

---

## 4. Dataset Attributes Available

The current dataset contains the following attributes:

| Attribute          | How We Use It                                                    |
| ------------------ | ---------------------------------------------------------------- |
| `user_id`          | Unique identifier of the user.                                   |
| `book_id`          | Unique identifier of the book.                                   |
| `is_read`          | Indicates whether the book is marked/read by the user.           |
| `rating`           | Rating given by the user; used as an explicit preference signal. |
| `started_at`       | Recorded date when the user started the book, if available.      |
| `read_at`          | Recorded date when the book was read/finished, if available.     |
| `date_added`       | Date when the user added the book to their shelf/library.        |
| `num_pages`        | Number of pages in the book.                                     |
| `authors`          | Author(s) of the book.                                           |
| `genre`            | Category information used as genre or topic information.         |
| `publication_year` | Year in which the book was published.                            |

---

## 5. Shelf-to-Read Conversion

For every user-book record, `date_added` is treated as the starting anchor.

We select a fixed horizon **T** (for example, **180 days**) and check whether the book is recorded as read within that period.

### Target

$$
P(\text{Read within T} \mid \text{User, Book, Historical Behavior})
$$

### Example

| `date_added` | `read_at`    | Days After Added |                  T = 180 Days | Target |
| ------------ | ------------ | ---------------: | ----------------------------: | -----: |
| 1 Jan        | 15 Feb       |               45 |                      Within T |      1 |
| 1 Jan        | 20 Dec       |              353 |                     Outside T |      0 |
| 1 Jan        | Not observed |                — | After full observation window |      — |

---

## 6. How We Build a Behavioral Profile for Each User

The system processes each user's historical interactions separately.

### A. Overall Conversion Behavior

For each user, calculate how many previously added books became read within **T**.

$$
Overall Conversion Rate = \frac{{Converted Books within T}} {{Eligible Books Added}}
$$

### B. Genre-Specific Behavior

For every genre, calculate how often the user converted books in that genre into reads.

For example, a user may convert **Fantasy** books much more often than **Suspense** books.

### C. Rating Preference

Use historical ratings to identify what the user liked.

Features can include:

* Average rating by genre.
* Average rating for previously read books.

### D. Author Preference

Count previous interactions with authors and identify authors whose books the user has frequently read or rated highly.

### E. Book-Length Preference

Using `num_pages`, identify the book-length ranges that the user historically converts and rates highly.

---

## 7. Example: One User

Suppose User **U1** has the following historical profile:

* Overall shelf-to-read conversion rate: **70%**
* Fantasy conversion rate: **85%**
* Mystery conversion rate: **70%**
* Suspense conversion rate: **30%**
* Preferred book length: approximately **200–450 pages**
* Fantasy average rating: **4.6/5**
* Frequently interacted with **Author A** and **Author B**

If a new candidate book is a **Fantasy** book, around **350 pages**, and written by a preferred author, it receives a high behavioral and preference score.

---

## 8. Recommendation Method

### Step 1: Candidate Generation

Generate a set of candidate books using content-based similarity from:

* Genre
* Authors
* Other book metadata

### Step 2: Preference Score

Calculate how well each candidate matches the user's historical interests, such as:

* Genre preference
* Author preference
* Rating history
* Book-length preference

**Content-Based Filtering**

### Step 3: Conversion Prediction

For each user-candidate pair, create behavioral features from the user's past history and predict:

**Probability that the user will read the book within T — Logistic Regression**

### Step 4: Final Ranking

Combine the user's preference score with the predicted conversion probability.

$$
\text{Final Score}
={XGBoost}({Preference Score},{Conversion Probability})
$$

Please correc this

The top-ranked books become the final personalized recommendations.

---

## 9. Application / Model Flow

```text
RAW DATASET
    ↓
Data Cleaning and Date Parsing
    ↓
Create Shelf-to-Read Target Using date_added and Horizon T
    ↓
Build Historical Profile for Each User
    ↓
Candidate Book Generation
    ↓
Preference Scoring
    ↓
Conversion Probability Prediction
    ↓
Combine Scores and Rank Books
    ↓
Top-N Personalized Recommendations
    ↓
Rule-Based "Why This Book?" Explanation
```

---

## 10. Explainability

Explanations will be produced using transparent rules based directly on calculated features.

### Example

> "Recommended because you have historically converted Fantasy books at a high rate, rated similar books highly, and this book matches your preferred book-length range."

---

## 11. Project Novelty

* The system does not only recommend books that look similar to a user's interests.
* It models whether the user historically converts added/saved books into actual reads.
* It learns different behavioral patterns for each user, including genre, author, rating, and book-length behavior.
* Recommendations are explainable through transparent rules and features.

---

## 12. Why Users May Use This Instead of Asking an LLM

An LLM can provide general book suggestions based on a prompt.

Our system is designed to work with a persistent history of user-book interactions and calculate measurable, user-specific behavioral patterns.

Its main purpose is not simply to suggest a good book, but to rank books based on both:

1. Personal preference
2. Probability score

---

## 13. Short Proposal Summary

Our project is a **behavior-aware personalized book recommendation system**.

Based on the available dataset, we do not use the traditional completion-rate definition because Goodreads-style data does not reliably show which books were started and then abandoned.

Instead, we define an observable target called **shelf-to-read conversion**.

Using `date_added` as the anchor and a horizon **T**, we predict whether a user will read a book within that period.

For every user, we build a historical profile using:

* Overall conversion behavior
* Genre-specific conversion
* Ratings
* Author preference
* Book-length preference

Candidate books are generated using book metadata, then ranked using both **preference score** and **predicted conversion probability**.

The recommendation explanation is **rule-based**.
