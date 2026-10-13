"""Section C - Mini Capstone: Food Delivery Review Intelligence System (M11-A1, NLP)

Console app: add reviews, classify them (sentiment + issue category) and view a summary.
Run:  python review_intelligence_system.py
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Folder that holds this project (works both as a .py script and inside Jupyter)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = BASE_DIR / "food_delivery_reviews.csv"


# ---------------------------------------------------------------
# TEXT PREPROCESSING PIPELINE
# lowercase -> expand contractions -> remove punctuation/digits/emoji
# -> tokenise -> stopword removal -> stemming
# ---------------------------------------------------------------
# Negation words are KEPT: "not good" must not become "good" (it would flip the sentiment).
NEGATIONS = {"no", "not", "nor", "never", "cannot", "nothing", "none", "nobody", "neither"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS
STEMMER = PorterStemmer()


def clean_tokens(text):
    """Steps 1-5: returns the content words (lowercased, no punctuation, no stopwords)."""
    text = str(text).lower()                                  # 1. lowercase
    text = text.replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)                     # 2. didn't -> did not
    text = re.sub(r"'(s|m|re|ve|ll|d)\b", "", text)           #    it's -> it, I'm -> I
    text = re.sub(r"[^a-z\s]", " ", text)                     # 3. drop punctuation, digits, emojis
    tokens = text.split()                                     # 4. tokenise
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]   # 5. stopword removal


def preprocess(text):
    """Full pipeline: returns (content_words, stemmed_text)."""
    words = clean_tokens(text)
    stems = [STEMMER.stem(w) for w in words]                  # 6. stemming
    return words, " ".join(stems)


# ---------------------------------------------------------------
# MODELS: TF-IDF + classifier, one pipeline per output
#   sentiment      -> Logistic Regression (Positive / Negative)
#   issue category -> Multinomial Naive Bayes (Delivery / Food Quality / App / General)
# ---------------------------------------------------------------
MIN_TRAIN_SAMPLES = 20          # below this, classification is disabled with a clear message
REQUIRED_COLUMNS = {"review_text", "sentiment", "issue_category"}


def make_sentiment_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=10)),
    ])


def make_category_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def load_training_data(path):
    """Read the labelled CSV. Returns a DataFrame, or None if the file is missing/unreadable."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


# ---------------------------------------------------------------
# THE APPLICATION CORE
# ---------------------------------------------------------------
SENTIMENTS = ["Positive", "Negative"]
CATEGORIES = ["Delivery", "Food Quality", "App", "General"]


class InvalidReview(ValueError):
    """The review text is empty or contains no usable words."""


class ClassifierUnavailable(RuntimeError):
    """Not enough labelled training data to classify yet."""


class ReviewIntelligenceSystem:
    def __init__(self, train_df=None, min_samples=MIN_TRAIN_SAMPLES):
        self.min_samples = min_samples
        self.reviews = []                      # stored, pre-processed reviews
        self.ready = False
        self.status = ""
        self.train(train_df)

    # ---------- training ----------
    def train(self, df):
        """Train both models. Never raises: sets self.ready / self.status instead."""
        self.ready = False
        if df is None:
            self.status = "Training data could not be loaded."
            return
        if not REQUIRED_COLUMNS.issubset(df.columns):
            self.status = f"Training data must contain the columns: {sorted(REQUIRED_COLUMNS)}."
            return
        df = df.dropna(subset=list(REQUIRED_COLUMNS))
        if len(df) < self.min_samples:
            self.status = (f"Only {len(df)} labelled training samples found; "
                           f"at least {self.min_samples} are needed to classify reviews.")
            return
        if df["sentiment"].nunique() < 2 or df["issue_category"].nunique() < 2:
            self.status = "Training data needs at least 2 different sentiment labels and 2 issue categories."
            return

        X = df["review_text"].map(lambda t: preprocess(t)[1])
        keep = X.str.len() > 0
        X, df = X[keep], df[keep]
        self.sentiment_model = make_sentiment_model().fit(X, df["sentiment"])
        self.category_model = make_category_model().fit(X, df["issue_category"])
        self.n_train = len(df)
        self.ready = True
        self.status = f"Models trained on {len(df)} labelled reviews."

    # ---------- classification ----------
    def classify(self, raw_text):
        """Return a dict with sentiment + category (+ confidences). Raises on bad input."""
        words, stemmed = preprocess(raw_text)
        if not stemmed:
            raise InvalidReview("The review has no usable words after preprocessing "
                                "(it was empty or only punctuation, numbers or emojis).")
        if not self.ready:
            raise ClassifierUnavailable(self.status)
        s_probs = self.sentiment_model.predict_proba([stemmed])[0]
        c_probs = self.category_model.predict_proba([stemmed])[0]
        return {
            "sentiment": self.sentiment_model.classes_[s_probs.argmax()],
            "sentiment_conf": float(s_probs.max()),
            "category": self.category_model.classes_[c_probs.argmax()],
            "category_conf": float(c_probs.max()),
        }

    # ---------- storing ----------
    def add_review(self, raw_text):
        """Preprocess, classify (if possible) and store a review. Returns the stored record."""
        words, stemmed = preprocess(raw_text)
        if not stemmed:
            raise InvalidReview("The review has no usable words after preprocessing "
                                "(it was empty or only punctuation, numbers or emojis).")
        record = {"id": len(self.reviews) + 1, "raw": raw_text.strip(),
                  "words": words, "stemmed": stemmed,
                  "sentiment": "Unclassified", "category": "Unclassified"}
        if self.ready:
            res = self.classify(raw_text)
            record["sentiment"], record["category"] = res["sentiment"], res["category"]
            record["sentiment_conf"], record["category_conf"] = res["sentiment_conf"], res["category_conf"]
        self.reviews.append(record)
        return record

    # ---------- report ----------
    def summary(self, top_n=5):
        sent = Counter(r["sentiment"] for r in self.reviews)
        cat = Counter(r["category"] for r in self.reviews)
        words = Counter(w for r in self.reviews for w in r["words"])
        return {
            "total": len(self.reviews),
            "sentiments": {k: sent.get(k, 0) for k in SENTIMENTS + (["Unclassified"] if sent.get("Unclassified") else [])},
            "categories": {k: cat.get(k, 0) for k in CATEGORIES + (["Unclassified"] if cat.get("Unclassified") else [])},
            "top_words": words.most_common(top_n),
        }


# ---------------------------------------------------------------
# CONSOLE MENU
# ---------------------------------------------------------------
def show_result(res):
    print(f"   Sentiment     : {res['sentiment']}  (confidence {res['sentiment_conf']:.0%})")
    print(f"   Issue category: {res['category']}  (confidence {res['category_conf']:.0%})")


def option_add(system, input_fn):
    text = input_fn("Enter the customer review: ").strip()
    try:
        rec = system.add_review(text)
    except InvalidReview as e:
        print(f"   Could not add review: {e}")
        return
    print(f"   Review #{rec['id']} added.")
    print(f"   Stored (pre-processed): {rec['stemmed']}")
    if system.ready:
        show_result(rec)
    else:
        print(f"   Note: not classified yet. {system.status}")


def option_classify(system, input_fn):
    text = input_fn("Enter review text (or a stored review number): ").strip()
    if text.isdigit():                                    # classify an already-stored review
        idx = int(text)
        if not 1 <= idx <= len(system.reviews):
            print(f"   No stored review with number {idx}. Stored reviews: {len(system.reviews)}.")
            return
        text = system.reviews[idx - 1]["raw"]
        print(f"   Review #{idx}: {text}")
    try:
        show_result(system.classify(text))
    except InvalidReview as e:
        print(f"   Cannot classify: {e}")
    except ClassifierUnavailable as e:
        print(f"   Classification is not available. {e}")


def option_report(system):
    s = system.summary()
    print("\n" + "=" * 46)
    print("  REVIEW SUMMARY REPORT")
    print("=" * 46)
    print(f"Total reviews added: {s['total']}")
    if s["total"] == 0:
        print("No reviews yet. Use option 1 to add some.")
        return
    print("\nBy sentiment:")
    for k, v in s["sentiments"].items():
        print(f"   {k:<14}{v}")
    print("\nBy issue category:")
    for k, v in s["categories"].items():
        print(f"   {k:<14}{v}")
    print("\nTop 5 most frequent content words:")
    for rank, (w, n) in enumerate(s["top_words"], start=1):
        print(f"   {rank}. {w:<14}{n}")


def main(system=None, input_fn=input):
    if system is None:
        system = ReviewIntelligenceSystem(load_training_data(DATA_PATH))
    print(f"[{system.status}]")
    while True:
        print("\n=== Food Delivery Review Intelligence System ===")
        print("1. Add a new review")
        print("2. Classify a review")
        print("3. View summary report")
        print("4. Exit")
        try:
            choice = input_fn("Choose an option (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInput closed. Goodbye!")
            break
        if choice == "1":
            option_add(system, input_fn)
        elif choice == "2":
            option_classify(system, input_fn)
        elif choice == "3":
            option_report(system)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")


if __name__ == "__main__":
    main()
