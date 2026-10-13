# Section C — Food Delivery Review Intelligence System (M11-A1, NLP)

A console-based NLP application: staff add customer reviews, each review is preprocessed and classified by
**sentiment** (Positive / Negative) and **issue category** (Delivery / Food Quality / App / General), and a
summary report is produced.

## Files
| File | Purpose |
|---|---|
| `Section_C_Review_Intelligence_System.ipynb` | The project as a Jupyter notebook: dataset analysis, preprocessing, model comparison, the application, demo, edge cases, requirement checks |
| `review_intelligence_system.py` | The same application as a stand-alone program (`python review_intelligence_system.py`) |
| `food_delivery_reviews.csv` | Labelled training dataset (320 reviews) |
| `requirements.txt` | Python packages needed |

## Dataset (`food_delivery_reviews.csv`)
| Column | Meaning |
|---|---|
| `review_id` | Row number |
| `review_text` | Raw customer review |
| `sentiment` | `Positive` / `Negative` (50 % each) |
| `issue_category` | `Delivery` / `Food Quality` / `App` / `General` (25 % each) |
| `source` | `handwritten` (80) or `generated` (240, built from sentiment openers + category sentences) |

Every sentiment x category combination has 40 reviews. The data is **synthetic**, created for this assessment;
swap in real labelled data with the same columns and nothing else needs to change.

## How to run
1. `pip install -r requirements.txt`
2. Notebook: `jupyter notebook`, open the `.ipynb`, choose *Kernel > Restart & Run All*.
   Terminal: `python review_intelligence_system.py`

No NLTK data download is needed (only the Porter stemmer, which ships with the package).

## Menu
1. Add a new review - preprocesses, classifies and stores it
2. Classify a review - type new text, or a stored review number
3. View summary report - total, per sentiment, per category, top 5 content words
4. Exit

## Approach
Preprocessing: lowercase -> expand contractions -> remove punctuation/digits/emojis -> tokenise -> stopword removal
(negations such as "not" are kept) -> Porter stemming.
Models: TF-IDF (unigrams + bigrams) -> Logistic Regression (sentiment) and Multinomial Naive Bayes (issue category).
Edge cases: fewer than 20 training samples, missing/invalid training file, one-class data, empty or emoji-only
reviews and closed input all print a clear message instead of crashing.
