from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]
X_val = dataset["validation"]["text"]
y_val = dataset["validation"]["label"]
X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

# 2. Build the single optimized pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),      # Looks at single words and 2-word phrases ("not good")
        max_features=30000,      # Keeps the 30,000 most important words
        stop_words='english',    # Removes useless words ("the", "and", "is")
        sublinear_tf=True,       # Stops long reviews from dominating the math
        min_df=2                 # Ignores typos that appear only once
    )),
    ('clf', LogisticRegression(max_iter=2000, C=1.0))
])

# 3. Train the model
print("Training the TF-IDF + Logistic Regression model...")
model.fit(X_train, y_train)

# 4. Evaluate on Validation and Test data
val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

print("\n--- RESULTS ---")
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, test_predictions) * 100:.2f}%")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_predictions, target_names=["Negative", "Positive"]))